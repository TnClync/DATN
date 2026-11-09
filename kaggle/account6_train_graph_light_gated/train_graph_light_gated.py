#!/usr/bin/env python
"""Kaggle: Train Graph Light Gated (single model). Loads LightGCN+SBERT from graph kernel output."""
import os, sys, time, json, math, gc, subprocess
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass
from pathlib import Path
try:
    import torch, faiss, pyarrow.parquet as pq, implicit, sentence_transformers
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet",
        "sentence-transformers>=2.7", "faiss-cpu", "implicit>=0.7", "pyarrow", "scipy", "tqdm"])
    import torch, faiss, pyarrow.parquet as pq, implicit, sentence_transformers
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

MIN_INTERACTIONS = 3
CATALOG = 2464371
SEED = 42
OUT_DIR = Path("/kaggle/working" if Path("/kaggle/working").exists() else "artifacts")
OUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EVAL_BATCH_USERS = 128   # eval: fuse this many users in ONE forward pass


def find_base_dir():
    for c in [Path("/kaggle/input/lastone/amazon_recs_output"), Path("/kaggle/input/lastone"),
              Path("/kaggle/input/datasets/nguynteo/lastone"), Path("data")]:
        if (c / "train").exists() or (c / "test").exists():
            return c
    if Path("/kaggle/input").exists():
        for p in Path("/kaggle/input").rglob("train"):
            if p.is_dir():
                return p.parent
    return Path("data")


def find_file(name):
    hits = sorted(Path("/kaggle/input").rglob(name))
    # prefer graph kernel output
    pref = [p for p in hits if "train-lightgcn" in str(p) or "graph-deep-hybrid" in str(p) or "deep-hybrid-recommender" in str(p)]
    return (pref[0] if pref else hits[0]) if hits else None

def find_sbert_file(dim=768):
    """Locate the 768-d combined SBERT artifact (older kernels also wrote 384-d copies)."""
    hits = sorted(Path("/kaggle/input").rglob("sbert_item_embeddings_f16.npy"))
    hits.sort(key=lambda p: 0 if "train-sbert" in str(p) else 1)
    for p in hits:
        try:
            a = np.load(p, mmap_mode="r")
            if a.ndim == 2 and a.shape[1] == dim:
                return p
        except Exception:
            continue
    return None


BASE_DIR = find_base_dir()

# ---- Load factors + SBERT + graph embeddings from kernel outputs ----
user_cf = np.load(find_file("als_user_factors_f16.npy")).astype(np.float32)
item_cf = np.load(find_file("als_item_factors_f16.npy")).astype(np.float32)
graph_user = np.load(find_file("graph_user_factors_f16.npy")).astype(np.float32)
graph_item = np.load(find_file("graph_item_factors_f16.npy")).astype(np.float32)
sbert_path = find_sbert_file()
assert sbert_path is not None, "768-d sbert_item_embeddings_f16.npy not found - run the SBERT kernel first"
item_sbert = np.asarray(np.load(sbert_path, mmap_mode="r")[:CATALOG], dtype=np.float32)
print(f"SBERT item embeddings: {item_sbert.shape} | {sbert_path}")
u_ids = np.load(find_file("user_ids.npy"))
i_ids = np.load(find_file("item_ids.npy"))
n_users, n_items = len(u_ids), len(i_ids)
SBERT_COVERAGE = n_items

# ---- Train history + test ----
train_files = sorted(BASE_DIR.glob("train/*.parquet"))
dfs = []
for f in train_files:
    d = pq.read_table(f, columns=["user_idx", "item_idx", "rating"]).to_pandas()
    d = d[d["rating"] >= 4.0]
    dfs.append(d[["user_idx", "item_idx"]])
train_df = pd.concat(dfs, ignore_index=True)
user_counts = train_df["user_idx"].value_counts()
active = user_counts[user_counts >= MIN_INTERACTIONS].index
train_df = train_df[train_df["user_idx"].isin(active)]
u_raw = train_df["user_idx"].to_numpy(np.int64)
i_raw = train_df["item_idx"].to_numpy(np.int64)
u_uniq, u_inv = np.unique(u_raw, return_inverse=True)
i_uniq, i_inv = np.unique(i_raw, return_inverse=True)

test_files = sorted(BASE_DIR.glob("test/*.parquet"))
dfs_test = []
for f in test_files:
    t = pq.read_table(f, columns=["user_idx", "item_idx", "rating"])
    d = t.to_pandas()
    d = d[d["rating"] >= 4.0]
    dfs_test.append(d[["user_idx", "item_idx"]])
test_df = pd.concat(dfs_test, ignore_index=True)
test_df = test_df[test_df["user_idx"].isin(set(u_uniq.tolist()))]
u_map = {orig: ci for ci, orig in enumerate(u_uniq)}
i_map = {orig: ci for ci, orig in enumerate(i_uniq)}
test_df["cu"] = test_df["user_idx"].map(u_map)
test_df["ci"] = test_df["item_idx"].map(i_map)
test_df = test_df.dropna(subset=["cu", "ci"])
test_df["cu"] = test_df["cu"].astype(np.int64)
test_df["ci"] = test_df["ci"].astype(np.int64)
user_gt = test_df.groupby("cu")["ci"].apply(set).to_dict()

hist_by_user_all = {}
for uu, ii in zip(u_inv, i_inv):
    if ii < SBERT_COVERAGE:
        hist_by_user_all.setdefault(int(uu), []).append(int(ii))

user_sbert = {}
for u in range(len(u_uniq)):
    h = hist_by_user_all.get(u, [])
    if h:
        v = item_sbert[h].mean(axis=0)
        nrm = float(np.linalg.norm(v))
        v = v / nrm if nrm > 1e-8 else np.zeros(768, dtype=np.float32)
    else:
        v = np.zeros(768, dtype=np.float32)
    user_sbert[u] = v.astype(np.float32)

user_inter_count = np.bincount(u_inv, minlength=len(u_uniq))


def compute_metrics(ranked, gt):
    res = {k: 0.0 for k in ["HR@5", "HR@10", "HR@20", "Recall@5", "Recall@10", "Recall@20",
                             "NDCG@5", "NDCG@10", "NDCG@20", "MRR"]}
    pos_list = np.where(np.asarray(ranked) == gt)[0]
    if len(pos_list) == 0:
        return res
    pos = int(pos_list[0])
    if pos < 5: res["HR@5"] = 1.0
    if pos < 10: res["HR@10"] = 1.0
    if pos < 20: res["HR@20"] = 1.0
    if pos < 5: res["Recall@5"] = 1.0
    if pos < 10: res["Recall@10"] = 1.0
    if pos < 20: res["Recall@20"] = 1.0
    if pos < 5: res["NDCG@5"] = 1.0 / math.log2(pos + 2)
    if pos < 10: res["NDCG@10"] = 1.0 / math.log2(pos + 2)
    if pos < 20: res["NDCG@20"] = 1.0 / math.log2(pos + 2)
    res["MRR"] = 1.0 / (pos + 1)
    return res


def mm(x):
    x = np.asarray(x, dtype=np.float32)
    mn, mx = x.min(), x.max()
    if mx - mn < 1e-8:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)


def cf_scores(uf, itf, u, cands):
    return np.asarray(itf[cands] @ uf[u], dtype=np.float32)


def sb_scores(u, cands):
    return np.asarray(item_sbert[cands] @ user_sbert[u], dtype=np.float32)


def build_instances(eval_users):
    out = []
    rng = np.random.default_rng(42)
    CAND_POOL = np.arange(SBERT_COVERAGE)
    for u in eval_users:
        u = int(u)
        gts = [it for it in user_gt[u] if it < SBERT_COVERAGE]
        if not gts:
            continue
        gt = int(rng.choice(gts))
        banned = (set(hist_by_user_all.get(u, [])) | set(user_gt[u])) - {gt}
        negs = []
        for _ in range(10):
            draw = rng.choice(CAND_POOL, size=300, replace=False)
            negs = [int(x) for x in draw if int(x) not in banned][:99]
            if len(negs) == 99:
                break
        if len(negs) < 99:
            continue
        cands = np.array([gt] + negs, dtype=np.int64)
        rng.shuffle(cands)
        out.append({"user": u, "gt": gt, "candidates": cands,
                    "banned": np.array(sorted(banned), dtype=np.int64)})
    return out


test_users = np.array([u for u in user_gt.keys() if user_inter_count[u] >= MIN_INTERACTIONS])
test_users = np.array([u for u in test_users if any(it < SBERT_COVERAGE for it in user_gt[u])])
eval_users = np.random.default_rng(42).choice(test_users, size=min(50000, len(test_users)), replace=False)
eval_instances = build_instances(eval_users)
print(f"Prepared {len(eval_instances):,} instances")
TEST_N = 5000
test_inst = eval_instances[:TEST_N]
val_inst = eval_instances[TEST_N:TEST_N + 2000]
train_for_fit = eval_instances[TEST_N + 2000:TEST_N + 2000 + 50000]


def eval_scorer(fn, instances):
    agg = {k: 0.0 for k in ["HR@5", "HR@10", "HR@20", "Recall@5", "Recall@10", "Recall@20",
                             "NDCG@5", "NDCG@10", "NDCG@20", "MRR"]}
    for inst in instances:
        scores = fn(inst["user"], inst["candidates"])
        ranked = inst["candidates"][np.argsort(-scores)]
        m = compute_metrics(ranked, inst["gt"])
        for k in agg:
            agg[k] += m[k]
    return {k: round(v / max(1, len(instances)), 4) for k, v in agg.items()}


# ---- Baselines ----
print("\n[STEP 5/8] BASELINES")
res_als = eval_scorer(lambda u, c: cf_scores(user_cf, item_cf, u, c), test_inst)
print("ALS:", res_als)
res_graph_cf = eval_scorer(lambda u, c: cf_scores(graph_user, graph_item, u, c), test_inst)
print("LightGCN:", res_graph_cf)
res_sbert = eval_scorer(lambda u, c: sb_scores(u, c), test_inst)
print("SBERT:", res_sbert)


# ---- Model: Graph Light Gated ----
print("\n[STEP 6/8] TRAIN: GRAPH LIGHT GATED")
class LightGate(nn.Module):
    def __init__(self, in_features=32 + 32 + 768):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(in_features, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 1), nn.Sigmoid())
    def forward(self, u_cf, i_cf, i_sbert, cf_score, sbert_score):
        feat = torch.cat([u_cf, i_cf, i_sbert], dim=-1)
        alpha = self.gate_net(feat).squeeze(-1)
        return alpha * cf_score + (1.0 - alpha) * sbert_score, alpha

gate = LightGate().to(DEVICE)
gate_Xu, gate_Xi, gate_Xs, gate_cf, gate_sb, gate_y = [], [], [], [], [], []
for inst in train_for_fit:
    u = inst["user"]; pos = inst["gt"]
    negs = [int(x) for x in inst["candidates"] if int(x) != pos][:4]
    for it, label in [(pos, 1.0)] + [(n, 0.0) for n in negs]:
        gate_Xu.append(graph_user[u]); gate_Xi.append(graph_item[it]); gate_Xs.append(item_sbert[it])
        gate_cf.append(float(graph_item[it] @ graph_user[u]))
        gate_sb.append(float(item_sbert[it] @ user_sbert[u]))
        gate_y.append(label)
t_u = torch.tensor(np.array(gate_Xu), dtype=torch.float32)
t_i = torch.tensor(np.array(gate_Xi), dtype=torch.float32)
t_s = torch.tensor(np.array(gate_Xs), dtype=torch.float32)
t_cf = torch.tensor(np.array(gate_cf), dtype=torch.float32)
t_sb = torch.tensor(np.array(gate_sb), dtype=torch.float32)
t_y = torch.tensor(np.array(gate_y), dtype=torch.float32)
ds = TensorDataset(t_u, t_i, t_s, t_cf, t_sb, t_y)
loader = DataLoader(ds, batch_size=512, shuffle=True, num_workers=0)
opt = torch.optim.Adam(gate.parameters(), lr=1e-3, weight_decay=1e-5)
bce = nn.BCEWithLogitsLoss()
gate.train()
for ep in range(30):
    tot = 0.0
    for bu, bi, bs, bcf, bsb, by in loader:
        bu, bi, bs = bu.to(DEVICE), bi.to(DEVICE), bs.to(DEVICE)
        bcf, bsb, by = bcf.to(DEVICE), bsb.to(DEVICE), by.to(DEVICE)
        pred, _ = gate(bu, bi, bs, bcf, bsb)
        loss = bce(pred, by)
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item()
    print(f"Gate Epoch {ep+1}/30 Loss: {tot/len(loader):.4f}")
torch.save(gate.state_dict(), OUT_DIR / "graph_light_gated_fusion.pt")


def eval_gate(instances, batch_users=EVAL_BATCH_USERS):
    """Batched eval: fuse B users x C candidates in ONE forward pass.

    Gate is pointwise over (user, item), so scores are identical to the
    per-user loop.
    """
    gate.eval()
    keys = ["HR@5", "HR@10", "HR@20", "Recall@5", "Recall@10", "Recall@20",
            "NDCG@5", "NDCG@10", "NDCG@20", "MRR"]
    agg = {k: 0.0 for k in keys}
    n = len(instances)
    n_batches = (n + batch_users - 1) // batch_users
    t0 = time.time()
    with torch.no_grad():
        for bi, start in enumerate(range(0, n, batch_users)):
            batch = instances[start:start + batch_users]
            b = len(batch)
            c = len(batch[0]["candidates"])
            users = np.array([inst["user"] for inst in batch], dtype=np.int64)
            cands = np.stack([inst["candidates"] for inst in batch])           # (B, C)

            i_graph = graph_item[cands]                                        # (B, C, 32)
            i_sb = item_sbert[cands]                                           # (B, C, 768)
            u_graph = graph_user[users]                                        # (B, 32)
            u_sb = np.stack([user_sbert[int(u)] for u in users])               # (B, 768)

            bcf = np.einsum("bcd,bd->bc", i_graph, u_graph, optimize=True)
            bsb = np.einsum("bcd,bd->bc", i_sb, u_sb, optimize=True)

            fused, _ = gate(
                torch.from_numpy(np.repeat(u_graph[:, None, :], c, axis=1)).to(DEVICE),
                torch.from_numpy(i_graph).to(DEVICE),
                torch.from_numpy(i_sb).to(DEVICE),
                torch.from_numpy(bcf.astype(np.float32)).to(DEVICE),
                torch.from_numpy(bsb.astype(np.float32)).to(DEVICE),
            )
            scores = fused.cpu().numpy()
            for row in range(b):
                ranked = cands[row][np.argsort(-scores[row])]
                m = compute_metrics(ranked, batch[row]["gt"])
                for k in keys:
                    agg[k] += m[k]
            if bi % 10 == 0 or bi == n_batches - 1:
                print(f"  eval batch {bi + 1}/{n_batches} "
                      f"({start + b:,}/{n:,} users) {time.time() - t0:.1f}s", flush=True)
    print(f"  eval done: {n:,} users in {time.time() - t0:.1f}s "
          f"({n / max(time.time() - t0, 1e-9):,.0f} users/s)", flush=True)
    return {k: round(v / max(1, n), 4) for k, v in agg.items()}


res_gate = eval_gate(test_inst)
print("Graph Light Gated:", res_gate)

summary = {
    "model": "graph_light_gated",
    "n_eval_users": len(test_inst),
    "min_interactions": MIN_INTERACTIONS,
    "als": res_als,
    "lightgcn": res_graph_cf,
    "sbert": res_sbert,
    "graph_light_gated": res_gate,
}
(OUT_DIR / "graph_light_gated_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
print("DONE", flush=True)
print("[ARTIFACT SAVED]", flush=True)
