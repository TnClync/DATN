#!/usr/bin/env python
"""Kaggle: Train Light Gated Late Fusion (single model).

Deep hybrid: neural gate learns per-(user,item) fusion weight between
ALS (CF) score and SBERT (content) score.
"""
import os, sys, time, json, math, gc, subprocess
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
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
from sentence_transformers import SentenceTransformer
# tqdm removed for clearer Kaggle logs

MIN_INTERACTIONS = 3
CATALOG = 2464371
SEED = 42
OUT_DIR = Path("/kaggle/working" if Path("/kaggle/working").exists() else "artifacts")
OUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EVAL_BATCH_USERS = 128   # eval: fuse this many users in ONE forward pass

_T0 = time.time()

def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')} +{time.time() - _T0:7.1f}s] {msg}", flush=True)

def mem_info() -> str:
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return f"{rss / 1e9:.1f} GB" if rss > 1e9 else f"{rss / 1e6:.1f} MB"
    except Exception:
        return "n/a"

def gpu_mem() -> str:
    if not torch.cuda.is_available():
        return "CPU"
    return " | ".join(f"GPU{i} alloc={torch.cuda.memory_allocated(i)/1e9:.1f}G res={torch.cuda.memory_reserved(i)/1e9:.1f}G" for i in range(torch.cuda.device_count()))

log(f"CONFIG: MIN_INTERACTIONS={MIN_INTERACTIONS} EVAL_BATCH_USERS={EVAL_BATCH_USERS} DEVICE={DEVICE} | {gpu_mem()}")


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


def find_factor_file(filename):
    hits = sorted(Path("/kaggle/input").rglob(filename))
    return hits[0] if hits else None

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
log(f"Dataset: {BASE_DIR}")


# ---- Load factors + SBERT from deep-hybrid kernel output ----
def load(name):
    up = find_factor_file(f"{name}_user_factors_f16.npy")
    ip = find_factor_file(f"{name}_item_factors_f16.npy")
    if up is None or ip is None:
        return None
    return np.load(up).astype(np.float32), np.load(ip).astype(np.float32)


als = load("als")
assert als is not None, "ALS factors not found"
user_cf, item_cf = als
log(f"ALS factors: {user_cf.shape} / {item_cf.shape} | RAM {mem_info()}")

# ---- Compact indices (same protocol as the SBERT kernel) ----
train_files = sorted(BASE_DIR.glob("train/*.parquet"))
log(f"[STEP 2] Reading {len(train_files)} train parquet file(s)...")
dfs = []
for i, f in enumerate(train_files, 1):
    log(f"[STEP 2]   train {i}/{len(train_files)}: {f.name}")
    d = pq.read_table(f, columns=["user_idx", "item_idx", "rating"]).to_pandas()
    d = d[d["rating"] >= 4.0]
    dfs.append(d[["user_idx", "item_idx"]])
    log(f"[STEP 2]   -> kept {len(dfs[-1]):,} positive rows")
train_df = pd.concat(dfs, ignore_index=True)
user_counts = train_df["user_idx"].value_counts()
active = user_counts[user_counts >= MIN_INTERACTIONS].index
train_df = train_df[train_df["user_idx"].isin(active)]
u_raw = train_df["user_idx"].to_numpy(np.int64)
i_raw = train_df["item_idx"].to_numpy(np.int64)
u_uniq, u_inv = np.unique(u_raw, return_inverse=True)
i_uniq, i_inv = np.unique(i_raw, return_inverse=True)
n_users, n_items = len(u_uniq), len(i_uniq)
SBERT_COVERAGE = min(n_items, CATALOG)
compact_to_orig = {idx: int(i_uniq[idx]) for idx in range(n_items)}

# ---- SBERT item embeddings: 768-d (metadata | reviewer), built once by the SBERT kernel ----
sbert_path = find_sbert_file()
assert sbert_path is not None, "sbert_item_embeddings_f16.npy not found - run the SBERT kernel first"
sbert_rows = np.load(sbert_path, mmap_mode="r").shape[0]
assert sbert_rows >= n_items, f"SBERT rows {sbert_rows:,} < items {n_items:,} - index mismatch"
item_sbert = np.asarray(np.load(sbert_path, mmap_mode="r")[:n_items], dtype=np.float32)
assert item_sbert.shape[1] == 768, f"expected 768-d SBERT, got {item_sbert.shape}"
log(f"SBERT item embeddings: {item_sbert.shape} | RAM {mem_info()}")

# User ids
np.save(OUT_DIR / "user_ids.npy", u_uniq)
np.save(OUT_DIR / "item_ids.npy", i_uniq)
np.save(OUT_DIR / "als_user_factors_f16.npy", user_cf.astype(np.float16))
np.save(OUT_DIR / "als_item_factors_f16.npy", item_cf.astype(np.float16))

# ---- Test + history ----
test_files = sorted(BASE_DIR.glob("test/*.parquet"))
log(f"Reading {len(test_files)} test parquet file(s)...")
dfs_test = []
for i, f in enumerate(test_files, 1):
    log(f"  test {i}/{len(test_files)}: {f.name}")
    t = pq.read_table(f, columns=["user_idx", "item_idx", "rating"])
    d = t.to_pandas()
    d = d[d["rating"] >= 4.0]
    dfs_test.append(d[["user_idx", "item_idx"]])
    log(f"  -> kept {len(dfs_test[-1]):,} positive rows")
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
for u in range(n_users):
    h = hist_by_user_all.get(u, [])
    if h:
        v = item_sbert[h].mean(axis=0)
        nrm = float(np.linalg.norm(v))
        v = v / nrm if nrm > 1e-8 else np.zeros(768, dtype=np.float32)
    else:
        v = np.zeros(768, dtype=np.float32)
    user_sbert[u] = v.astype(np.float32)


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


CAND_POOL = np.arange(SBERT_COVERAGE)
user_inter_count = np.bincount(u_inv, minlength=len(u_uniq))
test_users = np.array([u for u in user_gt.keys() if user_inter_count[u] >= MIN_INTERACTIONS])
test_users = np.array([u for u in test_users if any(it < SBERT_COVERAGE for it in user_gt[u])])
eval_users = np.random.default_rng(42).choice(test_users, size=min(50000, len(test_users)), replace=False)
eval_instances = build_instances(eval_users)
log(f"Prepared {len(eval_instances):,} instances | RAM {mem_info()}")
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
print("\n" + "=" * 50)
print("[STEP 5/7] BASELINES")
print("=" * 50)
res_als = eval_scorer(lambda u, c: cf_scores(user_cf, item_cf, u, c), test_inst)
log(f"[BASELINE] ALS: {res_als}")
res_sbert = eval_scorer(lambda u, c: sb_scores(u, c), test_inst)
log(f"[BASELINE] SBERT: {res_sbert}")


# ---- Model: Light Gated Late Fusion ----
print("\n" + "=" * 50)
print("[STEP 6/7] TRAIN: LIGHT GATED LATE FUSION")
print("=" * 50)


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
        gate_Xu.append(user_cf[u]); gate_Xi.append(item_cf[it]); gate_Xs.append(item_sbert[it])
        gate_cf.append(float(item_cf[it] @ user_cf[u]))
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
_t_train = time.time()
for ep in range(30):
    tot = 0.0
    for bu, bi, bs, bcf, bsb, by in loader:
        bu, bi, bs = bu.to(DEVICE), bi.to(DEVICE), bs.to(DEVICE)
        bcf, bsb, by = bcf.to(DEVICE), bsb.to(DEVICE), by.to(DEVICE)
        pred, _ = gate(bu, bi, bs, bcf, bsb)
        loss = bce(pred, by)
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item()
    if (ep + 1) % 5 == 0 or ep == 0 or ep == 29:
        log(f"[TRAIN] Gate Epoch {ep+1}/30 Loss: {tot/len(loader):.4f} | {time.time()-_t_train:.1f}s | {gpu_mem()}")
log(f"[TRAIN] Light Gated training done in {time.time()-_t_train:.1f}s")
torch.save(gate.state_dict(), OUT_DIR / "light_gated_fusion.pt")


def eval_gate(instances, batch_users=EVAL_BATCH_USERS):
    """Batched eval: fuse B users x C candidates in ONE forward pass.

    Gate is pointwise over (user, item), so batching only changes how many rows
    are pushed through the MLP — scores are identical to the per-user loop.
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

            i_cf = item_cf[cands]                                              # (B, C, 32)
            i_sb = item_sbert[cands]                                           # (B, C, 768)
            u_cf = user_cf[users]                                              # (B, 32)
            u_sb = np.stack([user_sbert[int(u)] for u in users])               # (B, 768)

            bcf = np.einsum("bcd,bd->bc", i_cf, u_cf, optimize=True)
            bsb = np.einsum("bcd,bd->bc", i_sb, u_sb, optimize=True)

            fused, _ = gate(
                torch.from_numpy(np.repeat(u_cf[:, None, :], c, axis=1)).to(DEVICE),
                torch.from_numpy(i_cf).to(DEVICE),
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
log(f"[RESULT] Light Gated: {res_gate}")

summary = {
    "model": "light_gated",
    "n_eval_users": len(test_inst),
    "min_interactions": MIN_INTERACTIONS,
    "als": res_als,
    "sbert": res_sbert,
    "light_gated": res_gate,
}
(OUT_DIR / "light_gated_summary.json").write_text(json.dumps(summary, indent=2))
log(f"[SAVE] {OUT_DIR / 'light_gated_summary.json'} | summary={json.dumps(summary)}")
log(f"DONE in {time.time()-_T0:.1f}s | RAM {mem_info()} | {gpu_mem()}")
print("[ARTIFACT SAVED]", flush=True)
