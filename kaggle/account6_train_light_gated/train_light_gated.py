#!/usr/bin/env python
"""Kaggle: Train Light Gated Late Fusion (single model).

Deep hybrid: neural gate learns per-(user,item) fusion weight between
ALS (CF) score and SBERT (content) score.
"""
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
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

MIN_INTERACTIONS = 3
CATALOG = 2464371
SEED = 42
OUT_DIR = Path("/kaggle/working" if Path("/kaggle/working").exists() else "artifacts")
OUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def find_item_profile_dir():
    if Path("/kaggle/input").exists():
        for p in Path("/kaggle/input").rglob("*.parquet"):
            try:
                import pyarrow.parquet as pq
                cols = pq.read_schema(p).names
                if "title" in cols and "item_idx" in cols:
                    return p.parent
            except Exception:
                pass
    return None


BASE_DIR = find_base_dir()
print("Dataset:", BASE_DIR)


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
print(f"ALS factors: {user_cf.shape} / {item_cf.shape}")

# SBERT with real text
profile_dir = find_item_profile_dir()
item_text_map = {}
if profile_dir is not None:
    for f in sorted(profile_dir.glob("*.parquet")):
        try:
            t = pq.read_table(f, columns=["item_idx", "title", "brand", "category", "description"])
            d = t.to_pandas()
            for _, r in d.iterrows():
                parts = []
                if isinstance(r.get("title"), str) and r["title"].strip():
                    parts.append(r["title"].strip())
                if isinstance(r.get("brand"), str) and r["brand"].strip() and "amazon" not in r["brand"].lower():
                    parts.append(r["brand"].strip())
                if isinstance(r.get("category"), str) and r["category"].strip():
                    parts.append(r["category"].strip())
                desc = r.get("description")
                if isinstance(desc, str) and desc.strip():
                    parts.append(desc.strip()[:200])
                txt = " | ".join(parts)
                if txt:
                    item_text_map[int(r["item_idx"])] = txt
        except Exception as e:
            print("profile read err", f, e)
print(f"Loaded real text for {len(item_text_map):,} items")

# Compact indices
train_files = sorted(BASE_DIR.glob("train/*.parquet"))
dfs = []
for f in tqdm(train_files, desc="Reading train"):
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
n_users, n_items = len(u_uniq), len(i_uniq)
SBERT_COVERAGE = min(n_items, CATALOG)
compact_to_orig = {idx: int(i_uniq[idx]) for idx in range(n_items)}

item_texts = []
for ci in range(SBERT_COVERAGE):
    orig = compact_to_orig[ci]
    item_texts.append(item_text_map.get(orig, f"Product item {orig}"))

print(f"Encoding {len(item_texts)} items...")
sbert_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=str(DEVICE))
sbert_embs = sbert_model.encode(item_texts, batch_size=512, show_progress_bar=True,
                                convert_to_numpy=True, normalize_embeddings=True)
item_sbert = np.zeros((n_items, 384), dtype=np.float32)
item_sbert[:SBERT_COVERAGE] = sbert_embs
del sbert_embs
gc.collect()

# User ids
np.save(OUT_DIR / "user_ids.npy", u_uniq)
np.save(OUT_DIR / "item_ids.npy", i_uniq)
np.save(OUT_DIR / "als_user_factors_f16.npy", user_cf.astype(np.float16))
np.save(OUT_DIR / "als_item_factors_f16.npy", item_cf.astype(np.float16))
np.save(OUT_DIR / "sbert_item_embeddings_f16.npy", item_sbert.astype(np.float16))

# Train history & test
test_files = sorted(BASE_DIR.glob("test/*.parquet"))
dfs_test = []
for f in tqdm(test_files, desc="Reading test"):
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
print("\n" + "=" * 50)
print("[STEP 5/7] BASELINES")
print("=" * 50)
res_als = eval_scorer(lambda u, c: cf_scores(user_cf, item_cf, u, c), test_inst)
print("ALS:", res_als)
res_sbert = eval_scorer(lambda u, c: sb_scores(u, c), test_inst)
print("SBERT:", res_sbert)


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
torch.save(gate.state_dict(), OUT_DIR / "light_gated_fusion.pt")


def eval_gate(instances):
    gate.eval()
    agg = {k: 0.0 for k in ["HR@5", "HR@10", "HR@20", "Recall@5", "Recall@10", "Recall@20",
                             "NDCG@5", "NDCG@10", "NDCG@20", "MRR"]}
    with torch.no_grad():
        for inst in instances:
            u = inst["user"]; cands = inst["candidates"]; gt = inst["gt"]
            bu = torch.tensor(np.repeat(user_cf[u][None, :], len(cands), axis=0), dtype=torch.float32).to(DEVICE)
            bi = torch.tensor(item_cf[cands], dtype=torch.float32).to(DEVICE)
            bs = torch.tensor(item_sbert[cands], dtype=torch.float32).to(DEVICE)
            bcf = torch.tensor(cf_scores(user_cf, item_cf, u, cands), dtype=torch.float32).to(DEVICE)
            bsb = torch.tensor(sb_scores(u, cands), dtype=torch.float32).to(DEVICE)
            fused, _ = gate(bu, bi, bs, bcf, bsb)
            ranked = cands[np.argsort(-fused.cpu().numpy())]
            m = compute_metrics(ranked, gt)
            for k in agg:
                agg[k] += m[k]
    return {k: round(v / max(1, len(instances)), 4) for k, v in agg.items()}


res_gate = eval_gate(test_inst)
print("Light Gated:", res_gate)

summary = {
    "model": "light_gated",
    "n_eval_users": len(test_inst),
    "min_interactions": MIN_INTERACTIONS,
    "als": res_als,
    "sbert": res_sbert,
    "light_gated": res_gate,
}
(OUT_DIR / "light_gated_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
print("DONE", flush=True)
print("[ARTIFACT SAVED]", flush=True)
