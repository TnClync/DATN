#!/usr/bin/env python
"""Evaluation-only Kaggle kernel: compare all trained models, no retraining.

Mounts the output of `nguynteo/deep-hybrid-recommender-datn` (CF factors, SBERT
embeddings, gate/early-fusion weights) and evaluates on the same protocol:
  1 positive + 99 random negatives on a shared 200k-item catalog,
  leak-free SBERT user profiles (mean of train-history item embeddings).
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from tqdm import tqdm

MIN_INTERACTIONS = 3
CATALOG = 200_000
TEST_N, VAL_N = 500, 300
SEED = 42


# --------------------------------------------------------------------- inputs
def find_base_dir() -> Path:
    candidates = [
        Path("/kaggle/input/lastone/amazon_recs_output"),
        Path("/kaggle/input/datasets/nguynteo/lastone/amazon_recs_output"),
        Path("/kaggle/input/lastone"),
        Path("/kaggle/input/datasets/nguynteo/lastone"),
        Path("data"),
    ]
    for c in candidates:
        if (c / "train").exists() or (c / "test").exists():
            return c
    if Path("/kaggle/input").exists():
        for p in Path("/kaggle/input").rglob("train"):
            if p.is_dir():
                return p.parent
    return Path("data")


def find_file(name: str) -> Path:
    hits = sorted(Path("/kaggle/input").rglob(name))
    if not hits:
        print("MISSING input file:", name)
        for root in sorted(Path("/kaggle/input").iterdir()):
            print("  input:", root)
        raise FileNotFoundError(name)
    print(f"Using {name}: {hits[0]}")
    return hits[0]


BASE_DIR = find_base_dir()
print("Dataset:", BASE_DIR)
OUT_DIR = Path("/kaggle/working")
OUT_DIR.mkdir(parents=True, exist_ok=True)

t0 = time.time()
print("\n" + "=" * 50)
print("1. LOADING ARTIFACTS (no retraining)")
print("=" * 50)
als_u_path = find_file("als_user_factors_f16.npy")
als_i_path = find_file("als_item_factors_f16.npy")
bpr_u_path = find_file("bpr_user_factors_f16.npy")
bpr_i_path = find_file("bpr_item_factors_f16.npy")
sbert_path = find_file("sbert_item_embeddings_f16.npy")
gate_path = find_file("light_gated_fusion.pt")
deep_path = find_file("deep_early_fusion.pt")

u_ids = np.load(find_file("user_ids.npy"))
i_ids = np.load(find_file("item_ids.npy"))
n_users, n_items = len(u_ids), len(i_ids)
print(f"Compact ids: {n_users:,} users x {n_items:,} items")

als_uf = np.load(als_u_path, mmap_mode="r")
als_itf = np.load(als_i_path).astype(np.float32)
bpr_uf = np.load(bpr_u_path, mmap_mode="r")
bpr_itf = np.load(bpr_i_path).astype(np.float32)
item_sbert = np.load(sbert_path, mmap_mode="r")
print("ALS:", als_uf.shape, als_itf.shape, "| BPR:", bpr_uf.shape, bpr_itf.shape)
print("SBERT:", item_sbert.shape)


# ----------------------------------------------------------------------- data
def read_split(directory: Path, columns: list[str]) -> pd.DataFrame:
    parts = sorted(directory.glob("*.parquet"))
    frames = [pq.read_table(p, columns=columns).to_pandas() for p in tqdm(parts, desc=str(directory.name))]
    df = pd.concat(frames, ignore_index=True)
    if "rating" in df.columns:
        df = df[df["rating"] >= 4.0]
    return df


def map_ids(values: np.ndarray, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pos = np.searchsorted(ids, values)
    pos = np.clip(pos, 0, len(ids) - 1)
    ok = ids[pos] == values
    return pos, ok


print("\n" + "=" * 50)
print("2. DATA + EVAL INSTANCES")
print("=" * 50)
train = read_split(BASE_DIR / "train", ["user_idx", "item_idx", "rating"])
cu, ok_u = map_ids(train["user_idx"].to_numpy(), u_ids)
ci, ok_i = map_ids(train["item_idx"].to_numpy(), i_ids)
keep = ok_u & ok_i
cu, ci = cu[keep], ci[keep]
user_inter_count = np.bincount(cu, minlength=n_users)
del train

test = read_split(BASE_DIR / "test", ["user_idx", "item_idx", "rating"])
tcu, tok_u = map_ids(test["user_idx"].to_numpy(), u_ids)
tci, tok_i = map_ids(test["item_idx"].to_numpy(), i_ids)
tkeep = tok_u & tok_i
gt_df = pd.DataFrame({"u": tcu[tkeep], "i": tci[tkeep]})
user_gt = gt_df.groupby("u")["i"].apply(set).to_dict()

rng = np.random.default_rng(SEED)
test_users = np.array([
    u for u in user_gt
    if user_inter_count[u] >= MIN_INTERACTIONS and any(it < CATALOG for it in user_gt[u])
])
eval_users = rng.choice(test_users, size=min(2000, len(test_users)), replace=False)
eval_set = set(int(u) for u in eval_users)
print(f"Eval users: {len(eval_users):,}")

mask = np.isin(cu, np.fromiter(eval_set, dtype=np.int64))
hist: dict[int, list[int]] = {}
for u, i in zip(cu[mask], ci[mask]):
    if i < CATALOG:
        hist.setdefault(int(u), []).append(int(i))

user_sbert: dict[int, np.ndarray] = {}
for u in eval_set:
    h = hist.get(u, [])
    if h:
        v = np.asarray(item_sbert[h], dtype=np.float32).mean(0)
        n = float(np.linalg.norm(v))
        v = v / n if n > 1e-8 else np.zeros(384, dtype=np.float32)
    else:
        v = np.zeros(384, dtype=np.float32)
    user_sbert[u] = v.astype(np.float32)

pool = np.arange(CATALOG)
eval_instances = []
for u in eval_users:
    u = int(u)
    gts = [it for it in user_gt[u] if it < CATALOG]
    gt = int(rng.choice(gts))
    banned = (set(hist.get(u, [])) | set(user_gt[u])) - {gt}
    negs = []
    for _ in range(10):
        draw = rng.choice(pool, size=300, replace=False)
        negs = [int(x) for x in draw if int(x) not in banned][:99]
        if len(negs) == 99:
            break
    if len(negs) < 99:
        continue
    cands = np.array([gt] + negs, dtype=np.int64)
    rng.shuffle(cands)
    eval_instances.append({"user": u, "gt": gt, "candidates": cands,
                           "banned": np.array(sorted(banned), dtype=np.int64)})

test_inst = eval_instances[:TEST_N]
val_inst = eval_instances[TEST_N:TEST_N + VAL_N]
print(f"Instances: {len(eval_instances)} (test={len(test_inst)}, val={len(val_inst)})")


# -------------------------------------------------------------------- scoring
def compute_metrics(ranked, gt):
    res = {k: 0.0 for k in ["HR@5", "HR@10", "HR@20", "NDCG@5", "NDCG@10", "NDCG@20", "MRR"]}
    pos_list = np.where(np.asarray(ranked) == gt)[0]
    if len(pos_list) == 0:
        return res
    pos = int(pos_list[0])
    if pos < 5:
        res["HR@5"] = 1.0
        res["NDCG@5"] = 1.0 / math.log2(pos + 2)
    if pos < 10:
        res["HR@10"] = 1.0
        res["NDCG@10"] = 1.0 / math.log2(pos + 2)
    if pos < 20:
        res["HR@20"] = 1.0
        res["NDCG@20"] = 1.0 / math.log2(pos + 2)
    res["MRR"] = 1.0 / (pos + 1)
    return res


def mm(x):
    x = np.asarray(x, dtype=np.float32)
    mn, mx = x.min(), x.max()
    return np.zeros_like(x) if mx - mn < 1e-8 else (x - mn) / (mx - mn)


def cf_scores(uf, itf, u, cands):
    return np.asarray(itf[cands] @ np.asarray(uf[u], dtype=np.float32), dtype=np.float32)


def sb_scores(u, cands):
    return np.asarray(item_sbert[cands] @ user_sbert[u], dtype=np.float32)


def fuse(cf, sb, alpha):
    return alpha * mm(cf) + (1.0 - alpha) * mm(sb)


def eval_scorer(fn, instances):
    agg = {k: 0.0 for k in ["HR@5", "HR@10", "HR@20", "NDCG@5", "NDCG@10", "NDCG@20", "MRR"]}
    for inst in instances:
        ranked = inst["candidates"][np.argsort(-fn(inst["user"], inst["candidates"]))]
        for k, v in compute_metrics(ranked, inst["gt"]).items():
            agg[k] += v
    return {k: round(v / max(1, len(instances)), 4) for k, v in agg.items()}


print("\n" + "=" * 50)
print("3. BASELINES")
print("=" * 50)
res_als = eval_scorer(lambda u, c: cf_scores(als_uf, als_itf, u, c), test_inst)
res_bpr = eval_scorer(lambda u, c: cf_scores(bpr_uf, bpr_itf, u, c), test_inst)
res_sbert = eval_scorer(sb_scores, test_inst)
print("ALS   :", res_als)
print("BPR   :", res_bpr)
print("SBERT :", res_sbert)


# ---------------------------------------------------------------- late fusion
print("\n" + "=" * 50)
print("4. LATE FUSION (alpha tuned on VAL)")
print("=" * 50)
late_alphas = [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
grid_val = {}
best_alpha, best_hr = 0.9, -1.0
for a in late_alphas:
    m = eval_scorer(lambda u, c, a=a: fuse(cf_scores(als_uf, als_itf, u, c), sb_scores(u, c), a), val_inst)
    grid_val[a] = m
    print(f"alpha={a}: HR@10={m['HR@10']:.4f}")
    if m["HR@10"] > best_hr:
        best_hr, best_alpha = m["HR@10"], a
res_late = eval_scorer(
    lambda u, c: fuse(cf_scores(als_uf, als_itf, u, c), sb_scores(u, c), best_alpha), test_inst)
print(f"Best alpha={best_alpha} -> {res_late}")


# --------------------------------------------------------------- two-stage
print("\n" + "=" * 50)
print("5. TWO-STAGE RETRIEVAL")
print("=" * 50)


def topk_full(mat, vec, k, banned):
    scores = mat @ vec
    if len(banned):
        b = banned[(banned >= 0) & (banned < len(scores))]
        scores[b] = -np.inf
    k = min(k, len(scores) - 1)
    idx = np.argpartition(-scores, k)[:k]
    return idx[np.argsort(-scores[idx])]


def two_stage_rank(inst, alpha):
    u = inst["user"]
    cf_top = topk_full(als_itf, np.asarray(als_uf[u], dtype=np.float32), 50, inst["banned"])
    sb_top = topk_full(item_sbert[:CATALOG], user_sbert[u], 50, inst["banned"])
    union = np.array(list(dict.fromkeys(list(cf_top) + list(sb_top))), dtype=np.int64)
    fused = fuse(cf_scores(als_uf, als_itf, u, union), sb_scores(u, union), alpha)
    return union[np.argsort(-fused)]


def eval_two_stage(alpha, instances):
    agg = {k: 0.0 for k in ["HR@5", "HR@10", "HR@20", "NDCG@5", "NDCG@10", "NDCG@20", "MRR"]}
    for inst in tqdm(instances, desc="Two-stage"):
        ranked = two_stage_rank(inst, alpha)
        for k, v in compute_metrics(ranked, inst["gt"]).items():
            agg[k] += v
    return {k: round(v / max(1, len(instances)), 4) for k, v in agg.items()}


ts_alphas = [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
ts_grid = {}
ts_best, ts_best_hr = 1.0, -1.0
for a in ts_alphas:
    m = eval_two_stage(a, val_inst)
    ts_grid[a] = m
    print(f"two-stage alpha={a}: val HR@10={m['HR@10']:.4f}")
    if m["HR@10"] > ts_best_hr:
        ts_best_hr, ts_best = m["HR@10"], a
res_two_stage = eval_two_stage(ts_best, test_inst)
print(f"Best two-stage alpha={ts_best} -> {res_two_stage}")


# ------------------------------------------------------------ deep heads
class LightGate(nn.Module):
    def __init__(self, in_features=32 + 32 + 384):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(in_features, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, u_cf, i_cf, i_sbert, cf_score, sbert_score):
        feat = torch.cat([u_cf, i_cf, i_sbert], dim=-1)
        alpha = self.gate_net(feat).squeeze(-1)
        return alpha * cf_score + (1.0 - alpha) * sbert_score, alpha


class DeepEarlyFusion(nn.Module):
    def __init__(self, in_dim, cross_layers=3, hidden=128):
        super().__init__()
        self.cross = nn.ModuleList([nn.Linear(in_dim, in_dim) for _ in range(cross_layers)])
        self.deep = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                                  nn.Linear(hidden, hidden), nn.ReLU(),
                                  nn.Linear(hidden, 1))

    def forward(self, x):
        x0 = x
        xi = x
        for layer in self.cross:
            xi = x0 * torch.sigmoid(layer(xi)) + xi
        deep_out = self.deep(x)
        cross_out = xi.mean(dim=-1, keepdim=True)
        return (cross_out + deep_out).squeeze(-1)


gate_model = LightGate()
gate_model.load_state_dict(torch.load(gate_path, map_location="cpu"))
gate_model.eval()
deep_model = DeepEarlyFusion(32 + 32 + 384)
deep_model.load_state_dict(torch.load(deep_path, map_location="cpu"))
deep_model.eval()


def eval_gate(instances):
    agg = {k: 0.0 for k in ["HR@5", "HR@10", "HR@20", "NDCG@5", "NDCG@10", "NDCG@20", "MRR"]}
    with torch.no_grad():
        for inst in instances:
            u, cands = inst["user"], inst["candidates"]
            u_cf = np.asarray(als_uf[u], dtype=np.float32)
            i_cf = np.asarray(als_itf[cands], dtype=np.float32)
            i_sb = np.asarray(item_sbert[cands], dtype=np.float32)
            cf = np.asarray(i_cf @ u_cf, dtype=np.float32)
            sb = np.asarray(i_sb @ user_sbert[u], dtype=np.float32)
            fused, _ = gate_model(
                torch.tensor(np.repeat(u_cf[None, :], len(cands), axis=0)),
                torch.tensor(i_cf), torch.tensor(i_sb),
                torch.tensor(cf), torch.tensor(sb))
            ranked = cands[np.argsort(-fused.numpy())]
            for k, v in compute_metrics(ranked, inst["gt"]).items():
                agg[k] += v
    return {k: round(v / max(1, len(instances)), 4) for k, v in agg.items()}


def eval_deep(instances):
    agg = {k: 0.0 for k in ["HR@5", "HR@10", "HR@20", "NDCG@5", "NDCG@10", "NDCG@20", "MRR"]}
    with torch.no_grad():
        for inst in instances:
            u, cands = inst["user"], inst["candidates"]
            u_cf = np.asarray(als_uf[u], dtype=np.float32)
            feats = np.concatenate([
                np.repeat(u_cf[None, :], len(cands), axis=0),
                np.asarray(als_itf[cands], dtype=np.float32),
                np.asarray(item_sbert[cands], dtype=np.float32),
            ], axis=1)
            logits = deep_model(torch.tensor(feats)).numpy()
            ranked = cands[np.argsort(-logits)]
            for k, v in compute_metrics(ranked, inst["gt"]).items():
                agg[k] += v
    return {k: round(v / max(1, len(instances)), 4) for k, v in agg.items()}


print("\n" + "=" * 50)
print("6. DEEP HYBRID HEADS")
print("=" * 50)
res_gate = eval_gate(test_inst)
res_deep = eval_deep(test_inst)
print("Light Gated:", res_gate)
print("Deep Early :", res_deep)


# ------------------------------------------------------------------- summary
summary = {
    "protocol": "1 positive + 99 random negatives, shared 200k-item catalog, leak-free SBERT user profiles",
    "eval_catalog_items": CATALOG,
    "n_eval_users": len(test_inst),
    "n_val_users": len(val_inst),
    "min_interactions": MIN_INTERACTIONS,
    "als": res_als,
    "bpr": res_bpr,
    "sbert": res_sbert,
    "late_fusion": {"best_alpha": best_alpha, "val_HR10": round(best_hr, 4),
                    "test": res_late, "grid_val": {str(a): m for a, m in grid_val.items()}},
    "two_stage": {"best_alpha": ts_best, "val_HR10": round(ts_best_hr, 4),
                  "test": res_two_stage, "grid_val": {str(a): m for a, m in ts_grid.items()}},
    "light_gated": res_gate,
    "deep_early_fusion": res_deep,
    "eval_time_sec": round(time.time() - t0, 1),
}
(OUT_DIR / "eval_summary.json").write_text(json.dumps(summary, indent=2))

print("\n" + "=" * 74)
print(f"FINAL COMPARISON (TEST SET, N={len(test_inst)})")
print("=" * 74)
print(f"{'Model':<26} {'HR@5':>7} {'HR@10':>7} {'HR@20':>7} {'NDCG@10':>8} {'MRR':>7}")
rows = [
    ("ALS", res_als),
    ("BPR", res_bpr),
    ("SBERT", res_sbert),
    (f"Late Fusion (a={best_alpha})", res_late),
    (f"Two-Stage (a={ts_best})", res_two_stage),
    ("Light Gated Fusion", res_gate),
    ("Deep Early Fusion", res_deep),
]
for name, m in rows:
    print(f"{name:<26} {m['HR@5']:>7.4f} {m['HR@10']:>7.4f} {m['HR@20']:>7.4f} "
          f"{m['NDCG@10']:>8.4f} {m['MRR']:>7.4f}")
print("=" * 74)
print(json.dumps(summary, indent=2))
print("DONE in", round(time.time() - t0, 1), "s")
