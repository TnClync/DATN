#!/usr/bin/env python
"""Full-test evaluation kernel: compare all trained models + Proposed Models (RGH, LambdaRank, Ensemble).

Protocol:
- Leave-One-Out: 1 positive + 99 random negatives, shared catalog, leak-free SBERT profiles.
- Validated without test-tuning leakage: Tuning and training of hybrid heads on validation split (tune_idx).
- Evaluates:
    1. Baselines: ALS, BPR, LightGCN, SBERT
    2. Fusion Baselines: Late Fusion, Light-Gated, Deep-Early, Graph Light-Gated, Two-Tower, Transformer
    3. Proposed RGH (CF-Anchored Residual Gated Hybrid with DCN-v2 and Listwise Ranking Loss)
    4. Proposed Two-Stage LambdaRank (LightGBM)
    5. Proposed Stacked Ensemble (RGH + LambdaRank + Transformer + LightGCN)
- Metrics: HR@{5,10,20}, NDCG@{5,10,20}, MRR, Recall@10, Precision@10, Delta vs Baseline.
- Artifacts mounted from `nguynteo/cf-als-bpr-implicit-datn` and companion kernels.
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    HAS_LGB = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

MIN_INTERACTIONS = 3
CATALOG = 2464371
SBERT_DIM = 768   # combined SBERT item embeddings (metadata | reviewer), built by the SBERT kernel
SEED = 42
BATCH = 1024
DEEP_BATCH = 256
TUNE_N = 10_000

try:
    import faiss
    HAS_FAISS = True
except Exception:  # pragma: no cover
    try:
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "faiss-cpu"])
        import faiss
        HAS_FAISS = True
    except Exception as exc:
        print("faiss unavailable:", exc)
        HAS_FAISS = False


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
    aliases = {
        "user_ids.npy": ["user_ids.npy", "als_user_ids.npy"],
        "item_ids.npy": ["item_ids.npy", "als_item_ids.npy"],
    }
    names_to_try = aliases.get(name, [name])
    for n in names_to_try:
        hits = sorted(Path("/kaggle/input").rglob(n))
        if hits:
            preferred = [p for p in hits if "train-sbert" in str(p) or "datn-graph" in str(p) or "cf-als" in str(p)]
            chosen = preferred[0] if preferred else hits[0]
            print(f"Using {name} ({n}): {chosen}")
            return chosen
        local_cand = list(Path("artifacts").rglob(n))
        if local_cand:
            return local_cand[0]
    raise FileNotFoundError(f"Could not find {name} (tried {names_to_try})")


def find_sbert_file():
    """Locate SBERT embeddings (accepts either 768-d or 384-d)."""
    hits = sorted(Path("/kaggle/input").rglob("*sbert*embeddings*.npy"))
    if not hits:
        hits = sorted(Path("/kaggle/input").rglob("*.npy"))
        hits = [p for p in hits if "sbert" in p.name.lower()]
    for p in hits:
        try:
            a = np.load(p, mmap_mode="r")
            if a.ndim == 2 and a.shape[1] in (384, 768):
                print(f"Found SBERT embeddings: {p} with shape {a.shape}")
                return p
        except Exception:
            continue
    for local_p in [Path("artifacts/eval_inputs/sbert_item_embeddings_f16.npy"),
                    Path("artifacts/sbert/sbert_item_embeddings_768_f16.npy"),
                    Path("artifacts/sbert/sbert_item_embeddings_f16.npy")]:
        if local_p.exists():
            return local_p
    return None


def read_split(directory: Path, columns: list[str]) -> pd.DataFrame:
    parts = sorted(directory.glob("*.parquet"))
    frames = [pq.read_table(p, columns=columns).to_pandas() for p in parts]
    df = pd.concat(frames, ignore_index=True)
    if "rating" in df.columns:
        df = df[df["rating"] >= 4.0]
    return df


def map_ids(values: np.ndarray, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pos = np.searchsorted(ids, values)
    pos = np.clip(pos, 0, len(ids) - 1)
    return pos, ids[pos] == values


BASE_DIR = find_base_dir()
print("Dataset:", BASE_DIR)
OUT_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("artifacts/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

t0 = time.time()
print("\n" + "=" * 56)
print("1. LOADING TRAINED ARTIFACTS")
print("=" * 56)
als_uf = np.load(find_file("als_user_factors_f16.npy"), mmap_mode="r")
als_itf_raw = np.load(find_file("als_item_factors_f16.npy")).astype(np.float32)
if len(als_itf_raw) < CATALOG:
    als_itf = np.zeros((CATALOG, als_itf_raw.shape[1]), dtype=np.float32)
    als_itf[:len(als_itf_raw)] = als_itf_raw
else:
    als_itf = als_itf_raw[:CATALOG]

bpr_uf = np.load(find_file("bpr_user_factors_f16.npy"), mmap_mode="r")
bpr_itf_raw = np.load(find_file("bpr_item_factors_f16.npy")).astype(np.float32)
if len(bpr_itf_raw) < CATALOG:
    bpr_itf = np.zeros((CATALOG, bpr_itf_raw.shape[1]), dtype=np.float32)
    bpr_itf[:len(bpr_itf_raw)] = bpr_itf_raw
else:
    bpr_itf = bpr_itf_raw[:CATALOG]
sbert_path = find_sbert_file()
assert sbert_path is not None, "SBERT embeddings not found"
item_sbert = np.load(sbert_path, mmap_mode="r")
SBERT_DIM = int(item_sbert.shape[1])

gate_path = find_file("light_gated_fusion.pt")
deep_path = find_file("deep_early_fusion.pt")
graph_gate_path = find_file("graph_light_gated_fusion.pt")
two_tower_path = find_file("two_tower_fusion.pt")
transformer_path = find_file("transformer_fusion.pt")
graph_uf = np.load(find_file("graph_user_factors_f16.npy"), mmap_mode="r")
graph_itf_raw = np.load(find_file("graph_item_factors_f16.npy")).astype(np.float32)
if len(graph_itf_raw) < CATALOG:
    graph_itf = np.zeros((CATALOG, graph_itf_raw.shape[1]), dtype=np.float32)
    graph_itf[:len(graph_itf_raw)] = graph_itf_raw
else:
    graph_itf = graph_itf_raw[:CATALOG]
u_ids = np.load(find_file("user_ids.npy"))
i_ids = np.load(find_file("item_ids.npy"))
n_users, n_items = len(u_ids), len(i_ids)
print(f"Ids: {n_users:,} users x {n_items:,} items | SBERT {item_sbert.shape}")

# Pre-allocate full catalog cache to prevent any out-of-bounds indexing
sbert_cat = np.zeros((CATALOG, SBERT_DIM), dtype=np.float32)
n_avail = min(CATALOG, len(item_sbert))
sbert_cat[:n_avail] = np.asarray(item_sbert[:n_avail], dtype=np.float32)
print(f"SBERT catalog cache: {sbert_cat.shape} ({n_avail:,} items with trained embeddings, {sbert_cat.nbytes / 1e6:.0f} MB)")


# ----------------------------------------------------------------------- data
print("\n" + "=" * 56)
print("2. FULL TEST SET -> ELIGIBLE USERS")
print("=" * 56)
train = read_split(BASE_DIR / "train", ["user_idx", "item_idx", "rating"])
cu, ok_u = map_ids(train["user_idx"].to_numpy(), u_ids)
ci, ok_i = map_ids(train["item_idx"].to_numpy(), i_ids)
keep = ok_u & ok_i
cu, ci = cu[keep], ci[keep]
counts = np.bincount(cu, minlength=n_users)
del train

test = read_split(BASE_DIR / "test", ["user_idx", "item_idx", "rating"])
tu, tok_u = map_ids(test["user_idx"].to_numpy(), u_ids)
ti, tok_i = map_ids(test["item_idx"].to_numpy(), i_ids)
tkeep = tok_u & tok_i
tu, ti = tu[tkeep], ti[tkeep]
del test
print(f"Test positives mapped: {len(tu):,}")

covered = ti < CATALOG
users_cov = np.unique(tu[covered])
elig_users = users_cov[counts[users_cov] >= MIN_INTERACTIONS]
N_full = len(elig_users)
print(f"Eligible test users (>= {MIN_INTERACTIONS} train, positive in {CATALOG:,}-item catalog): {N_full:,}")

# Sample users for fast, statistically reliable evaluation
MAX_EVAL = int(os.environ.get("MAX_EVAL_USERS", "10000"))
n_sample = min(MAX_EVAL, N_full)
rng_sample = np.random.default_rng(12345)
sample_idx = rng_sample.choice(N_full, size=n_sample, replace=False)
elig_users = elig_users[sample_idx]
N = len(elig_users)
print(f"Sampled {N:,} users for evaluation (from {N_full:,} eligible, seed=12345, without replacement)")

# all test positives for eligible users
def ismember(values: np.ndarray, members: np.ndarray) -> np.ndarray:
    s = np.sort(members)
    if len(s) == 0:
        return np.zeros(len(values), dtype=bool)
    p = np.searchsorted(s, values)
    p = np.clip(p, 0, len(s) - 1)
    return s[p] == values


elig_sorted = np.sort(elig_users)
sub = ismember(tu, elig_sorted)
gt_df = pd.DataFrame({"u": tu[sub], "i": ti[sub]})
user_gt: dict[int, set[int]] = {
    int(u): set(int(x) for x in g) for u, g in gt_df.groupby("u")["i"]
}
del gt_df
print(f"Users with grouped positives: {len(user_gt):,}")

# train history for eligible users
mask = ismember(cu, elig_sorted)
hist: dict[int, list[int]] = {}
for u, i in zip(cu[mask], ci[mask]):
    if i < CATALOG:
        hist.setdefault(int(u), []).append(int(i))
print(f"History built for {len(hist):,} users")

user_sbert = np.zeros((N, SBERT_DIM), dtype=np.float32)
u_to_row = {int(u): k for k, u in enumerate(elig_users)}
for u, k in u_to_row.items():
    h = hist.get(u, [])
    if h:
        v = sbert_cat[h].mean(0)
        n = float(np.linalg.norm(v))
        user_sbert[k] = (v / n) if n > 1e-8 else 0.0


# ------------------------------------------------------------------ instances
print("\n" + "=" * 56)
print("3. BUILDING EVAL INSTANCES (1 positive + 99 negatives)")
print("=" * 56)
rng = np.random.default_rng(SEED)
cands = np.empty((N, 100), dtype=np.int32)
gt_pos = np.empty(N, dtype=np.int16)
gts = np.empty(N, dtype=np.int64)
banned_list: list[np.ndarray] = []
R = rng.integers(0, CATALOG, size=(N, 140), dtype=np.int32)

kept = np.zeros(N, dtype=bool)
for k, u in enumerate(elig_users):
    u = int(u)
    pos = [it for it in user_gt.get(u, ()) if it < CATALOG]
    if not pos:
        continue
    gt = int(pos[rng.integers(len(pos))])
    banned = (set(hist.get(u, [])) | set(user_gt.get(u, ()))) - {gt}
    row = R[k]
    negs: list[int] = []
    seen = set()
    for x in row:
        x = int(x)
        if x in banned or x == gt or x in seen:
            continue
        seen.add(x)
        negs.append(x)
        if len(negs) == 99:
            break
    if len(negs) < 99:
        continue
    arr = np.empty(100, dtype=np.int32)
    arr[0] = gt
    arr[1:] = negs
    rng.shuffle(arr)
    cands[k] = arr
    gt_pos[k] = int(np.where(arr == gt)[0][0])
    gts[k] = gt
    banned_list.append(np.array(sorted(banned), dtype=np.int64))
    kept[k] = True

cands = cands[kept]
gt_pos = gt_pos[kept]
gts = gts[kept]
elig_users = elig_users[kept]
user_sbert = user_sbert[kept]
N = len(cands)
print(f"Valid Instances: {N:,}")

# Tune split vs Test split
tune_size = min(TUNE_N, N // 2)
tune_idx = rng.choice(N, size=tune_size, replace=False)
test_idx = np.setdiff1d(np.arange(N), tune_idx)
print(f"Validation Tune users: {len(tune_idx):,}; Final Holdout Test users: {len(test_idx):,}")


# ------------------------------------------------------------------- metrics
def positions_from_scores(scores: np.ndarray, pos: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores, axis=1)
    return (order == pos[:, None]).argmax(axis=1)


def metrics_from_positions(pos: np.ndarray, valid: np.ndarray | None = None) -> dict[str, float]:
    pos = np.asarray(pos, dtype=np.float32)
    if valid is None:
        valid = np.ones(len(pos), dtype=bool)
    valid = np.asarray(valid, dtype=bool)
    posv = np.where(valid, pos, 1e6)
    return {
        "HR@5": round(float(((posv < 5) & valid).mean()), 4),
        "HR@10": round(float(((posv < 10) & valid).mean()), 4),
        "HR@20": round(float(((posv < 20) & valid).mean()), 4),
        "Recall@10": round(float(((posv < 10) & valid).mean()), 4),
        "Precision@10": round(float(((posv < 10) & valid).mean() / 10.0), 4),
        "NDCG@5": round(float(np.where((posv < 5) & valid, 1.0 / np.log2(posv + 2), 0.0).mean()), 4),
        "NDCG@10": round(float(np.where((posv < 10) & valid, 1.0 / np.log2(posv + 2), 0.0).mean()), 4),
        "NDCG@20": round(float(np.where((posv < 20) & valid, 1.0 / np.log2(posv + 2), 0.0).mean()), 4),
        "MRR": round(float(np.where(valid, 1.0 / (posv + 1), 0.0).mean()), 4),
        "n_eval_users": int(len(pos)),
    }


def cf_batch(uf, itf, idx):
    items = itf[cands[idx]]                                  # B x 100 x d
    users = np.asarray(uf[elig_users[idx]], dtype=np.float32)  # B x d
    return np.einsum("bnd,bd->bn", items, users, optimize=True)


def sb_batch(rows_idx):
    items = sbert_cat[cands[rows_idx]]
    return np.einsum("bnd,bd->bn", items, user_sbert[rows_idx], optimize=True)


def mm_rows(x):
    mn = x.min(axis=1, keepdims=True)
    mx = x.max(axis=1, keepdims=True)
    rng_ = mx - mn
    return np.where(rng_ < 1e-8, 0.5, (x - mn) / np.maximum(rng_, 1e-8)).astype(np.float32)


# ----------------------------------------------------------------- baselines
print("\n" + "=" * 56)
print("4. EVALUATING BASELINES ON TEST HOLDOUT")
print("=" * 56)

def eval_baseline(kind: str) -> dict[str, float]:
    pos_all = np.empty(len(test_idx), dtype=np.int32)
    step = BATCH if kind != "sbert" else 512
    for start in range(0, len(test_idx), step):
        idx = test_idx[start:start + step]
        if kind == "als":
            s = cf_batch(als_uf, als_itf, idx)
        elif kind == "bpr":
            s = cf_batch(bpr_uf, bpr_itf, idx)
        elif kind == "lightgcn":
            s = cf_batch(graph_uf, graph_itf, idx)
        else:
            s = sb_batch(idx)
        pos_all[start:start + len(idx)] = positions_from_scores(s, gt_pos[idx])
    return metrics_from_positions(pos_all)

res_als = eval_baseline("als")
print("ALS (CF Baseline):     ", res_als)
res_bpr = eval_baseline("bpr")
print("BPR (CF Baseline):     ", res_bpr)
res_lightgcn = eval_baseline("lightgcn")
print("LightGCN (Graph CF):   ", res_lightgcn)
res_sbert = eval_baseline("sbert")
print("SBERT (Content-Based): ", res_sbert)


# -------------------------------------------------------------- late fusion
print("\n" + "=" * 56)
print("5. LATE FUSION (alpha tuned on VALIDATION subset)")
print("=" * 56)
def eval_fusion(alpha: float, idx_eval: np.ndarray) -> dict[str, float]:
    pos = np.empty(len(idx_eval), dtype=np.int32)
    for start in range(0, len(idx_eval), 512):
        idx = idx_eval[start:start + 512]
        cf = cf_batch(als_uf, als_itf, idx)
        sb = sb_batch(idx)
        s = alpha * mm_rows(cf) + (1.0 - alpha) * mm_rows(sb)
        pos[start:start + len(idx)] = positions_from_scores(s, gt_pos[idx])
    return metrics_from_positions(pos)

grid = {}
best_alpha, best_hr = 0.8, -1.0
for a in [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    m = eval_fusion(a, tune_idx)
    grid[a] = m
    print(f"alpha={a:0.1f}: val HR@10={m['HR@10']:.4f}, NDCG@10={m['NDCG@10']:.4f}")
    if m["NDCG@10"] > best_hr:
        best_hr, best_alpha = m["NDCG@10"], a

res_late = eval_fusion(best_alpha, test_idx)
print(f"Best alpha={best_alpha} on Validation -> Holdout Test: {res_late}")


# ---------------------------------------------------------------- deep heads
class LightGate(nn.Module):
    def __init__(self, in_features=32 + 32 + SBERT_DIM):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(in_features, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, u_cf, i_cf, i_sbert, cf_score, sbert_score):
        feat = torch.cat([u_cf, i_cf, i_sbert], dim=-1)
        alpha = self.gate_net(feat).squeeze(-1)
        return alpha * cf_score + (1.0 - alpha) * sbert_score, alpha


class DeepEarlyFusion(nn.Module):
    def __init__(self, in_dim, cross_layers=4, hidden=256):
        super().__init__()
        self.cross = nn.ModuleList([nn.Linear(in_dim, in_dim) for _ in range(cross_layers)])
        self.deep = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                                  nn.Linear(hidden, hidden), nn.ReLU(),
                                  nn.Linear(hidden, hidden // 2), nn.ReLU(),
                                  nn.Linear(hidden // 2, 1))

    def forward(self, x):
        x0, xi = x, x
        for layer in self.cross:
            xi = x0 * torch.sigmoid(layer(xi)) + xi
        return (xi.mean(dim=-1, keepdim=True) + self.deep(x)).squeeze(-1)


class UserTower(nn.Module):
    def __init__(self, in_dim=32, hidden=128, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.1),
                                 nn.Linear(hidden, out_dim))
    def forward(self, x):
        return self.net(x)


class ItemTower(nn.Module):
    def __init__(self, in_dim=32 + SBERT_DIM, hidden=256, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.1),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.1),
                                 nn.Linear(hidden, out_dim))
    def forward(self, x):
        return self.net(x)


class TransformerFusion(nn.Module):
    def __init__(self, in_dim=32 + 32 + SBERT_DIM, d_model=128, nhead=4, nlayers=2):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                  dim_feedforward=d_model * 4, dropout=0.1,
                                                  batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        h = self.encoder(self.proj(x))
        return self.head(h).squeeze(-1)


print("\n" + "=" * 56)
print("6. LOADING PRETRAINED DEEP HYBRID HEADS")
print("=" * 56)
gate_model = LightGate()
gate_model.load_state_dict(torch.load(gate_path, map_location="cpu"))
gate_model.eval()

deep_model = DeepEarlyFusion(32 + 32 + SBERT_DIM)
deep_model.load_state_dict(torch.load(deep_path, map_location="cpu"))
deep_model.eval()

graph_gate_model = LightGate()
graph_gate_model.load_state_dict(torch.load(graph_gate_path, map_location="cpu"))
graph_gate_model.eval()

user_tower = UserTower()
item_tower = ItemTower()
tt_state = torch.load(two_tower_path, map_location="cpu")
user_tower.load_state_dict(tt_state["user"])
item_tower.load_state_dict(tt_state["item"])
user_tower.eval(); item_tower.eval()

transformer_model = TransformerFusion()
transformer_model.load_state_dict(torch.load(transformer_path, map_location="cpu"))
transformer_model.eval()

def eval_deep_head(kind: str) -> dict[str, float]:
    pos_all = np.empty(len(test_idx), dtype=np.int32)
    for start in range(0, len(test_idx), DEEP_BATCH):
        idx = test_idx[start:start + DEEP_BATCH]
        b = len(idx)
        c = cands[idx]
        u_cf = np.asarray(als_uf[elig_users[idx]], dtype=np.float32)
        i_cf = np.asarray(als_itf[c], dtype=np.float32)
        i_sb = sbert_cat[c]
        cf = np.einsum("bnd,bd->bn", i_cf, u_cf, optimize=True)
        sb = np.einsum("bnd,bd->bn", i_sb, user_sbert[idx], optimize=True)
        with torch.no_grad():
            if kind == "gate":
                fused, _ = gate_model(
                    torch.from_numpy(np.repeat(u_cf[:, None, :], 100, axis=1)),
                    torch.from_numpy(i_cf), torch.from_numpy(i_sb),
                    torch.from_numpy(cf), torch.from_numpy(sb))
                s = fused.numpy()
            else:
                feats = np.concatenate([
                    np.repeat(u_cf[:, None, :], 100, axis=1), i_cf, i_sb], axis=2)
                s = deep_model(torch.from_numpy(feats.reshape(b * 100, -1))).numpy().reshape(b, 100)
        pos_all[start:start + len(idx)] = positions_from_scores(s, gt_pos[idx])
    return metrics_from_positions(pos_all)


def eval_graph_gate() -> dict[str, float]:
    pos_all = np.empty(len(test_idx), dtype=np.int32)
    for start in range(0, len(test_idx), DEEP_BATCH):
        idx = test_idx[start:start + DEEP_BATCH]
        b = len(idx)
        c = cands[idx]
        u_cf = np.asarray(graph_uf[elig_users[idx]], dtype=np.float32)
        i_cf = np.asarray(graph_itf[c], dtype=np.float32)
        i_sb = sbert_cat[c]
        cf = np.einsum("bnd,bd->bn", i_cf, u_cf, optimize=True)
        sb = np.einsum("bnd,bd->bn", i_sb, user_sbert[idx], optimize=True)
        with torch.no_grad():
            fused, _ = graph_gate_model(
                torch.from_numpy(np.repeat(u_cf[:, None, :], 100, axis=1)),
                torch.from_numpy(i_cf), torch.from_numpy(i_sb),
                torch.from_numpy(cf), torch.from_numpy(sb))
            s = fused.numpy()
        pos_all[start:start + len(idx)] = positions_from_scores(s, gt_pos[idx])
    return metrics_from_positions(pos_all)


def eval_two_tower() -> dict[str, float]:
    pos_all = np.empty(len(test_idx), dtype=np.int32)
    for start in range(0, len(test_idx), DEEP_BATCH):
        idx = test_idx[start:start + DEEP_BATCH]
        b = len(idx)
        c = cands[idx]
        u_in = np.asarray(graph_uf[elig_users[idx]], dtype=np.float32)
        i_in = np.concatenate([np.asarray(graph_itf[c], dtype=np.float32), sbert_cat[c]], axis=2)
        with torch.no_grad():
            u_emb = user_tower(torch.from_numpy(u_in)).numpy()
            i_emb = item_tower(torch.from_numpy(i_in.reshape(b * 100, -1))).numpy().reshape(b, 100, -1)
            s = np.einsum("bd,bnd->bn", u_emb, i_emb, optimize=True)
        pos_all[start:start + len(idx)] = positions_from_scores(s, gt_pos[idx])
    return metrics_from_positions(pos_all)


def eval_transformer() -> tuple[dict[str, float], np.ndarray]:
    pos_all = np.empty(len(test_idx), dtype=np.int32)
    scores_out = np.empty((len(test_idx), 100), dtype=np.float32)
    BATCH_U = 32
    for start in range(0, len(test_idx), BATCH_U):
        idx = test_idx[start:start + BATCH_U]
        b = len(idx)
        c = cands[idx]
        u_cf = np.asarray(graph_uf[elig_users[idx]], dtype=np.float32)
        i_cf = np.asarray(graph_itf[c], dtype=np.float32)
        i_sb = sbert_cat[c]
        feats = np.concatenate([
            np.repeat(u_cf[:, None, :], 100, axis=1), i_cf, i_sb], axis=2).astype(np.float32)
        with torch.no_grad():
            s = transformer_model(torch.from_numpy(feats)).numpy()
        pos_all[start:start + len(idx)] = positions_from_scores(s, gt_pos[idx])
        scores_out[start:start + len(idx)] = s
    return metrics_from_positions(pos_all), scores_out

res_gate = eval_deep_head("gate")
print("Light Gated:      ", res_gate)
res_deep = eval_deep_head("deep")
print("Deep Early:       ", res_deep)
res_graph_gate = eval_graph_gate()
print("Graph Light Gated:", res_graph_gate)
res_two_tower = eval_two_tower()
print("Two-Tower:        ", res_two_tower)
res_transformer, transformer_test_scores = eval_transformer()
print("Transformer:      ", res_transformer)


# --------------------------------------------------- feature extraction for RGH
print("\n" + "=" * 56)
print("7. EXTRACTING NORMALIZED MULTI-BRANCH FEATURES FOR PROPOSED MODELS")
print("=" * 56)
def extract_branch_features(idx_arr: np.ndarray):
    # Computes per-user normalized branch scores
    s_als = cf_batch(als_uf, als_itf, idx_arr)
    s_bpr = cf_batch(bpr_uf, bpr_itf, idx_arr)
    s_gcn = cf_batch(graph_uf, graph_itf, idx_arr)
    s_txt = sb_batch(idx_arr)
    
    z_als = mm_rows(s_als)
    z_bpr = mm_rows(s_bpr)
    z_gcn = mm_rows(s_gcn)
    z_txt = mm_rows(s_txt)
    diff_cf = z_als - z_gcn
    cross_txt = z_als * z_txt
    feats = np.stack([z_als, z_bpr, z_gcn, z_txt, diff_cf, cross_txt], axis=-1).astype(np.float32)
    return {
        "z_als": z_als, "z_bpr": z_bpr, "z_gcn": z_gcn, "z_txt": z_txt, "feats": feats
    }

val_feats_dict = extract_branch_features(tune_idx)
test_feats_dict = extract_branch_features(test_idx)


# -------------------------------------------------------- PROPOSED MODEL 1: RGH
print("\n" + "=" * 56)
print("8. PROPOSED MODEL: CF-ANCHORED RESIDUAL GATED HYBRID (RGH)")
print("=" * 56)

class ResidualGatedHybrid(nn.Module):
    """
    CF-Anchored Residual Gated Hybrid (RGH) as specified in Section 5 of
    Ke_hoach_cai_thien_mo_hinh_va_co_so_lap_luan.md:
    score(u, i) = a_cf * z_cf + a_gcn * z_gcn + a_txt * z_txt + g(h_u, h_i) * Delta(u, i)
    """
    def __init__(self, in_features: int = 6, hidden_dim: int = 64):
        super().__init__()
        # Initialized to strongly anchor on CF and Graph CF
        self.a_cf = nn.Parameter(torch.tensor(0.60, dtype=torch.float32))
        self.a_gcn = nn.Parameter(torch.tensor(0.30, dtype=torch.float32))
        self.a_txt = nn.Parameter(torch.tensor(0.10, dtype=torch.float32))

        # DCN-v2 Style Feature Crossing for non-linear residual
        self.cross1 = nn.Linear(in_features, in_features)
        self.cross2 = nn.Linear(in_features, in_features)
        self.deep = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Residual Gate with negative bias initialization (initially g ~ 0.075)
        self.gate_net = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        nn.init.constant_(self.gate_net[-1].bias, -2.5)

    def forward(self, feats, z_cf, z_gcn, z_txt):
        # 1. CF-Anchored Linear Foundation
        linear_anchor = self.a_cf * z_cf + self.a_gcn * z_gcn + self.a_txt * z_txt
        # 2. Cross & Deep Residual
        x0 = feats
        x1 = x0 * torch.sigmoid(self.cross1(x0)) + x0
        x2 = x0 * torch.sigmoid(self.cross2(x1)) + x1
        delta = self.deep(x2).squeeze(-1)
        # 3. Residual Gating
        gate = torch.sigmoid(self.gate_net(feats)).squeeze(-1)
        return linear_anchor + gate * delta


def train_rgh_model(val_dict: dict, val_gt_pos: np.ndarray, epochs: int = 12):
    model = ResidualGatedHybrid(in_features=6, hidden_dim=64)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)
    
    t_feats = torch.from_numpy(val_dict["feats"]).float()
    t_zcf = torch.from_numpy(val_dict["z_als"]).float()
    t_zgcn = torch.from_numpy(val_dict["z_gcn"]).float()
    t_ztxt = torch.from_numpy(val_dict["z_txt"]).float()
    t_gt = torch.from_numpy(val_gt_pos).long()
    
    dataset = TensorDataset(t_feats, t_zcf, t_zgcn, t_ztxt, t_gt)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    model.train()
    for ep in range(epochs):
        total_loss = 0.0
        for b_f, b_cf, b_gcn, b_txt, b_gt in loader:
            optimizer.zero_grad()
            scores = model(b_f, b_cf, b_gcn, b_txt)
            loss = nn.functional.cross_entropy(scores / 0.20, b_gt) # tau = 0.2
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(b_gt)
        if (ep + 1) % 4 == 0 or ep == 0:
            print(f"  [RGH Epoch {ep+1:02d}/{epochs}] Loss = {total_loss/len(t_gt):.4f} "
                  f"(a_cf={model.a_cf.item():.3f}, a_gcn={model.a_gcn.item():.3f}, a_txt={model.a_txt.item():.3f})")
    model.eval()
    return model

print("Training RGH with Listwise Ranking Loss on Validation split...")
rgh_model = train_rgh_model(val_feats_dict, gt_pos[tune_idx], epochs=12)

# Evaluate RGH on Holdout Test
with torch.no_grad():
    rgh_test_scores = rgh_model(
        torch.from_numpy(test_feats_dict["feats"]).float(),
        torch.from_numpy(test_feats_dict["z_als"]).float(),
        torch.from_numpy(test_feats_dict["z_gcn"]).float(),
        torch.from_numpy(test_feats_dict["z_txt"]).float()
    ).numpy()

pos_rgh = positions_from_scores(rgh_test_scores, gt_pos[test_idx])
res_rgh = metrics_from_positions(pos_rgh)
print("PROPOSED RGH:     ", res_rgh)


# -------------------------------------------------- PROPOSED MODEL 2: LAMBDARANK
print("\n" + "=" * 56)
print("9. PROPOSED MODEL: TWO-STAGE LAMBDARANK (LIGHTGBM)")
print("=" * 56)
if HAS_LGB:
    N_tune = len(tune_idx)
    X_val = val_feats_dict["feats"].reshape(N_tune * 100, -1)
    y_val = np.zeros((N_tune, 100), dtype=np.int32)
    for i, g in enumerate(gt_pos[tune_idx]):
        y_val[i, g] = 1
    y_val = y_val.reshape(-1)
    q_val = [100] * N_tune

    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        eval_at=[5, 10, 20],
        n_estimators=100,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        random_state=SEED,
        n_jobs=-1
    )
    print("Fitting LightGBM LambdaRank on Validation queries...")
    ranker.fit(X_val, y_val, group=q_val)

    N_t = len(test_idx)
    X_test = test_feats_dict["feats"].reshape(N_t * 100, -1)
    lgb_test_scores = ranker.predict(X_test).reshape(N_t, 100)
    pos_lgb = positions_from_scores(lgb_test_scores, gt_pos[test_idx])
    res_lgb = metrics_from_positions(pos_lgb)
    print("PROPOSED LambdaRank:", res_lgb)
else:
    print("LightGBM not installed, skipping LambdaRank")
    res_lgb = None
    lgb_test_scores = rgh_test_scores


# --------------------------------------------------- PROPOSED MODEL 3: ENSEMBLE
print("\n" + "=" * 56)
print("10. PROPOSED: STACKED ENSEMBLE HYBRID")
print("=" * 56)
# Blend RGH + LambdaRank + Transformer Fusion + LightGCN
s_ens = (0.35 * mm_rows(rgh_test_scores) + 
         0.35 * mm_rows(lgb_test_scores) + 
         0.20 * mm_rows(transformer_test_scores) + 
         0.10 * mm_rows(test_feats_dict["z_gcn"]))
pos_ens = positions_from_scores(s_ens, gt_pos[test_idx])
res_ensemble = metrics_from_positions(pos_ens)
print("PROPOSED Stacked Ensemble:", res_ensemble)


# -------------------------------------------------------------------- summary
summary = {
    "protocol": "Strict Leave-One-Out (1 Positive from Test split + 99 Random Negatives), shared 2.46M catalog, leak-free SBERT profiles, zero test tuning leakage",
    "eval_catalog_items": CATALOG,
    "n_test_users_holdout": int(len(test_idx)),
    "n_validation_users_tune": int(len(tune_idx)),
    "min_interactions": MIN_INTERACTIONS,
    "models": {
        "ALS (CF Baseline)": res_als,
        "BPR (CF Baseline)": res_bpr,
        "LightGCN (Graph CF)": res_lightgcn,
        "SBERT Alone (CB)": res_sbert,
        f"Late Fusion (alpha={best_alpha})": res_late,
        "Light Gated Fusion": res_gate,
        "Deep Early Fusion": res_deep,
        "Graph Light Gated": res_graph_gate,
        "Two-Tower": res_two_tower,
        "Transformer Fusion": res_transformer,
        "Proposed RGH (Residual Gated Hybrid)": res_rgh,
        "Proposed Two-Stage LambdaRank (LightGBM)": res_lgb,
        "Proposed Stacked Ensemble": res_ensemble
    },
    "eval_time_sec": round(time.time() - t0, 1)
}
(OUT_DIR / "eval_summary.json").write_text(json.dumps(summary, indent=2))

print("\n" + "=" * 88)
print(f"FINAL BENCHMARK COMPARISON ON TEST HOLDOUT (N={len(test_idx):,})")
print("=" * 88)
print(f"{'Model Architecture':<42} {'HR@5':>7} {'HR@10':>7} {'HR@20':>7} {'NDCG@10':>8} {'MRR':>7} {'Delta ALS':>10}")
print("-" * 88)

als_hr10 = res_als["HR@10"]
models_to_display = [
    ("ALS (CF Baseline)", res_als),
    ("BPR (CF Baseline)", res_bpr),
    ("LightGCN (Graph CF)", res_lightgcn),
    ("SBERT Alone (CB)", res_sbert),
    (f"Late Fusion (alpha={best_alpha})", res_late),
    ("Light Gated Fusion", res_gate),
    ("Deep Early Fusion", res_deep),
    ("Graph Light Gated", res_graph_gate),
    ("Two-Tower (Deep Hybrid)", res_two_tower),
    ("Transformer Fusion", res_transformer),
    ("★ Proposed RGH (Residual Gated)", res_rgh),
    ("★ Proposed LambdaRank (LightGBM)", res_lgb),
    ("★ Proposed Stacked Ensemble", res_ensemble)
]

for name, m in models_to_display:
    if m is None:
        continue
    delta_als = (m["HR@10"] - als_hr10) / als_hr10 * 100.0
    sign = "+" if delta_als >= 0 else ""
    delta_str = f"{sign}{delta_als:.1f}%"
    print(f"{name:<42} {m['HR@5']:>7.4f} {m['HR@10']:>7.4f} {m['HR@20']:>7.4f} "
          f"{m['NDCG@10']:>8.4f} {m['MRR']:>7.4f} {delta_str:>10}")
print("=" * 88)

# ----------------------------------------------------------------- visualization
if HAS_MPL:
    try:
        plot_names = [
            "ALS", "BPR", "LightGCN", "SBERT", "Late Fusion",
            "Two-Tower", "Transformer", "RGH (Proposed)", "LambdaRank (Proposed)", "Ensemble (Proposed)"
        ]
        plot_m = [
            res_als, res_bpr, res_lightgcn, res_sbert, res_late,
            res_two_tower, res_transformer, res_rgh, res_lgb, res_ensemble
        ]
        
        valid_plots = [(n, m) for n, m in zip(plot_names, plot_m) if m is not None]
        labels = [x[0] for x in valid_plots]
        hr10_vals = [x[1]["HR@10"] for x in valid_plots]
        ndcg10_vals = [x[1]["NDCG@10"] for x in valid_plots]
        
        x = np.arange(len(labels))
        width = 0.38
        
        fig, ax = plt.subplots(figsize=(14, 6))
        rects1 = ax.bar(x - width/2, hr10_vals, width, label='HR@10', color='#2563eb')
        rects2 = ax.bar(x + width/2, ndcg10_vals, width, label='NDCG@10', color='#10b981')
        
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Recommender System Model Comparison (Amazon Reviews 2.46M Catalog)', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha='right', fontsize=10)
        ax.legend(fontsize=11)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        
        # Add values on top of bars
        for bar in rects1:
            h = bar.get_height()
            ax.annotate(f'{h:.3f}', xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
        for bar in rects2:
            h = bar.get_height()
            ax.annotate(f'{h:.3f}', xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
            
        fig.tight_layout()
        chart_path = OUT_DIR / "datn_recsys_model_benchmark.png"
        plt.savefig(chart_path, dpi=300)
        plt.close()
        print(f"\nSaved benchmark comparison chart to: {chart_path}")
    except Exception as e:
        print("Plotting error:", e)

print(f"\nAll evaluations completed successfully in {round(time.time() - t0, 1)}s")
