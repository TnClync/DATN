#!/usr/bin/env python
"""Full-test evaluation kernel: compare all trained models, no retraining.

Evaluates on EVERY eligible test user (users with >=3 train interactions and at
least one test positive inside the SBERT-covered 200k-item catalog), not a
sample. Protocol: 1 positive + 99 random negatives, leak-free SBERT profiles.

Artifacts are mounted from `nguynteo/deep-hybrid-recommender-datn`.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn

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
    hits = sorted(Path("/kaggle/input").rglob(name))
    if not hits:
        print("MISSING input file:", name)
        for root in sorted(Path("/kaggle/input").iterdir()):
            print("  input:", root)
        raise FileNotFoundError(name)
    # Prefer artifacts produced by the dedicated single-model training kernels.
    preferred = []
    if name == "sbert_item_embeddings_f16.npy":
        preferred = [p for p in hits if "train-sbert" in str(p)]
    elif name in ("light_gated_fusion.pt", "deep_early_fusion.pt", "graph_light_gated_fusion.pt",
                  "two_tower_fusion.pt", "transformer_fusion.pt"):
        pref_key = {"light_gated_fusion.pt": "train-light_gated",
                    "deep_early_fusion.pt": "train-deep_early",
                    "graph_light_gated_fusion.pt": "train-graph_light_gated",
                    "two_tower_fusion.pt": "train-two_tower",
                    "transformer_fusion.pt": "train-transformer"}[name]
        preferred = [p for p in hits if pref_key in str(p)]
    if not preferred:
        preferred = [p for p in hits if "deep-hybrid" in str(p)]
    chosen = preferred[0] if preferred else hits[0]
    print(f"Using {name}: {chosen}")
    return chosen


def find_sbert_file(dim: int = SBERT_DIM):
    """Locate the SBERT artifact with the expected dimension (768 = meta | reviewer)."""
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
OUT_DIR = Path("/kaggle/working")
OUT_DIR.mkdir(parents=True, exist_ok=True)

t0 = time.time()
print("\n" + "=" * 56)
print("1. LOADING TRAINED ARTIFACTS (no retraining)")
print("=" * 56)
als_uf = np.load(find_file("als_user_factors_f16.npy"), mmap_mode="r")
als_itf = np.load(find_file("als_item_factors_f16.npy")).astype(np.float32)
bpr_uf = np.load(find_file("bpr_user_factors_f16.npy"), mmap_mode="r")
bpr_itf = np.load(find_file("bpr_item_factors_f16.npy")).astype(np.float32)
item_sbert = np.load(find_sbert_file(), mmap_mode="r")
sbert_path = find_sbert_file()
assert sbert_path is not None, "768-d sbert_item_embeddings_f16.npy not found - run the SBERT kernel first"
gate_path = find_file("light_gated_fusion.pt")
deep_path = find_file("deep_early_fusion.pt")
u_ids = np.load(find_file("user_ids.npy"))
i_ids = np.load(find_file("item_ids.npy"))
n_users, n_items = len(u_ids), len(i_ids)
print(f"Ids: {n_users:,} users x {n_items:,} items | SBERT {item_sbert.shape}")
# Only the first CATALOG rows are ever used; cache them in RAM as float32.
sbert_cat = np.asarray(item_sbert[:CATALOG], dtype=np.float32)
assert sbert_cat.shape[1] == SBERT_DIM, f"expected {SBERT_DIM}-d SBERT, got {sbert_cat.shape}"
print(f"SBERT catalog cache: {sbert_cat.shape} ({sbert_cat.nbytes / 1e6:.0f} MB)")


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
N = len(elig_users)
print(f"Eligible test users (>= {MIN_INTERACTIONS} train, positive in {CATALOG:,}-item catalog): {N:,}")

# all test positives for eligible users (catalog + non-catalog, to ban properly)
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
print("3. BUILDING EVAL INSTANCES (all users)")
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
print(f"Instances: {N:,}")

tune_idx = rng.choice(N, size=min(TUNE_N, N), replace=False)
tune_mask = np.zeros(N, dtype=bool)
tune_mask[tune_idx] = True


# ------------------------------------------------------------------- metrics
def positions_from_scores(scores: np.ndarray, pos: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores, axis=1)
    return (order == pos[:, None]).argmax(axis=1)


def metrics_from_positions(pos: np.ndarray, valid: np.ndarray | None = None) -> dict[str, float]:
    """Rank -> metrics. `valid=False` rows count as misses (all metrics 0), matching
    the retrieval protocol where an un-retrieved ground truth is a total miss."""
    pos = np.asarray(pos, dtype=np.float32)
    if valid is None:
        valid = np.ones(len(pos), dtype=bool)
    valid = np.asarray(valid, dtype=bool)
    posv = np.where(valid, pos, 1e6)
    return {
        "HR@5": round(float(((posv < 5) & valid).mean()), 4),
        "HR@10": round(float(((posv < 10) & valid).mean()), 4),
        "HR@20": round(float(((posv < 20) & valid).mean()), 4),
        "NDCG@5": round(float(np.where((posv < 5) & valid, 1.0 / np.log2(posv + 2), 0.0).mean()), 4),
        "NDCG@10": round(float(np.where((posv < 10) & valid, 1.0 / np.log2(posv + 2), 0.0).mean()), 4),
        "NDCG@20": round(float(np.where((posv < 20) & valid, 1.0 / np.log2(posv + 2), 0.0).mean()), 4),
        "MRR": round(float(np.where(valid, 1.0 / (posv + 1), 0.0).mean()), 4),
        "n_eval_users": int(len(pos)),
    }


def cf_batch(uf, itf, idx):
    """Row index -> factor index via elig_users (rows are NOT compact user ids)."""
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
    return np.where(rng_ < 1e-8, 0.0, (x - mn) / np.maximum(rng_, 1e-8))


def eval_baseline(kind: str) -> dict[str, float]:
    pos_all = np.empty(N, dtype=np.int32)
    step = BATCH if kind != "sbert" else 512
    for start in range(0, N, step):
        idx = np.arange(start, min(start + step, N))
        if kind == "als":
            s = cf_batch(als_uf, als_itf, idx)
        elif kind == "bpr":
            s = cf_batch(bpr_uf, bpr_itf, idx)
        else:
            s = sb_batch(idx)
        pos_all[idx] = positions_from_scores(s, gt_pos[idx])
    return metrics_from_positions(pos_all)


def eval_fusion(alpha: float, only_tune: bool = False) -> dict[str, float]:
    rows = tune_idx if only_tune else rows_all
    pos = np.empty(len(rows), dtype=np.int32)
    for start in range(0, len(rows), 512):
        idx = rows[start:start + 512]
        cf = cf_batch(als_uf, als_itf, idx)
        sb = sb_batch(idx)
        s = alpha * mm_rows(cf) + (1.0 - alpha) * mm_rows(sb)
        pos[start:start + len(idx)] = positions_from_scores(s, gt_pos[idx])
    return metrics_from_positions(pos)


rows_all = np.arange(N)
print("\n" + "=" * 56)
print("4. BASELINES (all test users)")
print("=" * 56)
res_als = eval_baseline("als")
print("ALS   :", res_als)
res_bpr = eval_baseline("bpr")
print("BPR   :", res_bpr)
res_sbert = eval_baseline("sbert")
print("SBERT :", res_sbert)

print("\n" + "=" * 56)
print("5. LATE FUSION (alpha tuned on 10k subset)")
print("=" * 56)
grid = {}
best_alpha, best_hr = 0.9, -1.0
for a in [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    m = eval_fusion(a, only_tune=True)
    grid[a] = m
    print(f"alpha={a}: tune HR@10={m['HR@10']:.4f}")
    if m["HR@10"] > best_hr:
        best_hr, best_alpha = m["HR@10"], a
res_late = eval_fusion(best_alpha)
print(f"Best alpha={best_alpha} -> {res_late}")


# ------------------------------------------------------------------ two-stage
print("\n" + "=" * 56)
print("6. TWO-STAGE RETRIEVAL (all test users)")
print("=" * 56)
ts_res = None
ts_best = None
if HAS_FAISS:
    # Both retrievers search the SAME shared 200k-item catalog used by every model.
    cf_index = faiss.IndexFlatIP(als_itf.shape[1])
    cf_index.add(als_itf[:CATALOG])
    sb_index = faiss.IndexFlatIP(SBERT_DIM)
    sb_index.add(sbert_cat)
    print("FAISS indexes:", cf_index.ntotal, sb_index.ntotal)

    # ---- retrieve once for every user, then reuse across alpha values ----------
    print("FAISS searching all users (CF top-60 + SBERT top-60)...")
    I_cf = np.empty((N, 60), dtype=np.int64)
    I_sb = np.empty((N, 60), dtype=np.int64)
    for start in range(0, N, 2048):
        idx = rows_all[start:start + 2048]
        _, ic = cf_index.search(np.asarray(als_uf[elig_users[idx]], dtype=np.float32), 60)
        _, isb = sb_index.search(np.asarray(user_sbert[idx], dtype=np.float32), 60)
        I_cf[start:start + len(idx)] = ic
        I_sb[start:start + len(idx)] = isb
        if start % 40960 == 0:
            print(f"  search {start}/{N}")

    def ts_rerank(alpha: float, idx: np.ndarray, with_hits: bool = False):
        out = np.empty(len(idx), dtype=np.int32)
        valid = np.zeros(len(idx), dtype=bool)
        hits = np.zeros((len(idx), 3), dtype=bool)  # cf@50, sbert@50, union@100
        for j, row in enumerate(idx):
            row = int(row)
            ban = banned_list[row]
            gt = gts[row]
            cf_ids = I_cf[row]
            sb_ids = I_sb[row]
            cf_top = cf_ids[~ismember(cf_ids, ban)][:50]
            sb_top = sb_ids[~ismember(sb_ids, ban)][:50]
            hits[j, 0] = bool(np.any(cf_top == gt))
            hits[j, 1] = bool(np.any(sb_top == gt))
            union = np.union1d(cf_top, sb_top)
            if len(union) == 0 or not bool(np.any(union == gt)):
                out[j] = 0
                continue
            hits[j, 2] = True
            uf = np.asarray(als_uf[elig_users[row]], dtype=np.float32)
            cfu = np.asarray(als_itf[union] @ uf, dtype=np.float32)
            usb = np.asarray(user_sbert[row], dtype=np.float32)
            sbu = sbert_cat[union] @ usb
            fused = alpha * mm_rows(cfu[None, :])[0] + (1.0 - alpha) * mm_rows(sbu[None, :])[0]
            gt_score = fused[np.where(union == gt)[0][0]]
            out[j] = int((fused > gt_score).sum())
            valid[j] = True
        if with_hits:
            return out, valid, hits
        return out, valid

    # tune two-stage alpha on the same 10k subset (cached retrieval)
    ts_best, ts_best_hr = 1.0, -1.0
    for a in [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        pos, ok = ts_rerank(a, tune_idx)
        m = metrics_from_positions(pos, ok)
        print(f"two-stage alpha={a}: tune HR@10={m['HR@10']:.4f}")
        if m["HR@10"] > ts_best_hr:
            ts_best_hr, ts_best = m["HR@10"], a

    print(f"Two-stage best alpha={ts_best} -> full evaluation...")
    pos_all = np.empty(N, dtype=np.int32)
    ok_all = np.zeros(N, dtype=bool)
    hits_all = np.zeros((N, 3), dtype=bool)
    for start in range(0, N, 4096):
        idx = rows_all[start:start + 4096]
        p, ok, hits = ts_rerank(ts_best, idx, with_hits=True)
        pos_all[start:start + len(idx)] = p
        ok_all[start:start + len(idx)] = ok
        hits_all[start:start + len(idx)] = hits
        if start % 40960 == 0:
            print(f"  two-stage {start}/{N}")
    ts_res = metrics_from_positions(pos_all, ok_all)
    ts_recall = {
        "cf@50": round(float(hits_all[:, 0].mean()), 4),
        "sbert@50": round(float(hits_all[:, 1].mean()), 4),
        "union@100": round(float(hits_all[:, 2].mean()), 4),
    }
    print("Two-stage retrieval recall:", ts_recall)
else:
    print("Skipping two-stage: faiss not available")
    ts_recall = None


# ---------------------------------------------------------------- deep heads
class LightGate(nn.Module):
    def __init__(self, in_features=32 + 32 + SBERT_DIM):
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
        x0, xi = x, x
        for layer in self.cross:
            xi = x0 * torch.sigmoid(layer(xi)) + xi
        return (xi.mean(dim=-1, keepdim=True) + self.deep(x)).squeeze(-1)


gate_model = LightGate()
gate_model.load_state_dict(torch.load(gate_path, map_location="cpu"))
gate_model.eval()
deep_model = DeepEarlyFusion(32 + 32 + SBERT_DIM)
deep_model.load_state_dict(torch.load(deep_path, map_location="cpu"))
deep_model.eval()


def eval_deep_head(kind: str) -> dict[str, float]:
    pos_all = np.empty(N, dtype=np.int32)
    for start in range(0, N, DEEP_BATCH):
        idx = np.arange(start, min(start + DEEP_BATCH, N))
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
        pos_all[idx] = positions_from_scores(s, gt_pos[idx])
    return metrics_from_positions(pos_all)


print("\n" + "=" * 56)
print("7. DEEP HYBRID HEADS (all test users)")
print("=" * 56)
res_gate = eval_deep_head("gate")
print("Light Gated:", res_gate)
res_deep = eval_deep_head("deep")
print("Deep Early :", res_deep)


# -------------------------------------------------------------------- summary
summary = {
    "protocol": "ALL eligible test users, 1 positive + 99 random negatives, shared 200k-item catalog, leak-free SBERT profiles",
    "eval_catalog_items": CATALOG,
    "n_test_positive_rows": int(len(tu)),
    "n_eval_users": int(N),
    "min_interactions": MIN_INTERACTIONS,
    "als": res_als,
    "bpr": res_bpr,
    "sbert": res_sbert,
    "late_fusion": {"best_alpha": best_alpha, "tune_HR10": round(best_hr, 4),
                    "test": res_late, "grid_tune": {str(a): m for a, m in grid.items()}},
    "two_stage": ({"best_alpha": ts_best, "tune_HR10": round(ts_best_hr, 4),
                   "test": ts_res, "retrieval_recall": ts_recall}
                  if ts_res is not None else None),
    "light_gated": res_gate,
    "deep_early_fusion": res_deep,
    "eval_time_sec": round(time.time() - t0, 1),
}
(OUT_DIR / "eval_summary.json").write_text(json.dumps(summary, indent=2))

print("\n" + "=" * 76)
print(f"FINAL COMPARISON (ALL TEST SET, N={N:,})")
print("=" * 76)
print(f"{'Model':<26} {'HR@5':>7} {'HR@10':>7} {'HR@20':>7} {'NDCG@10':>8} {'MRR':>7}")
rows = [
    ("ALS", res_als),
    ("BPR", res_bpr),
    ("SBERT", res_sbert),
    (f"Late Fusion (a={best_alpha})", res_late),
    ((f"Two-Stage (a={ts_best})", ts_res) if ts_res else ("Two-Stage", None)),
    ("Light Gated Fusion", res_gate),
    ("Deep Early Fusion", res_deep),
]
for name, m in rows:
    if m is None:
        print(f"{name:<26} {'n/a':>7}")
        continue
    print(f"{name:<26} {m['HR@5']:>7.4f} {m['HR@10']:>7.4f} {m['HR@20']:>7.4f} "
          f"{m['NDCG@10']:>8.4f} {m['MRR']:>7.4f}")
print("=" * 76)
print(json.dumps(summary, indent=2))
print("DONE in", round(time.time() - t0, 1), "s")
