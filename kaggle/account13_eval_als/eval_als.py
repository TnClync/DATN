#!/usr/bin/env python
"""Kaggle: Train Late Fusion V3 (ALS + BPR + SBERT, validation-tuned weights).

Khác bản Late Fusion cũ (chỉ ALS + SBERT, min-max toàn cục):
  1. Bổ sung nhánh **BPR** (ensemble ALS+BPR đã thắng ALS standalone).
  2. Chuẩn hoá **per-user z-score** trên tập candidate thay vì min-max toàn cục.
  3. Trọng số (w_als, w_bpr, w_txt) được **tune trên validation** (simplex grid),
     chọn theo HR@10(val), tie-break NDCG@10(val); test chỉ đánh giá một lần.

Single model per kernel. Output:
  - late_fusion_v3_summary.json  (metrics + best weights)
  - als/bpr/sbert references báo cáo trên cùng protocol
"""
import os, sys, time, json, math, gc, subprocess
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass
from pathlib import Path

try:
    import faiss, pyarrow.parquet as pq, implicit, sentence_transformers
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet",
        "sentence-transformers>=2.7", "faiss-cpu", "implicit>=0.7", "pyarrow", "scipy", "tqdm"])
    import faiss, pyarrow.parquet as pq, implicit, sentence_transformers

import numpy as np
import pandas as pd
# tqdm removed for cleaner Kaggle logs

MIN_INTERACTIONS = 3
CATALOG = 2464371
SEED = 42
OUT_DIR = Path("/kaggle/working" if Path("/kaggle/working").exists() else "artifacts")
OUT_DIR.mkdir(parents=True, exist_ok=True)

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

log(f"CONFIG: MIN_INTERACTIONS={MIN_INTERACTIONS} SEED={SEED} | RAM {mem_info()}")


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
log(f"[STEP 1] Base dir: {BASE_DIR}")


def load(name):
    up = find_factor_file(f"{name}_user_factors_f16.npy")
    ip = find_factor_file(f"{name}_item_factors_f16.npy")
    if up is None or ip is None:
        return None
    return np.load(up).astype(np.float32), np.load(ip).astype(np.float32)


als = load("als")
assert als is not None
user_cf, item_cf = als
bpr = load("bpr")
if bpr is not None:
    user_bpr, item_bpr = bpr
    HAS_BPR = True
else:
    log("BPR factors not found - running ALS + SBERT only")
    HAS_BPR = False
    user_bpr = item_bpr = None

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
log(f"[STEP 2] Compact: users={n_users:,} items={n_items:,} SBERT_COVERAGE={SBERT_COVERAGE:,}")

# ---- SBERT item embeddings: 768-d, built once by the SBERT kernel ----
sbert_path = find_sbert_file()
assert sbert_path is not None, "sbert_item_embeddings_f16.npy not found - run the SBERT kernel first"
sbert_rows = np.load(sbert_path, mmap_mode="r").shape[0]
assert sbert_rows >= n_items, f"SBERT rows {sbert_rows:,} < items {n_items:,} - index mismatch"
item_sbert = np.asarray(np.load(sbert_path, mmap_mode="r")[:n_items], dtype=np.float32)
assert item_sbert.shape[1] == 768, f"expected 768-d SBERT, got {item_sbert.shape}"
log(f"[STEP 2] SBERT item embeddings: {item_sbert.shape} | RAM {mem_info()}")

# ---- Test + history ----
test_files = sorted(BASE_DIR.glob("test/*.parquet"))
log(f"[STEP 3] Reading {len(test_files)} test parquet file(s)...")
dfs_test = []
for i, f in enumerate(test_files, 1):
    log(f"[STEP 3]   test {i}/{len(test_files)}: {f.name}")
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

log(f"[STEP 3] Building user content profiles (mean train history) ... RAM {mem_info()}")
user_sbert = np.zeros((n_users, 768), dtype=np.float32)
for u in range(n_users):
    h = hist_by_user_all.get(u)
    if h:
        v = item_sbert[h].mean(axis=0)
        n = float(np.linalg.norm(v))
        user_sbert[u] = v / n if n > 1e-8 else v
user_sbert = user_sbert.astype(np.float32)
log(f"[STEP 3] User content profiles: {user_sbert.shape}")


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


def build_instances(eval_users):
    out = []
    rng = np.random.default_rng(SEED)
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
        out.append({"user": u, "gt": gt, "candidates": cands})
    return out


CAND_POOL = np.arange(SBERT_COVERAGE)
user_inter_count = np.bincount(u_inv, minlength=len(u_uniq))
test_users = np.array([u for u in user_gt.keys() if user_inter_count[u] >= MIN_INTERACTIONS])
test_users = np.array([u for u in test_users if any(it < SBERT_COVERAGE for it in user_gt[u])])
eval_users = np.random.default_rng(SEED).choice(test_users, size=min(200000, len(test_users)), replace=False)
eval_instances = build_instances(eval_users)
TEST_N = 5000
test_inst = eval_instances[:TEST_N]
val_inst = eval_instances[TEST_N:TEST_N + 2000]
log(f"[STEP 4] Prepared {len(eval_instances):,} instances | test={len(test_inst)} val={len(val_inst)}")


# ---- Score helpers (per-user z-score on the instance's candidate set) ----
def zscore(v):
    v = np.asarray(v, dtype=np.float32)
    m, s = float(v.mean()), float(v.std())
    if s < 1e-8:
        return np.zeros_like(v)
    return (v - m) / (s + 1e-8)


def fused_scores(u, cands, w_als, w_bpr, w_txt):
    z_als = zscore(item_cf[cands] @ user_cf[u])
    z_txt = zscore(item_sbert[cands] @ user_sbert[u])
    if HAS_BPR:
        z_bpr = zscore(item_bpr[cands] @ user_bpr[u])
    else:
        z_bpr = z_als
    return w_als * z_als + w_bpr * z_bpr + w_txt * z_txt


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




# ===========================================================================
# EVAL-ONLY notebook: exactly ONE model (ALS) on the shared protocol
# ===========================================================================

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

print("\n" + "=" * 50)
print("[EVAL] als  (test 5000 users / val 2000 users, 1 pos + 99 negs)")
print("=" * 50)

SCORER = lambda u, c: item_cf[c] @ user_cf[u]

res_val = eval_scorer(SCORER, val_inst)
res_test = eval_scorer(SCORER, test_inst)
log(f"[EVAL] als VAL : {res_val}")
log(f"[EVAL] als TEST: {res_test}")

summary = {
    "model": "als",
    "protocol": "1 positive + 99 random negatives, shared SBERT-covered catalog, leak-free SBERT user profiles",
    "n_test_users": len(test_inst),
    "n_val_users": len(val_inst),
    "min_interactions": MIN_INTERACTIONS,
    "val": res_val,
    "test": res_test,
}
(OUT_DIR / "eval_als.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
log(f"DONE in {time.time()-_T0:.1f}s | RAM {mem_info()}")
print("[ARTIFACT SAVED]", flush=True)
