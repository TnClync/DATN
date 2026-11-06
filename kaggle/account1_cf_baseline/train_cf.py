#!/usr/bin/env python
"""Kaggle Account 1 (nguynteo): Train CF Models (ALS & BPR) using `implicit` library.

Outputs:
  - als_user_factors_f16.npy, als_item_factors_f16.npy
  - bpr_user_factors_f16.npy, bpr_item_factors_f16.npy
  - cf_evaluation_metrics.json (HR@5/10/20, NDCG@5/10/20, MRR)
"""
import os
import sys
import time
import json
import math
from pathlib import Path

# Ensure required libraries
import subprocess
try:
    import implicit
    import pyarrow.parquet as pq
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "implicit>=0.7", "pyarrow", "scipy", "tqdm"])
    import implicit
    import pyarrow.parquet as pq

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking
from tqdm import tqdm

print("Python:", sys.version)
print("implicit version:", implicit.__version__)

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
            print(f"Using dataset directory: {c}")
            return c
    if Path("/kaggle/input").exists():
        for p in Path("/kaggle/input").rglob("train"):
            if p.is_dir():
                print(f"Discovered train at: {p.parent}")
                return p.parent
    fallback = Path("data")
    print(f"Falling back to {fallback}")
    return fallback

BASE_DIR = find_base_dir()
OUT_DIR = Path("/kaggle/working" if Path("/kaggle/working").exists() else "artifacts/factors")
OUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_USERS = 22_048_170
NUM_ITEMS = 3_686_940

# 1. LOAD TRAINING DATA
print("\n" + "="*50)
print("1. LOADING TRAINING DATA")
print("="*50)
train_files = sorted(BASE_DIR.glob("train/*.parquet"))
if not train_files:
    raise FileNotFoundError(f"No train parquet files found in {BASE_DIR}/train")

dfs = []
for f in tqdm(train_files, desc="Reading train files"):
    tbl = pq.read_table(f, columns=["user_idx", "item_idx", "rating"])
    df = tbl.to_pandas()
    # Filter positive implicit interactions
    df = df[df["rating"] >= 4.0]
    dfs.append(df[["user_idx", "item_idx"]])

train_df = pd.concat(dfs, ignore_index=True)
n_interactions = len(train_df)
n_users_train = train_df["user_idx"].nunique()
n_items_train = train_df["item_idx"].nunique()
print(f"Train Interactions (>=4.0): {n_interactions:,}")
print(f"Active Users: {n_users_train:,} | Active Items: {n_items_train:,}")

# Build CSR Matrix (Users x Items)
print("Constructing Sparse CSR Interaction Matrix...")
u_arr = train_df["user_idx"].to_numpy(dtype=np.int64)
i_arr = train_df["item_idx"].to_numpy(dtype=np.int64)
del train_df

max_u = max(NUM_USERS, int(u_arr.max()) + 1)
max_i = max(NUM_ITEMS, int(i_arr.max()) + 1)

data_ones = np.ones(len(u_arr), dtype=np.float32)
user_item_matrix = csr_matrix((data_ones, (u_arr, i_arr)), shape=(max_u, max_i))
print(f"CSR Matrix Shape: {user_item_matrix.shape}, NNZ: {user_item_matrix.nnz:,}")

# 2. TRAIN ALS (implicit)
print("\n" + "="*50)
print("2. TRAINING ALS WITH `implicit`")
print("="*50)
t0 = time.time()
als_model = AlternatingLeastSquares(
    factors=64,
    regularization=0.05,
    iterations=15,
    calculate_training_loss=True,
    num_threads=0,
    random_state=42
)
als_model.fit(user_item_matrix)
als_train_time = time.time() - t0
print(f"ALS Training completed in {als_train_time:.1f}s")

als_user_factors = np.asarray(als_model.user_factors, dtype=np.float32)
als_item_factors = np.asarray(als_model.item_factors, dtype=np.float32)

print("Exporting ALS factors (float16 + float32)...")
np.save(OUT_DIR / "als_user_factors_f16.npy", als_user_factors.astype(np.float16))
np.save(OUT_DIR / "als_item_factors_f16.npy", als_item_factors.astype(np.float16))
np.save(OUT_DIR / "als_user_factors_f32.npy", als_user_factors)
np.save(OUT_DIR / "als_item_factors_f32.npy", als_item_factors)

# 3. TRAIN BPR (implicit)
print("\n" + "="*50)
print("3. TRAINING BPR WITH `implicit`")
print("="*50)
t0 = time.time()
bpr_model = BayesianPersonalizedRanking(
    factors=64,
    learning_rate=0.05,
    regularization=0.01,
    iterations=25,
    num_threads=0,
    random_state=42
)
bpr_model.fit(user_item_matrix)
bpr_train_time = time.time() - t0
print(f"BPR Training completed in {bpr_train_time:.1f}s")

bpr_user_factors = np.asarray(bpr_model.user_factors, dtype=np.float32)
bpr_item_factors = np.asarray(bpr_model.item_factors, dtype=np.float32)

print("Exporting BPR factors (float16 + float32)...")
np.save(OUT_DIR / "bpr_user_factors_f16.npy", bpr_user_factors.astype(np.float16))
np.save(OUT_DIR / "bpr_item_factors_f16.npy", bpr_item_factors.astype(np.float16))
np.save(OUT_DIR / "bpr_user_factors_f32.npy", bpr_user_factors)
np.save(OUT_DIR / "bpr_item_factors_f32.npy", bpr_item_factors)

del user_item_matrix, als_model, bpr_model

# 4. EVALUATION ON TEST SET
print("\n" + "="*50)
print("4. EVALUATING MODELS ON TEST SET (HR@5/10/20, NDCG@5/10/20, MRR)")
print("="*50)
test_files = sorted(BASE_DIR.glob("test/*.parquet"))
if not test_files:
    print("Warning: No test files found in BASE_DIR/test. Checking alternative paths...")
    test_files = list(BASE_DIR.rglob("*test*.parquet"))

dfs_test = []
for f in tqdm(test_files, desc="Reading test set"):
    tbl = pq.read_table(f, columns=["user_idx", "item_idx", "rating"])
    d = tbl.to_pandas()
    d = d[d["rating"] >= 4.0]
    dfs_test.append(d[["user_idx", "item_idx"]])

test_df = pd.concat(dfs_test, ignore_index=True)
print(f"Test Positive Interactions: {len(test_df):,}")

# Sample 5,000 users for standardized ranking evaluation
rng = np.random.default_rng(42)
test_users = test_df["user_idx"].unique()
eval_users = rng.choice(test_users, size=min(5000, len(test_users)), replace=False)
user_gt = test_df.groupby("user_idx")["item_idx"].apply(set).to_dict()

all_items_pool = np.arange(min(NUM_ITEMS, len(als_item_factors)))

def evaluate_cf_model(user_factors, item_factors, model_name="Model"):
    metrics = {
        "HR@5": 0.0, "HR@10": 0.0, "HR@20": 0.0,
        "NDCG@5": 0.0, "NDCG@10": 0.0, "NDCG@20": 0.0,
        "MRR": 0.0
    }
    cnt = 0
    for u in tqdm(eval_users, desc=f"Evaluating {model_name}"):
        gt_items = list(user_gt.get(u, set()))
        if not gt_items:
            continue
        gt_item = gt_items[0]
        if gt_item >= len(item_factors) or u >= len(user_factors):
            continue
        
        # Sample 99 negatives
        negs = rng.choice(all_items_pool, size=150, replace=False)
        negs = [item for item in negs if item not in user_gt[u]][:99]
        candidates = np.array([gt_item] + negs, dtype=np.int64)
        
        # Dot product scores
        u_vec = user_factors[u]
        scores = item_factors[candidates] @ u_vec
        ranked_indices = candidates[np.argsort(-scores)]
        
        # Rank of target
        pos_list = np.where(ranked_indices == gt_item)[0]
        if len(pos_list) == 0:
            continue
        pos = int(pos_list[0])
        
        # Hit Rate
        if pos < 5: metrics["HR@5"] += 1.0
        if pos < 10: metrics["HR@10"] += 1.0
        if pos < 20: metrics["HR@20"] += 1.0
        
        # NDCG
        if pos < 5: metrics["NDCG@5"] += 1.0 / math.log2(pos + 2)
        if pos < 10: metrics["NDCG@10"] += 1.0 / math.log2(pos + 2)
        if pos < 20: metrics["NDCG@20"] += 1.0 / math.log2(pos + 2)
        
        # MRR
        metrics["MRR"] += 1.0 / (pos + 1)
        cnt += 1

    if cnt > 0:
        for k in metrics:
            metrics[k] /= cnt
    metrics["n_users_evaluated"] = cnt
    return metrics

als_results = evaluate_cf_model(als_user_factors, als_item_factors, "ALS (implicit)")
bpr_results = evaluate_cf_model(bpr_user_factors, bpr_item_factors, "BPR (implicit)")

print("\n" + "="*50)
print("BENCHMARK RESULTS:")
print(f"ALS: HR@10={als_results['HR@10']:.4f}, NDCG@10={als_results['NDCG@10']:.4f}, MRR={als_results['MRR']:.4f}")
print(f"BPR: HR@10={bpr_results['HR@10']:.4f}, NDCG@10={bpr_results['NDCG@10']:.4f}, MRR={bpr_results['MRR']:.4f}")
print("="*50)

final_summary = {
    "dataset": "Amazon Reviews (Books + Electronics)",
    "n_train_interactions": int(n_interactions),
    "models": {
        "ALS": {
            "train_time_sec": als_train_time,
            "metrics": als_results
        },
        "BPR": {
            "train_time_sec": bpr_train_time,
            "metrics": bpr_results
        }
    }
}

metrics_path = OUT_DIR / "cf_evaluation_metrics.json"
metrics_path.write_text(json.dumps(final_summary, indent=2))
print(f"Saved evaluation metrics to {metrics_path}")
print("All tasks for Account 1 completed successfully!")
