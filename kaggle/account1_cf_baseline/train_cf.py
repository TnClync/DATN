#!/usr/bin/env python
"""Kaggle Account 1 (nguynteo): Train CF Models (ALS & BPR) using `implicit`.

- Chỉ dùng user có >= 5 tương tác (implicit positive).
- Compact re-indexing để tiết kiệm RAM và xuất user_ids/item_ids mapping.
- Xuất factors float16 + ids cho serving pipeline.
"""
import os
import sys
import time
import json
import math
from pathlib import Path

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
    return Path("data")

BASE_DIR = find_base_dir()
OUT_DIR = Path("/kaggle/working" if Path("/kaggle/working").exists() else "artifacts/factors")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_INTERACTIONS = 3

# 1. LOAD + FILTER
print("\n" + "="*50)
print("1. LOADING TRAINING DATA (min 5 interactions)")
print("="*50)
train_files = sorted(BASE_DIR.glob("train/*.parquet"))
if not train_files:
    raise FileNotFoundError(f"No train parquet files found in {BASE_DIR}/train")

dfs = []
for f in tqdm(train_files, desc="Reading train files"):
    tbl = pq.read_table(f, columns=["user_idx", "item_idx", "rating"])
    df = tbl.to_pandas()
    df = df[df["rating"] >= 4.0]
    dfs.append(df[["user_idx", "item_idx"]])

train_df = pd.concat(dfs, ignore_index=True)
# filter users with >= MIN_INTERACTIONS
user_counts = train_df["user_idx"].value_counts()
active_users = user_counts[user_counts >= MIN_INTERACTIONS].index
train_df = train_df[train_df["user_idx"].isin(active_users)].copy()
print(f"After filter (>= {MIN_INTERACTIONS} interactions): users={train_df.user_idx.nunique():,} interactions={len(train_df):,}")

# compact reindex
u_raw = train_df["user_idx"].to_numpy(dtype=np.int64)
i_raw = train_df["item_idx"].to_numpy(dtype=np.int64)
u_uniq, u_inv = np.unique(u_raw, return_inverse=True)
i_uniq, i_inv = np.unique(i_raw, return_inverse=True)
n_users = len(u_uniq)
n_items = len(i_uniq)
print(f"Compact matrix: {n_users:,} users x {n_items:,} items")

data_ones = np.ones(len(u_inv), dtype=np.float32)
user_item_matrix = csr_matrix((data_ones, (u_inv, i_inv)), shape=(n_users, n_items))

# 2. ALS
print("\n" + "="*50)
print("2. TRAINING ALS")
print("="*50)
t0 = time.time()
als_model = AlternatingLeastSquares(factors=32, regularization=0.05, iterations=12,
                                     calculate_training_loss=True, num_threads=0, random_state=42)
als_model.fit(user_item_matrix)
als_time = time.time() - t0
print(f"ALS done in {als_time:.1f}s")

als_uf = np.asarray(als_model.user_factors, dtype=np.float32)
als_it = np.asarray(als_model.item_factors, dtype=np.float32)

np.save(OUT_DIR / "als_user_factors_f16.npy", als_uf.astype(np.float16))
np.save(OUT_DIR / "als_item_factors_f16.npy", als_it.astype(np.float16))
np.save(OUT_DIR / "als_user_factors_f32.npy", als_uf)
np.save(OUT_DIR / "als_item_factors_f32.npy", als_it)
np.save(OUT_DIR / "als_user_ids.npy", u_uniq)
np.save(OUT_DIR / "als_item_ids.npy", i_uniq)

# 3. BPR
print("\n" + "="*50)
print("3. TRAINING BPR")
print("="*50)
t0 = time.time()
bpr_model = BayesianPersonalizedRanking(factors=32, learning_rate=0.05, regularization=0.05,
                                        iterations=30, num_threads=0, random_state=42)
bpr_model.fit(user_item_matrix)
bpr_time = time.time() - t0
print(f"BPR done in {bpr_time:.1f}s")

bpr_uf = np.asarray(bpr_model.user_factors, dtype=np.float32)
bpr_it = np.asarray(bpr_model.item_factors, dtype=np.float32)

np.save(OUT_DIR / "bpr_user_factors_f16.npy", bpr_uf.astype(np.float16))
np.save(OUT_DIR / "bpr_item_factors_f16.npy", bpr_it.astype(np.float16))
np.save(OUT_DIR / "bpr_user_factors_f32.npy", bpr_uf)
np.save(OUT_DIR / "bpr_item_factors_f32.npy", bpr_it)
np.save(OUT_DIR / "bpr_user_ids.npy", u_uniq)
np.save(OUT_DIR / "bpr_item_ids.npy", i_uniq)

# meta
meta = {"min_interactions": MIN_INTERACTIONS, "n_users": int(n_users), "n_items": int(n_items),
        "als_train_time": als_time, "bpr_train_time": bpr_time}
with open(OUT_DIR / "cf_meta.json", "w") as f:
    json.dump(meta, f, indent=2)
print("DONE. Artifacts saved to", OUT_DIR)
