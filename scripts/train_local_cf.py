#!/usr/bin/env python
"""Local Training Script for Collaborative Filtering using `implicit` library.

Optimized for 8GB RAM CPU machines:
  - Supports ALS and BPR
  - Memory-efficient Parquet streaming
  - Contiguous re-indexing to train 10x - 50x faster with < 500MB RAM
  - Direct export to artifacts/factors/{model}_{user,item}_factors_{f16,f32}.npy
  - Full evaluation: HR@5/10/20, NDCG@5/10/20, MRR

Usage:
  # Fast local train (active users/items, takes ~30-60 seconds on CPU):
  python scripts/train_local_cf.py --model als --max-users 100000 --factors 32 --iters 10

  # Train BPR locally:
  python scripts/train_local_cf.py --model bpr --max-users 100000 --factors 32 --iters 15
"""
from __future__ import annotations

import os
# Configure OpenBLAS/MKL threads before importing numpy/implicit to avoid threadpool contention
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.sparse import csr_matrix
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import TRAIN_DIR, TEST_DIR, FACTORS_DIR, NUM_USERS, NUM_ITEMS
from src.models.cf import train_cf, save_factors

def load_local_train_data(max_users: int = 100_000, min_rating: float = 4.0):
    parts = sorted(TRAIN_DIR.glob("*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No train parquet files found in {TRAIN_DIR}")
    
    print(f"Reading {len(parts)} parquet parts from {TRAIN_DIR}...")
    dfs = []
    for p in tqdm(parts, desc="Streaming Parquet parts"):
        t = pq.read_table(p, columns=["user_idx", "item_idx", "rating"])
        d = t.to_pandas()
        d = d[d["rating"] >= min_rating]
        dfs.append(d[["user_idx", "item_idx"]])
    
    df = pd.concat(dfs, ignore_index=True)
    total_pos = len(df)
    print(f"Total positive interactions (rating >= {min_rating}): {total_pos:,}")
    
    # Optional active-user filter to protect 8GB RAM
    if max_users > 0 and df["user_idx"].nunique() > max_users:
        print(f"Filtering to Top {max_users:,} most active users for ultra-fast 8GB RAM training...")
        top_users = df["user_idx"].value_counts().nlargest(max_users).index
        df = df[df["user_idx"].isin(top_users)].copy()
        print(f"Active subset interactions: {len(df):,} ({len(df)/total_pos*100:.1f}% retained)")
    
    return df

def build_compact_matrix(df: pd.DataFrame):
    u_raw = df["user_idx"].to_numpy(dtype=np.int64)
    i_raw = df["item_idx"].to_numpy(dtype=np.int64)
    
    u_uniq, u_inv = np.unique(u_raw, return_inverse=True)
    i_uniq, i_inv = np.unique(i_raw, return_inverse=True)
    
    n_users = len(u_uniq)
    n_items = len(i_uniq)
    data = np.ones(len(u_inv), dtype=np.float32)
    
    print(f"Constructing Compact CSR Matrix: ({n_users:,} users x {n_items:,} items)...")
    mat = csr_matrix((data, (u_inv, i_inv)), shape=(n_users, n_items))
    print(f"Matrix built successfully. Non-zero entries: {mat.nnz:,}")
    return mat, u_uniq, i_uniq

def evaluate_on_test(user_factors: np.ndarray, item_factors: np.ndarray, u_uniq: np.ndarray, i_uniq: np.ndarray, sample_n: int = 2000):
    test_parts = sorted(TEST_DIR.glob("*.parquet"))
    if not test_parts:
        print("No test files found, skipping eval.")
        return {}
    
    print(f"Loading test set for ranking evaluation ({sample_n} users)...")
    dfs = []
    for p in test_parts[:5]:  # read subset of test for quick local eval
        t = pq.read_table(p, columns=["user_idx", "item_idx", "rating"])
        d = t.to_pandas()
        d = d[d["rating"] >= 4.0]
        dfs.append(d[["user_idx", "item_idx"]])
    test_df = pd.concat(dfs, ignore_index=True)
    
    # Map raw IDs to compact IDs
    u_to_idx = {u: idx for idx, u in enumerate(u_uniq)}
    i_to_idx = {it: idx for idx, it in enumerate(i_uniq)}
    
    test_df["mapped_u"] = test_df["user_idx"].map(u_to_idx)
    test_df["mapped_i"] = test_df["item_idx"].map(i_to_idx)
    test_clean = test_df.dropna(subset=["mapped_u", "mapped_i"]).copy()
    test_clean["mapped_u"] = test_clean["mapped_u"].astype(np.int64)
    test_clean["mapped_i"] = test_clean["mapped_i"].astype(np.int64)
    
    user_gt = test_clean.groupby("mapped_u")["mapped_i"].apply(set).to_dict()
    eval_users_list = list(user_gt.keys())
    if not eval_users_list:
        return {}
    
    rng = np.random.default_rng(42)
    chosen_users = rng.choice(eval_users_list, size=min(sample_n, len(eval_users_list)), replace=False)
    all_items_pool = np.arange(len(item_factors))
    
    metrics = {
        "HR@5": 0.0, "HR@10": 0.0, "HR@20": 0.0,
        "NDCG@5": 0.0, "NDCG@10": 0.0, "NDCG@20": 0.0,
        "MRR": 0.0
    }
    cnt = 0
    for u in tqdm(chosen_users, desc="Local Test Ranking Eval"):
        gts = list(user_gt[u])
        if not gts: continue
        gt = gts[0]
        
        negs = rng.choice(all_items_pool, size=150, replace=False)
        negs = [it for it in negs if it not in user_gt[u]][:99]
        cands = np.array([gt] + negs, dtype=np.int64)
        
        scores = item_factors[cands] @ user_factors[u]
        ranked = cands[np.argsort(-scores)]
        pos_list = np.where(ranked == gt)[0]
        if len(pos_list) == 0: continue
        pos = int(pos_list[0])
        
        if pos < 5: metrics["HR@5"] += 1.0
        if pos < 10: metrics["HR@10"] += 1.0
        if pos < 20: metrics["HR@20"] += 1.0
        if pos < 5: metrics["NDCG@5"] += 1.0 / math.log2(pos + 2)
        if pos < 10: metrics["NDCG@10"] += 1.0 / math.log2(pos + 2)
        if pos < 20: metrics["NDCG@20"] += 1.0 / math.log2(pos + 2)
        metrics["MRR"] += 1.0 / (pos + 1)
        cnt += 1
        
    if cnt > 0:
        for k in metrics: metrics[k] /= cnt
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Train implicit CF models locally on CPU (8GB RAM optimized)")
    parser.add_argument("--model", choices=["als", "bpr"], default="als", help="Model architecture")
    parser.add_argument("--factors", type=int, default=32, help="Latent factors (32 recommended for 8GB RAM)")
    parser.add_argument("--iters", type=int, default=10, help="Training iterations")
    parser.add_argument("--reg", type=float, default=0.05, help="Regularization parameter")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate (for BPR)")
    parser.add_argument("--threads", type=int, default=0, help="CPU threads (0 = all cores)")
    parser.add_argument("--max-users", type=int, default=100000, 
                        help="Active users to keep (default 100k for fast ~30s execution; set 500k or 0 as needed)")
    args = parser.parse_args()

    print("=" * 60)
    print(f"LOCAL CF TRAINING WITH `implicit` ({args.model.upper()})")
    print(f"Factors: {args.factors} | Iterations: {args.iters} | Reg: {args.reg}")
    print("=" * 60)

    t0 = time.time()
    df = load_local_train_data(max_users=args.max_users)
    
    mat, u_uniq, i_uniq = build_compact_matrix(df)
    del df
    
    print(f"\nStarting {args.model.upper()} training with implicit...")
    t_train = time.time()
    model, uf_compact, itf_compact = train_cf(
        mat,
        name=args.model,
        factors=args.factors,
        iterations=args.iters,
        regularization=args.reg,
        learning_rate=args.lr,
        num_threads=args.threads
    )
    train_duration = time.time() - t_train
    print(f"Training completed in {train_duration:.1f}s")
    print(f"User factors: {uf_compact.shape} | Item factors: {itf_compact.shape}")
    
    del mat
    
    print("\nPopulating factors and saving to artifacts/factors/...")
    # Save compact factors + ID maps
    np.save(FACTORS_DIR / f"{args.model}_user_ids.npy", u_uniq)
    np.save(FACTORS_DIR / f"{args.model}_item_ids.npy", i_uniq)
    
    # Save standard factor files for the serving pipeline
    save_factors(args.model, uf_compact, itf_compact, meta={
        "model": args.model,
        "library": "implicit",
        "factors": args.factors,
        "iterations": args.iters,
        "train_time_sec": train_duration,
        "n_users": len(u_uniq),
        "n_items": len(i_uniq)
    })
    
    # Run ranking eval on test set
    print("\nRunning test evaluation...")
    metrics = evaluate_on_test(uf_compact, itf_compact, u_uniq, i_uniq)
    if metrics:
        print("\n" + "=" * 40)
        print(f"LOCAL {args.model.upper()} EVALUATION METRICS:")
        for k, v in metrics.items():
            print(f"  {k:8s}: {v:.4f}")
        print("=" * 40)
        
        out_eval = FACTORS_DIR / f"{args.model}_local_eval.json"
        out_eval.write_text(json.dumps(metrics, indent=2))
        print(f"Saved eval metrics to {out_eval}")
        
    print(f"\nAll done in {time.time() - t0:.1f}s! Factors are ready in artifacts/factors/ for local serving.")

if __name__ == "__main__":
    main()
