#!/usr/bin/env python
"""Step 1: Train ALS / BPR with `implicit` and export factors.

Run on Kaggle (GPU not required; CPU is fine for implicit ALS). 
Produces artifacts/factors/{als,bpr}_{user,item}_factors_{f16,f32}.npy

Usage:
    python scripts/01_train_cf.py --model als --factors 64 --iters 15
    python scripts/01_train_cf.py --model bpr --factors 64 --iters 30
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.sparse import csr_matrix
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import TRAIN_DIR, FACTORS_DIR, NUM_USERS, NUM_ITEMS
from src.models.cf import train_cf, save_factors


def load_train_df() -> pd.DataFrame:
    import glob
    parts = sorted(glob.glob(str(TRAIN_DIR / "*.parquet")))
    dfs = []
    for p in tqdm(parts, desc="reading train parts"):
        t = pq.read_table(p, columns=["user_idx", "item_idx", "rating"])
        dfs.append(t.to_pandas())
    df = pd.concat(dfs, ignore_index=True)
    # implicit positives
    df = df[df["rating"] >= 4.0]
    return df


def build_matrix(df: pd.DataFrame) -> csr_matrix:
    rows = df["user_idx"].to_numpy(dtype=np.int64)
    cols = df["item_idx"].to_numpy(dtype=np.int64)
    data = np.ones(len(df), dtype=np.float32)
    mat = csr_matrix((data, (rows, cols)), shape=(NUM_USERS, NUM_ITEMS))
    return mat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["als", "bpr"], default="als")
    ap.add_argument("--factors", type=int, default=64)
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--reg", type=float, default=0.01)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    print("Loading train interactions...")
    df = load_train_df()
    print(f"positives: {len(df):,}  users: {df.user_idx.nunique():,}  items: {df.item_idx.nunique():,}")

    print("Building user x item matrix...")
    mat = build_matrix(df)
    print(f"matrix shape: {mat.shape} nnz: {mat.nnz:,}")

    print(f"Training {args.model} (factors={args.factors}, iters={args.iters})...")
    model, uf, itf = train_cf(
        mat, name=args.model, factors=args.factors, iterations=args.iters,
        regularization=args.reg, learning_rate=args.lr, num_threads=args.threads,
    )
    print(f"user_factors: {uf.shape}  item_factors: {itf.shape}")

    meta = {"factors": args.factors, "iterations": args.iters, "reg": args.reg,
            "train_positives": int(len(df))}
    paths = save_factors(args.model, uf, itf, meta=meta)
    print("Saved:", paths)


if __name__ == "__main__":
    main()
