#!/usr/bin/env python
"""Step 3: Baseline Late Fusion (ALS + SBERT) evaluation on validation set.

Evaluates HR@K, NDCG@K, MRR on a sample of users.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import VAL_DIR, TRAIN_DIR, FACTORS_DIR, NUM_USERS, NUM_ITEMS
from src.data import ground_truth_by_user
from src.metrics import evaluate_rankings
from src.models.cf import load_factors, topk_cf
from src.models.sbert import load_sbert, load_faiss_index, sbert_topk
from src.models.late_fusion import classic_late_fusion


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cf", default="als")
    ap.add_argument("--users", type=int, default=500, help="num eval users")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--candidates", type=int, default=200)
    ap.add_argument("--alpha", type=float, default=0.6)
    args = ap.parse_args()

    print("Loading val ground truth...")
    import pyarrow.parquet as pq, glob
    parts = sorted(glob.glob(str(VAL_DIR / "*.parquet")))
    dfs = [pq.read_table(p, columns=["user_idx", "item_idx", "rating"]).to_pandas() for p in parts]
    val = pd.concat(dfs, ignore_index=True)
    gt = ground_truth_by_user(val)

    # load train history for sample users
    print("Loading train history (sample users)...")
    train_parts = sorted(glob.glob(str(TRAIN_DIR / "*.parquet"))) if False else sorted(glob.glob(str(Path(str(TRAIN_DIR)) / "*.parquet")))
    # build history dict lazily
    history_by_user = {}
    # We'll fill after sampling
    users = list(gt.keys())
    import random
    random.seed(42)
    sample_users = random.sample(users, min(args.users, len(users)))
    sample_set = set(sample_users)
    for p in tqdm(train_parts, desc="train parts"):
        t = pq.read_table(p, columns=["user_idx", "item_idx", "rating"]).to_pandas()
        t = t[t["user_idx"].isin(sample_set)]
        for u, g in t.groupby("user_idx"):
            history_by_user.setdefault(int(u), set()).update(g["item_idx"].tolist())

    print("Loading factors...")
    uf, itf = load_factors(args.cf, dtype="float16", mmap=True)
    uf = np.asarray(uf, dtype=np.float32)
    itf = np.asarray(itf, dtype=np.float32)
    sbert_emb = load_sbert(mmap=False)
    faiss_index = load_faiss_index()

    def user_sbert_vec(hist):
        if sbert_emb is None or not hist:
            return None
        v = sbert_emb[hist].mean(axis=0)
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    # user history from train? Use val items as proxy history for retrieval query
    preds = {}
    for u in tqdm(sample_users, desc="eval"):
        rel = gt[u]
        hist = list(history_by_user.get(u, []))
        hist_set = set(hist)
        cf_items, cf_scores = topk_cf(uf, itf, u, k=args.candidates, exclude=hist_set)
        sb_items, sb_scores = [], []
        q = user_sbert_vec(hist)
        if q is not None and faiss_index is not None:
            sb_items, sb_scores = sbert_topk(faiss_index, q, k=args.candidates)
        cand_set = set(cf_items) | set(sb_items)
        cands = list(cand_set)
        cf_map = {it: 0.0 for it in cands}
        for it, sc in zip(cf_items, cf_scores):
            cf_map[it] = float(sc)
        sb_map = {it: 0.0 for it in cands}
        for it, sc in zip(sb_items, sb_scores):
            sb_map[it] = float(sc)
        cf_arr = np.array([cf_map[it] for it in cands], dtype=np.float32)
        sb_arr = np.array([sb_map[it] for it in cands], dtype=np.float32)
        fused = classic_late_fusion(cf_arr, sb_arr, alpha=args.alpha, beta=1-args.alpha)
        order = np.argsort(-fused)
        preds[u] = [cands[i] for i in order if cands[i] not in hist_set][:args.k]

    gt_sample = {u: gt[u] for u in sample_users}
    metrics = evaluate_rankings(preds, gt_sample, ks=(5, 10, 20))
    print(metrics)
    import json
    out = ROOT / "artifacts" / "results" / "baseline_late_fusion.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": "late_fusion", "alpha": args.alpha, "metrics": metrics}, indent=2))
    print("Saved", out)


if __name__ == "__main__":
    main()
