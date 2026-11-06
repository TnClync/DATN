#!/usr/bin/env python
"""Step 4: Two-Stage Retrieval + Cross-Encoder Reranker evaluation."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import VAL_DIR, TRAIN_DIR, NUM_USERS, NUM_ITEMS
from src.data import ground_truth_by_user
from src.metrics import evaluate_rankings
from src.models.cf import load_factors, topk_cf
from src.models.sbert import load_sbert, load_faiss_index, sbert_topk
from src.models.two_stage import candidate_union, rerank_cross_encoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cf", default="als")
    ap.add_argument("--users", type=int, default=200)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--candidates", type=int, default=200)
    ap.add_argument("--reranker", default="BAAI/bge-reranker-v2-m3")
    ap.add_argument("--no-rerank", action="store_true")
    args = ap.parse_args()

    import pyarrow.parquet as pq, glob, random
    parts = sorted(glob.glob(str(VAL_DIR / "*.parquet")))
    dfs = [pq.read_table(p, columns=["user_idx", "item_idx", "rating"]).to_pandas() for p in parts]
    val = pd.concat(dfs, ignore_index=True)
    gt = ground_truth_by_user(val)
    users = list(gt.keys())
    random.seed(42)
    sample_users = random.sample(users, min(args.users, len(users)))
    sample_set = set(sample_users)

    # train history
    history_by_user = {}
    for p in tqdm(sorted(glob.glob(str(TRAIN_DIR / "*.parquet"))), desc="train parts"):
        t = pq.read_table(p, columns=["user_idx", "item_idx"]).to_pandas()
        t = t[t["user_idx"].isin(sample_set)]
        for u, g in t.groupby("user_idx"):
            history_by_user.setdefault(int(u), set()).update(g["item_idx"].tolist())

    uf, itf = load_factors(args.cf, dtype="float16", mmap=True)
    uf = np.asarray(uf, dtype=np.float32)
    itf = np.asarray(itf, dtype=np.float32)
    sbert_emb = load_sbert(mmap=False)
    faiss_index = load_faiss_index()

    reranker = None
    if not args.no_rerank:
        try:
            from src.models.two_stage import load_reranker
            reranker = load_reranker(args.reranker)
        except Exception as e:
            print("reranker load failed:", e)

    def item_text_fn(i):
        return f"item {i}"

    preds = {}
    for u in tqdm(sample_users, desc="eval"):
        hist = list(history_by_user.get(u, []))
        hist_set = set(hist)
        cf_items, _ = topk_cf(uf, itf, u, k=args.candidates, exclude=hist_set)
        sb_items, _ = [], []
        if sbert_emb is not None and faiss_index is not None and hist:
            v = sbert_emb[hist].mean(axis=0)
            n = np.linalg.norm(v)
            if n > 0:
                sb_items, _ = sbert_topk(faiss_index, v / n, k=args.candidates)
        cands = candidate_union(cf_items, sb_items, max_candidates=args.candidates)
        if reranker is not None:
            scored = rerank_cross_encoder(reranker, f"user {u}", cands, item_text_fn)
            preds[u] = [it for it, _ in scored if it not in hist_set][:args.k]
        else:
            preds[u] = [it for it in cands if it not in hist_set][:args.k]

    gt_sample = {u: gt[u] for u in sample_users}
    metrics = evaluate_rankings(preds, gt_sample, ks=(5, 10, 20))
    print(metrics)
    import json
    out = ROOT / "artifacts" / "results" / "two_stage_reranker.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": "two_stage", "reranker": args.reranker,
                               "no_rerank": args.no_rerank, "metrics": metrics}, indent=2))
    print("Saved", out)


if __name__ == "__main__":
    main()
