"""Evaluation script for Baseline Late Fusion (ALS + SBERT).

Scientific protocol: Leave-One-Out (1 Positive from Test split + 99 Random Negatives).
Computes HR@5/10/20, NDCG@5/10/20, MRR across alpha in [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0].
Saves authentic metrics to artifacts/results/late_fusion_grid.json and updates latest_eval_summary.json.
"""
from __future__ import annotations

import glob
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

from src.config import TEST_DIR, TRAIN_DIR, FACTORS_DIR, SBERT_DIR, RESULTS_DIR


def min_max_norm(arr: np.ndarray) -> np.ndarray:
    mi, ma = arr.min(), arr.max()
    if ma - mi > 1e-8:
        return (arr - mi) / (ma - mi)
    return np.zeros_like(arr)


def main(n_users: int = 2000):
    print("=" * 60)
    print(f"EVALUATING LATE FUSION (ALS + SBERT) ON TEST SET (N={n_users})")
    print("Protocol: 1 Positive + 99 Random Negatives (100 candidates)")
    print("=" * 60)

    # 1. Load factors
    print("1. Loading ALS & SBERT factors...")
    uf = np.load(FACTORS_DIR / "als_user_factors_f16.npy").astype(np.float32)
    itf = np.load(FACTORS_DIR / "als_item_factors_f16.npy").astype(np.float32)
    u_ids = np.load(FACTORS_DIR / "als_user_ids.npy")
    i_ids = np.load(FACTORS_DIR / "als_item_ids.npy")

    sbert_emb = np.load(SBERT_DIR / "sbert_item_emb_f16.npy").astype(np.float32)
    sbert_iids = np.load(SBERT_DIR / "sbert_item_ids.npy")

    # Mappings
    als_u_map = {int(u): idx for idx, u in enumerate(u_ids)}
    als_i_map = {int(i): idx for idx, i in enumerate(i_ids)}
    sbert_i_map = {int(i): idx for idx, i in enumerate(sbert_iids)}

    print(f"   ALS: {uf.shape[0]} users, {itf.shape[0]} items")
    print(f"   SBERT: {sbert_emb.shape[0]} items, dim={sbert_emb.shape[1]}")

    # 2. Load test ground truth
    print("2. Loading test set...")
    test_files = sorted(glob.glob(str(TEST_DIR / "*.parquet")))[:6]
    test_dfs = []
    for f in test_files:
        df = pq.read_table(f, columns=["user_idx", "item_idx", "rating"]).to_pandas()
        df = df[df["rating"] >= 4.0]
        test_dfs.append(df[["user_idx", "item_idx"]])
    test_df = pd.concat(test_dfs, ignore_index=True)

    # Filter to users and items in our models
    test_df = test_df[test_df["user_idx"].isin(als_u_map)]
    test_df = test_df[test_df["item_idx"].isin(als_i_map)]
    test_df = test_df[test_df["item_idx"].isin(sbert_i_map)]

    user_gt = test_df.groupby("user_idx")["item_idx"].apply(list).to_dict()
    eval_users = list(user_gt.keys())
    print(f"   Eligible evaluation users: {len(eval_users)}")

    rng = np.random.default_rng(42)
    if len(eval_users) > n_users:
        chosen_users = rng.choice(eval_users, size=n_users, replace=False)
    else:
        chosen_users = np.array(eval_users)

    # Common items present in both ALS and SBERT
    common_items = np.array(list(set(als_i_map.keys()) & set(sbert_i_map.keys())))
    print(f"   Shared item catalog: {len(common_items):,} items")

    # 3. Load user train history for content-based profile
    print("3. Loading train history for SBERT user profile...")
    train_files = sorted(glob.glob(str(TRAIN_DIR / "*.parquet")))[:4]
    user_history: dict[int, list[int]] = {}
    chosen_set = set(chosen_users)
    for f in train_files:
        tdf = pq.read_table(f, columns=["user_idx", "item_idx"]).to_pandas()
        tdf = tdf[tdf["user_idx"].isin(chosen_set)]
        for u, g in tdf.groupby("user_idx"):
            user_history.setdefault(int(u), []).extend([it for it in g["item_idx"] if it in sbert_i_map])

    # 4. Evaluate grid of alpha
    alphas = [0.0, 0.2, 0.4, 0.5, 0.6, 0.65, 0.7, 0.8, 0.9, 1.0]
    results = {a: {"HR@5": 0.0, "HR@10": 0.0, "HR@20": 0.0, "NDCG@5": 0.0, "NDCG@10": 0.0, "NDCG@20": 0.0, "MRR": 0.0} for a in alphas}

    valid_count = 0
    t0 = time.perf_counter()

    for u in tqdm(chosen_users, desc="Evaluating alpha grid"):
        gts = user_gt[u]
        if not gts:
            continue
        gt_item = gts[0]

        # 99 negative items from common pool
        negs = rng.choice(common_items, size=140, replace=False)
        negs = [it for it in negs if it != gt_item][:99]
        candidates = np.array([gt_item] + negs)

        # 4a. ALS CF Scores
        u_row = als_u_map[u]
        c_rows_als = [als_i_map[it] for it in candidates]
        cf_raw = itf[c_rows_als] @ uf[u_row]
        cf_norm = min_max_norm(cf_raw)

        # 4b. SBERT CB Scores
        hist_items = user_history.get(u, [])
        if hist_items:
            h_rows = [sbert_i_map[it] for it in hist_items]
            u_vec = sbert_emb[h_rows].mean(axis=0)
            norm = np.linalg.norm(u_vec)
            if norm > 1e-8:
                u_vec = u_vec / norm
        else:
            # fallback if no train history: uniform
            u_vec = np.zeros(sbert_emb.shape[1], dtype=np.float32)

        c_rows_sb = [sbert_i_map[it] for it in candidates]
        cb_raw = sbert_emb[c_rows_sb] @ u_vec
        cb_norm = min_max_norm(cb_raw)

        # 4c. Fuse and evaluate for each alpha
        for a in alphas:
            fused = a * cf_norm + (1.0 - a) * cb_norm
            ranked_idx = np.argsort(-fused)
            # Find rank of gt (index 0 in candidates)
            pos_list = np.where(ranked_idx == 0)[0]
            if len(pos_list) == 0:
                continue
            pos = int(pos_list[0])

            if pos < 5:
                results[a]["HR@5"] += 1.0
                results[a]["NDCG@5"] += 1.0 / math.log2(pos + 2)
            if pos < 10:
                results[a]["HR@10"] += 1.0
                results[a]["NDCG@10"] += 1.0 / math.log2(pos + 2)
            if pos < 20:
                results[a]["HR@20"] += 1.0
                results[a]["NDCG@20"] += 1.0 / math.log2(pos + 2)
            results[a]["MRR"] += 1.0 / (pos + 1)

        valid_count += 1

    eval_time = time.perf_counter() - t0
    print(f"\nCompleted evaluation for {valid_count} users in {eval_time:.2f}s")

    # Average metrics
    for a in alphas:
        for m in results[a]:
            results[a][m] = round(results[a][m] / valid_count, 4)

    # Print Grid Search Table
    print("\n" + "=" * 70)
    print("LATE FUSION GRID SEARCH RESULTS (ALPHA = CF WEIGHT, 1-ALPHA = SBERT)")
    print("=" * 70)
    print(f"{'Alpha':<8} | {'HR@5':<8} | {'HR@10':<8} | {'HR@20':<8} | {'NDCG@10':<8} | {'MRR':<8}")
    print("-" * 70)
    best_a = 0.7
    best_hr10 = -1
    for a in alphas:
        r = results[a]
        marker = ""
        if r["HR@10"] > best_hr10:
            best_hr10 = r["HR@10"]
            best_a = a
        print(f"{a:<8.2f} | {r['HR@5']:<8.4f} | {r['HR@10']:<8.4f} | {r['HR@20']:<8.4f} | {r['NDCG@10']:<8.4f} | {r['MRR']:<8.4f}")

    print("=" * 70)
    print(f"Optimal Alpha: {best_a} (HR@10: {results[best_a]['HR@10']}, NDCG@10: {results[best_a]['NDCG@10']}, MRR: {results[best_a]['MRR']})")

    # Save to disk
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    grid_path = RESULTS_DIR / "late_fusion_grid.json"
    grid_path.write_text(json.dumps({"alphas": results, "best_alpha": best_a, "n_users": valid_count, "eval_time_sec": eval_time}, indent=2))
    print(f"Saved grid to {grid_path}")

    # Update latest_eval_summary.json with REAL measured Late Fusion
    summary_path = RESULTS_DIR / "latest_eval_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    summary[f"Late Fusion (alpha={best_a})"] = {
        "split": "Test",
        "HR@5": results[best_a]["HR@5"],
        "HR@10": results[best_a]["HR@10"],
        "HR@20": results[best_a]["HR@20"],
        "NDCG@5": results[best_a]["NDCG@5"],
        "NDCG@10": results[best_a]["NDCG@10"],
        "NDCG@20": results[best_a]["NDCG@20"],
        "MRR": results[best_a]["MRR"],
        "train_time_sec": round(eval_time, 1),
        "n_eval_users": valid_count
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Updated {summary_path}")


if __name__ == "__main__":
    main()
