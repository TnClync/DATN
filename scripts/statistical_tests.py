#!/usr/bin/env python
"""Bootstrap CI95 + paired Wilcoxon/McNemar từ per_user_ranks.npz.

Dùng:
    python scripts/statistical_tests.py artifacts/eval_inputs/per_user_ranks.npz

Sinh ra bảng markdown (dán trực tiếp vào docs/Bao_cao_day_du.md):
  - CI95 của HR@10 / NDCG@10 / MRR cho từng model
  - paired tests giữa các cặp model trên cùng user
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, wilcoxon

DISPLAY = {
    "als": "ALS",
    "bpr": "BPR",
    "sbert": "SBERT (content-only)",
    "als_bpr": "ALS+BPR (50/50)",
    "late_fusion_v3": "Late Fusion V3",
    "early_fusion_v3": "Early Fusion V3",
}
PAIRS = [
    ("early_fusion_v3", "als"),
    ("early_fusion_v3", "als_bpr"),
    ("early_fusion_v3", "late_fusion_v3"),
    ("late_fusion_v3", "als"),
    ("als", "bpr"),
    ("als", "sbert"),
]


def ndcg(ranks: np.ndarray, k: int = 10) -> np.ndarray:
    r = ranks.astype(np.float64)
    return np.where(r < k, 1.0 / np.log2(r + 2.0), 0.0)


def mrr(ranks: np.ndarray) -> np.ndarray:
    return 1.0 / (ranks.astype(np.float64) + 1.0)


def metric_values(ranks: np.ndarray, metric: str) -> np.ndarray:
    if metric.startswith("HR@"):
        return (ranks < int(metric.split("@")[1])).astype(np.float64)
    if metric.startswith("NDCG@"):
        return ndcg(ranks, int(metric.split("@")[1]))
    if metric == "MRR":
        return mrr(ranks)
    raise ValueError(metric)


def bootstrap_ci(ranks: np.ndarray, metric: str, n_boot: int = 2000, seed: int = 42):
    rng = np.random.default_rng(seed)
    vals = metric_values(ranks, metric)
    n = len(vals)
    boot = np.empty(n_boot)
    for s in range(0, n_boot, 250):
        e = min(s + 250, n_boot)
        boot[s:e] = vals[rng.integers(0, n, size=(e - s, n))].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(vals.mean()), float(lo), float(hi)


def paired(ranks_a: np.ndarray, ranks_b: np.ndarray):
    a, b = ndcg(ranks_a), ndcg(ranks_b)
    p_w = 1.0 if np.allclose(a, b) else float(
        wilcoxon(a, b, zero_method="wilcox", alternative="two-sided").pvalue
    )
    ha, hb = ranks_a < 10, ranks_b < 10
    n10, n01 = int(np.sum(ha & ~hb)), int(np.sum(~ha & hb))
    p_m = float(binomtest(n10, n10 + n01, 0.5).pvalue) if n10 + n01 else 1.0
    return {
        "d_ndcg10": float(a.mean() - b.mean()),
        "p_wilcoxon": p_w,
        "n_a_only": n10,
        "n_b_only": n01,
        "p_mcnemar": p_m,
        "wins": int(np.sum(ranks_a < ranks_b)),
        "losses": int(np.sum(ranks_a > ranks_b)),
        "ties": int(np.sum(ranks_a == ranks_b)),
    }


def fmt_p(p: float) -> str:
    if p < 1e-12:
        return "< 1e-12"
    if p < 0.001:
        return f"{p:.2e}"
    return f"{p:.4f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", type=Path)
    ap.add_argument("--split", default="ext", choices=["test", "ext"])
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    models = [str(m) for m in z["models"]]
    ranks = {m: z[f"rank_{args.split}_{m}"] for m in models}
    suffix = "test" if args.split == "test" else "ext"
    n = len(ranks[models[0]])

    print(f"### Split `{suffix}` — n = {n:,} user\n")
    print("| Model | HR@10 | CI95 HR@10 | NDCG@10 | CI95 NDCG@10 | MRR | CI95 MRR |")
    print("|---|---:|---|---:|---|---:|---|")
    for m in models:
        cells = []
        for metric in ("HR@10", "NDCG@10", "MRR"):
            mean, lo, hi = bootstrap_ci(ranks[m], metric)
            cells.append(f"{mean:.4f}")
            cells.append(f"[{lo:.4f}, {hi:.4f}]")
        # cells: mean, ci, mean, ci, mean, ci
        print(f"| {DISPLAY.get(m, m)} | {cells[0]} | {cells[1]} | {cells[2]} | "
              f"{cells[3]} | {cells[4]} | {cells[5]} |")

    print(f"\n**Paired tests (n = {n:,}, cùng user/candidate)**\n")
    print("| So sánh | ΔNDCG@10 | Wilcoxon p | HR@10 chỉ A đúng | HR@10 chỉ B đúng | McNemar p | W/L/T |")
    print("|---|---:|---:|---:|---:|---:|---|")
    for a, b in PAIRS:
        if a not in ranks or b not in ranks:
            continue
        r = paired(ranks[a], ranks[b])
        print(f"| {DISPLAY.get(a, a)} vs {DISPLAY.get(b, b)} | {r['d_ndcg10']:+.4f} | "
              f"{fmt_p(r['p_wilcoxon'])} | {r['n_a_only']:,} | {r['n_b_only']:,} | "
              f"{fmt_p(r['p_mcnemar'])} | {r['wins']:,}/{r['losses']:,}/{r['ties']:,} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
