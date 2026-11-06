"""Ranking metrics: HR@K, NDCG@K, MRR.

All functions accept a list/array of ranked item ids and a set of relevant items.
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence


def hit_rate_at_k(ranked: Sequence[int], relevant: set[int], k: int) -> float:
    topk = ranked[:k]
    return 1.0 if any(item in relevant for item in topk) else 0.0


def ndcg_at_k(ranked: Sequence[int], relevant: set[int], k: int) -> float:
    dcg = 0.0
    for i, item in enumerate(ranked[:k]):
        if item in relevant:
            dcg += 1.0 / math.log2(i + 2)
    # ideal DCG for up to min(k, len(relevant)) hits
    ideal_hits = min(k, len(relevant))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def mrr(ranked: Sequence[int], relevant: set[int]) -> float:
    for i, item in enumerate(ranked):
        if item in relevant:
            return 1.0 / (i + 1)
    return 0.0


def evaluate_rankings(
    predictions: dict[int, list[int]],
    ground_truth: dict[int, set[int]],
    ks: tuple[int, ...] = (5, 10, 20),
) -> dict[str, float]:
    """Aggregate metrics over users present in both predictions and ground truth."""
    users = [u for u in ground_truth if u in predictions]
    if not users:
        return {f"HR@{k}": 0.0 for k in ks} | {f"NDCG@{k}": 0.0 for k in ks} | {"MRR": 0.0}
    out = {f"HR@{k}": 0.0 for k in ks}
    out.update({f"NDCG@{k}": 0.0 for k in ks})
    out["MRR"] = 0.0
    n = len(users)
    for u in users:
        ranked = predictions[u]
        rel = ground_truth[u]
        for k in ks:
            out[f"HR@{k}"] += hit_rate_at_k(ranked, rel, k)
            out[f"NDCG@{k}"] += ndcg_at_k(ranked, rel, k)
        out["MRR"] += mrr(ranked, rel)
    for key in out:
        out[key] /= n
    return out
