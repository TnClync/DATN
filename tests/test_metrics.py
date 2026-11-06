import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metrics import hit_rate_at_k, ndcg_at_k, mrr, evaluate_rankings


def test_hr():
    assert hit_rate_at_k([1,2,3], {2}, 5) == 1.0
    assert hit_rate_at_k([1,2,3], {9}, 5) == 0.0


def test_ndcg():
    v = ndcg_at_k([1,2,3], {2}, 5)
    assert 0 < v <= 1.0


def test_mrr():
    assert mrr([5, 2, 3], {2}) == 0.5
    assert mrr([5, 2, 3], {9}) == 0.0


def test_eval():
    preds = {1: [1,2,3], 2: [4,5,6]}
    gt = {1: {2}, 2: {6}}
    m = evaluate_rankings(preds, gt, ks=(5,10))
    assert "HR@5" in m and "MRR" in m
    assert m["HR@5"] == 1.0
