"""Evaluate Two-Stage retrieval (CF top-50 ∪ SBERT top-50 + linear rerank).

This reproduces the evaluation protocol used by the Kaggle deep-hybrid kernel
locally, so the Two-Stage metric is directly comparable to ALS/BPR/SBERT and
the fusion models. It intentionally does NOT retrain any model.
"""
from __future__ import annotations

import glob
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

from src.config import TRAIN_DIR, TEST_DIR, FACTORS_DIR, SBERT_DIR, RESULTS_DIR

MIN_INTERACTIONS = 3
CATALOG = 200_000
TEST_N, VAL_N = 500, 300
SEED = 42


def read_split(directory: Path, columns: list[str]) -> pd.DataFrame:
    parts = sorted(glob.glob(str(directory / "*.parquet")))
    frames = [pq.read_table(p, columns=columns).to_pandas() for p in parts]
    df = pd.concat(frames, ignore_index=True)
    if "rating" in df.columns:
        df = df[df["rating"] >= 4.0]
    return df


def map_ids(values: np.ndarray, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pos = np.searchsorted(ids, values)
    pos = np.clip(pos, 0, len(ids) - 1)
    ok = ids[pos] == values
    return pos, ok


def metrics(ranked: np.ndarray, gt: int) -> dict[str, float]:
    res = {k: 0.0 for k in ["HR@5", "HR@10", "HR@20", "NDCG@5", "NDCG@10", "NDCG@20", "MRR"]}
    hit = np.where(ranked == gt)[0]
    if len(hit) == 0:
        return res
    pos = int(hit[0])
    if pos < 5:
        res["HR@5"] = 1.0
        res["NDCG@5"] = 1.0 / math.log2(pos + 2)
    if pos < 10:
        res["HR@10"] = 1.0
        res["NDCG@10"] = 1.0 / math.log2(pos + 2)
    if pos < 20:
        res["HR@20"] = 1.0
        res["NDCG@20"] = 1.0 / math.log2(pos + 2)
    res["MRR"] = 1.0 / (pos + 1)
    return res


def mm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mn, mx = x.min(), x.max()
    return np.zeros_like(x) if mx - mn < 1e-8 else (x - mn) / (mx - mn)


def topk_full(mat, vec, k: int, banned: np.ndarray) -> np.ndarray:
    scores = mat @ vec
    if len(banned):
        b = banned[(banned >= 0) & (banned < len(scores))]
        scores[b] = -np.inf
    k = min(k, len(scores) - 1)
    idx = np.argpartition(-scores, k)[:k]
    return idx[np.argsort(-scores[idx])]


def main() -> None:
    print("Loading factors...")
    uf = np.load(FACTORS_DIR / "als_user_factors_f16.npy", mmap_mode="r")
    itf = np.load(FACTORS_DIR / "als_item_factors_f16.npy")
    u_ids = np.load(FACTORS_DIR / "als_user_ids.npy")
    i_ids = np.load(FACTORS_DIR / "als_item_ids.npy")
    sbert = np.load(SBERT_DIR / "sbert_item_embeddings_f16.npy", mmap_mode="r")

    print("Loading train/test...")
    train = read_split(TRAIN_DIR, ["user_idx", "item_idx", "rating"])
    cu, ok_u = map_ids(train["user_idx"].to_numpy(), u_ids)
    ci, ok_i = map_ids(train["item_idx"].to_numpy(), i_ids)
    keep = ok_u & ok_i
    cu, ci = cu[keep], ci[keep]
    counts = np.bincount(cu, minlength=len(u_ids))

    test = read_split(TEST_DIR, ["user_idx", "item_idx", "rating"])
    tcu, tok_u = map_ids(test["user_idx"].to_numpy(), u_ids)
    tci, tok_i = map_ids(test["item_idx"].to_numpy(), i_ids)
    tkeep = tok_u & tok_i
    gt_df = pd.DataFrame({"u": tcu[tkeep], "i": tci[tkeep]})
    user_gt = gt_df.groupby("u")["i"].apply(set).to_dict()

    eval_users = np.array([
        u for u in user_gt
        if counts[u] >= MIN_INTERACTIONS and any(it < CATALOG for it in user_gt[u])
    ])
    rng = np.random.default_rng(SEED)
    eval_users = rng.choice(eval_users, size=min(2000, len(eval_users)), replace=False)
    eval_set = set(int(u) for u in eval_users)

    hist: dict[int, list[int]] = {}
    mask = np.isin(cu, np.fromiter(eval_set, dtype=np.int64))
    for u, i in zip(cu[mask], ci[mask]):
        if i < CATALOG:
            hist.setdefault(int(u), []).append(int(i))

    user_sbert = {}
    for u in eval_set:
        h = hist.get(u, [])
        v = np.asarray(sbert[h], dtype=np.float32).mean(0) if h else np.zeros(768, dtype=np.float32)
        n = np.linalg.norm(v)
        user_sbert[u] = (v / n).astype(np.float32) if n > 1e-8 else v.astype(np.float32)

    pool = np.arange(CATALOG)
    instances = []
    for u in eval_users:
        u = int(u)
        gts = [it for it in user_gt[u] if it < CATALOG]
        gt = int(rng.choice(gts))
        banned = (set(hist.get(u, [])) | set(user_gt[u])) - {gt}
        negs = []
        for _ in range(10):
            draw = rng.choice(pool, size=300, replace=False)
            negs = [int(x) for x in draw if int(x) not in banned][:99]
            if len(negs) == 99:
                break
        if len(negs) < 99:
            continue
        cands = np.array([gt] + negs, dtype=np.int64)
        rng.shuffle(cands)
        instances.append({
            "user": u,
            "gt": gt,
            "candidates": cands,
            "banned": np.array(sorted(banned), dtype=np.int64),
        })

    test_inst = instances[:TEST_N]
    val_inst = instances[TEST_N:TEST_N + VAL_N]
    print(f"Instances: {len(instances)} (test={len(test_inst)}, val={len(val_inst)})")

    def cf(u, c):
        return np.asarray(itf[c] @ np.asarray(uf[u], dtype=np.float32), dtype=np.float32)

    def sb(u, c):
        return np.asarray(sbert[c] @ user_sbert[u], dtype=np.float32)

    def evaluate(fn, rows):
        agg = {k: 0.0 for k in ["HR@5", "HR@10", "HR@20", "NDCG@5", "NDCG@10", "NDCG@20", "MRR"]}
        for inst in rows:
            ranked = fn(inst)
            for k, v in metrics(ranked, inst["gt"]).items():
                agg[k] += v
        return {k: round(v / max(1, len(rows)), 4) for k, v in agg.items()}

    def als_rank(inst):
        cands = inst["candidates"]
        return cands[np.argsort(-cf(inst["user"], cands))]

    als_metrics = evaluate(als_rank, test_inst)

    def two_stage(alpha):
        def rank(inst):
            u = inst["user"]
            cf_top = topk_full(itf, np.asarray(uf[u], dtype=np.float32), 50, inst["banned"])
            sb_top = topk_full(sbert[:CATALOG], user_sbert[u], 50, inst["banned"])
            union = np.array(list(dict.fromkeys(list(cf_top) + list(sb_top))), dtype=np.int64)
            fused = alpha * mm(cf(u, union)) + (1 - alpha) * mm(sb(u, union))
            return union[np.argsort(-fused)]
        return rank

    best_alpha, best_hr, grid = 1.0, -1.0, {}
    for alpha in [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        m = evaluate(two_stage(alpha), val_inst)
        grid[alpha] = m
        print(f"alpha={alpha}: val HR@10={m['HR@10']:.4f}")
        if m["HR@10"] > best_hr:
            best_hr, best_alpha = m["HR@10"], alpha

    ts_metrics = evaluate(two_stage(best_alpha), test_inst)
    out = {
        "protocol": "1 positive + 99 random negatives, shared 200k-item catalog",
        "n_eval_users": len(test_inst),
        "min_interactions": MIN_INTERACTIONS,
        "als_baseline": als_metrics,
        "two_stage": ts_metrics,
        "best_alpha": best_alpha,
        "val_HR10": round(best_hr, 4),
        "grid_val": {str(a): m for a, m in grid.items()},
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "two_stage_eval.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
