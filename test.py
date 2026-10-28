"""
Test script cho các mô hình của DATN.

Chạy:
    python test.py                # test toàn bộ (serving + hybrid + content-based)
    python test.py --model hybrid # chỉ test hybrid
    python test.py --model cb     # chỉ test content-based similarity
    python test.py --model serving# chỉ test serving store
    python test.py --user A1IPEJDMQNXJYF --top_k 5

Yêu cầu: đã build serving cache (models/serving/). Nếu chưa, chạy:
    python scripts/build_serving_cache.py
"""

from __future__ import annotations

import argparse
import os
import random
import sqlite3
import sys
import time
from pathlib import Path

# Đảm bảo PySpark (nếu dùng) dùng cùng phiên bản Python với driver
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.models.hybrid_recommender import HybridRecommender
from src.models.serving_store import ServingStore


SAMPLE_USERS = [
    "A1IPEJDMQNXJYF",
    "A12C7JY236MJ6U",
    "A2ER9SRSFRM5IO",
]


def test_serving_store() -> bool:
    print("=" * 70)
    print("TEST 1: ServingStore")
    print("=" * 70)
    store = ServingStore()
    if not store.is_built():
        print("FAIL: serving cache chưa build. Chạy scripts/build_serving_cache.py")
        return False
    t0 = time.time()
    store.load()
    print(f"  load_seconds      : {store.load_seconds:.2f}s (wall {time.time()-t0:.2f}s)")
    print(f"  catalog_size      : {store.catalog_size:,}")
    print(f"  vocab_size        : {store.vocab_size:,}")
    print(f"  pop_score len     : {len(store.pop_score):,}")
    assert store.catalog_size > 0, "catalog rỗng"
    assert store.vocab_size > 0, "vocab rỗng"

    for uid in SAMPLE_USERS:
        idx = store.user_index(uid)
        print(f"  user_index({uid}) = {idx}")

    idx = store.user_index(SAMPLE_USERS[0])
    assert idx is not None, "user đầu tiên không tìm thấy"
    seen = store.seen_rows(idx)
    print(f"  seen_rows(user0)  : {len(seen)} items")

    indices, values = store.row_vector(0)
    print(f"  row_vector(0)     : nnz={len(indices)}")
    assert len(indices) == len(values)

    meta = store.meta_rows([0, 1, 2])
    print(f"  meta_rows([0,1,2]) : {len(meta)} rows, cols={list(meta.columns)}")
    assert len(meta) == 3

    print("  summary:", store.summary())
    print("PASS: ServingStore\n")
    return True


def test_hybrid(top_k: int = 5, alpha: float = 0.65, domain=None) -> bool:
    print("=" * 70)
    print("TEST 2: HybridRecommender")
    print("=" * 70)
    rec = HybridRecommender()
    t0 = time.time()
    rec.load()
    print(f"  load_seconds      : {rec.get_status()['load_seconds']}s (wall {time.time()-t0:.2f}s)")
    status = rec.get_status()
    print(f"  mode              : {status['mode']} ({status['mode_label']})")
    print(f"  catalog_size      : {status['catalog_size']:,}")
    print(f"  vocab_size        : {status['vocab_size']:,}")
    assert status["mode"] in ("hybrid_pop_cb", "cb_only"), f"mode không hợp lệ: {status['mode']}"

    for uid in SAMPLE_USERS:
        print("-" * 70)
        print(f"  user: {uid}  (known={rec.user_known(uid)})")
        t0 = time.time()
        results = rec.recommend(
            user_raw_id=uid,
            top_k=top_k,
            domain=domain,
            alpha=alpha,
            exclude_seen=True,
        )
        dt = time.time() - t0
        print(f"  recommend()       : {len(results)} results in {dt:.2f}s")
        assert isinstance(results, list)
        assert len(results) <= top_k
        if results:
            r0 = results[0]
            print(f"  top1: {r0['asin']} | {r0['title'][:50]} | hybrid={r0['hybrid_score']}")
            required_keys = {"rank", "asin", "title", "brand", "category", "price",
                             "hybrid_score", "cf_score", "cb_score", "explanation"}
            assert required_keys.issubset(r0.keys()), f"thiếu keys: {required_keys - set(r0.keys())}"
            for r in results:
                assert 0.0 <= r["hybrid_score"] <= 1.0
                assert 0.0 <= r["cf_score"] <= 1.0
                assert 0.0 <= r["cb_score"] <= 1.0
            assert [r["rank"] for r in results] == list(range(1, len(results) + 1))
        for r in results[: min(3, len(results))]:
            print(f"    #{r['rank']} {r['asin']:<12} H={r['hybrid_score']:.4f} "
                  f"CF={r['cf_score']:.4f} CB={r['cb_score']:.4f} {r['title'][:30]}")

    print("-" * 70)
    print("  cold-start user (không tồn tại):")
    results = rec.recommend(user_raw_id="NONEXISTENT_USER_12345", top_k=top_k, alpha=alpha)
    print(f"  recommend()       : {len(results)} results (popularity fallback)")
    assert len(results) > 0, "cold-start phải trả về popularity"

    print("-" * 70)
    print("  domain filter = electronics:")
    results = rec.recommend(user_raw_id=SAMPLE_USERS[0], top_k=top_k, domain="electronics", alpha=alpha)
    print(f"  recommend()       : {len(results)} results")

    for a in (0.0, 0.5, 1.0):
        results = rec.recommend(user_raw_id=SAMPLE_USERS[0], top_k=3, alpha=a)
        print(f"  alpha={a:.2f}: {len(results)} results, top hybrid={results[0]['hybrid_score'] if results else None}")

    print("PASS: HybridRecommender\n")
    return True


def _find_row_id_for_asin(store: ServingStore, asin: str) -> int | None:
    conn = sqlite3.connect(f"file:{store.meta_path}?mode=ro", uri=True)
    try:
        cur = conn.execute("SELECT row_id FROM catalog WHERE asin=? LIMIT 1", (asin,))
        row = cur.fetchone()
        return int(row[0]) if row else None
    finally:
        conn.close()


def test_content_based(asin: str = "B0006NAB96", top_n: int = 5, sample_size: int = 20000) -> bool:
    print("=" * 70)
    print("TEST 3: Content-Based (TF-IDF) similarity (lightweight, serving cache)")
    print("=" * 70)
    store = ServingStore()
    store.load()

    source_row = _find_row_id_for_asin(store, asin)
    print(f"  source asin       : {asin}  (row_id={source_row})")
    assert source_row is not None, f"ASIN {asin} không có trong catalog"

    src_indices, src_values = store.row_vector(source_row)
    assert len(src_indices) > 0, "source vector rỗng"
    src_norm = float(store.item_norm[source_row])
    assert src_norm > 0

    import numpy as np
    rng = np.random.default_rng(42)
    total = store.catalog_size
    cand_rows = rng.choice(total, size=min(sample_size, total), replace=False)
    cand_rows = [int(r) for r in cand_rows if int(r) != source_row]

    scores: list[tuple[int, float]] = []
    for row_id in cand_rows:
        indices, values = store.row_vector(row_id)
        if len(indices) == 0:
            continue
        denom = float(store.item_norm[row_id]) * src_norm
        if denom <= 0:
            continue
        vmap = {int(i): float(v) for i, v in zip(indices, values)}
        dot = 0.0
        for i, v in zip(src_indices, src_values):
            w = vmap.get(int(i))
            if w is not None:
                dot += float(v) * w
        sim = dot / denom
        if sim > 0:
            scores.append((row_id, sim))

    scores.sort(key=lambda x: x[1], reverse=True)
    top = scores[:top_n]
    print(f"  candidates scored : {len(cand_rows):,}  (sim>0: {len(scores)})")
    print(f"  top-{top_n} similar:")
    meta = store.meta_rows([r for r, _ in top])
    for (row_id, sim), mrow in zip(top, meta.itertuples(index=False)):
        print(f"    row={row_id:<8} sim={sim:.4f}  {str(mrow.asin):<12} {str(mrow.title)[:40]}")
    assert len(top) > 0, "không tìm được sản phẩm tương tự"
    print("PASS: Content-Based similarity (lightweight)\n")

    print("Ghi chú: script gốc test_content_similar.py (Spark UDF trên toàn catalog)")
    print("  có thể chạy riêng với PYSPARK_PYTHON trỏ về .venv:")
    print("  PYSPARK_PYTHON=.venv/bin/python PYSPARK_DRIVER_PYTHON=.venv/bin/python \\")
    print("      .venv/bin/python scripts/test_content_similar.py --asin B0006NAB96 --top-n 10")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Test models DATN")
    parser.add_argument("--model", type=str, default="all",
                        choices=["all", "serving", "hybrid", "cb"],
                        help="Mô hình cần test")
    parser.add_argument("--user", type=str, default=None, help="reviewerID test")
    parser.add_argument("--top_k", type=int, default=5, help="Top-K")
    parser.add_argument("--alpha", type=float, default=0.65, help="alpha")
    parser.add_argument("--domain", type=str, default=None, help="domain filter")
    parser.add_argument("--asin", type=str, default="B0006NAB96", help="ASIN nguồn cho CB")
    args = parser.parse_args()

    if args.user:
        global SAMPLE_USERS
        SAMPLE_USERS = [args.user]

    ok = True
    if args.model in ("all", "serving"):
        try:
            ok = test_serving_store() and ok
        except Exception as e:
            print(f"FAIL serving: {e}")
            ok = False
    if args.model in ("all", "hybrid"):
        try:
            ok = test_hybrid(top_k=args.top_k, alpha=args.alpha, domain=args.domain) and ok
        except Exception as e:
            print(f"FAIL hybrid: {e}")
            import traceback; traceback.print_exc()
            ok = False
    if args.model in ("all", "cb"):
        try:
            ok = test_content_based(asin=args.asin, top_n=args.top_k) and ok
        except Exception as e:
            print(f"FAIL cb: {e}")
            import traceback; traceback.print_exc()
            ok = False

    print("=" * 70)
    print("RESULT:", "ALL PASS" if ok else "SOME FAILED")
    print("=" * 70)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
