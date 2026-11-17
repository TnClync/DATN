#!/usr/bin/env python
"""Chạy demo inference với artifact tải từ Hugging Face.

Usage:
    # 1. Tải artifact (chỉ cần 1 lần)
    python scripts/download_demo_artifacts.py

    # 2. Chạy demo
    python scripts/run_demo.py --customer-id 12345 --top-k 10
    python scripts/run_demo.py --customer-id 12345 --model early_fusion --top-k 10
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure project root is on path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import ARTIFACTS_DIR
from src.serving.pipeline import RecommenderPipeline


def main():
    ap = argparse.ArgumentParser(description="Demo inference recommender")
    ap.add_argument("--customer-id", type=int, default=0, help="User index")
    ap.add_argument("--top-k", type=int, default=10, help="Số gợi ý")
    ap.add_argument("--model", type=str, default="als", help="Model: als, bpr, late_fusion, early_fusion")
    args = ap.parse_args()

    if not ARTIFACTS_DIR.exists():
        print("ERROR: Không tìm thấy thư mục artifacts/.")
        print("Chạy trước: python scripts/download_demo_artifacts.py")
        sys.exit(1)

    print(f"Đang tải pipeline (model={args.model})...")
    pipe = RecommenderPipeline(cf_model_name=args.model, use_reranker=False)
    pipe.load()

    if not pipe._available:
        print("ERROR: Pipeline không khả dụng. Kiểm tra artifacts.")
        sys.exit(1)

    print(f"\nGợi ý cho user_idx={args.customer_id}, top-{args.top_k}:")
    print("-" * 60)

    # Simple recommendation using CF factors
    user_vec = pipe._user_cf_vec(args.customer_id)
    if pipe.item_factors is not None:
        scores = pipe.item_factors @ user_vec
        top_idx = np.argsort(-scores)[:args.top_k]
        for rank, idx in enumerate(top_idx, 1):
            item_orig = int(pipe.item_ids[idx]) if pipe.item_ids is not None else int(idx)
            print(f"  {rank:2d}. item_idx={idx:8d}  (orig={item_orig})  score={scores[idx]:.4f}")
    else:
        print("  Không có item factors.")

    print("-" * 60)
    print(f"Head status: {pipe.head_status}")


if __name__ == "__main__":
    import numpy as np
    main()
