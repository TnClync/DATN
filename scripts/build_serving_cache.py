#!/usr/bin/env python
"""
Build serving cache cho Hybrid Recommender.

Chạy MỘT LẦN sau khi đã có dữ liệu trong:
    models/als/{item_mapping,user_mapping,train}
    models/content_based/item_vectors

    python scripts/build_serving_cache.py

Cache được ghi vào models/serving/ và giúp app khởi động trong < 1 giây.
Dùng --force để build lại từ đầu.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.models.serving_store import ServingStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Build serving cache cho recommender")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Build lại dù cache đã tồn tại",
    )
    args = parser.parse_args()

    store = ServingStore()
    print("=" * 70)
    print("BUILD SERVING CACHE")
    print(f"  ALS      : {store.als_dir}")
    print(f"  Content  : {store.cb_dir}")
    print(f"  Output   : {store.serving_dir}")
    print("=" * 70)

    try:
        info = store.build(force=args.force)
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1

    print("\nHoàn tất!")
    for key, value in info.items():
        print(f"  {key:16s}: {value}")
    print("\nBây giờ chạy: streamlit run app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
