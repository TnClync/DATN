#!/usr/bin/env python
"""Tải artifact demo từ Hugging Face Hub về máy local.

Usage:
    python scripts/download_demo_artifacts.py [--target-dir ./artifacts]
    python scripts/download_demo_artifacts.py --target-dir /path/to/project/artifacts

Sau khi tải xong, đặt DATN_ROOT trỏ tới thư mục chứa artifacts/ và chạy demo.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "Clync/datn-deep-hybrid-recommender-data"
REPO_TYPE = "dataset"


def main():
    ap = argparse.ArgumentParser(description="Tải artifact demo từ Hugging Face")
    ap.add_argument("--target-dir", type=str, default="artifacts",
                    help="Thư mục đích (mặc định: ./artifacts)")
    ap.add_argument("--token", type=str, default=None,
                    help="HF token (hoặc đặt env HF_TOKEN)")
    args = ap.parse_args()

    target = Path(args.target_dir).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    print(f"Đang tải từ {REPO_ID} ...")
    print(f"Thư mục đích: {target}")
    print("Lưu ý: tổng dung lượng ~3.9 GB, có thể mất 10-30 phút tuỳ tốc độ mạng.")
    print()

    local_dir = snapshot_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        local_dir=str(target.parent),
        token=args.token or os.environ.get("HF_TOKEN"),
        ignore_patterns=["*.log", "*.tmp"],
    )

    print(f"\nTải xong! Artifact nằm tại: {local_dir}/artifacts/")
    print("\nChạy demo:")
    print(f"  cd {target.parent}")
    print("  uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000")
    print("  # Web UI: http://localhost:8000")


if __name__ == "__main__":
    main()
