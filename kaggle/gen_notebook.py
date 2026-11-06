"""Generate a Kaggle-ready notebook (ipynb) for training."""
import json, os

cells = []

def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in text.split("\n")]})

def code(text):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": [line + "\n" for line in text.split("\n")]})

md("""# Deep Hybrid Recommender — Kaggle Training

Train ALS/BPR (implicit) + SBERT + Fusion models trên Kaggle.

## Chuẩn bị
1. Add dataset chứa thư mục `data/` (train/validation/test parquet + mappings) vào notebook (Input).
2. Bật GPU (T4/P100) để encode SBERT nhanh (optional).
3. Chạy lần lượt các cell.
""")

code("""# 1. Setup
!pip -q install implicit sentence-transformers faiss-cpu 2>&1 | tail -3
import os, sys
DATA_ROOT = "/kaggle/input/datn-data"   # <-- sửa tên dataset của bạn
WORK = "/kaggle/working/DATN"
if not os.path.exists(WORK):
    !git clone https://github.com/TnClync/DATN.git $WORK
os.chdir(WORK)
sys.path.insert(0, WORK)
os.environ["DATN_ROOT"] = WORK
os.environ["DATN_DATA_DIR"] = os.path.join(DATA_ROOT, "data")
print("cwd", os.getcwd())
print("data exists:", os.path.exists(os.environ["DATN_DATA_DIR"]))
""")

code("""# 2. Train ALS
!python scripts/01_train_cf.py --model als --factors 64 --iters 15 --threads 8
""")

code("""# 3. Train BPR
!python scripts/01_train_cf.py --model bpr --factors 64 --iters 30 --threads 8
""")

code("""# 4. Build SBERT + FAISS (dùng GPU nếu có)
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device", device)
!python scripts/02_build_sbert.py --model-name sentence-transformers/all-MiniLM-L6-v2 --batch-size 512 --device $device
""")

code("""# 5. Baseline Late Fusion eval
!python scripts/03_baseline_late_fusion_eval.py --users 500 --k 20
""")

code("""# 6. Two-Stage eval (không reranker để nhanh)
!python scripts/04_two_stage_rerank_eval.py --users 200 --no-rerank
""")

code("""# 7. Light Gated Fusion train
!python scripts/05_train_gated_fusion.py --epochs 5 --batch 2048
""")

code("""# 8. Deep Early Fusion train
!python scripts/06_train_deep_early_fusion.py --epochs 3 --batch 1024 --samples 200000
""")

code("""# 9. Zip artifacts để tải về
!cd /kaggle/working && zip -r DATN_artifacts.zip DATN/artifacts 2>&1 | tail -3
print("Done. Download DATN_artifacts.zip từ Output.")
""")

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = "/home/clyde/DATN/kaggle/DATN_train.ipynb"
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as f:
    json.dump(nb, f, indent=2)
print("written", out)
