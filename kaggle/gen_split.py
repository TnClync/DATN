"""Generate per-account Kaggle notebooks to split training across 2 accounts.

Usage:
    python kaggle/gen_split.py A   # -> kaggle/DATN_train_A.ipynb
    python kaggle/gen_split.py B
"""
import json, os, sys

account = (sys.argv[1] if len(sys.argv) > 1 else "A").upper()
assert account in ("A", "B")

cells = []
def md(t): cells.append({"cell_type":"markdown","metadata":{},"source":[l+"\n" for l in t.split("\n")]})
def code(t): cells.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[l+"\n" for l in t.split("\n")]})

md(f"""# Deep Hybrid Recommender — Kaggle Training (Account {account})

Phân chia công việc giữa 2 tài khoản để tránh timeout và tận dụng quota GPU.

## Account {account} phụ trách
""" + ("""- Train ALS
- Build SBERT + FAISS
- Baseline Late Fusion eval
""" if account == "A" else """- Train BPR
- Two-Stage eval
- Light Gated Fusion train
- Deep Early Fusion train
"""))

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

if account == "A":
    code("""# 2. Train ALS
!python scripts/01_train_cf.py --model als --factors 64 --iters 15 --threads 8
""")
    code("""# 3. Build SBERT + FAISS
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device", device)
!python scripts/02_build_sbert.py --model-name sentence-transformers/all-MiniLM-L6-v2 --batch-size 512 --device $device
""")
    code("""# 4. Baseline Late Fusion eval
!python scripts/03_baseline_late_fusion_eval.py --users 500 --k 20
""")
    code("""# 5. Zip artifacts A
!cd /kaggle/working && zip -r DATN_artifacts_A.zip DATN/artifacts 2>&1 | tail -3
print("Download DATN_artifacts_A.zip")
""")
else:
    code("""# 2. Train BPR
!python scripts/01_train_cf.py --model bpr --factors 64 --iters 30 --threads 8
""")
    code("""# 3. Two-Stage eval (no reranker để nhanh; bật reranker nếu còn GPU time)
!python scripts/04_two_stage_rerank_eval.py --users 200 --no-rerank
""")
    code("""# 4. Light Gated Fusion train
!python scripts/05_train_gated_fusion.py --epochs 5 --batch 2048
""")
    code("""# 5. Deep Early Fusion train
!python scripts/06_train_deep_early_fusion.py --epochs 3 --batch 1024 --samples 200000
""")
    code("""# 6. Zip artifacts B
!cd /kaggle/working && zip -r DATN_artifacts_B.zip DATN/artifacts 2>&1 | tail -3
print("Download DATN_artifacts_B.zip")
""")

nb = {"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.10"}},"nbformat":4,"nbformat_minor":5}
out = f"/home/clyde/DATN/kaggle/DATN_train_{account}.ipynb"
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out,"w") as f: json.dump(nb, f, indent=2)
print("written", out)
