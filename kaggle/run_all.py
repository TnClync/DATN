"""Kaggle one-file runner: train CF + build SBERT + eval baseline.

Usage on Kaggle:
    !python kaggle/run_all.py
"""
import subprocess, sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

steps = [
    "python scripts/01_train_cf.py --model als --factors 64 --iters 15 --threads 8",
    "python scripts/01_train_cf.py --model bpr --factors 64 --iters 30 --threads 8",
    "python scripts/02_build_sbert.py --batch-size 256 --device cuda",
    "python scripts/03_baseline_late_fusion_eval.py --users 500",
    "python scripts/04_two_stage_rerank_eval.py --users 200 --no-rerank",
]
for s in steps:
    print(">>>", s)
    r = subprocess.run(s, shell=True)
    if r.returncode != 0:
        print("FAILED:", s)
        sys.exit(r.returncode)
print("ALL DONE")
