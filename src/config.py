"""Central configuration for the Deep Hybrid Recommender System.

All paths are relative to the project root. Override via environment variables
when running on Kaggle / different machines.
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("DATN_ROOT", Path(__file__).resolve().parents[1]))

DATA_DIR = Path(os.environ.get("DATN_DATA_DIR", PROJECT_ROOT / "data"))
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "validation"
TEST_DIR = DATA_DIR / "test"
USER_MAP_DIR = DATA_DIR / "user_mapping"
ITEM_MAP_DIR = DATA_DIR / "item_mapping"
SPLIT_META = DATA_DIR / "split_metadata.json"

ARTIFACTS_DIR = Path(os.environ.get("DATN_ARTIFACTS_DIR", PROJECT_ROOT / "artifacts"))
FACTORS_DIR = ARTIFACTS_DIR / "factors"
SBERT_DIR = ARTIFACTS_DIR / "sbert"
FAISS_DIR = ARTIFACTS_DIR / "faiss"
MODELS_DIR = ARTIFACTS_DIR / "models"
RESULTS_DIR = ARTIFACTS_DIR / "results"

for d in (ARTIFACTS_DIR, FACTORS_DIR, SBERT_DIR, FAISS_DIR, MODELS_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Dimensions
NUM_USERS = 22_048_170
NUM_ITEMS = 3_686_940
CF_FACTORS = 32
SBERT_DIM = 384

# Reproducibility
SEED = 42
""" 
"""
