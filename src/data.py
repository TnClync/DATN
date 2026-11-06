"""Data loading helpers for the Amazon Reviews implicit dataset.

The dataset is stored as Spark-style zstd-compressed parquet parts. We read them
with pyarrow and convert to pandas / scipy sparse matrices for `implicit`.

Memory strategy for 8GB RAM (deploy):
  * Train on Kaggle (full 45.6M).
  * Export compact factors (.npy float16) + item metadata.
  * Local inference only mmaps user factors and keeps item factors in RAM.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.sparse import csr_matrix

from .config import (
    TRAIN_DIR, VAL_DIR, TEST_DIR, USER_MAP_DIR, ITEM_MAP_DIR, SPLIT_META,
    NUM_USERS, NUM_ITEMS,
)


def list_parquet_parts(directory: Path) -> list[str]:
    return sorted(glob.glob(str(directory / "*.parquet")))


def read_parquet_dir(directory: Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Read all parquet parts in a directory into a single DataFrame."""
    parts = list_parquet_parts(directory)
    if not parts:
        raise FileNotFoundError(f"No parquet parts in {directory}")
    tables = [pq.read_table(p, columns=columns) for p in parts]
    import pyarrow as pa
    table = pa.concat_tables(tables)
    return table.to_pandas()


def load_split(name: str, columns: list[str] | None = None) -> pd.DataFrame:
    base = {"train": TRAIN_DIR, "validation": VAL_DIR, "test": TEST_DIR}[name]
    return read_parquet_dir(base, columns=columns)


def load_user_mapping() -> pd.DataFrame:
    return read_parquet_dir(USER_MAP_DIR)


def load_item_mapping() -> pd.DataFrame:
    return read_parquet_dir(ITEM_MAP_DIR)


def split_metadata() -> dict:
    with open(SPLIT_META) as f:
        return json.load(f)


def build_interaction_matrix(
    df: pd.DataFrame,
    num_users: int = NUM_USERS,
    num_items: int = NUM_ITEMS,
    implicit: bool = True,
    min_user_interactions: int = 0,
) -> csr_matrix:
    """Build a user x item CSR matrix suitable for `implicit`.

    `implicit=True` converts ratings to binary confidence (rating>=4 -> 1).
    """
    d = df.copy()
    if implicit:
        d = d[d["rating"] >= 4.0] if "rating" in d.columns else d
    if min_user_interactions > 0:
        counts = d.groupby("user_idx").size()
        keep = counts[counts >= min_user_interactions].index
        d = d[d["user_idx"].isin(keep)]

    rows = d["user_idx"].to_numpy(dtype=np.int64)
    cols = d["item_idx"].to_numpy(dtype=np.int64)
    data = np.ones(len(d), dtype=np.float32) if implicit else d["rating"].to_numpy(dtype=np.float32)

    mat = csr_matrix((data, (rows, cols)), shape=(num_users, num_items))
    # implicit expects (items x users) for some algos? implicit ALS expects item_users? 
    # implicit's fit expects (user_items) CSR user x item for recommend(user_id,...).
    return mat


def user_item_sets(df: pd.DataFrame) -> dict[int, set[int]]:
    g = df.groupby("user_idx")["item_idx"].apply(lambda s: set(s.tolist()))
    return g.to_dict()


def ground_truth_by_user(df: pd.DataFrame) -> dict[int, set[int]]:
    """Validation/test ground truth: items per user (implicit positives)."""
    pos = df[df["rating"] >= 4.0] if "rating" in df.columns else df
    return pos.groupby("user_idx")["item_idx"].apply(lambda s: set(s.tolist())).to_dict()
