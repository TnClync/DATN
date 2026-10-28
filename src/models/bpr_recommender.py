"""
BPR-MF (Bayesian Personalized Ranking Matrix Factorization) recommender.

Sử dụng checkpoint đã train trên Kaggle/Colab:
    models/bpr_mf/model/bpr_mf_best.pt  (hoặc bpr_mf_checkpoint.pt)

Checkpoint chứa user_emb / item_emb (embedding_dim=64) cùng id remapping
trong models/bpr_mf/data/id_mapping. Ánh xạ từ reviewerID (Amazon) đi qua:
    reviewerID -> (als) user_idx -> (bpr) new_user_idx
    asin       -> (als) item_idx -> (bpr) new_item_idx
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

BASE_DIR = Path(__file__).resolve().parents[2]


def _read_parquet_parts(directory: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(directory / "*.parquet")))
    if not files:
        raise FileNotFoundError(f"Không tìm thấy parquet trong {directory}")
    frames = [pq.read_table(f).to_pandas() for f in files]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _clean(value: Any, default: str = "N/A") -> str:
    if value is None:
        return default
    try:
        import math
        if isinstance(value, float) and math.isnan(value):
            return default
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text if text and text.lower() != "nan" else default


class BPRRecommender:
    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        als_dir: str | Path | None = None,
        bpr_data_dir: str | Path | None = None,
        serving_dir: str | Path | None = None,
    ):
        self.als_dir = Path(als_dir) if als_dir else BASE_DIR / "models" / "als"
        self.bpr_data_dir = Path(bpr_data_dir) if bpr_data_dir else BASE_DIR / "models" / "bpr_mf" / "data"
        self.serving_dir = Path(serving_dir) if serving_dir else BASE_DIR / "models" / "serving"

        if checkpoint_path is None:
            for name in ("bpr_mf_best.pt", "bpr_mf_checkpoint.pt"):
                cand = BASE_DIR / "models" / "bpr_mf" / "model" / name
                if cand.exists():
                    checkpoint_path = cand
                    break
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None

        self._loaded = False
        self._user_emb: Optional[torch.Tensor] = None
        self._item_emb: Optional[torch.Tensor] = None
        self._num_users = 0
        self._num_items = 0
        self._emb_dim = 0

        self._als_user_map: Optional[pd.DataFrame] = None   # reviewerID -> user_idx
        self._als_item_map: Optional[pd.DataFrame] = None   # asin -> item_idx
        self._bpr_user_map: Optional[dict[int, int]] = None # user_idx -> new_user_idx
        self._bpr_item_map: Optional[dict[int, int]] = None # item_idx -> new_item_idx
        self._rev_item_map: Optional[dict[int, int]] = None # new_item_idx -> item_idx
        self._seen_by_user: dict[int, set[int]] = {}

        self._meta_lookup = None  # ServingStore (lazy)

    # ------------------------------------------------------------------
    def load(self) -> "BPRRecommender":
        if self._loaded:
            return self
        if self.checkpoint_path is None or not self.checkpoint_path.exists():
            raise FileNotFoundError("Không tìm thấy checkpoint BPR-MF (.pt)")

        ck = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        sd = ck["model_state_dict"]
        self._user_emb = sd["user_emb.weight"].detach().cpu()
        self._item_emb = sd["item_emb.weight"].detach().cpu()
        self._num_users = int(ck["num_users"])
        self._num_items = int(ck["num_items"])
        self._emb_dim = int(ck["embedding_dim"])

        # ALS mappings
        self._als_user_map = _read_parquet_parts(self.als_dir / "user_mapping")
        self._als_item_map = _read_parquet_parts(self.als_dir / "item_mapping")

        # BPR remappings
        bu = _read_parquet_parts(self.bpr_data_dir / "id_mapping" / "users")
        bi = _read_parquet_parts(self.bpr_data_dir / "id_mapping" / "items")
        self._bpr_user_map = dict(zip(bu["user_idx"].astype(int), bu["new_user_idx"].astype(int)))
        self._bpr_item_map = dict(zip(bi["item_idx"].astype(int), bi["new_item_idx"].astype(int)))
        self._rev_item_map = {v: k for k, v in self._bpr_item_map.items()}

        self._loaded = True
        return self

    # ------------------------------------------------------------------
    def _als_user_idx(self, reviewer_id: str) -> Optional[int]:
        df = self._als_user_map
        sub = df[df["reviewerID"] == reviewer_id]
        if sub.empty:
            return None
        return int(sub.iloc[0]["user_idx"])

    def _als_item_idx_to_asin(self) -> dict[int, str]:
        df = self._als_item_map
        return dict(zip(df["item_idx"].astype(int), df["asin"].astype(str)))

    def _seen_new_items(self, user_idx: int) -> set[int]:
        if user_idx in self._seen_by_user:
            return self._seen_by_user[user_idx]
        result: set[int] = set()
        # đọc split train của bpr (user_idx,item_idx) — lọc theo user
        split_dir = self.bpr_data_dir / "split_data"
        train_dirs = [p for p in split_dir.glob("_split=train") if p.is_dir()]
        for td in train_dirs:
            df = _read_parquet_parts(td)
            sub = df[df["user_idx"] == user_idx]
            for it in sub["item_idx"].tolist():
                ni = self._bpr_item_map.get(int(it))
                if ni is not None:
                    result.add(ni)
        self._seen_by_user[user_idx] = result
        return result

    def _meta_for_asin(self, asin: str) -> dict:
        if self._meta_lookup is None:
            from .serving_store import ServingStore
            store = ServingStore(serving_dir=self.serving_dir)
            store.load()
            self._meta_lookup = store
        store = self._meta_lookup
        # tìm row_id theo asin
        import sqlite3
        conn = sqlite3.connect(f"file:{store.meta_path}?mode=ro", uri=True)
        try:
            cur = conn.execute("SELECT title, brand, category, price FROM catalog WHERE asin=? LIMIT 1", (asin,))
            row = cur.fetchone()
        finally:
            conn.close()
        if not row:
            return {"title": "N/A", "brand": "N/A", "category": "N/A", "price": "N/A"}
        return {"title": _clean(row[0]), "brand": _clean(row[1]),
                "category": _clean(row[2]), "price": _clean(row[3])}

    # ------------------------------------------------------------------
    def user_known(self, reviewer_id: str) -> bool:
        if not self._loaded:
            self.load()
        ui = self._als_user_idx(reviewer_id)
        if ui is None:
            return False
        return ui in self._bpr_user_map

    def get_status(self) -> Dict[str, Any]:
        return {
            "mode": "bpr_mf" if self._loaded else "not_loaded",
            "mode_label": "BPR-MF (Bayesian Personalized Ranking)",
            "num_users": self._num_users,
            "num_items": self._num_items,
            "embedding_dim": self._emb_dim,
            "loaded": self._loaded,
        }

    def recommend(
        self,
        user_raw_id: str,
        top_k: int = 10,
        domain: Optional[str] = None,
        exclude_seen: bool = True,
    ) -> List[Dict[str, Any]]:
        if not self._loaded:
            self.load()

        top_k = max(1, int(top_k))
        user_idx = self._als_user_idx(user_raw_id)
        if user_idx is None or user_idx not in self._bpr_user_map:
            return []

        new_user = self._bpr_user_map[user_idx]
        with torch.no_grad():
            uvec = self._user_emb[new_user]  # (D,)
            scores = (self._item_emb @ uvec).cpu().numpy().astype(np.float32)

        if exclude_seen:
            seen = self._seen_new_items(user_idx)
            if seen:
                scores[list(seen)] = -np.inf

        # domain filter: map new_item -> item_idx -> asin -> category
        domain_key = (domain or "").strip().lower()
        if domain_key not in ("", "all", "tất cả", "tat ca"):
            item_idx_of = self._rev_item_map
            asin_of_itemidx = self._als_item_idx_to_asin()
            # build mask
            keep = np.ones(self._num_items, dtype=bool)
            for ni in range(self._num_items):
                ii = item_idx_of.get(ni)
                if ii is None:
                    keep[ni] = False
                    continue
                asin = asin_of_itemidx.get(ii)
                if asin is None:
                    keep[ni] = False
                    continue
                meta = self._meta_for_asin(asin)
                cat = meta["category"].lower()
                if domain_key not in cat:
                    keep[ni] = False
            scores[~keep] = -np.inf

        if not np.isfinite(scores).any():
            return []

        top_positions = np.argsort(scores)[::-1][:top_k]
        asin_of_itemidx = self._als_item_idx_to_asin()

        results: List[Dict[str, Any]] = []
        for rank, ni in enumerate(top_positions, start=1):
            ni = int(ni)
            if not np.isfinite(scores[ni]):
                continue
            item_idx = self._rev_item_map.get(ni)
            asin = asin_of_itemidx.get(item_idx, "N/A") if item_idx is not None else "N/A"
            meta = self._meta_for_asin(asin) if asin != "N/A" else {
                "title": "N/A", "brand": "N/A", "category": "N/A", "price": "N/A"
            }
            results.append({
                "rank": rank,
                "asin": _clean(asin),
                "title": _clean(meta["title"])[:80],
                "brand": _clean(meta["brand"]),
                "category": _clean(meta["category"]),
                "price": _clean(meta["price"]),
                "score": round(float(scores[ni]), 4),
            })
        # re-rank ranks contiguous
        for i, r in enumerate(results, start=1):
            r["rank"] = i
        return results
