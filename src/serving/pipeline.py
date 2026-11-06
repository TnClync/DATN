"""End-to-end inference pipeline for local CPU deployment (8GB RAM friendly)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..config import FACTORS_DIR, SBERT_DIR, FAISS_DIR, MODELS_DIR, NUM_ITEMS
from ..models import cf as cf_mod
from ..models import late_fusion as lf
from ..models import sbert as sbert_mod
from ..models import two_stage as ts


class RecommenderPipeline:
    def __init__(self, cf_model_name: str = "als", use_reranker: bool = False,
                 reranker_name: str = "BAAI/bge-reranker-v2-m3"):
        self.cf_name = cf_model_name
        self.use_reranker = use_reranker
        # lazy
        self.user_factors = None  # float32 (mmap-backed)
        self.item_factors = None
        self.user_ids = None
        self.item_ids = None
        self.user_id_to_row = {}
        self.sbert_emb = None
        self.faiss_index = None
        self.reranker = None
        self.popular_items = None
        self.item_text_fn = lambda i: f"item {i}"
        self._loaded = False
        self._available = False

    def load(self):
        if self._loaded:
            return
        self._available = False
        try:
            uf, itf = cf_mod.load_factors(self.cf_name, dtype="float16", mmap=True)
            self.user_factors = np.asarray(uf, dtype=np.float32)
            self.item_factors = np.asarray(itf, dtype=np.float32)
            self._available = True
        except Exception:
            self.user_factors = None
            self.item_factors = None

        # Load IDs mapping if available
        try:
            uid_p = FACTORS_DIR / f"{self.cf_name}_user_ids.npy"
            iid_p = FACTORS_DIR / f"{self.cf_name}_item_ids.npy"
            if uid_p.exists():
                self.user_ids = np.load(uid_p)
                self.user_id_to_row = {int(uid): idx for idx, uid in enumerate(self.user_ids)}
            if iid_p.exists():
                self.item_ids = np.load(iid_p)
        except Exception:
            pass

        # SBERT
        try:
            self.sbert_emb = sbert_mod.load_sbert(mmap=False)
            self.faiss_index = sbert_mod.load_faiss_index()
        except Exception:
            self.sbert_emb = None
            self.faiss_index = None
        if self.use_reranker:
            try:
                self.reranker = ts.load_reranker()
            except Exception:
                self.reranker = None
        self._loaded = True

    def _user_row_idx(self, user_idx: int) -> int:
        if self.user_id_to_row and user_idx in self.user_id_to_row:
            return self.user_id_to_row[user_idx]
        if self.user_factors is not None and len(self.user_factors) > 0:
            if 0 <= user_idx < len(self.user_factors):
                return user_idx
            return user_idx % len(self.user_factors)
        return 0

    def _user_cf_vec(self, user_idx: int) -> np.ndarray:
        row = self._user_row_idx(user_idx)
        return np.asarray(self.user_factors[row], dtype=np.float32)

    def _user_sbert_vec(self, user_history: list[int] | None) -> np.ndarray | None:
        if self.sbert_emb is None or not user_history:
            return None
        vecs = self.sbert_emb[user_history]
        v = vecs.mean(axis=0)
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def cf_topk(self, user_idx: int, k: int = 200, exclude: set[int] | None = None):
        row = self._user_row_idx(user_idx)
        return cf_mod.topk_cf(self.user_factors, self.item_factors, row, k=k, exclude=exclude)

    def sbert_topk(self, user_history: list[int] | None, k: int = 200):
        if self.faiss_index is None:
            return [], []
        q = self._user_sbert_vec(user_history)
        if q is None:
            return [], []
        return sbert_mod.sbert_topk(self.faiss_index, q, k=k)

    def recommend(
        self,
        user_idx: int,
        user_history: list[int] | None = None,
        k: int = 10,
        model: str = "late_fusion",
        alpha: float = 0.6,
        candidate_k: int = 300,
        exclude_history: bool = True,
    ) -> list[int]:
        self.load()
        if not self._available:
            return []
        exclude = set(user_history or []) if exclude_history else set()
        cf_items, cf_scores_list = self.cf_topk(user_idx, k=candidate_k, exclude=exclude)
        sb_items, sb_scores_list = self.sbert_topk(user_history, k=candidate_k)

        raw_recs = []
        if model in ("cf", "als", "bpr"):
            raw_recs = cf_items[:k]
        elif model == "sbert":
            raw_recs = sb_items[:k] if sb_items else cf_items[:k]
        elif model == "two_stage":
            cands = ts.candidate_union(cf_items, sb_items, self.popular_items, max_candidates=candidate_k)
            if self.reranker is not None:
                q = f"user {user_idx}"
                scored = ts.rerank_cross_encoder(self.reranker, q, cands, self.item_text_fn)
                raw_recs = [it for it, _ in scored[:k]]
            else:
                raw_recs = cands[:k]
        else:
            # late fusion (classic)
            cand_set = set(cf_items) | set(sb_items)
            cands = list(cand_set)
            cf_score_map = {it: 0.0 for it in cands}
            for it, sc in zip(cf_items, cf_scores_list):
                cf_score_map[it] = float(sc)
            sb_score_map = {it: 0.0 for it in cands}
            for it, sc in zip(sb_items, sb_scores_list):
                sb_score_map[it] = float(sc)
            cf_arr = np.array([cf_score_map[it] for it in cands], dtype=np.float32)
            sb_arr = np.array([sb_score_map[it] for it in cands], dtype=np.float32)
            fused = lf.classic_late_fusion(cf_arr, sb_arr, alpha=alpha, beta=1 - alpha)
            order = np.argsort(-fused)
            out = [cands[i] for i in order if cands[i] not in exclude]
            raw_recs = out[:k]

        # Map internal item indices to original item IDs if mapping exists
        if self.item_ids is not None and len(self.item_ids) > 0:
            final_recs = [int(self.item_ids[i]) if 0 <= i < len(self.item_ids) else int(i) for i in raw_recs]
        else:
            final_recs = [int(i) for i in raw_recs]
        return final_recs
