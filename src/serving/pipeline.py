"""End-to-end inference pipeline for local CPU deployment (8GB RAM friendly).

All item indices returned by the public API are ORIGINAL Amazon item ids
(`item_ids.npy`), while internal scoring works on compact row indices.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import FACTORS_DIR, SBERT_DIR, FAISS_DIR, MODELS_DIR, NUM_ITEMS
from ..models import cf as cf_mod
from ..models import late_fusion as lf
from ..models import sbert as sbert_mod
from ..models import two_stage as ts

try:  # torch is optional: CF/SBERT/late-fusion still work without it
    import torch
    from ..models.gated_fusion import AdaptiveLightGate
    from ..models.early_fusion import DeepCrossFusion
except Exception:  # pragma: no cover
    torch = None
    AdaptiveLightGate = None
    DeepCrossFusion = None


class RecommenderPipeline:
    def __init__(self, cf_model_name: str = "als", use_reranker: bool = False,
                 reranker_name: str = "BAAI/bge-reranker-v2-m3"):
        self.cf_name = cf_model_name
        self.use_reranker = use_reranker
        # lazy
        self.user_factors = None  # float16 mmap
        self.item_factors = None  # float32
        self.user_ids = None      # sorted original user ids
        self.item_ids = None      # sorted original item ids (compact -> original)
        self.sbert_emb = None     # float16 mmap (compact item rows)
        self.sbert_index_ids = None  # FAISS row -> compact item row
        self.faiss_index = None
        self.gate_model = None
        self.deep_model = None
        self.reranker = None
        self.popular_items = None
        self.item_text_fn = lambda i: f"item {i}"
        self._loaded = False
        self._available = False

    # ------------------------------------------------------------------ load
    def load(self):
        if self._loaded:
            return
        self._available = False
        try:
            uf, itf = cf_mod.load_factors(self.cf_name, dtype="float16", mmap=True)
            self.user_factors = uf
            self.item_factors = itf
            self._available = True
        except Exception:
            self.user_factors = None
            self.item_factors = None

        # ID mappings
        try:
            uid_p = FACTORS_DIR / f"{self.cf_name}_user_ids.npy"
            iid_p = FACTORS_DIR / f"{self.cf_name}_item_ids.npy"
            if uid_p.exists():
                self.user_ids = np.load(uid_p)
            if iid_p.exists():
                self.item_ids = np.load(iid_p)
        except Exception:
            pass

        # SBERT embeddings + FAISS index
        try:
            self.sbert_emb = sbert_mod.load_sbert(mmap=True)
            self.faiss_index = sbert_mod.load_faiss_index()
            idx_ids = SBERT_DIR / "sbert_index_ids.npy"
            if idx_ids.exists():
                self.sbert_index_ids = np.load(idx_ids)
        except Exception:
            self.sbert_emb = None
            self.faiss_index = None
            self.sbert_index_ids = None

        # Deep hybrid heads (trained on Kaggle, reuse CF factors)
        if torch is not None:
            gate_p = MODELS_DIR / "light_gated_fusion.pt"
            deep_p = MODELS_DIR / "deep_early_fusion.pt"
            try:
                if gate_p.exists() and AdaptiveLightGate is not None:
                    self.gate_model = AdaptiveLightGate()
                    self.gate_model.load_state_dict(torch.load(gate_p, map_location="cpu"))
                    self.gate_model.eval()
            except Exception:
                self.gate_model = None
            try:
                if deep_p.exists() and DeepCrossFusion is not None:
                    self.deep_model = DeepCrossFusion()
                    self.deep_model.load_state_dict(torch.load(deep_p, map_location="cpu"))
                    self.deep_model.eval()
            except Exception:
                self.deep_model = None

        if self.use_reranker:
            try:
                self.reranker = ts.load_reranker()
            except Exception:
                self.reranker = None
        self._loaded = True

    # -------------------------------------------------------------- id lookup
    def _user_row_idx(self, user_idx: int) -> int:
        if self.user_ids is not None and len(self.user_ids) > 0:
            pos = int(np.searchsorted(self.user_ids, user_idx))
            if pos < len(self.user_ids) and int(self.user_ids[pos]) == int(user_idx):
                return pos
        if self.user_factors is not None and len(self.user_factors) > 0:
            if 0 <= user_idx < len(self.user_factors):
                return user_idx
            return user_idx % len(self.user_factors)
        return 0

    def _item_row(self, item_orig: int) -> int | None:
        if self.item_ids is not None and len(self.item_ids) > 0:
            pos = int(np.searchsorted(self.item_ids, item_orig))
            if pos < len(self.item_ids) and int(self.item_ids[pos]) == int(item_orig):
                return pos
            return None
        return item_orig if 0 <= item_orig < NUM_ITEMS else None

    # ------------------------------------------------------------------ score
    def _user_cf_vec(self, user_idx: int) -> np.ndarray:
        row = self._user_row_idx(user_idx)
        return np.asarray(self.user_factors[row], dtype=np.float32)

    def _user_sbert_vec(self, hist_rows: list[int] | None) -> np.ndarray | None:
        if self.sbert_emb is None or not hist_rows:
            return None
        vecs = np.asarray(self.sbert_emb[hist_rows], dtype=np.float32)
        v = vecs.mean(axis=0)
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def cf_topk(self, user_idx: int, k: int = 200, exclude: set[int] | None = None):
        row = self._user_row_idx(user_idx)
        return cf_mod.topk_cf(self.user_factors, self.item_factors, row, k=k, exclude=exclude)

    def sbert_topk(self, hist_rows: list[int] | None, k: int = 200):
        if self.faiss_index is None:
            return [], []
        q = self._user_sbert_vec(hist_rows)
        if q is None:
            return [], []
        items, scores = sbert_mod.sbert_topk(self.faiss_index, q, k=k)
        if self.sbert_index_ids is not None and len(items):
            items = [int(self.sbert_index_ids[i]) if 0 <= i < len(self.sbert_index_ids) else int(i)
                     for i in items]
        return items, scores

    # -------------------------------------------------------------- recommend
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

        hist_rows = [r for r in (self._item_row(i) for i in (user_history or [])) if r is not None]
        exclude = set(hist_rows) if exclude_history else set()

        cf_items, cf_scores_list = self.cf_topk(user_idx, k=candidate_k, exclude=exclude)
        sb_items, sb_scores_list = self.sbert_topk(hist_rows, k=candidate_k)

        raw_recs: list[int]
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
                # fallback reranker: weighted CF score
                cands_arr = np.asarray(cands)
                cf_map = dict(zip(cf_items, cf_scores_list))
                scores = np.array([cf_map.get(int(i), 0.0) for i in cands_arr], dtype=np.float32)
                raw_recs = [int(cands_arr[i]) for i in np.argsort(-scores)[:k]]
        elif model in ("light_gated", "deep_early"):
            raw_recs = self._deep_hybrid_recs(user_idx, hist_rows, cf_items, cf_scores_list,
                                              sb_items, sb_scores_list, model, k, alpha)
        else:
            # classic late fusion
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

        # compact row -> original item id
        if self.item_ids is not None and len(self.item_ids) > 0:
            final_recs = [int(self.item_ids[i]) if 0 <= i < len(self.item_ids) else int(i) for i in raw_recs]
        else:
            final_recs = [int(i) for i in raw_recs]
        return final_recs

    # ------------------------------------------------------- deep hybrid heads
    def _deep_hybrid_recs(self, user_idx, hist_rows, cf_items, cf_scores_list,
                          sb_items, sb_scores_list, model, k, alpha):
        cands = list(dict.fromkeys(list(cf_items) + list(sb_items)))
        if not cands:
            return []
        cands_arr = np.asarray(cands, dtype=np.int64)
        u_cf = self._user_cf_vec(user_idx)
        i_cf = np.asarray(self.item_factors[cands_arr], dtype=np.float32)
        cf_raw = i_cf @ u_cf

        u_sb = self._user_sbert_vec(hist_rows)
        if u_sb is not None and self.sbert_emb is not None:
            i_sb = np.asarray(self.sbert_emb[cands_arr], dtype=np.float32)
            sb_raw = i_sb @ u_sb
        else:
            i_sb = np.zeros((len(cands), 384), dtype=np.float32)
            sb_raw = np.zeros(len(cands), dtype=np.float32)

        if model == "light_gated" and self.gate_model is not None and torch is not None:
            with torch.no_grad():
                bu = torch.tensor(np.repeat(u_cf[None, :], len(cands), axis=0), dtype=torch.float32)
                bi = torch.tensor(i_cf, dtype=torch.float32)
                bs = torch.tensor(i_sb, dtype=torch.float32)
                bcf = torch.tensor(cf_raw, dtype=torch.float32)
                bsb = torch.tensor(sb_raw, dtype=torch.float32)
                fused, _ = self.gate_model(bu, bi, bs, bcf, bsb)
                scores = fused.numpy()
        elif model == "deep_early" and self.deep_model is not None and torch is not None:
            with torch.no_grad():
                feats = np.concatenate([
                    np.repeat(u_cf[None, :], len(cands), axis=0), i_cf, i_sb
                ], axis=1)
                scores = self.deep_model(torch.tensor(feats, dtype=torch.float32)).numpy()
        else:
            fused = lf.classic_late_fusion(cf_raw, sb_raw, alpha=alpha, beta=1 - alpha)
            scores = fused

        order = np.argsort(-scores)
        return [int(cands_arr[i]) for i in order][:k]
