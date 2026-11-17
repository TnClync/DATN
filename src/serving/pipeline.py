"""End-to-end inference pipeline for local CPU deployment (8GB RAM friendly).

All item indices returned by the public API are ORIGINAL Amazon item ids
(`item_ids.npy`), while internal scoring works on compact row indices.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..config import FACTORS_DIR, SBERT_DIR, FAISS_DIR, MODELS_DIR, NUM_ITEMS, SBERT_DIM
from ..models import cf as cf_mod
from ..models import late_fusion as lf
from ..models import sbert as sbert_mod
from ..models import two_stage as ts

try:  # torch is optional: CF/SBERT/late-fusion still work without it
    import torch
    from ..models.gated_fusion import AdaptiveLightGate
    from ..models.early_fusion import DeepCrossFusion
    from ..models.graph_models import (GraphLightGate, UserTower, ItemTower,
                                      TransformerFusion)
except Exception:  # pragma: no cover
    torch = None
    AdaptiveLightGate = None
    DeepCrossFusion = None
    GraphLightGate = None
    UserTower = None
    ItemTower = None
    TransformerFusion = None


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
        self.sbert_emb = None     # float16 mmap (compact item rows, 384-d)
        self.sbert_emb_768 = None  # float16 mmap (combined metadata|reviewer, 768-d)
        self.sbert_index_ids = None  # FAISS row -> compact item row
        self.faiss_index = None
        self.gate_model = None
        self.deep_model = None
        self.graph_gate_model = None
        self.user_tower = None
        self.item_tower = None
        self.transformer_model = None
        self.graph_user_factors = None
        self.graph_item_factors = None
        self.als_factors = None    # (user, item) mmap pair for ensemble scoring
        self.bpr_factors = None
        self.ensemble_weights = None
        self.ensemble_late_alpha = 0.9
        self.reranker = None
        self.popular_items = None
        self.item_text_fn = lambda i: f"item {i}"
        self._loaded = False
        self._available = False
        self.head_status = {}

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
            # popular items for cold-start (top by factor norm)
            try:
                norms = np.linalg.norm(np.asarray(itf, dtype=np.float32), axis=1)
                self.popular_items = np.argsort(-norms)[:200].tolist()
            except Exception:
                self.popular_items = list(range(min(200, len(itf))))
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

        # 768-d combined (metadata | reviewer) embeddings used by the graph hybrids
        try:
            p768 = SBERT_DIR / "sbert_item_embeddings_768_f16.npy"
            if p768.exists():
                self.sbert_emb_768 = np.load(p768, mmap_mode="r")
        except Exception:
            self.sbert_emb_768 = None

        # Graph factors (LightGCN) — needed for graph-based deep hybrids
        try:
            gu_p = FACTORS_DIR / "graph_user_factors_f16.npy"
            gi_p = FACTORS_DIR / "graph_item_factors_f16.npy"
            if gu_p.exists() and gi_p.exists():
                self.graph_user_factors = np.load(gu_p, mmap_mode="r")
                self.graph_item_factors = np.load(gi_p).astype(np.float32)
        except Exception:
            self.graph_user_factors = None
            self.graph_item_factors = None

        # Deep hybrid heads (trained on Kaggle, reuse CF factors) — infer dims from checkpoint
        self.head_status = {}
        if torch is not None:
            gate_p = MODELS_DIR / "light_gated_fusion.pt"
            deep_p = MODELS_DIR / "deep_early_fusion.pt"
            try:
                if gate_p.exists() and AdaptiveLightGate is not None:
                    sd = torch.load(gate_p, map_location="cpu")
                    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
                    in_f = int(sd["gate_net.0.weight"].shape[1]) if "gate_net.0.weight" in sd else 32 + 32 + (int(self.sbert_emb.shape[1]) if self.sbert_emb is not None else SBERT_DIM)
                    self.gate_model = AdaptiveLightGate(in_features=in_f)
                    self.gate_model.load_state_dict(sd)
                    self.gate_model.eval()
                    self.head_status["light_gated"] = f"loaded(in_features={in_f})"
            except Exception as e:
                self.gate_model = None
                self.head_status["light_gated"] = f"unavailable: {e}"
            try:
                if deep_p.exists() and DeepCrossFusion is not None:
                    sd = torch.load(deep_p, map_location="cpu")
                    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
                    in_d = int(sd["cross.0.weight"].shape[1]) if "cross.0.weight" in sd else 32 + 32 + (int(self.sbert_emb.shape[1]) if self.sbert_emb is not None else SBERT_DIM)
                    self.deep_model = DeepCrossFusion(in_dim=in_d)
                    self.deep_model.load_state_dict(sd)
                    self.deep_model.eval()
                    self.head_status["deep_early"] = f"loaded(in_dim={in_d})"
            except Exception as e:
                self.deep_model = None
                self.head_status["deep_early"] = f"unavailable: {e}"

            # Graph Light Gated
            try:
                gp = MODELS_DIR / "graph_light_gated_fusion.pt"
                if gp.exists() and GraphLightGate is not None:
                    sd = torch.load(gp, map_location="cpu")
                    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
                    in_f = int(sd["gate_net.0.weight"].shape[1]) if "gate_net.0.weight" in sd else 32 + 32 + (int(self.sbert_emb.shape[1]) if self.sbert_emb is not None else SBERT_DIM)
                    self.graph_gate_model = GraphLightGate(in_features=in_f)
                    self.graph_gate_model.load_state_dict(sd)
                    self.graph_gate_model.eval()
                    self.head_status["graph_light_gated"] = f"loaded(in_features={in_f})"
            except Exception as e:
                self.graph_gate_model = None
                self.head_status["graph_light_gated"] = f"unavailable: {e}"

            # Two-Tower
            try:
                tp = MODELS_DIR / "two_tower_fusion.pt"
                if tp.exists() and UserTower is not None and ItemTower is not None:
                    sd = torch.load(tp, map_location="cpu")
                    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
                    user_sd = sd.get("user", sd)
                    item_sd = sd.get("item", sd)
                    self.user_tower = UserTower()
                    self.user_tower.load_state_dict(user_sd)
                    self.user_tower.eval()
                    self.item_tower = ItemTower()
                    self.item_tower.load_state_dict(item_sd)
                    self.item_tower.eval()
                    self.head_status["two_tower"] = "loaded"
            except Exception as e:
                self.user_tower = None
                self.item_tower = None
                self.head_status["two_tower"] = f"unavailable: {e}"

            # Transformer
            try:
                tp = MODELS_DIR / "transformer_fusion.pt"
                if tp.exists() and TransformerFusion is not None:
                    sd = torch.load(tp, map_location="cpu")
                    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
                    self.transformer_model = TransformerFusion()
                    self.transformer_model.load_state_dict(sd)
                    self.transformer_model.eval()
                    self.head_status["transformer"] = "loaded"
            except Exception as e:
                self.transformer_model = None
                self.head_status["transformer"] = f"unavailable: {e}"

        # Stacked ensemble weights (produced by the eval kernel)
        try:
            ep = MODELS_DIR / "ensemble_weights.json"
            if ep.exists():
                data = json.loads(ep.read_text())
                w = {k: float(v) for k, v in (data.get("weights") or {}).items()}
                self.ensemble_weights = {k: v for k, v in w.items() if v > 0} or None
                self.ensemble_late_alpha = float(data.get("late_alpha", 0.9))
                if self.ensemble_weights:
                    self.head_status["ensemble"] = f"loaded({len(self.ensemble_weights)} base models)"
        except Exception as e:
            self.ensemble_weights = None
            self.head_status["ensemble"] = f"unavailable: {e}"

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
            return -1
        if self.user_factors is not None and len(self.user_factors) > 0:
            if 0 <= user_idx < len(self.user_factors):
                return user_idx
        return -1

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
        if row < 0 or self.user_factors is None:
            d = self.item_factors.shape[1] if self.item_factors is not None else 32
            return np.zeros(d, dtype=np.float32)
        return np.asarray(self.user_factors[row], dtype=np.float32)

    def _user_sbert_vec(self, hist_rows: list[int] | None,
                        emb: np.ndarray | None = None) -> np.ndarray | None:
        mat = self.sbert_emb if emb is None else emb
        if mat is None or not hist_rows:
            return None
        vecs = np.asarray(mat[hist_rows], dtype=np.float32)
        v = vecs.mean(axis=0)
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def cf_topk(self, user_idx: int, k: int = 200, exclude: set[int] | None = None):
        row = self._user_row_idx(user_idx)
        if row < 0 or self.user_factors is None or self.item_factors is None:
            pop = self.popular_items if self.popular_items else list(range(min(k, len(self.item_factors) if self.item_factors is not None else 0)))
            out = [i for i in pop if i not in (exclude or set())][:k]
            return out, [0.0] * len(out)
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
        elif model in ("light_gated", "deep_early", "graph_light_gated", "two_tower", "transformer"):
            raw_recs = self._deep_hybrid_recs(user_idx, hist_rows, cf_items, cf_scores_list,
                                              sb_items, sb_scores_list, model, k, alpha)
        elif model == "ensemble":
            raw_recs = self._ensemble_recs(user_idx, hist_rows, cf_items, sb_items, k, alpha)
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
        scores = self._deep_scores(model, user_idx, hist_rows, cands_arr, alpha)
        if scores is None:
            return []
        order = np.argsort(-scores)
        return [int(cands_arr[i]) for i in order][:k]

    @staticmethod
    def _mm(x: np.ndarray) -> np.ndarray:
        """Min-max normalise one candidate list (same transform used in eval)."""
        x = np.asarray(x, dtype=np.float32)
        if x.size == 0:
            return x
        lo, hi = float(x.min()), float(x.max())
        r = hi - lo
        return (x - lo) / r if r > 1e-8 else np.zeros_like(x)

    def _cf_mats(self, name: str):
        """Lazily mmap the ALS/BPR factor pair used by the ensemble."""
        cache = self.als_factors if name == "als" else self.bpr_factors
        if cache is not None:
            return cache
        try:
            uf = np.load(FACTORS_DIR / f"{name}_user_factors_f16.npy", mmap_mode="r")
            itf = np.load(FACTORS_DIR / f"{name}_item_factors_f16.npy", mmap_mode="r")
            cache = (uf, itf)
        except Exception:
            cache = (None, None)
        if name == "als":
            self.als_factors = cache
        else:
            self.bpr_factors = cache
        return cache

    def _deep_scores(self, model: str, user_idx: int, hist_rows, cands_arr: np.ndarray,
                     alpha: float) -> np.ndarray | None:
        """Raw candidate scores for one deep hybrid head (None when unavailable)."""
        graph_models = ("graph_light_gated", "two_tower", "transformer")
        sb_emb = self.sbert_emb_768 if model in graph_models else self.sbert_emb
        if model in graph_models and sb_emb is None:
            self.head_status[model] = "unavailable: sbert_item_embeddings_768_f16.npy not found"
            return None

        if model in graph_models:
            gu = self.graph_user_factors
            gi = self.graph_item_factors
            u_row = self._user_row_idx(user_idx)
            if gu is not None and gi is not None and u_row >= 0:
                u_cf = np.asarray(gu[u_row], dtype=np.float32).reshape(-1)
                i_cf = np.asarray(gi[cands_arr], dtype=np.float32)
            else:
                u_cf = self._user_cf_vec(user_idx)
                i_cf = np.asarray(self.item_factors[cands_arr], dtype=np.float32)
        else:
            u_cf = self._user_cf_vec(user_idx)
            i_cf = np.asarray(self.item_factors[cands_arr], dtype=np.float32)

        cf_raw = i_cf @ u_cf

        u_sb = self._user_sbert_vec(hist_rows, emb=sb_emb)
        if u_sb is not None and sb_emb is not None:
            i_sb = np.asarray(sb_emb[cands_arr], dtype=np.float32)
            sb_raw = i_sb @ u_sb
        else:
            sdim = int(sb_emb.shape[1]) if sb_emb is not None and hasattr(sb_emb, 'shape') else SBERT_DIM
            i_sb = np.zeros((len(cands_arr), sdim), dtype=np.float32)
            sb_raw = np.zeros(len(cands_arr), dtype=np.float32)

        if model == "light_gated" and self.gate_model is not None and torch is not None:
            with torch.no_grad():
                bu = torch.tensor(np.repeat(u_cf[None, :], len(cands_arr), axis=0), dtype=torch.float32)
                bi = torch.tensor(i_cf, dtype=torch.float32)
                bs = torch.tensor(i_sb, dtype=torch.float32)
                bcf = torch.tensor(cf_raw, dtype=torch.float32)
                bsb = torch.tensor(sb_raw, dtype=torch.float32)
                fused, _ = self.gate_model(bu, bi, bs, bcf, bsb)
                return fused.numpy()
        if model == "deep_early" and self.deep_model is not None and torch is not None:
            with torch.no_grad():
                feats = np.concatenate([
                    np.repeat(u_cf[None, :], len(cands_arr), axis=0), i_cf, i_sb
                ], axis=1)
                return self.deep_model(torch.tensor(feats, dtype=torch.float32)).numpy()
        if model == "graph_light_gated" and self.graph_gate_model is not None and torch is not None:
            with torch.no_grad():
                bu = torch.tensor(np.repeat(u_cf[None, :], len(cands_arr), axis=0), dtype=torch.float32)
                bi = torch.tensor(i_cf, dtype=torch.float32)
                bs = torch.tensor(i_sb, dtype=torch.float32)
                bcf = torch.tensor(cf_raw, dtype=torch.float32)
                bsb = torch.tensor(sb_raw, dtype=torch.float32)
                fused, _ = self.graph_gate_model(bu, bi, bs, bcf, bsb)
                return fused.numpy()
        if model == "two_tower" and self.user_tower is not None and self.item_tower is not None and torch is not None:
            with torch.no_grad():
                u_emb = self.user_tower(torch.tensor(u_cf[None, :], dtype=torch.float32)).numpy()
                i_in = np.concatenate([i_cf, i_sb], axis=1)
                i_emb = self.item_tower(torch.tensor(i_in, dtype=torch.float32)).numpy()
                return (i_emb @ u_emb[0]).astype(np.float32)
        if model == "transformer" and self.transformer_model is not None and torch is not None:
            with torch.no_grad():
                feats = np.concatenate([
                    np.repeat(u_cf[None, :], len(cands_arr), axis=0), i_cf, i_sb
                ], axis=1)
                return self.transformer_model(
                    torch.tensor(feats[None, :, :], dtype=torch.float32)).numpy().reshape(-1)
        # fall back to the classic linear fusion of this head's inputs
        return lf.classic_late_fusion(cf_raw, sb_raw, alpha=alpha, beta=1 - alpha)

    def _scalar_scores(self, name: str, user_idx: int, hist_rows, cands_arr: np.ndarray,
                       alpha: float) -> np.ndarray | None:
        """Scores for non-neural base models used by the ensemble."""
        if name == "sbert":
            if self.sbert_emb is None:
                return None
            u = self._user_sbert_vec(hist_rows, emb=self.sbert_emb)
            if u is None:
                return None
            return np.asarray(self.sbert_emb[cands_arr], dtype=np.float32) @ u
        if name in ("als", "bpr"):
            uf, itf = self._cf_mats(name)
            row = self._user_row_idx(user_idx)
            if uf is None or itf is None or row < 0 or row >= len(uf):
                return None
            return np.asarray(itf[cands_arr], dtype=np.float32) @ np.asarray(uf[row], dtype=np.float32)
        return None

    def _ensemble_recs(self, user_idx, hist_rows, cf_items, sb_items, k, alpha):
        """Weighted blend of the trained models (weights from the eval kernel)."""
        cands = list(dict.fromkeys(list(cf_items) + list(sb_items)))
        if not cands:
            return []
        cands_arr = np.asarray(cands, dtype=np.int64)

        weights = dict(self.ensemble_weights or {})
        if not weights:
            # sensible default until the eval kernel writes ensemble_weights.json
            weights = {n: 1.0 for n in ("late_fusion", "light_gated", "deep_early",
                                        "graph_light_gated", "two_tower", "transformer")}

        parts: list[tuple[float, np.ndarray]] = []
        used: list[str] = []
        for name, w in weights.items():
            if w <= 0:
                continue
            if name == "late_fusion":
                s = self._scalar_scores("als", user_idx, hist_rows, cands_arr, alpha)
                t = self._scalar_scores("sbert", user_idx, hist_rows, cands_arr, alpha)
                if s is None or t is None:
                    continue
                s = lf.classic_late_fusion(s, t, alpha=self.ensemble_late_alpha,
                                           beta=1 - self.ensemble_late_alpha)
            else:
                s = self._scalar_scores(name, user_idx, hist_rows, cands_arr, alpha)
                if s is None:
                    s = self._deep_scores(name, user_idx, hist_rows, cands_arr, alpha)
            if s is None:
                continue
            parts.append((float(w), self._mm(s)))
            used.append(name)

        if not parts:
            return []
        total = sum(w for w, _ in parts)
        fused = np.zeros(len(cands_arr), dtype=np.float32)
        for w, s in parts:
            fused += (w / total) * s
        order = np.argsort(-fused)
        self.head_status["ensemble"] = f"blend({len(used)}): " + ",".join(used)
        return [int(cands_arr[i]) for i in order][:k]
