"""
Hybrid Recommender System — Popularity-CF + TF-IDF Content-Based
Đồ án tốt nghiệp: Xây dựng mô hình lai ghép cho hệ thống khuyến nghị sản phẩm Amazon

Phiên bản "laptop-friendly":
  - Không đọc 1.2GB item_vectors + 45.6M interactions vào pandas mỗi lần chạy.
  - Dùng serving cache (xem src/models/serving_store.py), TF-IDF được memory-map.
  - Chỉ chấm CB trên một candidate pool nhỏ (mặc định 500) thay vì toàn bộ 3.7M item.
  - Kết quả trả về giữ nguyên API cũ để app.py / main.py không phải đổi cách gọi.

  CF proxy  : Item popularity score từ train interactions
              score_cf = 0.7·log1p(n_ratings) + 0.3·(avg_rating−1)/4
  CB        : TF-IDF cosine similarity giữa user profile và item vectors
              score_cb = cosine(mean_tfidf(seen_items), item_tfidf)
  Fusion    : score_hybrid = α·norm(CF) + (1−α)·norm(CB)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from .serving_store import ServingStore

_DOMAIN_CODES = {
    "books": 1,
    "book": 1,
    "sách": 1,
    "sach": 1,
    "electronics": 2,
    "electronic": 2,
    "điện tử": 2,
    "dien tu": 2,
}


def _minmax(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    lo, hi = float(arr.min()), float(arr.max())
    return (arr - lo) / (hi - lo) if hi > lo else np.full_like(arr, 0.5)


def _clean(value: Any, default: str = "N/A") -> str:
    if value is None:
        return default
    try:
        if isinstance(value, float) and np.isnan(value):
            return default
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text if text and text.lower() != "nan" else default


class HybridRecommender:
    """
    Hybrid Recommender (Popularity-CF + TF-IDF CB).

    Sử dụng:
        rec = HybridRecommender()
        rec.load()          # < 1 giây nếu serving cache đã được build
        results = rec.recommend("A1IPEJDMQNXJYF", top_k=10, alpha=0.65)
    """

    def __init__(
        self,
        als_dir: str | None = None,
        cb_dir: str | None = None,
        serving_dir: str | None = None,
        candidate_pool: int = 500,
    ):
        self._store = ServingStore(
            serving_dir=serving_dir,
            als_dir=als_dir,
            cb_dir=cb_dir,
        )
        self.candidate_pool = max(1, int(candidate_pool))

        self._mode = "not_loaded"
        self._loaded = False
        self._load_info: dict = {}

    # ─────────────────────────────────────────────────────────────────────
    # Load
    # ─────────────────────────────────────────────────────────────────────

    def load(self) -> "HybridRecommender":
        """
        Nạp serving cache. Có thể raise FileNotFoundError nếu cache chưa build;
        app sẽ bắt lỗi này và hiển thị nút xây dựng cache.
        """
        if self._loaded:
            return self

        self._store.load()
        info = self._store.build_info()
        has_popularity = (
            self._store.pop_score.size > 0
            and bool(np.any(self._store.pop_score > 0))
        )
        self._mode = (
            "hybrid_pop_cb"
            if has_popularity
            else ("cb_only" if self._store.catalog_size else "unavailable")
        )
        self._load_info = {
            "num_users": int(info.get("num_users", 0)),
            "nnz": int(info.get("nnz", 0)),
            "cache_built_at": info.get("built_at", "—"),
            "cache_elapsed_s": info.get("elapsed_s", 0),
        }
        self._loaded = True
        return self

    # ─────────────────────────────────────────────────────────────────────
    # Recommend
    # ─────────────────────────────────────────────────────────────────────

    def recommend(
        self,
        user_raw_id: str,
        top_k: int = 10,
        domain: Optional[str] = None,
        alpha: float = 0.65,
        exclude_seen: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Trả về danh sách Top-K sản phẩm gợi ý lai ghép.

        Tham số:
            user_raw_id : reviewerID từ Amazon dataset.
            top_k       : Số sản phẩm trả về.
            domain      : "books" | "electronics" | None (tất cả).
            alpha       : Trọng số CF/Popularity (α). CB weight = 1−α.
            exclude_seen: Loại bỏ sản phẩm đã tương tác.
        """
        if not self._loaded:
            self.load()

        store = self._store
        n_items = store.catalog_size
        if n_items == 0:
            return []

        top_k = max(1, int(top_k))
        alpha = float(np.clip(alpha, 0.0, 1.0))

        user_idx = store.user_index(user_raw_id)

        # ── 1. Lọc domain + item đã xem trên mảng popularity ──────────────
        scores = np.array(store.pop_score, dtype=np.float32, copy=True)

        domain_key = (domain or "").strip().lower()
        if domain_key in ("all", "tất cả", "tat ca", ""):
            domain_key = ""
        domain_code = _DOMAIN_CODES.get(domain_key)
        if domain_code is not None:
            scores[store.domain_code != domain_code] = -np.inf

        seen_rows = (
            store.seen_rows(user_idx)
            if exclude_seen and user_idx is not None
            else np.empty(0, dtype=np.int32)
        )
        if len(seen_rows):
            scores[seen_rows] = -np.inf

        finite = np.isfinite(scores)
        n_valid = int(finite.sum())
        if n_valid == 0:
            return []

        # ── 2. Candidate pool theo popularity ────────────────────────────
        pool = min(
            max(self.candidate_pool, top_k * 30, top_k),
            n_valid,
        )
        if pool < n_valid:
            candidates = np.argpartition(-scores, pool - 1)[:pool]
        else:
            candidates = np.flatnonzero(finite)

        candidates = candidates[np.isfinite(scores[candidates])]
        # Sắp xếp candidate theo popularity giảm dần (giúp kết quả ổn định)
        candidates = candidates[np.argsort(scores[candidates])[::-1]]
        if candidates.size == 0:
            return []

        # ── 3. CF score ──────────────────────────────────────────────────
        cf_raw = scores[candidates].astype(np.float32)

        # ── 4. CB score: chỉ chấm trên candidate pool ────────────────────
        cb_raw = np.zeros(len(candidates), dtype=np.float32)
        user_profile: Optional[np.ndarray] = None
        profile_norm = 0.0

        if user_idx is not None and len(seen_rows):
            profile_rows = seen_rows[:50]
            dense = np.zeros(store.vocab_size, dtype=np.float32)
            used = 0
            for row_id in profile_rows:
                indices, values = store.row_vector(int(row_id))
                if len(indices):
                    dense[indices] += values
                    used += 1
            if used:
                dense /= used
                norm = float(np.linalg.norm(dense))
                if norm > 0:
                    user_profile = dense
                    profile_norm = norm

        if user_profile is not None:
            item_norms = store.item_norm
            for i, row_id in enumerate(candidates):
                indices, values = store.row_vector(int(row_id))
                if not len(indices):
                    continue
                denominator = float(item_norms[row_id]) * profile_norm
                if denominator > 0:
                    dot = float(np.dot(values, user_profile[indices]))
                    cb_raw[i] = dot / denominator

        # ── 5. Normalize & fuse ──────────────────────────────────────────
        cf_norm = _minmax(cf_raw)
        cb_norm = _minmax(cb_raw)
        hybrid = alpha * cf_norm + (1.0 - alpha) * cb_norm

        # ── 6. Top-K + metadata (chỉ đọc vài dòng) ───────────────────────
        top_positions = np.argsort(hybrid)[::-1][:top_k]
        top_rows = candidates[top_positions]
        meta = store.meta_rows(top_rows)

        results: List[Dict[str, Any]] = []
        for rank, position in enumerate(top_positions, start=1):
            row_id = int(top_rows[rank - 1])
            row = meta.iloc[rank - 1] if rank - 1 < len(meta) else {}
            cb_percent = int(cb_norm[position] * 100)

            n_ratings = int(store.n_ratings[row_id]) if store.n_ratings.size else 0
            avg_rating = (
                float(store.avg_rating[row_id]) if store.avg_rating.size else 0.0
            )

            if user_profile is None:
                reason = (
                    f"Phổ biến: {n_ratings:,} lượt đánh giá, "
                    f"avg={avg_rating:.2f}★ · Chưa có lịch sử để tính CB"
                )
            else:
                reason = (
                    f"Phổ biến: {n_ratings:,} lượt đánh giá, "
                    f"avg={avg_rating:.2f}★ · "
                    f"Nội dung khớp CB {cb_percent}%"
                )

            results.append(
                {
                    "rank": rank,
                    "asin": _clean(row.get("asin")),
                    "title": _clean(row.get("title"))[:80],
                    "brand": _clean(row.get("brand")),
                    "category": _clean(row.get("category")),
                    "price": _clean(row.get("price")),
                    "hybrid_score": round(float(hybrid[position]), 4),
                    "cf_score": round(float(cf_norm[position]), 4),
                    "cb_score": round(float(cb_norm[position]), 4),
                    "n_ratings": n_ratings,
                    "avg_rating": round(avg_rating, 2),
                    "explanation": reason,
                }
            )

        return results

    # ─────────────────────────────────────────────────────────────────────
    # Status
    # ─────────────────────────────────────────────────────────────────────

    def user_known(self, reviewer_id: str) -> bool:
        """True nếu reviewerID có trong user_mapping."""
        if not self._loaded:
            self.load()
        return self._store.user_index(reviewer_id) is not None

    def get_status(self) -> Dict[str, Any]:
        labels = {
            "hybrid_pop_cb": "🟢 Hybrid (Popularity-CF + TF-IDF CB)",
            "cb_only": "🟡 Content-Based only (TF-IDF)",
            "unavailable": "🔴 Không tải được — kiểm tra thư mục models/",
            "not_loaded": "⚪ Chưa load — gọi .load()",
        }
        return {
            "mode": self._mode,
            "mode_label": labels.get(self._mode, self._mode),
            "message": labels.get(self._mode, self._mode),
            "catalog_size": self._store.catalog_size if self._loaded else 0,
            "vocab_size": self._store.vocab_size if self._loaded else 0,
            "als_rank": 0,
            "load_seconds": (
                round(self._store.load_seconds, 2) if self._loaded else 0
            ),
            "candidate_pool": self.candidate_pool,
            **self._load_info,
        }
