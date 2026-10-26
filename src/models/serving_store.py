"""
Serving store cho Hybrid Recommender.

Mục tiêu: laptop 8GB RAM không còn bị treo khi mở app.

Bản cũ đọc toàn bộ 3.7M item vectors (1.2GB Parquet) + 45.6M interactions
(260MB Parquet) vào pandas mỗi lần khởi động, sau đó còn chuyển từng vector
TF-IDF 200.000 chiều thành dense array bằng vòng lặp Python. Cách đó ngốn
nhiều GB RAM và gần như không thể chạy trên máy 8GB.

Module này tạo một "serving cache" gọn nhẹ, chỉ build MỘT LẦN:

    models/serving/
    ├── catalog_meta.sqlite      # asin, title, brand, category, price (tra cứu Top-K)
    ├── catalog_index.npz        # mảng số (item_idx, domain, popularity, norm)
    ├── tfidf_indptr.npy         # CSR row pointers
    ├── tfidf_indices.i32        # CSR column indices (memmap)
    ├── tfidf_data.f32           # CSR values (memmap)
    ├── users_sorted.s20         # reviewerID đã sort (memmap, tra cứu log N)
    ├── users_idx.i32            # user_idx tương ứng
    ├── seen_indptr.npy          # CSR row pointers cho item đã tương tác
    ├── seen_items.i32           # item_idx đã tương tác (memmap)
    └── build_info.json

Khi chạy app, chỉ các mảng nhỏ được nạp vào RAM. TF-IDF và danh sách item
đã xem được memory-map, nên chỉ những trang thực sự dùng mới bị đọc.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

BASE_DIR = Path(__file__).resolve().parents[2]

ProgressCallback = Callable[[float, str], None]

_META_SCHEMA = pa.schema(
    [
        ("asin", pa.string()),
        ("title", pa.string()),
        ("brand", pa.string()),
        ("category", pa.string()),
        ("price", pa.string()),
    ]
)


def _parquet_files(directory: Path) -> list[str]:
    files = sorted(str(p) for p in Path(directory).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"Không tìm thấy file .parquet trong {directory}")
    return files


def _report(progress: Optional[ProgressCallback], fraction: float, message: str) -> None:
    fraction = float(min(1.0, max(0.0, fraction)))
    if progress is not None:
        progress(fraction, message)
    else:
        print(f"[{fraction * 100:5.1f}%] {message}", flush=True)


def _encode_fixed(values: Sequence[str], width: int = 20) -> np.ndarray:
    """Chuyển list[str] sang numpy fixed-width bytes để binary-search/memmap."""
    return np.asarray(
        [v.encode("utf-8") if v is not None else b"" for v in values],
        dtype=f"S{width}",
    )


class ServingStore:
    """Truy cập catalog + TF-IDF + lịch sử tương tác theo cách tiết kiệm RAM."""

    INDEX_VERSION = 1

    def __init__(
        self,
        serving_dir: str | Path | None = None,
        als_dir: str | Path | None = None,
        cb_dir: str | Path | None = None,
    ) -> None:
        self.serving_dir = Path(serving_dir) if serving_dir else BASE_DIR / "models" / "serving"
        self.als_dir = Path(als_dir) if als_dir else BASE_DIR / "models" / "als"
        self.cb_dir = Path(cb_dir) if cb_dir else BASE_DIR / "models" / "content_based"

        self.meta_path = self.serving_dir / "catalog_meta.sqlite"
        self.meta_parquet_path = self.serving_dir / "catalog_meta.parquet"
        self.index_path = self.serving_dir / "catalog_index.npz"
        self.indptr_path = self.serving_dir / "tfidf_indptr.npy"
        self.indices_path = self.serving_dir / "tfidf_indices.i32"
        self.data_path = self.serving_dir / "tfidf_data.f32"
        self.users_sorted_path = self.serving_dir / "users_sorted.s20"
        self.users_idx_path = self.serving_dir / "users_idx.i32"
        self.seen_indptr_path = self.serving_dir / "seen_indptr.npy"
        self.seen_items_path = self.serving_dir / "seen_items.i32"
        self.info_path = self.serving_dir / "build_info.json"

        self._loaded = False
        self._user_cache: dict[str, Optional[int]] = {}
        self._load_seconds = 0.0

        # Dữ liệu nhỏ, nạp vào RAM
        self._item_idx_als: Optional[np.ndarray] = None
        self._domain_code: Optional[np.ndarray] = None
        self._item_norm: Optional[np.ndarray] = None
        self._pop_score: Optional[np.ndarray] = None
        self._n_ratings: Optional[np.ndarray] = None
        self._avg_rating: Optional[np.ndarray] = None
        self._row_id_by_item: Optional[np.ndarray] = None
        self._vocab_size = 0

        # Dữ liệu lớn, memory-map
        self._indptr: Optional[np.ndarray] = None
        self._indices: Optional[np.ndarray] = None
        self._data: Optional[np.ndarray] = None
        self._users_sorted: Optional[np.ndarray] = None
        self._users_idx: Optional[np.ndarray] = None
        self._seen_indptr: Optional[np.ndarray] = None
        self._seen_items: Optional[np.ndarray] = None

    # ─────────────────────────────────────────────────────────────────────
    # Trạng thái cache
    # ─────────────────────────────────────────────────────────────────────

    def is_built(self) -> bool:
        required = [
            self.index_path,
            self.indptr_path,
            self.indices_path,
            self.data_path,
            self.users_sorted_path,
            self.users_idx_path,
            self.seen_indptr_path,
            self.seen_items_path,
        ]
        has_meta = self.meta_path.exists() or self.meta_parquet_path.exists()
        return has_meta and all(p.exists() and p.stat().st_size > 0 for p in required)

    def build_info(self) -> dict:
        if not self.info_path.exists():
            return {}
        try:
            return json.loads(self.info_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _parquet_to_sqlite(
        self,
        parquet_path: Path,
        progress: Optional[ProgressCallback] = None,
        start: float = 0.0,
        end: float = 1.0,
    ) -> None:
        """
        Chuyển metadata sang SQLite để tra cứu Top-K mà không nạp 1.4GB vào RAM.

        Parquet 191MB khi đọc toàn bộ vào Arrow sẽ nở ra ~1.4GB vì title/brand
        là chuỗi dài. SQLite chỉ đọc đúng vài dòng cần hiển thị.
        """
        tmp_path = self.meta_path.with_suffix(".sqlite.tmp")
        tmp_path.unlink(missing_ok=True)

        conn = sqlite3.connect(tmp_path)
        try:
            conn.execute("PRAGMA journal_mode=OFF")
            conn.execute("PRAGMA synchronous=OFF")
            conn.execute(
                "CREATE TABLE catalog ("
                "row_id INTEGER PRIMARY KEY, "
                "asin TEXT, title TEXT, brand TEXT, category TEXT, price TEXT)"
            )

            parquet_file = pq.ParquetFile(parquet_path)
            total_rows = parquet_file.metadata.num_rows
            row_id = 0
            batch_size = 50_000
            for batch in parquet_file.iter_batches(
                batch_size=batch_size,
                columns=["asin", "title", "brand", "category", "price"],
            ):
                data = batch.to_pydict()
                size = batch.num_rows
                rows = [
                    (
                        row_id + i,
                        data["asin"][i],
                        data["title"][i],
                        data["brand"][i],
                        data["category"][i],
                        data["price"][i],
                    )
                    for i in range(size)
                ]
                conn.executemany(
                    "INSERT INTO catalog VALUES (?, ?, ?, ?, ?, ?)", rows
                )
                conn.commit()
                row_id += size
                if progress is not None and total_rows:
                    fraction = start + (end - start) * row_id / total_rows
                    _report(
                        progress,
                        fraction,
                        f"Đang tạo metadata SQLite {row_id:,}/{total_rows:,}...",
                    )
        finally:
            conn.close()

        os.replace(tmp_path, self.meta_path)

    # ─────────────────────────────────────────────────────────────────────
    # Build cache
    # ─────────────────────────────────────────────────────────────────────

    def build(
        self,
        progress: Optional[ProgressCallback] = None,
        force: bool = False,
    ) -> dict:
        """Tạo serving cache từ models/als + models/content_based."""
        if self.is_built() and not force:
            _report(progress, 1.0, "Serving cache đã tồn tại — bỏ qua build.")
            return self.build_info()

        self.serving_dir.mkdir(parents=True, exist_ok=True)
        t_start = time.time()

        # Dọn file tạm còn sót lại từ lần build lỗi trước
        for path in self.serving_dir.glob("*.tmp"):
            path.unlink(missing_ok=True)

        # ── 1. Đọc mapping asin → ALS item_idx ────────────────────────────
        _report(progress, 0.01, "Đang đọc item_mapping (asin → item_idx)...")
        mapping = pq.read_table(
            self.als_dir / "item_mapping", columns=["asin", "item_idx"]
        ).to_pandas()
        if mapping["asin"].duplicated().any():
            mapping = mapping.drop_duplicates("asin", keep="first")

        asin_keys = _encode_fixed(mapping["asin"].tolist(), width=10)
        als_values = mapping["item_idx"].to_numpy(dtype=np.int64)
        order = np.argsort(asin_keys, kind="stable")
        asin_keys = asin_keys[order]
        als_values = als_values[order]
        del mapping, order

        n_items_als = int(als_values.max()) + 1 if len(als_values) else 0

        # ── 2. Item vectors + metadata ────────────────────────────────────
        _report(progress, 0.03, "Đang build TF-IDF CSR + catalog metadata...")
        vector_files = _parquet_files(self.cb_dir / "item_vectors")
        total_files = len(vector_files)

        data_tmp = self.data_path.with_suffix(".f32.tmp")
        indices_tmp = self.indices_path.with_suffix(".i32.tmp")
        meta_tmp = self.serving_dir / "catalog_meta.parquet.tmp"

        indptr_parts: list[np.ndarray] = [np.zeros(1, dtype=np.int64)]
        item_idx_parts: list[np.ndarray] = []
        domain_parts: list[np.ndarray] = []
        norm_parts: list[np.ndarray] = []
        offset = 0
        vocab_size = 0
        total_rows = 0
        total_nnz = 0

        writer = pq.ParquetWriter(meta_tmp, _META_SCHEMA, compression="zstd")
        try:
            with open(data_tmp, "wb") as data_fh, open(indices_tmp, "wb") as indices_fh:
                for file_no, path in enumerate(vector_files, start=1):
                    table = pq.read_table(
                        path,
                        columns=[
                            "item_idx",
                            "asin",
                            "title",
                            "brand",
                            "category",
                            "price",
                            "tfidf",
                        ],
                    )
                    n_rows = table.num_rows
                    tfidf = table["tfidf"].combine_chunks()
                    indices_list = tfidf.field("indices")
                    values_list = tfidf.field("values")

                    lengths = pc.fill_null(
                        pc.list_value_length(indices_list), 0
                    ).to_numpy(zero_copy_only=False)
                    flat_indices = pc.list_flatten(indices_list).to_numpy(
                        zero_copy_only=False
                    )
                    flat_values = pc.list_flatten(values_list).to_numpy(
                        zero_copy_only=False
                    )

                    if len(flat_indices) != len(flat_values):
                        raise ValueError(
                            f"TF-IDF indices/values lệch kích thước ở {path}"
                        )

                    flat_indices = flat_indices.astype(np.int32, copy=False)
                    flat_values = flat_values.astype(np.float32, copy=False)
                    data_fh.write(flat_values.tobytes(order="C"))
                    indices_fh.write(flat_indices.tobytes(order="C"))

                    per_file = offset + np.cumsum(lengths, dtype=np.int64)
                    indptr_parts.append(per_file)
                    offset = int(per_file[-1])

                    # L2 norm cho từng item (dùng cho cosine similarity)
                    if len(flat_values):
                        row_ids = np.repeat(
                            np.arange(n_rows, dtype=np.int64), lengths
                        )
                        sumsq = np.bincount(
                            row_ids,
                            weights=flat_values.astype(np.float64) ** 2,
                            minlength=n_rows,
                        )
                        norms = np.sqrt(sumsq).astype(np.float32)
                    else:
                        norms = np.zeros(n_rows, dtype=np.float32)
                    norm_parts.append(norms)

                    # Map asin → ALS item_idx
                    batch_asin = _encode_fixed(
                        table["asin"].to_numpy(zero_copy_only=False).tolist(),
                        width=10,
                    )
                    positions = np.searchsorted(asin_keys, batch_asin)
                    positions = np.minimum(positions, max(len(asin_keys) - 1, 0))
                    found = (
                        (len(asin_keys) > 0) & (asin_keys[positions] == batch_asin)
                        if len(asin_keys)
                        else np.zeros(len(batch_asin), dtype=bool)
                    )
                    mapped = np.full(n_rows, -1, dtype=np.int32)
                    mapped[found] = als_values[positions[found]].astype(np.int32)
                    item_idx_parts.append(mapped)

                    # Domain code: 0=khác, 1=books, 2=electronics
                    categories = table["category"].to_numpy(zero_copy_only=False)
                    codes = np.zeros(n_rows, dtype=np.int8)
                    for i, category in enumerate(categories):
                        if not category:
                            continue
                        lowered = str(category).lower()
                        if "books" in lowered:
                            codes[i] = 1
                        elif "electronics" in lowered:
                            codes[i] = 2
                    domain_parts.append(codes)

                    meta_table = table.select(
                        ["asin", "title", "brand", "category", "price"]
                    ).cast(_META_SCHEMA)
                    writer.write_table(meta_table)

                    if vocab_size == 0 and n_rows:
                        vocab_size = int(
                            tfidf.field("size")[0].as_py() or 200_000
                        )

                    total_rows += n_rows
                    total_nnz += len(flat_indices)
                    if file_no % 4 == 0 or file_no == total_files:
                        frac = 0.03 + 0.52 * file_no / total_files
                        _report(
                            progress,
                            frac,
                            f"Đang xử lý item vectors {file_no}/{total_files} "
                            f"({total_rows:,} items)...",
                        )
        finally:
            writer.close()

        # Hoàn tất các file tạm
        os.replace(data_tmp, self.data_path)
        os.replace(indices_tmp, self.indices_path)
        os.replace(meta_tmp, self.meta_parquet_path)

        tfidf_indptr = np.concatenate(indptr_parts)
        np.save(self.indptr_path, tfidf_indptr)

        item_idx_als = np.concatenate(item_idx_parts)
        domain_code = np.concatenate(domain_parts)
        item_norm = np.concatenate(norm_parts)
        del indptr_parts, item_idx_parts, domain_parts, norm_parts

        row_id_by_item = np.full(n_items_als, -1, dtype=np.int32)
        valid = item_idx_als >= 0
        row_id_by_item[item_idx_als[valid]] = np.flatnonzero(valid).astype(np.int32)

        del asin_keys, als_values

        # ── 3. Tra cứu reviewerID đã sort (binary search trên memmap) ─────
        _report(progress, 0.58, "Đang build chỉ mục người dùng...")
        users_tmp = self.users_sorted_path.with_suffix(".s20.tmp")
        users_idx_tmp = self.users_idx_path.with_suffix(".i32.tmp")
        user_table = pq.read_table(
            self.als_dir / "user_mapping", columns=["reviewerID", "user_idx"]
        )
        user_ids = _encode_fixed(
            user_table["reviewerID"].to_numpy(zero_copy_only=False).tolist(),
            width=20,
        )
        user_idx = user_table["user_idx"].to_numpy(zero_copy_only=False).astype(np.int32)
        del user_table
        user_order = np.argsort(user_ids, kind="stable")
        user_ids = user_ids[user_order]
        user_idx = user_idx[user_order]
        user_ids.tofile(users_tmp)
        user_idx.tofile(users_idx_tmp)
        os.replace(users_tmp, self.users_sorted_path)
        os.replace(users_idx_tmp, self.users_idx_path)
        n_users = len(user_idx)
        del user_ids, user_idx, user_order

        # ── 4. Popularity + seen items từ train interactions ──────────────
        _report(progress, 0.65, "Đang đọc train interactions (lượt 1/2)...")
        train_files = _parquet_files(self.als_dir / "train")
        item_counts = np.zeros(n_items_als, dtype=np.int64)
        item_rating_sum = np.zeros(n_items_als, dtype=np.float64)
        user_counts = np.zeros(n_users, dtype=np.int64)

        for file_no, path in enumerate(train_files, start=1):
            table = pq.read_table(
                path, columns=["user_idx", "item_idx", "rating"]
            )
            u = table["user_idx"].to_numpy(zero_copy_only=False).astype(np.int64)
            i = table["item_idx"].to_numpy(zero_copy_only=False).astype(np.int64)
            r = table["rating"].to_numpy(zero_copy_only=False).astype(np.float64)
            item_counts += np.bincount(i, minlength=n_items_als)[:n_items_als]
            item_rating_sum += np.bincount(
                i, weights=r, minlength=n_items_als
            )[:n_items_als]
            user_counts += np.bincount(u, minlength=n_users)[:n_users]
            del table, u, i, r
            if file_no % 4 == 0 or file_no == len(train_files):
                _report(
                    progress,
                    0.65 + 0.10 * file_no / len(train_files),
                    f"Popularity {file_no}/{len(train_files)}...",
                )

        # ── 5. Build seen-items CSR (user_idx → item_idx) ─────────────────
        _report(progress, 0.76, "Đang build chỉ mục item đã tương tác...")
        seen_indptr = np.zeros(n_users + 1, dtype=np.int64)
        np.cumsum(user_counts, out=seen_indptr[1:])
        total_seen = int(seen_indptr[-1])
        seen_tmp = self.seen_items_path.with_suffix(".i32.tmp")
        seen = np.memmap(seen_tmp, dtype=np.int32, mode="w+", shape=(total_seen,))
        cursor = seen_indptr[:-1].astype(np.int32)

        for file_no, path in enumerate(train_files, start=1):
            table = pq.read_table(path, columns=["user_idx", "item_idx"])
            u = table["user_idx"].to_numpy(zero_copy_only=False).astype(np.int64)
            i = table["item_idx"].to_numpy(zero_copy_only=False).astype(np.int32)
            del table
            n = len(u)
            order = np.argsort(u, kind="stable")
            us = u[order]
            items = i[order]
            del u, i, order

            change = np.empty(n, dtype=bool)
            change[0] = True
            np.not_equal(us[1:], us[:-1], out=change[1:])
            group_starts = np.flatnonzero(change).astype(np.int64)
            group_ids = np.repeat(
                np.arange(len(group_starts), dtype=np.int64),
                np.diff(np.append(group_starts, n)),
            )
            positions = cursor[us] + (
                np.arange(n, dtype=np.int64) - group_starts[group_ids]
            )
            seen[positions] = items

            unique_users, counts = np.unique(us, return_counts=True)
            cursor[unique_users] += counts.astype(np.int32)
            del us, items, change, group_starts, group_ids, positions, unique_users, counts
            if file_no % 4 == 0 or file_no == len(train_files):
                _report(
                    progress,
                    0.76 + 0.20 * file_no / len(train_files),
                    f"Seen items {file_no}/{len(train_files)}...",
                )

        seen.flush()
        del seen, cursor
        os.replace(seen_tmp, self.seen_items_path)
        np.save(self.seen_indptr_path, seen_indptr)

        # ── 6. Popularity theo từng catalog row ───────────────────────────
        _report(progress, 0.96, "Đang hoàn tất chỉ mục...")
        mapped_mask = item_idx_als >= 0
        safe_item_idx = np.where(mapped_mask, item_idx_als, 0)

        n_ratings = np.zeros(total_rows, dtype=np.int32)
        avg_rating = np.zeros(total_rows, dtype=np.float32)
        pop_score = np.zeros(total_rows, dtype=np.float32)

        n_ratings[mapped_mask] = item_counts[safe_item_idx[mapped_mask]].astype(
            np.int32
        )
        rating_sum = np.zeros(total_rows, dtype=np.float64)
        rating_sum[mapped_mask] = item_rating_sum[safe_item_idx[mapped_mask]]
        has_rating = n_ratings > 0
        avg_rating[has_rating] = (
            rating_sum[has_rating] / n_ratings[has_rating]
        ).astype(np.float32)
        if has_rating.any():
            max_count = float(n_ratings.max())
            pop_score[has_rating] = (
                0.7
                * np.log1p(n_ratings[has_rating].astype(np.float32))
                / np.log1p(max_count)
                + 0.3 * (avg_rating[has_rating] - 1.0) / 4.0
            )

        del rating_sum, item_counts, item_rating_sum, user_counts, safe_item_idx

        np.savez(
            self.index_path,
            item_idx_als=item_idx_als.astype(np.int32, copy=False),
            domain_code=domain_code.astype(np.int8, copy=False),
            item_norm=item_norm.astype(np.float32, copy=False),
            pop_score=pop_score,
            n_ratings=n_ratings,
            avg_rating=avg_rating,
            row_id_by_item=row_id_by_item,
            vocab_size=np.asarray([vocab_size], dtype=np.int32),
        )

        # Metadata: Parquet → SQLite (tra cứu Top-K, không nạp cả GB vào RAM)
        if self.meta_parquet_path.exists():
            self._parquet_to_sqlite(
                self.meta_parquet_path,
                progress=progress,
                start=0.96,
                end=0.995,
            )
            self.meta_parquet_path.unlink(missing_ok=True)

        info = {
            "version": self.INDEX_VERSION,
            "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_s": round(time.time() - t_start, 1),
            "catalog_size": total_rows,
            "vocab_size": vocab_size,
            "nnz": total_nnz,
            "num_users": n_users,
            "num_seen": total_seen,
            "num_items_als": n_items_als,
        }
        self.info_path.write_text(
            json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        _report(
            progress,
            1.0,
            f"Xong sau {info['elapsed_s']}s · {total_rows:,} sản phẩm · "
            f"{total_nnz:,} giá trị TF-IDF",
        )
        self._loaded = False
        return info

    # ─────────────────────────────────────────────────────────────────────
    # Load cache
    # ─────────────────────────────────────────────────────────────────────

    def load(self) -> "ServingStore":
        if self._loaded:
            return self
        if not self.is_built():
            raise FileNotFoundError(
                "Chưa có serving cache. Chạy:\n"
                "    python scripts/build_serving_cache.py\n"
                "hoặc bấm nút 'Xây dựng chỉ mục' trong app."
            )

        # Cache cũ có catalog_meta.parquet → chuyển sang SQLite một lần.
        if not self.meta_path.exists() and self.meta_parquet_path.exists():
            self._parquet_to_sqlite(self.meta_parquet_path)
            self.meta_parquet_path.unlink(missing_ok=True)

        t_start = time.time()
        index = np.load(self.index_path, allow_pickle=False)
        self._item_idx_als = index["item_idx_als"]
        self._domain_code = index["domain_code"]
        self._item_norm = index["item_norm"]
        self._pop_score = index["pop_score"]
        self._n_ratings = index["n_ratings"]
        self._avg_rating = index["avg_rating"]
        self._row_id_by_item = index["row_id_by_item"]
        self._vocab_size = int(index["vocab_size"][0])

        self._indptr = np.load(self.indptr_path)
        self._indices = np.memmap(
            self.indices_path, dtype=np.int32, mode="r"
        )
        self._data = np.memmap(self.data_path, dtype=np.float32, mode="r")

        self._users_sorted = np.memmap(
            self.users_sorted_path, dtype="S20", mode="r"
        )
        self._users_idx = np.memmap(
            self.users_idx_path, dtype=np.int32, mode="r"
        )
        self._seen_indptr = np.memmap(
            self.seen_indptr_path, dtype=np.int64, mode="r"
        )
        self._seen_items = np.memmap(
            self.seen_items_path, dtype=np.int32, mode="r"
        )

        self._loaded = True
        self._load_seconds = time.time() - t_start
        return self

    # ─────────────────────────────────────────────────────────────────────
    # Truy vấn
    # ─────────────────────────────────────────────────────────────────────

    @property
    def catalog_size(self) -> int:
        return 0 if self._domain_code is None else len(self._domain_code)

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def load_seconds(self) -> float:
        return self._load_seconds

    @property
    def pop_score(self) -> np.ndarray:
        return self._pop_score if self._pop_score is not None else np.empty(0)

    @property
    def domain_code(self) -> np.ndarray:
        return self._domain_code if self._domain_code is not None else np.empty(0)

    @property
    def item_norm(self) -> np.ndarray:
        return self._item_norm if self._item_norm is not None else np.empty(0)

    @property
    def n_ratings(self) -> np.ndarray:
        return self._n_ratings if self._n_ratings is not None else np.empty(0)

    @property
    def avg_rating(self) -> np.ndarray:
        return self._avg_rating if self._avg_rating is not None else np.empty(0)

    def user_index(self, reviewer_id: str) -> Optional[int]:
        """Tra reviewerID → user_idx bằng binary search, không quét 226MB."""
        reviewer_id = (reviewer_id or "").strip()
        if not reviewer_id:
            return None
        if reviewer_id in self._user_cache:
            return self._user_cache[reviewer_id]

        result: Optional[int] = None
        encoded = reviewer_id.encode("utf-8")
        if self._users_sorted is not None and 0 < len(encoded) <= 20:
            key = np.asarray([encoded], dtype="S20")[0]
            position = int(np.searchsorted(self._users_sorted, key))
            if (
                position < len(self._users_sorted)
                and self._users_sorted[position] == key
            ):
                result = int(self._users_idx[position])
        else:
            # ID quá dài / không chuẩn: fallback quét Parquet (hiếm khi xảy ra).
            try:
                table = pq.read_table(
                    self.als_dir / "user_mapping",
                    columns=["user_idx"],
                    filters=[("reviewerID", "=", reviewer_id)],
                )
                if table.num_rows:
                    result = int(table["user_idx"][0].as_py())
            except Exception:
                result = None

        self._user_cache[reviewer_id] = result
        return result

    def seen_items(self, user_idx: int) -> np.ndarray:
        """item_idx (ALS) mà user đã tương tác trong train set."""
        if self._seen_indptr is None or self._seen_items is None:
            return np.empty(0, dtype=np.int32)
        if user_idx < 0 or user_idx + 1 >= len(self._seen_indptr):
            return np.empty(0, dtype=np.int32)
        start = int(self._seen_indptr[user_idx])
        end = int(self._seen_indptr[user_idx + 1])
        return np.asarray(self._seen_items[start:end])

    def seen_rows(self, user_idx: int) -> np.ndarray:
        """Catalog row id của các item user đã xem (đã lọc item không có vector)."""
        seen = self.seen_items(user_idx)
        if not len(seen) or self._row_id_by_item is None:
            return np.empty(0, dtype=np.int32)
        rows = self._row_id_by_item[seen]
        return rows[rows >= 0]

    def row_vector(self, row_id: int) -> tuple[np.ndarray, np.ndarray]:
        """Trả về (indices, values) của TF-IDF vector cho một catalog row."""
        if self._indptr is None or self._indices is None or self._data is None:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)
        start = int(self._indptr[row_id])
        end = int(self._indptr[row_id + 1])
        return self._indices[start:end], self._data[start:end]

    def meta_rows(self, rows: Sequence[int]) -> pd.DataFrame:
        row_ids = [int(r) for r in rows]
        if not row_ids:
            return pd.DataFrame(columns=_META_SCHEMA.names)

        placeholders = ",".join("?" for _ in row_ids)
        conn = sqlite3.connect(f"file:{self.meta_path}?mode=ro", uri=True)
        try:
            cursor = conn.execute(
                "SELECT row_id, asin, title, brand, category, price "
                f"FROM catalog WHERE row_id IN ({placeholders})",
                row_ids,
            )
            found = {row[0]: row[1:] for row in cursor.fetchall()}
        finally:
            conn.close()

        missing = (None, None, None, None, None)
        ordered = [found.get(row_id, missing) for row_id in row_ids]
        return pd.DataFrame(ordered, columns=_META_SCHEMA.names)

    # ─────────────────────────────────────────────────────────────────────
    # Thông tin
    # ─────────────────────────────────────────────────────────────────────

    def summary(self) -> dict:
        info = self.build_info()
        return {
            "catalog_size": self.catalog_size,
            "vocab_size": self._vocab_size,
            "load_seconds": round(self._load_seconds, 2),
            "build": info,
        }
