
#!/usr/bin/env python
"""
Train implicit BPR and export user/item factors.

Memory-conscious CPU implementation.

Default configuration:
    RAM target: approximately 4 GB (not guaranteed)
    CPU threads: 4
    Factors: 8
    Iterations: 15
    Positive threshold: 4.0
    Parquet batch: 50,000

Pipeline:
    Parquet
       |
       v
    SQLite deduplication
       |
       v
    Optional iterative K-core
       |
       v
    Disk-backed CSR
       |
       v
    implicit BPR
       |
       v
    save_factors()

Usage:
    uv run python scripts/bpr.py

    uv run python scripts/bpr.py \
        --factors 8 \
        --iters 15 \
        --lr 0.05 \
        --reg 0.01 \
        --threads 4

    uv run python scripts/bpr.py \
        --min-user 3 \
        --min-item 5 \
        --kcore-rounds 2

IMPORTANT:
    - Original user_idx and item_idx are preserved.
    - No full pandas DataFrame is created.
    - SQLite is used for disk-backed deduplication.
    - CSR is backed by NumPy memmap files.
    - BPR model factors are still allocated in RAM.
    - K-core changes the training interaction set.
"""

from __future__ import annotations

# =========================================================
# 0. THREAD LIMITS
# Must be set before importing NumPy / SciPy.
# =========================================================

import os

for key in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[key] = "4"


# =========================================================
# 1. IMPORTS
# =========================================================

import argparse
import gc
import shutil
import sqlite3
import sys
import time

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from scipy.sparse import csr_matrix
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import (
    TRAIN_DIR,
    FACTORS_DIR,
    NUM_USERS,
    NUM_ITEMS,
)

from src.models.cf import save_factors
from implicit.bpr import BayesianPersonalizedRanking


# =========================================================
# 2. UTILITIES
# =========================================================

def log(message=""):
    print(message, flush=True)


def print_memory():
    """Display process peak RSS on Linux."""

    try:
        import resource

        peak = resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss

        log(
            f"[MEMORY] Peak RSS: "
            f"{peak / 1024:.1f} MB"
        )

    except Exception as exc:
        log(f"[MEMORY] Cannot read RSS: {exc}")


def get_workdir():
    path = Path(FACTORS_DIR) / ".bpr_work"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_parquet_files(directory):
    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(
            f"Directory not found: {directory}"
        )

    files = sorted(directory.rglob("*.parquet"))

    if not files:
        raise FileNotFoundError(
            f"No Parquet files found in {directory}"
        )

    return files


# =========================================================
# 3. SQLITE DATABASE
# =========================================================

def create_database(db_path):
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(str(db_path))

    conn.execute("PRAGMA journal_mode=OFF")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA temp_store=FILE")
    conn.execute("PRAGMA cache_size=-32768")
    conn.execute("PRAGMA mmap_size=0")

    conn.execute("""
        CREATE TABLE interactions (
            user_idx INTEGER NOT NULL,
            item_idx INTEGER NOT NULL,
            PRIMARY KEY (user_idx, item_idx)
        ) WITHOUT ROWID
    """)

    conn.commit()

    return conn


# =========================================================
# 4. LOAD PARQUET -> SQLITE
# =========================================================

def load_train(
    conn,
    train_dir,
    batch_size,
    positive_threshold,
):
    files = get_parquet_files(train_dir)

    log(f"\nFound {len(files)} Parquet files")

    total_rows = 0
    total_positive = 0

    insert_sql = """
        INSERT OR IGNORE INTO interactions
        (user_idx, item_idx)
        VALUES (?, ?)
    """

    for path in tqdm(files, desc="Reading train"):

        parquet_file = pq.ParquetFile(path)

        for batch in parquet_file.iter_batches(
            columns=[
                "user_idx",
                "item_idx",
                "rating",
            ],
            batch_size=batch_size,
        ):
            users = batch.column("user_idx").to_numpy()
            items = batch.column("item_idx").to_numpy()
            ratings = batch.column("rating").to_numpy()

            mask = (
                np.isfinite(ratings)
                & (ratings >= positive_threshold)
                & (users >= 0)
                & (users < NUM_USERS)
                & (items >= 0)
                & (items < NUM_ITEMS)
            )

            total_rows += len(users)

            positive_count = int(mask.sum())
            total_positive += positive_count

            if positive_count == 0:
                continue

            selected_users = users[mask]
            selected_items = items[mask]

            records = (
                (int(u), int(i))
                for u, i in zip(
                    selected_users,
                    selected_items,
                )
            )

            conn.executemany(
                insert_sql,
                records,
            )

            del selected_users
            del selected_items
            del records

        conn.commit()
        gc.collect()

    unique_pairs = conn.execute(
        "SELECT COUNT(*) FROM interactions"
    ).fetchone()[0]

    log("\nTrain summary")
    log(f"Rows scanned:       {total_rows:,}")
    log(f"Positive rows:      {total_positive:,}")
    log(f"Unique pairs:       {unique_pairs:,}")
    log(
        f"Duplicate pairs:    "
        f"{total_positive - unique_pairs:,}"
    )

    if unique_pairs == 0:
        raise ValueError("No positive interactions found.")

    print_memory()

    return int(unique_pairs)


# =========================================================
# 5. OPTIONAL K-CORE FILTERING
# =========================================================

def apply_kcore(
    conn,
    min_user,
    min_item,
    rounds,
):
    """
    Iteratively remove users/items below thresholds.

    IDs are preserved. No re-indexing is performed.

    rounds:
        Number of user/item filtering rounds.
        This is not guaranteed to be full convergence.
    """

    if min_user <= 1 and min_item <= 1:
        log("\nK-core disabled.")
        return

    log("\n" + "=" * 60)
    log("K-CORE FILTERING")
    log("=" * 60)

    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_interactions_item
        ON interactions(item_idx, user_idx)
    """)

    for round_id in range(1, rounds + 1):

        before = conn.execute(
            "SELECT COUNT(*) FROM interactions"
        ).fetchone()[0]

        log(f"\nK-core round {round_id}/{rounds}")
        log(f"Interactions before: {before:,}")

        # Remove users with too few positive interactions.
        conn.execute("DROP TABLE IF EXISTS bad_users")

        conn.execute("""
            CREATE TEMP TABLE bad_users AS
            SELECT user_idx
            FROM interactions
            GROUP BY user_idx
            HAVING COUNT(*) < ?
        """, (min_user,))

        bad_users = conn.execute(
            "SELECT COUNT(*) FROM bad_users"
        ).fetchone()[0]

        conn.execute("""
            DELETE FROM interactions
            WHERE user_idx IN (
                SELECT user_idx FROM bad_users
            )
        """)

        conn.execute("DROP TABLE bad_users")

        # Remove items with too few positive interactions.
        conn.execute("DROP TABLE IF EXISTS bad_items")

        conn.execute("""
            CREATE TEMP TABLE bad_items AS
            SELECT item_idx
            FROM interactions
            GROUP BY item_idx
            HAVING COUNT(*) < ?
        """, (min_item,))

        bad_items = conn.execute(
            "SELECT COUNT(*) FROM bad_items"
        ).fetchone()[0]

        conn.execute("""
            DELETE FROM interactions
            WHERE item_idx IN (
                SELECT item_idx FROM bad_items
            )
        """)

        conn.execute("DROP TABLE bad_items")
        conn.commit()

        after = conn.execute(
            "SELECT COUNT(*) FROM interactions"
        ).fetchone()[0]

        removed = before - after

        log(f"Removed users:       {bad_users:,}")
        log(f"Removed items:       {bad_items:,}")
        log(f"Interactions after:  {after:,}")
        log(f"Removed interactions: {removed:,}")

        if after == 0:
            raise ValueError(
                "K-core removed all interactions. "
                "Check min_user/min_item."
            )

        if removed == 0:
            log("No more interactions removed. Stopping.")
            break

    log("\nK-core completed.")

    print_memory()


# =========================================================
# 6. BUILD DISK-BACKED CSR
# =========================================================

def build_csr(conn, workdir, nnz):

    log("\n" + "=" * 60)
    log("BUILDING DISK-BACKED CSR")
    log("=" * 60)

    if NUM_ITEMS >= np.iinfo(np.int32).max:
        raise ValueError("NUM_ITEMS exceeds int32 capacity.")

    if nnz >= np.iinfo(np.int32).max:
        raise ValueError(
            "NNZ exceeds int32 capacity. "
            "Use int64 CSR indices."
        )

    indptr = np.lib.format.open_memmap(
        workdir / "indptr.npy",
        mode="w+",
        dtype=np.int32,
        shape=(NUM_USERS + 1,),
    )

    indices = np.lib.format.open_memmap(
        workdir / "indices.npy",
        mode="w+",
        dtype=np.int32,
        shape=(nnz,),
    )

    data = np.lib.format.open_memmap(
        workdir / "data.npy",
        mode="w+",
        dtype=np.float32,
        shape=(nnz,),
    )

    indptr[:] = 0

    # Count rows per user.
    log("Counting interactions per user...")

    cursor = conn.execute("""
        SELECT user_idx, COUNT(*)
        FROM interactions
        GROUP BY user_idx
    """)

    for user_idx, count in cursor:
        indptr[int(user_idx) + 1] = int(count)

    # Cumulative offsets.
    np.cumsum(
        indptr,
        out=indptr,
        dtype=np.int64,
    )

    if int(indptr[-1]) != nnz:
        raise RuntimeError(
            f"CSR count mismatch: {indptr[-1]} != {nnz}"
        )

    # Fill item indices.
    log("Filling item indices...")

    cursor = conn.execute("""
        SELECT user_idx, item_idx
        FROM interactions
        ORDER BY user_idx, item_idx
    """)

    position = 0
    fetch_size = 100_000

    while True:
        records = cursor.fetchmany(fetch_size)

        if not records:
            break

        count = len(records)

        indices[position:position + count] = np.fromiter(
            (row[1] for row in records),
            dtype=np.int32,
            count=count,
        )

        data[position:position + count] = 1.0

        position += count

        if position % 1_000_000 < fetch_size:
            log(f"CSR: {position:,}/{nnz:,}")

        del records

    if position != nnz:
        raise RuntimeError(
            f"CSR fill mismatch: {position} != {nnz}"
        )

    indptr.flush()
    indices.flush()
    data.flush()

    mat = csr_matrix(
        (data, indices, indptr),
        shape=(NUM_USERS, NUM_ITEMS),
        copy=False,
    )

    log("\nCSR completed.")
    log(f"Shape: {mat.shape}")
    log(f"NNZ: {mat.nnz:,}")
    log(f"Data dtype: {mat.data.dtype}")
    log(f"Indices dtype: {mat.indices.dtype}")
    log(f"Indptr dtype: {mat.indptr.dtype}")

    print_memory()

    return mat, (data, indices, indptr)


# =========================================================
# 7. TRAIN BPR
# =========================================================

def train_bpr(
    mat,
    factors,
    iterations,
    learning_rate,
    regularization,
    threads,
    seed,
):
    log("\n" + "=" * 60)
    log("TRAINING IMPLICIT BPR")
    log("=" * 60)

    model = BayesianPersonalizedRanking(
        factors=factors,
        learning_rate=learning_rate,
        regularization=regularization,
        iterations=iterations,
        num_threads=threads,
        random_state=seed,
    )

    embedding_mb = (
        (NUM_USERS + NUM_ITEMS)
        * factors
        * 4
        / (1024 ** 2)
    )

    log("\nConfiguration:")
    log(f"Factors:        {factors}")
    log(f"Iterations:     {iterations}")
    log(f"Learning rate:  {learning_rate}")
    log(f"Regularization: {regularization}")
    log(f"Threads:        {threads}")
    log(f"Seed:           {seed}")

    log(
        f"Estimated raw embedding memory: "
        f"{embedding_mb:.1f} MB"
    )

    start = time.time()

    model.fit(
        mat,
        show_progress=True,
    )

    elapsed = time.time() - start

    user_factors = model.user_factors
    item_factors = model.item_factors

    log("\nBPR training completed.")
    log(f"Training time: {elapsed / 60:.2f} minutes")
    log(f"User factors: {user_factors.shape}")
    log(f"Item factors: {item_factors.shape}")

    print_memory()

    return model, user_factors, item_factors, elapsed


# =========================================================
# 8. EXPORT FACTORS
# =========================================================

def export_factors(
    user_factors,
    item_factors,
    conn,
    args,
    nnz,
    elapsed,
):
    log("\nExporting BPR factors...")

    active_users = conn.execute("""
        SELECT COUNT(*)
        FROM (
            SELECT user_idx
            FROM interactions
            GROUP BY user_idx
        )
    """).fetchone()[0]

    active_items = conn.execute("""
        SELECT COUNT(DISTINCT item_idx)
        FROM interactions
    """).fetchone()[0]

    metadata = {
        "model": "implicit_bpr",
        "library": "implicit",
        "factors": args.factors,
        "iterations": args.iters,
        "learning_rate": args.lr,
        "regularization": args.reg,
        "num_threads": args.threads,
        "random_state": args.seed,
        "positive_threshold": args.positive_threshold,
        "min_user": args.min_user,
        "min_item": args.min_item,
        "kcore_rounds": args.kcore_rounds,
        "num_users": NUM_USERS,
        "num_items": NUM_ITEMS,
        "active_users": int(active_users),
        "active_items": int(active_items),
        "train_positives": int(nnz),
        "training_time_seconds": float(elapsed),
    }

    paths = save_factors(
        "bpr",
        user_factors,
        item_factors,
        meta=metadata,
    )

    log("\nSaved artifacts:")
    log(str(paths))

    return paths


# =========================================================
# 9. MAIN
# =========================================================

def main():

    parser = argparse.ArgumentParser(
        description="Train implicit BPR and export factors."
    )

    parser.add_argument(
        "--factors",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--iters",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--reg",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=50_000,
    )

    parser.add_argument(
        "--positive-threshold",
        type=float,
        default=4.0,
    )

    parser.add_argument(
        "--min-user",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--min-item",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--kcore-rounds",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--keep-work",
        action="store_true",
        help="Keep SQLite and memmap files.",
    )

    args = parser.parse_args()

    if args.factors < 1:
        parser.error("--factors must be >= 1")

    if args.iters < 1:
        parser.error("--iters must be >= 1")

    if not 1 <= args.threads <= 4:
        parser.error("--threads must be between 1 and 4")

    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")

    if args.min_user < 1 or args.min_item < 1:
        parser.error("--min-user and --min-item must be >= 1")

    if args.kcore_rounds < 1:
        parser.error("--kcore-rounds must be >= 1")

    start_total = time.time()

    workdir = get_workdir()
    db_path = workdir / "interactions.sqlite"

    conn = None
    model = None
    mat = None
    backing_arrays = None

    success = False

    try:
        log("=" * 65)
        log("IMPLICIT BPR TRAINING")
        log("=" * 65)

        log(f"Train directory: {TRAIN_DIR}")
        log(f"Users:           {NUM_USERS:,}")
        log(f"Items:           {NUM_ITEMS:,}")
        log(f"Factors:         {args.factors}")
        log(f"Threads:         {args.threads}")
        log(f"Work directory:  {workdir}")

        # Step 1: Load and deduplicate train data.
        conn = create_database(db_path)

        nnz = load_train(
            conn=conn,
            train_dir=TRAIN_DIR,
            batch_size=args.batch_size,
            positive_threshold=args.positive_threshold,
        )

        # Step 2: Optional K-core.
        apply_kcore(
            conn=conn,
            min_user=args.min_user,
            min_item=args.min_item,
            rounds=args.kcore_rounds,
        )

        nnz = conn.execute(
            "SELECT COUNT(*) FROM interactions"
        ).fetchone()[0]

        log(f"\nFinal training pairs: {nnz:,}")

        # Step 3: Build CSR.
        mat, backing_arrays = build_csr(
            conn=conn,
            workdir=workdir,
            nnz=nnz,
        )

        # Step 4: Train BPR.
        model, user_factors, item_factors, elapsed = train_bpr(
            mat=mat,
            factors=args.factors,
            iterations=args.iters,
            learning_rate=args.lr,
            regularization=args.reg,
            threads=args.threads,
            seed=args.seed,
        )

        # Step 5: Export.
        paths = export_factors(
            user_factors=user_factors,
            item_factors=item_factors,
            conn=conn,
            args=args,
            nnz=nnz,
            elapsed=elapsed,
        )

        total_elapsed = time.time() - start_total

        log("\n" + "=" * 65)
        log("BPR TRAINING SUMMARY")
        log("=" * 65)

        log(f"Users:          {NUM_USERS:,}")
        log(f"Items:          {NUM_ITEMS:,}")
        log(f"Positive pairs: {nnz:,}")
        log(f"Factors:        {args.factors}")
        log(f"Iterations:     {args.iters}")
        log(f"Threads:        {args.threads}")
        log(f"Training time:  {elapsed / 60:.2f} min")
        log(f"Total time:     {total_elapsed / 60:.2f} min")

        log("\nExport paths:")
        log(str(paths))

        print_memory()

        log("\nBPR training completed successfully.")

        success = True

    except Exception:
        log("\nBPR training failed.")
        log(f"Temporary files preserved: {workdir}")
        raise

    finally:
        if conn is not None:
            conn.close()

        model = None
        mat = None
        backing_arrays = None

        gc.collect()

        if success and not args.keep_work:
            log("\nRemoving temporary files...")

            shutil.rmtree(
                workdir,
                ignore_errors=True,
            )

            log("Temporary files removed.")

        else:
            log(f"Temporary files retained: {workdir}")


if __name__ == "__main__":
    main()
