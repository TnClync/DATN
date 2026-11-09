#!/usr/bin/env python
"""Build a SQLite metadata DB from the raw Item_profile parquet files.

Stores FULL product metadata (all columns from nguynteo/item-profile / Item_profile
parquets) keyed by item_idx. The API reads from this DB (SELECT *), so new columns
are served automatically. No fabrication; metadata is read from the raw source.
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]


def find_source() -> Path:
    candidates = [
        Path("/tmp/opencode/ipall/Item_profile"),
        ROOT / "data" / "Item_profile",
        ROOT / "data" / "item_profile",
        ROOT / "data" / "raw" / "Item_profile",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise SystemExit("Item_profile source not found. Download raw nguynteo/item-profile first.")


COLS = [
    "item_idx", "asin", "title", "brand", "category", "feature", "description",
    "price", "also_buy", "also_view", "metadata_source", "product_text",
]


def main() -> None:
    src = find_source()
    db = ROOT / "artifacts" / "item_metadata.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    tmp = db.with_suffix(".full.tmp.db")
    if tmp.exists():
        tmp.unlink()

    conn = sqlite3.connect(tmp)
    conn.execute("PRAGMA journal_mode=OFF")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA temp_store=MEMORY")
    coldefs = ", ".join(("item_idx INTEGER PRIMARY KEY" if c == "item_idx" else f"{c} TEXT") for c in COLS)
    conn.execute(f"CREATE TABLE items ({coldefs})")
    q = f"INSERT OR REPLACE INTO items ({','.join(COLS)}) VALUES ({','.join(['?'] * len(COLS))})"

    t0 = time.time()
    total = 0
    files = [f for f in sorted(src.glob("*.parquet")) if f.suffix == ".parquet" and "part-" in f.name]
    for f in files:
        schema_names = pq.read_schema(f).names
        use = [c for c in COLS if c in schema_names]
        table = pq.read_table(f, columns=use)
        df = table.to_pandas()
        for c in COLS:
            if c not in df.columns:
                df[c] = ""
        df = df[COLS]
        df["item_idx"] = pd.to_numeric(df["item_idx"], errors="coerce").fillna(-1).astype("int64")
        for c in COLS:
            if c == "item_idx":
                continue
            s = df[c]
            df[c] = s.map(lambda x: "" if x is None or (isinstance(x, float) and pd.isna(x)) else str(x))
        conn.executemany(q, df.itertuples(index=False, name=None))
        conn.commit()
        total += len(df)
        print(f"{f.name}: +{len(df)} total={total} {time.time() - t0:.1f}s", flush=True)

    conn.execute("CREATE INDEX IF NOT EXISTS idx_asin ON items(asin)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_category ON items(category)")
    conn.commit()
    rows = conn.execute("SELECT COUNT(*) FROM items").fetchone()[0]
    print(f"final rows={rows}")
    conn.close()

    bak = db.with_suffix(".db.bak")
    if db.exists():
        if bak.exists():
            bak.unlink()
        db.rename(bak)
    tmp.rename(db)
    print(f"saved {db} bytes={db.stat().st_size} backup={bak.exists()}")


if __name__ == "__main__":
    main()
