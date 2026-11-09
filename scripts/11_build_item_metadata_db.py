#!/usr/bin/env python
"""Build a SQLite metadata DB from the Item_profile parquet files.

Stores real product metadata (title, brand, category, price, description) keyed by
item_idx. The API reads from this DB instead of fabricating metadata in code.
"""
from __future__ import annotations
from pathlib import Path
import sqlite3
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
SRC = Path("/tmp/opencode/ipall/Item_profile")
DB = ROOT / "artifacts" / "item_metadata.db"
DB.parent.mkdir(parents=True, exist_ok=True)
if DB.exists():
    DB.unlink()

conn = sqlite3.connect(DB)
conn.execute("""CREATE TABLE items (
    item_idx INTEGER PRIMARY KEY,
    asin TEXT,
    title TEXT,
    brand TEXT,
    category TEXT,
    price TEXT,
    description TEXT
)""")
conn.execute("CREATE INDEX idx_asin ON items(asin)")

total = 0
for f in sorted(SRC.glob("*.parquet")):
    t = pq.read_table(f, columns=["item_idx", "asin", "title", "brand", "category", "price", "description"])
    df = t.to_pandas()
    rows = []
    for _, r in df.iterrows():
        desc = r.get("description")
        if not isinstance(desc, str):
            desc = ""
        rows.append((
            int(r["item_idx"]),
            str(r.get("asin") or ""),
            str(r.get("title") or ""),
            str(r.get("brand") or ""),
            str(r.get("category") or ""),
            str(r.get("price") or ""),
            desc[:500],
        ))
    conn.executemany("INSERT OR REPLACE INTO items VALUES (?,?,?,?,?,?,?)", rows)
    conn.commit()
    total += len(rows)
    print(f"{f.name}: +{len(rows)} (total {total})", flush=True)

conn.commit()
print("rows:", conn.execute("SELECT COUNT(*) FROM items").fetchone()[0])
conn.close()
print("saved", DB)
