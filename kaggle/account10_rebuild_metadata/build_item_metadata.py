#!/usr/bin/env python
"""Kaggle: Rebuild item_metadata.db from nguynteo/finaldata raw Amazon metadata (Spark, CPU).

Bug being fixed: artifacts/item_metadata.db was built from a stale/truncated source —
schema shrank to 7 columns and 66-80% of text fields (title/brand/category/...) were
empty; asin duplicated over rows. Rebuild it properly:

  1. item_idx -> asin : nguynteo/lastone  (amazon_recs_output/item_mapping, asin<->item_idx)
  2. meta per asin      : nguynteo/finaldata (meta_Books.json, meta_Electronics.json;
                           Amazon 2023 schema, per line a JSON object)
  3. match via ASIN (stable key) with Spark (CPU), join, flatten,
  4. write item_metadata.db (sqlite) + item_meta_fill_stats.json

Requires: pyspark (installed in-kernel). GPU not needed.
"""
import json
import sqlite3
import time
from pathlib import Path
import shutil

OUT_DIR = Path("/kaggle/working" if Path("/kaggle/working").exists() else "artifacts")
OUT_DIR.mkdir(parents=True, exist_ok=True)
SCRATCH = OUT_DIR / "_meta_join"

_T0 = time.time()

def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')} +{time.time() - _T0:7.1f}s] {msg}", flush=True)

def mem_info() -> str:
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return f"{rss / 1e9:.1f} GB" if rss > 1e9 else f"{rss / 1e6:.1f} MB"
    except Exception:
        return "n/a"

# ---- install pyspark if needed
try:
    import pyspark  # noqa: F401
except ImportError:
    import subprocess, sys
    log("installing pyspark ...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "pyspark>=3.5"])

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType
import pyarrow.parquet as pq

log(f"=== Rebuild item_metadata.db (Spark, CPU) | RAM {mem_info()}")


def find_input(pattern: str):
    hits = sorted(Path("/kaggle/input").rglob(pattern))
    if not hits:
        return None
    chosen = [p for p in hits if "__MACOSX" not in p.parts]
    return chosen[0] if chosen else hits[0]


def find_input_multi(*patterns: str):
    for pat in patterns:
        p = find_input(pat)
        if p is not None:
            print(f"found {p} (size={p.stat().st_size:,} bytes)")
            return p
    raise FileNotFoundError(f"missing input: {patterns}")


# 1. SparkSession (CPU)
spark = (SparkSession.builder
         .appName("rebuild-item-metadata")
         .master("local[*]")
         .config("spark.driver.memory", "5g")
         .config("spark.executor.memory", "5g")
         .config("spark.sql.shuffle.partitions", "16")
         .config("spark.sql.caseSensitive", "false")
         .getOrCreate())
spark.sparkContext.setLogLevel("WARN")
log(f"1. Spark {spark.version} running, master={spark.sparkContext.master}")

# 2. Load mapping (asin -> item_idx)
mapping_path = find_input_multi("item_mapping*.parquet", "*/item_mapping/part-*.parquet")
mapping = spark.read.parquet(str(mapping_path)).select(
    F.col("asin").cast("string").alias("asin"),
    F.col("item_idx").cast("bigint").alias("item_idx"),
).dropna(subset=["asin"]).dropDuplicates(["asin"]).cache()
n_items = mapping.count()
log(f"2. mapping: {n_items:,} unique asin rows")
print("sample mapping (item_idx, asin):", [(int(r.item_idx), str(r.asin)) for r in mapping.take(5)])
log(f"2. mapping cached: {mapping.storageLevel}")

# 3. Parse raw metadata JSONL lines into flat strings
META_COLS = ["asin", "title", "brand", "category", "price", "description",
             "feature", "also_buy", "also_view", "imageURL", "domain", "product_text"]
META_SCHEMA = StructType([StructField(c, StringType()) for c in META_COLS])


def parse_line(line, domain):
    try:
        obj = json.loads(line)
    except Exception:
        return None
    asin = str(obj.get("parent_asin") or obj.get("asin") or "").strip()
    if not asin:
        return None

    def flat_cats(cats):
        flat = []
        if isinstance(cats, list):
            for part in cats:
                if isinstance(part, list):
                    flat.append(" > ".join(str(x) for x in part))
                else:
                    flat.append(str(part))
        elif cats is not None:
            flat.append(str(cats))
        return " | ".join(flat)

    def flat_list(lst):
        if not isinstance(lst, list):
            return str(lst) if lst is not None else ""
        return " | ".join(str(x) for x in lst)

    def clean(x):
        if x is None:
            return ""
        if isinstance(x, (list, dict)):
            return flat_list(x) if isinstance(x, list) else json.dumps(x, ensure_ascii=False)
        s = str(x).strip()
        return "" if s.lower() == "nan" else s

    title = clean(obj.get("title"))[:512]
    brand = clean(obj.get("brand") or obj.get("store"))[:128]
    category = flat_cats(obj.get("categories") or obj.get("category"))[:512]
    price = obj.get("price")
    price_s = (f"{price}" if isinstance(price, (int, float)) else str(price or ""))[:64]
    description = flat_list(obj.get("description") if isinstance(obj.get("description"), list)
                            else str(obj.get("description") or ""))[:2048]
    feature = flat_list(obj.get("features") if isinstance(obj.get("features"), list)
                        else str(obj.get("feature") or ""))[:2048]
    also_buy = flat_list(obj.get("also_buy") if isinstance(obj.get("also_buy"), list) else str(obj.get("also_buy") or ""))[:1024]
    also_view = flat_list(obj.get("also_viewed") if isinstance(obj.get("also_viewed"), list) else str(obj.get("also_view") or ""))[:1024]
    image = clean(obj.get("imageURL"))[:512]
    product_text = " | ".join(p for p in [title, brand, category, feature, description] if p)[:4096]
    return tuple([asin, title, brand, category, price_s, description, feature,
                  also_buy, also_view, image, domain, product_text])


parse_b = F.udf(lambda line: parse_line(line, "books"), META_SCHEMA)
parse_e = F.udf(lambda line: parse_line(line, "electronics"), META_SCHEMA)


def stream_file(pattern: str, parser, domain: str):
    src = find_input_multi(pattern, f"*/{pattern}", f"**/{pattern}")
    log(f"3. streaming {src.name} ({src.stat().st_size:,} bytes) via Spark")
    lines = spark.read.text(str(src))
    return (lines.select(parser(F.col("value")).alias("m"))
                 .select("m.*")
                 .dropna(subset=["asin"]))


books = stream_file("meta_Books.json", parse_b, "books")
electronics = stream_file("meta_Electronics.json", parse_e, "electronics")
meta = books.unionByName(electronics).dropDuplicates(["asin"])
log(f"3. meta rows -> {meta.count():,}")

# 4. Join mapping x meta (left) and write parquet for python merge
joined = (mapping.alias("m")
          .join(meta.alias("x"), F.col("m.asin") == F.col("x.asin"), "left")
          .select(F.col("m.item_idx").alias("item_idx"),
                  F.coalesce(F.col("x.asin"), F.col("m.asin")).alias("asin"),
                  F.coalesce(F.col("x.title"), F.lit("")).alias("title"),
                  F.coalesce(F.col("x.brand"), F.lit("")).alias("brand"),
                  F.coalesce(F.col("x.category"), F.lit("")).alias("category"),
                  F.coalesce(F.col("x.price"), F.lit("")).alias("price"),
                  F.coalesce(F.col("x.description"), F.lit("")).alias("description"),
                  F.coalesce(F.col("x.feature"), F.lit("")).alias("feature"),
                  F.coalesce(F.col("x.also_buy"), F.lit("")).alias("also_buy"),
                  F.coalesce(F.col("x.also_view"), F.lit("")).alias("also_view"),
                  F.coalesce(F.col("x.imageURL"), F.lit("")).alias("imageURL"),
                  F.coalesce(F.col("x.domain"), F.lit("")).alias("domain"),
                  F.coalesce(F.col("x.product_text"), F.lit("")).alias("product_text")))
if SCRATCH.exists():
    shutil.rmtree(SCRATCH)
joined.coalesce(16).write.mode("overwrite").parquet(str(SCRATCH))
spark.stop()
log(f"4. joined parquet written -> {SCRATCH}")
_part_files = list(SCRATCH.glob("part-*.parquet"))
_bytes = sum(f.stat().st_size for f in _part_files)
log(f"4. parquet partitions={len(_part_files)} bytes={_bytes:,}")

# 5. Python: write final sqlite DB + fill stats
COLS = ["item_idx", "asin", "title", "brand", "category", "price", "description",
        "feature", "also_buy", "also_view", "imageURL", "domain", "product_text"]
db = OUT_DIR / "item_metadata.db"
if db.exists():
    db.unlink()
conn = sqlite3.connect(str(db))
conn.execute("PRAGMA journal_mode=OFF")
conn.execute("PRAGMA synchronous=OFF")
conn.execute("PRAGMA temp_store=MEMORY")
conn.execute(
    "CREATE TABLE items ("
    "item_idx INTEGER PRIMARY KEY, asin TEXT, title TEXT, brand TEXT, category TEXT,"
    "price TEXT, description TEXT, feature TEXT, also_buy TEXT, also_view TEXT,"
    "imageURL TEXT, domain TEXT, product_text TEXT)"
)
Q = f"INSERT OR REPLACE INTO items ({','.join(COLS)}) VALUES ({','.join('?' * len(COLS))})"
stats = {k: 0 for k in ["total", "title", "brand", "category", "description", "price",
                        "product_text", "imageURL", "feature", "also_buy", "also_view",
                        "missing_meta"]}
def _iter_scratch():
    for _part in sorted(SCRATCH.glob("part-*.parquet")):
        yield from pq.ParquetFile(str(_part)).iter_batches(batch_size=50_000, columns=COLS)
for batch in _iter_scratch():
    batch_df = batch.to_pandas()
    for tup in batch_df.itertuples(index=False, name=None):
        conn.execute(Q, tuple(str(x) if x is not None else "" for x in tup))
        stats["total"] += 1
        for i, k in [(2, "title"), (3, "brand"), (4, "category"), (5, "price"),
                     (6, "description"), (7, "feature"), (8, "also_buy"), (9, "also_view"),
                     (10, "imageURL"), (12, "product_text")]:
            if str(tup[i]).strip():
                stats[k] += 1
        if not str(tup[2]).strip() and not str(tup[12]).strip():
            stats["missing_meta"] += 1
conn.commit()
conn.execute("CREATE INDEX IF NOT EXISTS idx_asin ON items(asin)")
conn.execute("CREATE INDEX IF NOT EXISTS idx_category ON items(category)")
conn.commit()
total = stats["total"]
stats_out = {
    "written_items": total,
    "fill_counts": {k: stats[k] for k in ["title", "brand", "category", "description",
                                          "price", "product_text", "imageURL", "feature", "also_buy"]},
    "fill_pct": {k: round(stats[k] / max(1, total) * 100, 2) for k in ["title", "brand", "category",
                  "description", "price", "product_text", "imageURL", "feature", "also_buy"]},
    "items_missing_meta": stats["missing_meta"],
}
(OUT_DIR / "item_meta_fill_stats.json").write_text(json.dumps(stats_out, indent=2))
print("=== FILL STATS ===")
print(json.dumps(stats_out, indent=2))
sample = conn.execute("""
    SELECT item_idx, asin, substr(title, 1, 60) AS title, domain, substr(product_text, 1, 120) AS product_text
    FROM items WHERE title != '' ORDER BY item_idx LIMIT 5
""").fetchall()
print("=== sample rows (item_idx, asin, title, domain, product_text) ===")
for row in sample:
    print(" ", row)
sample_missing = conn.execute("SELECT COUNT(*) FROM items WHERE product_text = ''").fetchone()[0]
print(f"=== items with empty product_text: {sample_missing:,} ({sample_missing / max(1, stats_out['written_items']) * 100:.2f}%)")
conn.close()
if SCRATCH.exists():
    shutil.rmtree(SCRATCH)
log(f"DONE. DB={OUT_DIR / 'item_metadata.db'} size={(OUT_DIR / 'item_metadata.db').stat().st_size:,} bytes | RAM {mem_info()} | elapsed={time.time() - _T0:.1f}s")
