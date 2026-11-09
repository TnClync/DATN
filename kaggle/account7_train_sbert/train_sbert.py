#!/usr/bin/env python
"""Build SBERT embeddings (metadata + reviewer text) once — FULL catalog (2.46M items).

Outputs (float16, written chunk-by-chunk with np.memmap so RAM stays low):
  - sbert_meta_embeddings_f16.npy     (N x 384, title | brand | category | description[:300])
  - sbert_review_embeddings_f16.npy   (N x 384, up to 3 review_content; meta emb reused when no review)
  - sbert_item_embeddings_f16.npy     (N x 768, concat[meta, review] — PRIMARY for downstream kernels)
  - user_ids.npy, item_ids.npy, compact_to_orig.npy, sbert_summary.json

Kaggle optimizations:
  - vectorized parquet text building (no slow iterrows)
  - chunked encode with live logs (steps, timings, texts/s, RAM/GPU usage)
  - fp16 forward pass on T4 tensor cores
  - dynamic token-bucketed batching (short fallback texts get 4x bigger batches)
  - multi-GPU: one shared worker pool on cuda:0 + cuda:1 (T4 x2)
  - no float32 concatenation of the full matrix (avoid OOM)
"""
import os, sys, gc, time, json, math, subprocess
from pathlib import Path

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass


def _install(pkgs):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + pkgs)


try:
    import torch
    import pyarrow.parquet as pq
    import sentence_transformers
except ImportError:
    _install(["sentence-transformers>=2.7", "pyarrow", "tqdm"])
    import torch
    import pyarrow.parquet as pq
    import sentence_transformers

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
from typing import Optional, Set

_T0 = time.time()


def log(msg: str):
    """Realtime log with wall clock + elapsed time."""
    print(f"[{time.strftime('%H:%M:%S')} +{time.time() - _T0:7.1f}s] {msg}", flush=True)


def mem_info() -> str:
    """Current process peak RSS (GB)."""
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return f"{rss / 1e9:.1f} GB" if rss > 1e9 else f"{rss / 1e6:.1f} GB"
    except Exception:
        return "n/a"


def gpu_mem() -> str:
    if not torch.cuda.is_available():
        return "CPU"
    parts = []
    for i in range(torch.cuda.device_count()):
        alloc = torch.cuda.memory_allocated(i) / 1e9
        reserved = torch.cuda.memory_reserved(i) / 1e9
        parts.append(f"GPU{i} alloc={alloc:.1f}G res={reserved:.1f}G")
    return " | ".join(parts)


MIN_INTERACTIONS = 3
SEED = 42
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMB_DIM = 384
ENCODE_CHUNK = 100_000          # rows encoded + written per iteration

OUT_DIR = Path("/kaggle/working" if Path("/kaggle/working").exists() else "artifacts/sbert")
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0
USE_FP16 = N_GPUS >= 1                 # fp16 inference on T4 tensor cores (~1.5-2x)
SHORT_MAX_TOKENS = 24                  # "Product item <id>" fallback texts are ~5 tokens
BATCH_SIZE_SHORT = 2048                # tiny sequences -> big batches (less launch overhead)
BATCH_SIZE_LONG = 512                  # real metadata/review text

if N_GPUS >= 1:
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

log(f"OUT_DIR       = {OUT_DIR}")
log(f"DEVICE        = {DEVICE}  (N_GPUS={N_GPUS})")
log(f"FP16          = {USE_FP16}")
log(f"BATCH (s/l)   = {BATCH_SIZE_SHORT}/{BATCH_SIZE_LONG}")
log(f"ENCODE_CHUNK  = {ENCODE_CHUNK:,}")
log(f"MODEL         = {MODEL_NAME}")


def find_base_dir() -> Path:
    candidates = [
        Path("/kaggle/input/lastone/amazon_recs_output"),
        Path("/kaggle/input/lastone"),
        Path("/kaggle/input/datasets/nguynteo/lastone"),
        Path("data"),
    ]
    for c in candidates:
        if (c / "train").exists() or (c / "test").exists():
            return c
    root = Path("/kaggle/input")
    if root.exists():
        for p in root.rglob("train"):
            if p.is_dir():
                return p.parent
    return Path("data")


def _find_parquet_dir(required_cols: Set[str]) -> Optional[Path]:
    root = Path("/kaggle/input")
    if not root.exists():
        return None
    seen = set()
    for p in root.rglob("*.parquet"):
        parent = p.parent
        if parent in seen:
            continue
        seen.add(parent)
        try:
            if required_cols.issubset(set(pq.read_schema(p).names)):
                return parent
        except Exception:
            continue
    return None


BASE_DIR = find_base_dir()
log(f"BASE_DIR = {BASE_DIR}")


# ---------------------------------------------------------------------------
# STEP 1/5 — compact indices (same protocol as the CF / eval kernels)
# ---------------------------------------------------------------------------
log("=" * 60)
log("STEP 1/5  Build compact user/item indices")
log("=" * 60)
_t = time.time()

train_files = sorted(BASE_DIR.glob("train/*.parquet"))
assert train_files, f"No train parquet under {BASE_DIR}"
log(f"train parquet files: {len(train_files)}")

dfs = []
for f in tqdm(train_files, desc="Reading train"):
    d = pq.read_table(f, columns=["user_idx", "item_idx", "rating"]).to_pandas()
    d = d.loc[d["rating"] >= 4.0, ["user_idx", "item_idx"]]
    dfs.append(d)
train_df = pd.concat(dfs, ignore_index=True)
del dfs
gc.collect()
log(f"raw positive interactions: {len(train_df):,}  | RAM {mem_info()}")

user_counts = train_df["user_idx"].value_counts()
active = user_counts[user_counts >= MIN_INTERACTIONS].index
train_df = train_df[train_df["user_idx"].isin(active)]
log(f"active users (>={MIN_INTERACTIONS} interactions): {len(active):,}")

u_raw = train_df["user_idx"].to_numpy(np.int64)
i_raw = train_df["item_idx"].to_numpy(np.int64)
del train_df, user_counts, active
gc.collect()

u_uniq = np.unique(u_raw)
i_uniq = np.unique(i_raw)
n_users, n_items = len(u_uniq), len(i_uniq)

compact_to_orig = i_uniq.astype(np.int64)

log(f"compact: {n_users:,} users x {n_items:,} items  ({time.time() - _t:.1f}s)  | RAM {mem_info()}")


# ---------------------------------------------------------------------------
# STEP 2/5 — real item metadata text
# ---------------------------------------------------------------------------
log("=" * 60)
log("STEP 2/5  Load real item metadata text")
log("=" * 60)
_t = time.time()

profile_dir = _find_parquet_dir({"item_idx", "title"})
log(f"profile dir: {profile_dir}")

item_text_map = {}
if profile_dir is not None:
    files = sorted(profile_dir.glob("*.parquet"))
    log(f"profile parquet files: {len(files)}")
    cols = ["item_idx", "title", "brand", "category", "description"]

    for f in tqdm(files, desc="Loading profiles"):
        try:
            available = set(pq.read_schema(f).names)
            if "item_idx" not in available:
                continue
            use_cols = [c for c in cols if c in available]
            d = pq.read_table(f, columns=use_cols).to_pandas()

            def col(name):
                if name in d:
                    return d[name].fillna("").astype(str).str.strip()
                return pd.Series("", index=d.index)

            title, brand, cat = col("title"), col("brand"), col("category")
            desc = col("description").str[:300]
            brand = brand.where(~brand.str.lower().str.contains("amazon", na=False), "")

            text = title.copy()
            for p in (brand, cat, desc):
                mask = p.str.len() > 0
                text = text.where(~mask, text + " | " + p)
            text = text.str.strip(" |").str.strip()

            keep = text.str.len() > 0
            for idx, txt in zip(d.loc[keep, "item_idx"].to_numpy(np.int64), text[keep].to_numpy()):
                item_text_map[int(idx)] = txt
            del d
        except Exception as e:
            log(f"profile err {f.name}: {e}")

log(f"real text loaded for {len(item_text_map):,} items  ({time.time() - _t:.1f}s)  | RAM {mem_info()}")


# ---------------------------------------------------------------------------
# STEP 3/5 — encode metadata into a float16 memmap (chunked, no big RAM array)
# ---------------------------------------------------------------------------
log("=" * 60)
log("STEP 3/5  Encode metadata embeddings")
log("=" * 60)

log(f"loading model {MODEL_NAME}")
sbert = SentenceTransformer(MODEL_NAME, device=(f"cuda:0" if N_GPUS else "cpu"))
if USE_FP16:
    sbert = sbert.half()
    torch.backends.cuda.matmul.allow_tf32 = True
log(f"model loaded | dtype={next(sbert.parameters()).dtype} | {gpu_mem()} | RAM {mem_info()}")

# Kaggle "GPU T4 x2" -> 2 devices. One shared pool: workers are created once and
# reused for every chunk (much cheaper than spawning a pool per encode call).
POOL = None
if N_GPUS >= 2:
    try:
        POOL = sbert.start_multi_process_pool([f"cuda:{i}" for i in range(N_GPUS)])
        log(f"multi-GPU encode pool started on {N_GPUS} devices")
    except Exception as e:
        POOL = None
        log(f"multi-GPU pool failed ({e}) -> falling back to single device")


def _token_lengths(texts):
    """Fast batch tokenization -> sequence lengths (drives dynamic batching)."""
    enc = sbert.tokenizer(texts, add_special_tokens=True, truncation=True,
                          max_length=sbert.max_seq_length)
    return np.fromiter((len(x) for x in enc["input_ids"]), dtype=np.int32, count=len(texts))


def _encode_raw(texts, batch_size):
    if POOL is not None:
        embs = sbert.encode(texts, pool=POOL, batch_size=batch_size, chunk_size=5_000,
                            show_progress_bar=True, convert_to_numpy=True)
    else:
        embs = sbert.encode(texts, batch_size=batch_size, show_progress_bar=True,
                            convert_to_numpy=True, device=DEVICE)
    return np.asarray(embs, dtype=np.float32)


def encode_texts(texts, desc: str):
    """Dynamic (token-bucketed) batching: short texts get much larger batches.

    Encoder inference has no decode loop, so "continuous batching" in the vLLM
    sense does not apply; the equivalent win is to size each forward batch by
    sequence length instead of using one fixed batch size for everything.
    """
    n = len(texts)
    if n == 0:
        return np.zeros((0, EMB_DIM), dtype=np.float32)

    _t = time.time()
    if N_GPUS == 0:
        # CPU fallback: dynamic bucketing only doubles tokenization cost here
        out = _encode_raw(texts, 128)
    else:
        lens = _token_lengths(texts)
        order = np.argsort(lens, kind="stable")
        short_idx = order[lens[order] <= SHORT_MAX_TOKENS]
        long_idx = order[lens[order] > SHORT_MAX_TOKENS]

        out = np.empty((n, EMB_DIM), dtype=np.float32)
        for idx, bs, label in ((short_idx, BATCH_SIZE_SHORT, "short"),
                               (long_idx, BATCH_SIZE_LONG, "long")):
            if idx.size == 0:
                continue
            out[idx] = _encode_raw([texts[i] for i in idx], bs)
            log(f"  {desc} [{label}] {idx.size:,} texts | batch={bs} | {gpu_mem()}")

    # L2-normalize in float32 (fp16 forward pass, stable cosine scores)
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    np.divide(out, np.maximum(norms, 1e-12), out=out)
    if not np.isfinite(out).all():
        log(f"  WARN {desc}: non-finite values detected")

    dt = max(time.time() - _t, 1e-9)
    log(f"  {desc}: {n:,} texts in {dt:.1f}s = {n / dt:,.0f} texts/s")
    return out


meta_path = OUT_DIR / "sbert_meta_embeddings_f16.npy"
meta_mm = np.lib.format.open_memmap(meta_path, mode="w+", dtype=np.float16, shape=(n_items, EMB_DIM))

n_fallback = 0
_t = time.time()
n_chunks = math.ceil(n_items / ENCODE_CHUNK)
for ci, start in enumerate(range(0, n_items, ENCODE_CHUNK)):
    end = min(start + ENCODE_CHUNK, n_items)
    texts = []
    for idx in range(start, end):
        orig = int(compact_to_orig[idx])
        txt = item_text_map.get(orig)
        if txt is None:
            txt = f"Product item {orig}"
            n_fallback += 1
        texts.append(txt)
    embs = encode_texts(texts, f"metadata chunk {ci + 1}/{n_chunks}")
    meta_mm[start:end] = embs.astype(np.float16)
    del texts, embs
    if N_GPUS:
        torch.cuda.empty_cache()
    log(f"  metadata {end:,}/{n_items:,} ({100.0 * end / n_items:5.1f}%)  | RAM {mem_info()}")
meta_mm.flush()
del meta_mm
gc.collect()
del item_text_map
gc.collect()
log(f"metadata done in {time.time() - _t:.1f}s  (fallback dummy text: {n_fallback:,}/{n_items:,})")


# ---------------------------------------------------------------------------
# STEP 4/5 — reviewer text embeddings (fallback rows reuse metadata embedding)
# ---------------------------------------------------------------------------
log("=" * 60)
log("STEP 4/5  Reviewer text embeddings")
log("=" * 60)
_t = time.time()

review_dir = _find_parquet_dir({"item_idx", "review_content"})
log(f"review dir: {review_dir}")

MAX_REVIEWS_PER_ITEM = 3
MAX_REVIEW_CHARS = 500
review_by_item = {}
n_review_rows = 0

if review_dir is not None:
    files = sorted(review_dir.glob("*.parquet"))
    log(f"review parquet files: {len(files)}")
    for f in tqdm(files, desc="Loading reviews"):
        try:
            d = pq.read_table(f, columns=["item_idx", "review_content"]).to_pandas()
            d = d.dropna(subset=["review_content"])
            d["review_content"] = d["review_content"].astype(str).str.strip().str[:MAX_REVIEW_CHARS]
            d = d[d["review_content"].str.len() > 0]
            if d.empty:
                continue
            d = d.groupby("item_idx", sort=False).head(MAX_REVIEWS_PER_ITEM)
            for item_idx, grp in d.groupby("item_idx", sort=False):
                key = int(item_idx)
                cur = review_by_item.setdefault(key, [])
                if len(cur) < MAX_REVIEWS_PER_ITEM:
                    cur.extend(grp["review_content"].tolist()[: MAX_REVIEWS_PER_ITEM - len(cur)])
            n_review_rows += len(d)
            del d
        except Exception as e:
            log(f"review err {f.name}: {e}")

log(f"{n_review_rows:,} review rows -> {len(review_by_item):,} items  ({time.time() - _t:.1f}s)  | RAM {mem_info()}")

meta_mm = np.load(meta_path, mmap_mode="r")
review_path = OUT_DIR / "sbert_review_embeddings_f16.npy"
review_mm = np.lib.format.open_memmap(review_path, mode="w+", dtype=np.float16, shape=(n_items, EMB_DIM))

_t = time.time()
n_chunks = math.ceil(n_items / ENCODE_CHUNK)
n_with_rev = 0
for ci, start in enumerate(range(0, n_items, ENCODE_CHUNK)):
    end = min(start + ENCODE_CHUNK, n_items)
    enc_idx, enc_txt = [], []
    for idx in range(start, end):
        revs = review_by_item.get(int(compact_to_orig[idx]))
        if revs:
            enc_idx.append(idx)
            enc_txt.append(" ".join(revs)[:1000])
    n_with_rev += len(enc_idx)

    # items without reviews: text == metadata text -> reuse the (identical) embedding
    if len(enc_idx) < (end - start):
        keep = np.ones(end - start, dtype=bool)
        keep[np.asarray(enc_idx, dtype=np.int64) - start] = False
        rows = np.arange(start, end, dtype=np.int64)[keep]
        review_mm[rows] = meta_mm[rows]

    embs = encode_texts(enc_txt, f"review chunk {ci + 1}/{n_chunks}")
    if len(enc_idx):
        review_mm[np.asarray(enc_idx, dtype=np.int64)] = embs.astype(np.float16)
    del enc_idx, enc_txt, embs
    if N_GPUS:
        torch.cuda.empty_cache()
    log(f"  reviewer {end:,}/{n_items:,} ({100.0 * end / n_items:5.1f}%)  | RAM {mem_info()}")

review_mm.flush()
del review_mm, meta_mm
gc.collect()
del review_by_item
gc.collect()
log(f"reviewer done in {time.time() - _t:.1f}s  (items with real reviews: {n_with_rev:,})")


# ---------------------------------------------------------------------------
# STEP 5/5 — concat into the primary 768-dim file (memmap, no float32 blow-up)
# ---------------------------------------------------------------------------
log("=" * 60)
log("STEP 5/5  Combine + save")
log("=" * 60)
_t = time.time()

meta_mm = np.load(meta_path, mmap_mode="r")
review_mm = np.load(review_path, mmap_mode="r")
comb_path = OUT_DIR / "sbert_item_embeddings_f16.npy"
comb_mm = np.lib.format.open_memmap(comb_path, mode="w+", dtype=np.float16, shape=(n_items, 2 * EMB_DIM))
for start in range(0, n_items, ENCODE_CHUNK):
    end = min(start + ENCODE_CHUNK, n_items)
    comb_mm[start:end, :EMB_DIM] = meta_mm[start:end]
    comb_mm[start:end, EMB_DIM:] = review_mm[start:end]
comb_mm.flush()
del comb_mm, meta_mm, review_mm
gc.collect()
log(f"combined saved in {time.time() - _t:.1f}s")

np.save(OUT_DIR / "user_ids.npy", u_uniq)
np.save(OUT_DIR / "item_ids.npy", i_uniq)
np.save(OUT_DIR / "compact_to_orig.npy", compact_to_orig)

if POOL is not None:
    try:
        sbert.stop_multi_process_pool(POOL)
        log("multi-GPU pool stopped")
    except Exception as e:
        log(f"pool stop warning: {e}")

summary = {
    "n_users": int(n_users),
    "n_items": int(n_items),
    "items_with_real_text": int(n_items - n_fallback),
    "items_with_fallback_text": int(n_fallback),
    "items_with_reviews": int(n_with_rev),
    "review_rows": int(n_review_rows),
    "dims_meta": EMB_DIM,
    "dims_review": EMB_DIM,
    "dims_combined": 2 * EMB_DIM,
    "dtype": "float16",
    "device": DEVICE,
    "cuda_devices": int(N_GPUS),
    "multi_gpu_pool": POOL is not None,
    "fp16": bool(USE_FP16),
    "dynamic_batching": True,
    "batch_size_short": int(BATCH_SIZE_SHORT),
    "batch_size_long": int(BATCH_SIZE_LONG),
    "short_max_tokens": int(SHORT_MAX_TOKENS),
    "elapsed_sec": round(time.time() - _T0, 1),
}
(OUT_DIR / "sbert_summary.json").write_text(json.dumps(summary, indent=2))

log("saved files:")
for p in sorted(OUT_DIR.glob("*")):
    if p.is_file():
        log(f"  {p.name:40s} {p.stat().st_size / 1e6:9.1f} MB")
log(f"summary: {json.dumps(summary)}")
log("DONE")
