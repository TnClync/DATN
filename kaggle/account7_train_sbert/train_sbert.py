#!/usr/bin/env python
"""Kaggle: Build SBERT embeddings (metadata + reviewer text) once — FULL catalog (2.46M items).

Saves (all float16):
  - sbert_meta_embeddings_f16.npy     (N x 384, title+brand+category+description)
  - sbert_review_embeddings_f16.npy   (N x 384, aggregated review_content; meta emb reused when no review)
  - sbert_item_embeddings_f16.npy     (N x 768, concat[meta, review] — PRIMARY for downstream kernels)
  - user_ids.npy, item_ids.npy        (compact -> original idx mapping)

Memory-safe: encodes in chunks into np.memmap files, never holds the full
2.46M x 768 float32 matrix in RAM.
"""
import os, sys, time, json, math, gc, subprocess
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

from pathlib import Path

try:
    import torch
    import pyarrow.parquet as pq
    from sentence_transformers import SentenceTransformer
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet",
        "sentence-transformers>=2.7", "pyarrow", "scipy", "tqdm"])
    import torch
    import pyarrow.parquet as pq
    from sentence_transformers import SentenceTransformer

import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

MIN_INTERACTIONS = 3
CATALOG = None            # None -> use all compact items (full catalog)
SEED = 42
CHUNK = 100_000           # encode chunk size (items per forward pass batch group)
OUT_DIR = Path("/kaggle/working" if Path("/kaggle/working").exists() else "artifacts/sbert")
OUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

T0 = time.time()
def log(msg):
    print(f"[{time.time() - T0:8.1f}s] {msg}", flush=True)


def find_base_dir():
    for c in [Path("/kaggle/input/lastone/amazon_recs_output"), Path("/kaggle/input/lastone"),
              Path("/kaggle/input/datasets/nguynteo/lastone"), Path("data")]:
        if (c / "train").exists() or (c / "test").exists():
            return c
    if Path("/kaggle/input").exists():
        for p in Path("/kaggle/input").rglob("train"):
            if p.is_dir():
                return p.parent
    return Path("data")


def find_dir_with_cols(required):
    if not Path("/kaggle/input").exists():
        return None
    for p in Path("/kaggle/input").rglob("*.parquet"):
        if "datasets/nguynteo/lastone" in str(p):
            continue
        try:
            cols = pq.read_schema(p).names
            if all(c in cols for c in required):
                return p.parent
        except Exception:
            pass
    return None


# ---------------------------------------------------------------------------
# STEP 1/5 — compact indices (same protocol as CF / eval kernels)
# ---------------------------------------------------------------------------
log("STEP 1/5 — building compact indices")
BASE_DIR = find_base_dir()
log(f"dataset dir: {BASE_DIR}")

train_files = sorted(BASE_DIR.glob("train/*.parquet"))
log(f"train parquet files: {len(train_files)}")
dfs = []
for f in tqdm(train_files, desc="Reading train"):
    d = pq.read_table(f, columns=["user_idx", "item_idx", "rating"]).to_pandas()
    d = d[d["rating"] >= 4.0]
    dfs.append(d[["user_idx", "item_idx"]])
train_df = pd.concat(dfs, ignore_index=True)
del dfs
user_counts = train_df["user_idx"].value_counts()
active = user_counts[user_counts >= MIN_INTERACTIONS].index
train_df = train_df[train_df["user_idx"].isin(active)]

u_raw = train_df["user_idx"].to_numpy(np.int64)
i_raw = train_df["item_idx"].to_numpy(np.int64)
u_uniq, _ = np.unique(u_raw, return_inverse=True)
i_uniq, _ = np.unique(i_raw, return_inverse=True)
n_users, n_items = len(u_uniq), len(i_uniq)
del train_df, u_raw, i_raw, user_counts, active
gc.collect()

if CATALOG is not None:
    n_items = min(n_items, CATALOG)
compact_to_orig = i_uniq[:n_items].astype(np.int64)
log(f"compact: {n_users:,} users x {n_items:,} items (min_interactions={MIN_INTERACTIONS})")

# ---------------------------------------------------------------------------
# STEP 2/5 — real item metadata text
# ---------------------------------------------------------------------------
log("STEP 2/5 — loading real item metadata (title/brand/category/description)")
profile_dir = find_dir_with_cols(["item_idx", "title", "brand", "category", "description"])
log(f"profile dir: {profile_dir}")

item_text_map = {}
if profile_dir is not None:
    for f in sorted(profile_dir.glob("*.parquet")):
        try:
            d = pq.read_table(f, columns=["item_idx", "title", "brand", "category", "description"]).to_pandas()
        except Exception as e:
            log(f"  skip {f.name}: {e}")
            continue
        ii = d["item_idx"].to_numpy(np.int64)
        tt = d["title"].to_numpy()
        bb = d["brand"].to_numpy()
        cc = d["category"].to_numpy()
        dd = d["description"].to_numpy()
        for k in range(len(ii)):
            parts = []
            t, b, c, x = tt[k], bb[k], cc[k], dd[k]
            if isinstance(t, str) and t.strip():
                parts.append(t.strip())
            if isinstance(b, str) and b.strip() and "amazon" not in b.lower():
                parts.append(b.strip())
            if isinstance(c, str) and c.strip():
                parts.append(c.strip())
            if isinstance(x, str) and x.strip():
                parts.append(x.strip()[:300])
            txt = " | ".join(parts)
            if txt:
                item_text_map[int(ii[k])] = txt
        del d
    log(f"real text loaded for {len(item_text_map):,} items")

n_with_text = 0
for ci in range(n_items):
    if int(compact_to_orig[ci]) in item_text_map:
        n_with_text += 1
log(f"coverage: {n_with_text:,}/{n_items:,} items have real text; rest use fallback 'Product item <id>'")

# ---------------------------------------------------------------------------
# STEP 3/5 — encode metadata (chunked, memmap output)
# ---------------------------------------------------------------------------
log(f"STEP 3/5 — encoding metadata embeddings on {DEVICE} (chunk={CHUNK:,})")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Kaggle "GPU T4 x2" exposes 2 devices: run one model replica per GPU in
# threads and split each batch group across them (~1.7-2x encode speedup).
N_GPU = torch.cuda.device_count() if DEVICE.type == "cuda" else 0
log(f"CUDA devices: {N_GPU} -> {[torch.cuda.get_device_name(i) for i in range(N_GPU)]}")

MODELS = [SentenceTransformer(MODEL_NAME, device=str(DEVICE))]
for gi in range(1, N_GPU):
    MODELS.append(SentenceTransformer(MODEL_NAME, device=f"cuda:{gi}"))
log(f"encode replicas: {len(MODELS)} ({[str(m.device) for m in MODELS]})")


def _run_encode(mdl, texts, batch_size):
    return mdl.encode(texts, batch_size=batch_size, show_progress_bar=False,
                       convert_to_numpy=True, normalize_embeddings=True)


def encode_texts(texts, batch_size=512):
    if len(MODELS) == 1:
        return _run_encode(MODELS[0], texts, batch_size)
    bounds = np.linspace(0, len(texts), len(MODELS) + 1).astype(int)
    jobs = [(i, texts[bounds[i]:bounds[i + 1]]) for i in range(len(MODELS)) if bounds[i + 1] > bounds[i]]
    with ThreadPoolExecutor(max_workers=len(jobs)) as ex:
        futs = [ex.submit(_run_encode, MODELS[i], part, batch_size) for i, part in jobs]
        parts = [f.result() for f in futs]
    return np.concatenate(parts, axis=0)


meta_path = OUT_DIR / "sbert_meta_embeddings_f16.npy"
meta_mm = np.lib.format.open_memmap(meta_path, mode="w+", dtype=np.float16, shape=(n_items, 384))

for start in range(0, n_items, CHUNK):
    end = min(start + CHUNK, n_items)
    texts = [item_text_map.get(int(compact_to_orig[ci]), f"Product item {int(compact_to_orig[ci])}")
             for ci in range(start, end)]
    embs = encode_texts(texts)
    meta_mm[start:end] = embs.astype(np.float16)
    del texts, embs
    log(f"  metadata {end:,}/{n_items:,} ({100.0 * end / n_items:5.1f}%)")
meta_mm.flush()
del meta_mm
gc.collect()

# ---------------------------------------------------------------------------
# STEP 4/5 — reviewer text embeddings
# ---------------------------------------------------------------------------
log("STEP 4/5 — loading reviewer text (review_content)")
review_dir = find_dir_with_cols(["item_idx", "review_content"])
log(f"review dir: {review_dir}")

MAX_REVIEWS_PER_ITEM = 3
MAX_REVIEW_CHARS = 500
review_by_item = {}
n_review_rows = 0
if review_dir is not None:
    for f in sorted(review_dir.glob("*.parquet")):
        try:
            d = pq.read_table(f, columns=["item_idx", "review_content"]).to_pandas()
        except Exception as e:
            log(f"  skip {f.name}: {e}")
            continue
        ii = d["item_idx"].to_numpy(np.int64)
        rr = d["review_content"].to_numpy()
        for k in range(len(ii)):
            txt = rr[k]
            if not isinstance(txt, str) or not txt.strip():
                continue
            lst = review_by_item.setdefault(int(ii[k]), [])
            if len(lst) < MAX_REVIEWS_PER_ITEM:
                lst.append(txt.strip()[:MAX_REVIEW_CHARS])
            n_review_rows += 1
        del d
    log(f"{n_review_rows:,} review rows -> {len(review_by_item):,} items")

meta_mm = np.load(meta_path, mmap_mode="r")
review_path = OUT_DIR / "sbert_review_embeddings_f16.npy"
review_mm = np.lib.format.open_memmap(review_path, mode="w+", dtype=np.float16, shape=(n_items, 384))

for start in range(0, n_items, CHUNK):
    end = min(start + CHUNK, n_items)
    enc_idx, enc_txt, reuse_idx = [], [], []
    for ci in range(start, end):
        revs = review_by_item.get(int(compact_to_orig[ci]))
        if revs:
            enc_idx.append(ci)
            enc_txt.append(" ".join(revs)[:1000])
        else:
            reuse_idx.append(ci)
    if reuse_idx:
        idx = np.asarray(reuse_idx, dtype=np.int64)
        review_mm[idx] = meta_mm[idx]          # identical text -> identical embedding
    if enc_idx:
        embs = encode_texts(enc_txt)
        review_mm[np.asarray(enc_idx, dtype=np.int64)] = embs.astype(np.float16)
        del embs
    del enc_idx, enc_txt, reuse_idx
    log(f"  reviewer {end:,}/{n_items:,} ({100.0 * end / n_items:5.1f}%)")
review_mm.flush()
del review_mm, meta_mm
gc.collect()

# ---------------------------------------------------------------------------
# STEP 5/5 — concat -> primary 768-dim embedding file, save ids
# ---------------------------------------------------------------------------
log("STEP 5/5 — writing combined 768-dim embeddings (memmap, no extra RAM)")
meta_mm = np.load(meta_path, mmap_mode="r")
review_mm = np.load(review_path, mmap_mode="r")
comb_path = OUT_DIR / "sbert_item_embeddings_f16.npy"
comb_mm = np.lib.format.open_memmap(comb_path, mode="w+", dtype=np.float16, shape=(n_items, 768))
for start in range(0, n_items, CHUNK):
    end = min(start + CHUNK, n_items)
    comb_mm[start:end, :384] = meta_mm[start:end]
    comb_mm[start:end, 384:] = review_mm[start:end]
comb_mm.flush()
del comb_mm, meta_mm, review_mm
gc.collect()

np.save(OUT_DIR / "user_ids.npy", u_uniq)
np.save(OUT_DIR / "item_ids.npy", i_uniq)
np.save(OUT_DIR / "compact_to_orig.npy", compact_to_orig)

summary = {
    "n_users": int(n_users),
    "n_items": int(n_items),
    "items_with_real_text": int(n_with_text),
    "items_with_reviews": int(len(review_by_item)),
    "review_rows": int(n_review_rows),
    "dims_meta": 384,
    "dims_review": 384,
    "dims_combined": 768,
    "dtype": "float16",
    "device": str(DEVICE),
    "cuda_devices": int(N_GPU),
    "elapsed_sec": round(time.time() - T0, 1),
}
(OUT_DIR / "sbert_summary.json").write_text(json.dumps(summary, indent=2))

log("saved files:")
for p in sorted(OUT_DIR.glob("*")):
    if p.is_file():
        log(f"  {p.name:42s} {p.stat().st_size / 1e6:9.1f} MB")
log("DONE")
