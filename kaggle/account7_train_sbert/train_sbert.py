#!/usr/bin/env python
"""Kaggle: Build SBERT embeddings (metadata + reviewer text) once.

Saves:
  - sbert_meta_embeddings_f16.npy     (384-dim, title+brand+category+description)
  - sbert_review_embeddings_f16.npy   (384-dim, aggregated review_content)
  - sbert_combined_embeddings_f16.npy (768-dim, concat)
  - sbert_item_embeddings_f16.npy     (768-dim combined, primary)
  - user_ids.npy, item_ids.npy
"""
import os, sys, time, json, math, gc, subprocess
from pathlib import Path
try:
    import torch, faiss, pyarrow.parquet as pq, sentence_transformers
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet",
        "sentence-transformers>=2.7", "faiss-cpu", "pyarrow", "scipy", "tqdm"])
    import torch, faiss, pyarrow.parquet as pq, sentence_transformers
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

MIN_INTERACTIONS = 3
CATALOG = 200_000
SEED = 42
OUT_DIR = Path("/kaggle/working" if Path("/kaggle/working").exists() else "artifacts")
OUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def find_item_profile_dir():
    if Path("/kaggle/input").exists():
        for p in Path("/kaggle/input").rglob("*.parquet"):
            try:
                import pyarrow.parquet as pq
                cols = pq.read_schema(p).names
                if "title" in cols and "item_idx" in cols:
                    return p.parent
            except Exception:
                pass
    return None


def find_review_dir():
    if Path("/kaggle/input").exists():
        for p in Path("/kaggle/input").rglob("*.parquet"):
            try:
                import pyarrow.parquet as pq
                cols = pq.read_schema(p).names
                if "review_content" in cols and "item_idx" in cols:
                    return p.parent
            except Exception:
                pass
    return None


BASE_DIR = find_base_dir()
print("Dataset:", BASE_DIR)

# ---- Compact indices ----
train_files = sorted(BASE_DIR.glob("train/*.parquet"))
dfs = []
for f in tqdm(train_files, desc="Reading train"):
    d = pq.read_table(f, columns=["user_idx", "item_idx", "rating"]).to_pandas()
    d = d[d["rating"] >= 4.0]
    dfs.append(d[["user_idx", "item_idx"]])
train_df = pd.concat(dfs, ignore_index=True)
user_counts = train_df["user_idx"].value_counts()
active = user_counts[user_counts >= MIN_INTERACTIONS].index
train_df = train_df[train_df["user_idx"].isin(active)]
u_raw = train_df["user_idx"].to_numpy(np.int64)
i_raw = train_df["item_idx"].to_numpy(np.int64)
u_uniq, u_inv = np.unique(u_raw, return_inverse=True)
i_uniq, i_inv = np.unique(i_raw, return_inverse=True)
n_users, n_items = len(u_uniq), len(i_uniq)
SBERT_COVERAGE = min(n_items, CATALOG)
compact_to_orig = {idx: int(i_uniq[idx]) for idx in range(n_items)}
print(f"Compact: {n_users:,} users x {n_items:,} items, coverage {SBERT_COVERAGE:,}")


# ---- SBERT METADATA ----
print("\n" + "=" * 50)
print("SBERT METADATA (title+brand+category+description)")
print("=" * 50)
profile_dir = find_item_profile_dir()
print("Profile dir:", profile_dir)
item_text_map = {}
if profile_dir is not None:
    for f in sorted(profile_dir.glob("*.parquet")):
        try:
            t = pq.read_table(f, columns=["item_idx", "title", "brand", "category", "description"])
            d = t.to_pandas()
            for _, r in d.iterrows():
                parts = []
                if isinstance(r.get("title"), str) and r["title"].strip():
                    parts.append(r["title"].strip())
                if isinstance(r.get("brand"), str) and r["brand"].strip() and "amazon" not in r["brand"].lower():
                    parts.append(r["brand"].strip())
                if isinstance(r.get("category"), str) and r["category"].strip():
                    parts.append(r["category"].strip())
                desc = r.get("description")
                if isinstance(desc, str) and desc.strip():
                    parts.append(desc.strip()[:300])
                txt = " | ".join(parts)
                if txt:
                    item_text_map[int(r["item_idx"])] = txt
        except Exception as e:
            print("profile read err", f, e)
print(f"Loaded real text for {len(item_text_map):,} items")

meta_texts = []
for ci in range(SBERT_COVERAGE):
    orig = compact_to_orig[ci]
    meta_texts.append(item_text_map.get(orig, f"Product item {orig}"))

print(f"Encoding {len(meta_texts)} items (metadata)...")
sbert_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=str(DEVICE))
meta_embs = sbert_model.encode(meta_texts, batch_size=512, show_progress_bar=True,
                               convert_to_numpy=True, normalize_embeddings=True)
item_sbert_meta = np.zeros((n_items, 384), dtype=np.float32)
item_sbert_meta[:SBERT_COVERAGE] = meta_embs
del meta_embs
gc.collect()


# ---- SBERT REVIEWER TEXT ----
print("\n" + "=" * 50)
print("SBERT REVIEWER TEXT (aggregated review_content)")
print("=" * 50)
review_dir = find_review_dir()
print("Review dir:", review_dir)
review_by_item = {}
if review_dir is not None:
    for f in sorted(review_dir.glob("*.parquet")):
        try:
            t = pq.read_table(f, columns=["item_idx", "review_content"])
            d = t.to_pandas()
            for _, r in d.iterrows():
                ci_raw = int(r["item_idx"])
                # map original item_idx -> compact
                # (review_dir uses original item_idx; need compact index)
                txt = r.get("review_content")
                if isinstance(txt, str) and txt.strip():
                    review_by_item.setdefault(ci_raw, []).append(txt.strip()[:500])
        except Exception as e:
            print("review read err", f, e)
print(f"Items with reviews (raw idx): {len(review_by_item):,}")

# Map original item_idx -> compact index
orig_to_compact = {int(i_uniq[idx]): idx for idx in range(n_items)}

review_texts = []
for ci in range(SBERT_COVERAGE):
    orig = compact_to_orig[ci]
    revs = review_by_item.get(orig, [])
    if revs:
        txt = " ".join(revs[:3])[:1000]
    else:
        txt = meta_texts[ci]  # fallback to metadata text
    review_texts.append(txt)

print(f"Encoding {len(review_texts)} items (reviewer)...")
review_embs = sbert_model.encode(review_texts, batch_size=512, show_progress_bar=True,
                                 convert_to_numpy=True, normalize_embeddings=True)
item_sbert_review = np.zeros((n_items, 384), dtype=np.float32)
item_sbert_review[:SBERT_COVERAGE] = review_embs
del review_embs
gc.collect()

# ---- Combined (768-dim) ----
item_sbert_combined = np.concatenate([item_sbert_meta, item_sbert_review], axis=1).astype(np.float32)

# ---- Save ----
np.save(OUT_DIR / "sbert_meta_embeddings_f16.npy", item_sbert_meta.astype(np.float16))
np.save(OUT_DIR / "sbert_review_embeddings_f16.npy", item_sbert_review.astype(np.float16))
np.save(OUT_DIR / "sbert_combined_embeddings_f16.npy", item_sbert_combined.astype(np.float16))
np.save(OUT_DIR / "sbert_item_embeddings_f16.npy", item_sbert_combined.astype(np.float16))  # primary = combined
np.save(OUT_DIR / "user_ids.npy", u_uniq)
np.save(OUT_DIR / "item_ids.npy", i_uniq)

print(f"\nSaved:")
print(f"  meta:     {item_sbert_meta.shape}")
print(f"  review:   {item_sbert_review.shape}")
print(f"  combined: {item_sbert_combined.shape}")
print("DONE")
PY