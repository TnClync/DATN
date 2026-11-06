#!/usr/bin/env python
"""Step 2: Build SBERT item embeddings + FAISS index.

Synthesizes item text from available metadata (domain). If data/item_text.parquet
exists (columns: item_idx, text), it is used instead.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import TRAIN_DIR, ITEM_MAP_DIR, FACTORS_DIR, SBERT_DIR, FAISS_DIR, NUM_ITEMS
from src.models.sbert import (load_item_texts, build_sbert_embeddings,
                               build_faiss_index, save_sbert)


def build_item_domain_text() -> list[str]:
    # most common domain per item from train
    import glob
    parts = sorted(glob.glob(str(TRAIN_DIR / "*.parquet")))
    domain_by_item = {}
    counts = {}
    for p in tqdm(parts, desc="reading train for domain"):
        t = pq.read_table(p, columns=["item_idx", "domain"])
        d = t.to_pandas()
        for it, dom in zip(d["item_idx"].to_numpy(), d["domain"].to_list()):
            if it not in counts:
                counts[it] = {}
            counts[it][dom] = counts[it].get(dom, 0) + 1
    for it, c in counts.items():
        domain_by_item[it] = max(c, key=c.get)
    texts = []
    for i in range(NUM_ITEMS):
        dom = domain_by_item.get(i, "product")
        texts.append(f"{dom} product item {i}")
    return texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--index-type", default="hnsw", choices=["hnsw", "flat", "ivfpq"])
    args = ap.parse_args()

    text_path = ROOT / "data" / "item_text.parquet"
    if text_path.exists():
        print("Using item_text.parquet")
        texts = load_item_texts(text_path)
    else:
        print("No item_text.parquet; synthesizing from domain")
        texts = build_item_domain_text()

    print(f"Encoding {len(texts)} items...")
    emb = build_sbert_embeddings(texts, model_name=args.model_name,
                                 batch_size=args.batch_size, device=args.device)
    print(f"embeddings: {emb.shape}")

    print("Building FAISS index...")
    index = build_faiss_index(emb, index_type=args.index_type)
    save_sbert(emb, index)
    print("Saved SBERT + FAISS.")


if __name__ == "__main__":
    main()
