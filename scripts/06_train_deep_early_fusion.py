#!/usr/bin/env python
"""Step 6: Deep Early Fusion training (Cross Network + MLP).

Trains on sampled implicit interactions. Saves to artifacts/models/deep_early_fusion.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import TRAIN_DIR, VAL_DIR, FACTORS_DIR, MODELS_DIR, NUM_ITEMS, CF_FACTORS, SBERT_DIM
from src.models.early_fusion import DeepEarlyFusion


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cf", default="als")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--samples", type=int, default=100000)
    ap.add_argument("--cross-layers", type=int, default=3)
    args = ap.parse_args()

    from src.models.cf import load_factors
    uf, itf = load_factors(args.cf, dtype="float16", mmap=False)
    uf = np.asarray(uf, dtype=np.float32)
    itf = np.asarray(itf, dtype=np.float32)
    from src.models.sbert import load_sbert
    sbert_emb = load_sbert(mmap=False)

    import pyarrow.parquet as pq, glob, random
    random.seed(42)
    np.random.seed(42)

    parts = sorted(glob.glob(str(VAL_DIR / "*.parquet")))
    dfs = [pq.read_table(p, columns=["user_idx", "item_idx", "rating"]).to_pandas() for p in parts]
    val = pd.concat(dfs, ignore_index=True)
    val = val[val["rating"] >= 4.0]

    pos = val[["user_idx", "item_idx"]].drop_duplicates().sample(
        n=min(args.samples // 2, len(val)), random_state=42)
    neg = pd.DataFrame({
        "user_idx": np.random.choice(pos["user_idx"].to_numpy(), size=len(pos)),
        "item_idx": np.random.randint(0, NUM_ITEMS, size=len(pos)),
    })
    pos["label"] = 1.0
    neg["label"] = 0.0
    data = pd.concat([pos, neg], ignore_index=True).sample(frac=1, random_state=42)

    def build(data):
        us = data["user_idx"].to_numpy(dtype=np.int64)
        its = data["item_idx"].to_numpy(dtype=np.int64)
        ucf = uf[us]
        icf = itf[its]
        isb = sbert_emb[its] if sbert_emb is not None else np.zeros((len(its), SBERT_DIM), dtype=np.float32)
        X = np.concatenate([ucf, icf, isb], axis=1).astype(np.float32)
        y = data["label"].to_numpy(dtype=np.float32)
        return X, y

    X, y = build(data)
    in_dim = X.shape[1]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DeepEarlyFusion(in_dim=in_dim, cross_layers=args.cross_layers, hidden=128).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    ds = TensorDataset(torch.tensor(X), torch.tensor(y))
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True)

    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for xb, yb in tqdm(loader, desc=f"epoch {epoch+1}"):
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
        print(f"epoch {epoch+1} loss {total/len(ds):.4f}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "in_dim": in_dim,
                "cross_layers": args.cross_layers}, MODELS_DIR / "deep_early_fusion.pt")
    print("Saved", MODELS_DIR / "deep_early_fusion.pt")


if __name__ == "__main__":
    main()
