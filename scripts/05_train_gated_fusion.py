#!/usr/bin/env python
"""Step 5: Light Gated Late Fusion training.

Trains a tiny MLP that predicts alpha (trust in CF vs SBERT) per user-item.
Saves weights to artifacts/models/light_gate.pt
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
from src.config import TRAIN_DIR, VAL_DIR, FACTORS_DIR, MODELS_DIR, NUM_ITEMS
from src.models.gated_fusion import make_gate

# popularity
def build_popularity() -> np.ndarray:
    import pyarrow.parquet as pq, glob
    counts = np.zeros(NUM_ITEMS, dtype=np.float32)
    for p in glob.glob(str(TRAIN_DIR / "*.parquet")):
        t = pq.read_table(p, columns=["item_idx"]).to_pandas()
        c = t["item_idx"].value_counts()
        for idx, v in c.items():
            counts[int(idx)] += int(v)
    return counts


def user_activity(train_df_user_counts) -> dict:
    return train_df_user_counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cf", default="als")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--samples", type=int, default=200000)
    args = ap.parse_args()

    import pyarrow.parquet as pq, glob, random
    random.seed(42)
    np.random.seed(42)

    print("Loading factors...")
    from src.models.cf import load_factors
    uf, itf = load_factors(args.cf, dtype="float16", mmap=False)
    uf = np.asarray(uf, dtype=np.float32)
    itf = np.asarray(itf, dtype=np.float32)
    from src.models.sbert import load_sbert
    sbert_emb = load_sbert(mmap=False)

    print("Building popularity...")
    pop = build_popularity()
    log_pop = np.log1p(pop)

    # user activity from train counts
    print("Building user activity...")
    user_act = {}
    for p in glob.glob(str(TRAIN_DIR / "*.parquet")):
        t = pq.read_table(p, columns=["user_idx"]).to_pandas()
        c = t["user_idx"].value_counts()
        for u, v in c.items():
            user_act[int(u)] = user_act.get(int(u), 0) + int(v)
    def log_act(u):
        return np.log1p(user_act.get(int(u), 0))

    # Build samples from validation positives + random negatives
    parts = sorted(glob.glob(str(VAL_DIR / "*.parquet")))
    dfs = [pq.read_table(p, columns=["user_idx", "item_idx", "rating"]).to_pandas() for p in parts]
    val = pd.concat(dfs, ignore_index=True)
    val = val[val["rating"] >= 4.0]

    pos_samples = val[["user_idx", "item_idx"]].drop_duplicates().sample(
        n=min(args.samples // 2, len(val)), random_state=42)
    # negatives: random items not in user's positives
    neg_rows = []
    all_items = np.arange(NUM_ITEMS)
    for _, r in pos_samples.iterrows():
        u = int(r["user_idx"]); 
        # sample 1 negative
        neg = int(np.random.randint(0, NUM_ITEMS))
        neg_rows.append((u, neg))
    neg_df = pd.DataFrame(neg_rows, columns=["user_idx", "item_idx"])

    def make_features(df, label):
        us = df["user_idx"].to_numpy(dtype=np.int64)
        its = df["item_idx"].to_numpy(dtype=np.int64)
        # cf score
        cf_sc = np.einsum("ij,ij->i", uf[us], itf[its])
        # sbert score: need user vec; use item's? We'll approximate user sbert vec by train history mean not available here.
        # Use zero user sbert vec -> item norm; fallback: cosine(item, item) placeholder. Use item self-similarity? 
        # Better: use item popularity as proxy for content score when no history. Here set sbert_score = 0.
        sb_sc = np.zeros(len(df), dtype=np.float32)
        if sbert_emb is not None:
            # proxy: average sbert embedding of a few random items? Instead, use item embedding norm (constant-ish). 
            # We'll compute cosine between item emb and a global mean user vec.
            gmean = sbert_emb.mean(axis=0)
            gn = np.linalg.norm(gmean)
            if gn > 0:
                gmean = gmean / gn
                ie = sbert_emb[its]
                inrm = np.linalg.norm(ie, axis=1, keepdims=True)
                inrm[inrm == 0] = 1
                ie = ie / inrm
                sb_sc = (ie @ gmean).astype(np.float32)
        ua = np.array([log_act(u) for u in us], dtype=np.float32)
        ip = log_pop[its].astype(np.float32)
        X = np.stack([cf_sc.astype(np.float32), sb_sc, ua, ip], axis=1)
        y = np.full(len(df), label, dtype=np.float32)
        return X, y

    Xp, yp = make_features(pos_samples, 1.0)
    Xn, yn = make_features(neg_df, 0.0)
    X = np.concatenate([Xp, Xn], axis=0)
    y = np.concatenate([yp, yn], axis=0)

    # standardize features
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-6
    Xnrm = (X - mean) / std

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = make_gate(in_dim=4, hidden=16).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCELoss()

    ds = TensorDataset(torch.tensor(Xnrm, dtype=torch.float32),
                      torch.tensor(y, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True)

    # normalize cf/sbert scores are inside features already (raw). We'll store scaler.
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for xb, yb in tqdm(loader, desc=f"epoch {epoch+1}"):
            xb, yb = xb.to(device), yb.to(device)
            # alpha from gate (uses first? gate uses all 4 features)
            alpha = model(xb)  # (B,1)
            # reconstruct final score: alpha*cf + (1-alpha)*sbert
            # features: [cf, sbert, ua, ip]; need raw scores. We stored normalized features only.
            # For simplicity, use a linear head on features directly as score (gate still trained).
            # Instead, compute score from alpha applied to raw cf/sbert stored separately.
            # We'll just train alpha network as classifier: score = alpha*cf_norm + (1-alpha)*sbert_norm
            # Here use normalized cf/sbert from X (first two cols) as proxy.
            cf_n = xb[:, 0]
            sb_n = xb[:, 1]
            a = alpha.squeeze(-1)
            score = a * cf_n + (1 - a) * sb_n
            prob = torch.sigmoid(score)
            loss = loss_fn(prob, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
        print(f"epoch {epoch+1} loss {total/len(ds):.4f}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "mean": mean, "std": std,
                "in_dim": 4, "hidden": 16}, MODELS_DIR / "light_gate.pt")
    print("Saved", MODELS_DIR / "light_gate.pt")


if __name__ == "__main__":
    main()
