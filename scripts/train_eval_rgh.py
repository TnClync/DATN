#!/usr/bin/env python
"""
Empirical Training and Evaluation of Proposed CF-Anchored Residual Gated Hybrid (RGH)
and Two-Stage LambdaRank according to docs/Ke_hoach_cai_thien_mo_hinh_va_co_so_lap_luan.md.

Protocol:
- Leave-One-Out: 1 positive + 99 sampled negatives from shared catalog (2,464,371 items).
- Strictly leak-free: History built only from Train split.
- Hyperparameter tuning and model checkpoint selection on VALIDATION split.
- Final evaluation reported ONCE on TEST split (holdout).
- External metrics: HR@{5,10,20}, NDCG@{5,10,20}, MRR, P@10, Recall@10.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Constants
CATALOG_SIZE = 2464371
SBERT_DIM = 768
CF_DIM = 32
GRAPH_DIM = 32
SEED = 42

def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def map_ids(values: np.ndarray, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pos = np.searchsorted(ids, values)
    pos = np.clip(pos, 0, len(ids) - 1)
    ok = ids[pos] == values
    return pos, ok

def read_split(directory: Path, columns: list[str]) -> pd.DataFrame:
    parts = sorted(directory.glob("*.parquet"))
    frames = [pq.read_table(p, columns=columns).to_pandas() for p in parts]
    df = pd.concat(frames, ignore_index=True)
    if "rating" in df.columns:
        df = df[df["rating"] >= 4.0]
    return df

def mm_rows(x: np.ndarray) -> np.ndarray:
    lo = x.min(axis=1, keepdims=True)
    hi = x.max(axis=1, keepdims=True)
    diff = hi - lo
    return np.where(diff > 1e-8, (x - lo) / diff, 0.5).astype(np.float32)

def compute_ranking_metrics(pos_arr: np.ndarray) -> dict[str, float]:
    n = len(pos_arr)
    if n == 0:
        return {
            "HR@5": 0.0, "HR@10": 0.0, "HR@20": 0.0,
            "NDCG@5": 0.0, "NDCG@10": 0.0, "NDCG@20": 0.0,
            "MRR": 0.0, "Precision@10": 0.0, "Recall@10": 0.0,
            "n_users": 0
        }
    
    hr5 = float(np.mean(pos_arr < 5))
    hr10 = float(np.mean(pos_arr < 10))
    hr20 = float(np.mean(pos_arr < 20))
    
    ndcg5 = float(np.mean(np.where(pos_arr < 5, 1.0 / np.log2(pos_arr + 2), 0.0)))
    ndcg10 = float(np.mean(np.where(pos_arr < 10, 1.0 / np.log2(pos_arr + 2), 0.0)))
    ndcg20 = float(np.mean(np.where(pos_arr < 20, 1.0 / np.log2(pos_arr + 2), 0.0)))
    
    mrr = float(np.mean(1.0 / (pos_arr + 1)))
    p10 = float(hr10 / 10.0) # exactly 1 positive per user
    r10 = float(hr10)        # recall@10 = 1 if positive in top 10 else 0
    
    return {
        "HR@5": round(hr5, 4),
        "HR@10": round(hr10, 4),
        "HR@20": round(hr20, 4),
        "NDCG@5": round(ndcg5, 4),
        "NDCG@10": round(ndcg10, 4),
        "NDCG@20": round(ndcg20, 4),
        "MRR": round(mrr, 4),
        "Precision@10": round(p10, 4),
        "Recall@10": round(r10, 4),
        "n_users": int(n)
    }

class ResidualGatedHybrid(nn.Module):
    """
    CF-Anchored Residual Gated Hybrid (RGH) as specified in Section 5:
    score(u, i) = a_cf * z_cf + a_gcn * z_gcn + a_txt * z_txt + g(h_u, h_i) * Delta(u, i)
    """
    def __init__(self, in_features: int = 6, hidden_dim: int = 64):
        super().__init__()
        # Learnable branch weights anchored on CF
        self.a_cf = nn.Parameter(torch.tensor(0.60, dtype=torch.float32))
        self.a_gcn = nn.Parameter(torch.tensor(0.30, dtype=torch.float32))
        self.a_txt = nn.Parameter(torch.tensor(0.10, dtype=torch.float32))
        
        # Non-linear cross layers & deep MLP for Delta(u, i)
        self.cross1 = nn.Linear(in_features, in_features)
        self.cross2 = nn.Linear(in_features, in_features)
        self.deep = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Residual Gate: g(h_u, h_i) with negative bias initialization
        # so initially g ~ 0.05, safely preserving the strong CF anchor
        self.gate_net = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        nn.init.constant_(self.gate_net[-1].bias, -2.5)

    def forward(self, feats: torch.Tensor, z_cf: torch.Tensor, z_gcn: torch.Tensor, z_txt: torch.Tensor) -> torch.Tensor:
        # feats: (batch, n_cands, in_features)
        # z_*: (batch, n_cands)
        # 1. CF-Anchored Linear Fusion
        linear_anchor = self.a_cf * z_cf + self.a_gcn * z_gcn + self.a_txt * z_txt
        
        # 2. DCN-v2 Style Feature Crossing
        x0 = feats
        x1 = x0 * torch.sigmoid(self.cross1(x0)) + x0
        x2 = x0 * torch.sigmoid(self.cross2(x1)) + x1
        delta = self.deep(x2).squeeze(-1) # (batch, n_cands)
        
        # 3. Residual Gating
        gate = torch.sigmoid(self.gate_net(feats)).squeeze(-1) # (batch, n_cands)
        
        # 4. Final RGH Score
        return linear_anchor + gate * delta

def train_rgh(model: ResidualGatedHybrid, val_feats: np.ndarray, val_z_cf: np.ndarray, 
              val_z_gcn: np.ndarray, val_z_txt: np.ndarray, val_labels: np.ndarray,
              epochs: int = 15, lr: float = 0.005, batch_size: int = 64, device: str = "cpu"):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    N = len(val_feats)
    # Convert to torch tensors
    t_feats = torch.from_numpy(val_feats).float()
    t_zcf = torch.from_numpy(val_z_cf).float()
    t_zgcn = torch.from_numpy(val_z_gcn).float()
    t_ztxt = torch.from_numpy(val_z_txt).float()
    t_gt = torch.from_numpy(val_labels).long() # position index of positive
    
    dataset = TensorDataset(t_feats, t_zcf, t_zgcn, t_ztxt, t_gt)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model.train()
    for ep in range(epochs):
        total_loss = 0.0
        for b_feats, b_zcf, b_zgcn, b_ztxt, b_gt in loader:
            b_feats = b_feats.to(device)
            b_zcf = b_zcf.to(device)
            b_zgcn = b_zgcn.to(device)
            b_ztxt = b_ztxt.to(device)
            b_gt = b_gt.to(device)
            
            optimizer.zero_grad()
            scores = model(b_feats, b_zcf, b_zgcn, b_ztxt) # (batch, 100)
            
            # Listwise Softmax Loss with temperature tau=0.2
            tau = 0.2
            logits = scores / tau
            loss = nn.functional.cross_entropy(logits, b_gt)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(b_gt)
            
        avg_loss = total_loss / N
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  [RGH Epoch {ep+1:02d}/{epochs}] Loss = {avg_loss:.4f} (a_cf={model.a_cf.item():.3f}, a_gcn={model.a_gcn.item():.3f}, a_txt={model.a_txt.item():.3f})")
    
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default="artifacts/eval_inputs")
    parser.add_argument("--data", default="data")
    parser.add_argument("--out", default="artifacts/results/rgh_evaluation_report.json")
    parser.add_argument("--val-users", type=int, default=2000)
    parser.add_argument("--test-users", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    set_seed(args.seed)
    A = Path(args.artifacts)
    D = Path(args.data)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    t0 = time.time()
    print("=" * 70)
    print("  EMPIRICAL EVALUATION: CF-ANCHORED RESIDUAL GATED HYBRID (RGH)")
    print("=" * 70)

    # 1. Load Factor Artifacts
    print("1. Loading factors and embeddings from", A)
    user_ids = np.load(A / "user_ids.npy")
    item_ids = np.load(A / "item_ids.npy")
    als_uf = np.load(A / "als_user_factors_f16.npy", mmap_mode="r")
    als_itf = np.load(A / "als_item_factors_f16.npy", mmap_mode="r")
    bpr_uf = np.load(A / "bpr_user_factors_f16.npy", mmap_mode="r")
    bpr_itf = np.load(A / "bpr_item_factors_f16.npy", mmap_mode="r")
    graph_uf = np.load(A / "graph_user_factors_f16.npy", mmap_mode="r")
    graph_itf = np.load(A / "graph_item_factors_f16.npy", mmap_mode="r")
    sbert_emb = np.load(A / "sbert_item_embeddings_f16.npy", mmap_mode="r")
    print(f"   Users: {len(user_ids):,}, Items: {len(item_ids):,}, SBERT dim: {sbert_emb.shape[1]}")

    # 2. Load User History for Leak-Free Negatives & Profiles
    hist_db_path = Path("artifacts/user_history.db")
    has_hist_db = hist_db_path.exists()
    con = sqlite3.connect(hist_db_path) if has_hist_db else None
    cur = con.cursor() if con else None

    # Compute item popularity from train
    print("2. Computing item popularity...")
    pop_cache_path = Path("artifacts/item_popularity.npy")
    if pop_cache_path.exists():
        item_pop = np.load(pop_cache_path)
    else:
        # Load train parquet to get popularity
        tr_df = read_split(D / "train", ["item_idx"])
        pos, ok = map_ids(tr_df["item_idx"].to_numpy(np.int64), item_ids)
        counts = np.bincount(pos[ok], minlength=len(item_ids))
        item_pop = np.log1p(counts).astype(np.float32)
        np.save(pop_cache_path, item_pop)
    print(f"   Item popularity cached. Max log pop = {item_pop.max():.2f}")

    # 3. Read Validation & Test Splits
    print("3. Reading Validation and Test splits...")
    val_raw = read_split(D / "validation", ["user_idx", "item_idx", "rating"])
    test_raw = read_split(D / "test", ["user_idx", "item_idx", "rating"])

    # Active users filter
    item_set = set(item_ids.tolist())

    def prep_split_users(df: pd.DataFrame, n_sample: int, seed: int):
        df = df[df["item_idx"].isin(item_set)].drop_duplicates("user_idx", keep="last")
        u_raw = df["user_idx"].to_numpy(np.int64)
        i_raw = df["item_idx"].to_numpy(np.int64)
        cu, oku = map_ids(u_raw, user_ids)
        ci, oki = map_ids(i_raw, item_ids)
        keep = oku & oki
        cu, ci = cu[keep], ci[keep]
        rng = np.random.default_rng(seed)
        sample_idx = rng.choice(len(cu), size=min(n_sample, len(cu)), replace=False)
        return cu[sample_idx], ci[sample_idx]

    cu_val, ci_val = prep_split_users(val_raw, args.val_users, args.seed)
    cu_test, ci_test = prep_split_users(test_raw, args.test_users, args.seed + 1)
    print(f"   Sampled Validation users: {len(cu_val):,}; Test users: {len(cu_test):,}")

    def get_user_history(u_idx: int) -> list[int]:
        if cur:
            # Query by user_ids[u_idx] (raw user_idx)
            raw_u = int(user_ids[u_idx])
            row = cur.execute("SELECT items FROM history WHERE user_idx = ?", (raw_u,)).fetchone()
            if row and row[0]:
                raw_items = np.frombuffer(row[0], dtype=np.int32)
                c_items, ok = map_ids(raw_items.astype(np.int64), item_ids)
                return c_items[ok].tolist()
        return []

    # 4. Construct Candidates (1 positive + 99 negatives)
    def build_candidates(cu, ci, label, seed):
        rng = np.random.default_rng(seed)
        N = len(cu)
        cands = np.empty((N, 100), dtype=np.int64)
        gt_pos = np.empty(N, dtype=np.int64)
        u_sbert_profiles = np.zeros((N, SBERT_DIM), dtype=np.float32)
        
        print(f"   Building {label} candidate sets (1+99) & SBERT user profiles...")
        for k in range(N):
            u = int(cu[k])
            g = int(ci[k])
            hist = get_user_history(u)
            ban = set(hist)
            ban.add(g)
            
            # User SBERT profile: average of train history items
            if hist:
                v = np.asarray(sbert_emb[hist], dtype=np.float32).mean(axis=0)
                norm = np.linalg.norm(v)
                if norm > 1e-8:
                    u_sbert_profiles[k] = v / norm
            
            negs = []
            while len(negs) < 99:
                draws = rng.integers(0, CATALOG_SIZE, size=150)
                for it in draws:
                    it = int(it)
                    if it not in ban and it not in negs:
                        negs.append(it)
                        if len(negs) == 99:
                            break
            
            arr = np.array([g] + negs, dtype=np.int64)
            rng.shuffle(arr)
            cands[k] = arr
            gt_pos[k] = int(np.where(arr == g)[0][0])
            
        return cands, gt_pos, u_sbert_profiles

    val_cands, val_gt, val_u_sbert = build_candidates(cu_val, ci_val, "VAL", args.seed)
    test_cands, test_gt, test_u_sbert = build_candidates(cu_test, ci_test, "TEST", args.seed + 2)

    # 5. Feature Extraction Function
    def extract_scores_and_features(cu, cands, u_sbert):
        N = len(cu)
        # ALS dot product
        u_als = np.asarray(als_uf[cu], dtype=np.float32)
        i_als = np.asarray(als_itf[cands], dtype=np.float32)
        als_scores = np.einsum("bd,bnd->bn", u_als, i_als)
        
        # BPR dot product
        u_bpr = np.asarray(bpr_uf[cu], dtype=np.float32)
        i_bpr = np.asarray(bpr_itf[cands], dtype=np.float32)
        bpr_scores = np.einsum("bd,bnd->bn", u_bpr, i_bpr)
        
        # LightGCN dot product
        u_gcn = np.asarray(graph_uf[cu], dtype=np.float32)
        i_gcn = np.asarray(graph_itf[cands], dtype=np.float32)
        gcn_scores = np.einsum("bd,bnd->bn", u_gcn, i_gcn)
        
        # SBERT cosine similarity
        i_sbert = np.asarray(sbert_emb[cands], dtype=np.float32)
        sbert_scores = np.einsum("bd,bnd->bn", u_sbert, i_sbert)
        
        # Popularity feature
        pop_scores = item_pop[cands]
        
        # Min-Max normalized scores (per user across 100 candidates)
        z_als = mm_rows(als_scores)
        z_bpr = mm_rows(bpr_scores)
        z_gcn = mm_rows(gcn_scores)
        z_txt = mm_rows(sbert_scores)
        z_pop = mm_rows(pop_scores)
        
        # Combined multi-branch feature tensor: (N, 100, 6)
        # [z_als, z_bpr, z_gcn, z_txt, z_pop, als_scores - gcn_scores]
        diff_cf = z_als - z_gcn
        feats = np.stack([z_als, z_bpr, z_gcn, z_txt, z_pop, diff_cf], axis=-1).astype(np.float32)
        
        return {
            "als_raw": als_scores, "bpr_raw": bpr_scores, "gcn_raw": gcn_scores, "sbert_raw": sbert_scores,
            "pop_raw": pop_scores,
            "z_als": z_als, "z_bpr": z_bpr, "z_gcn": z_gcn, "z_txt": z_txt, "z_pop": z_pop,
            "feats": feats
        }

    print("4. Extracting multi-branch scores and normalized features...")
    val_data = extract_scores_and_features(cu_val, val_cands, val_u_sbert)
    test_data = extract_scores_and_features(cu_test, test_cands, test_u_sbert)

    # Helper to evaluate ranking from score matrix
    def rank_from_scores(scores: np.ndarray, gt: np.ndarray) -> np.ndarray:
        # returns position of gt in sorted descending order
        # scores: (N, 100)
        # ranks: descending sort
        order = np.argsort(-scores, axis=1)
        positions = np.empty(len(gt), dtype=np.int64)
        for i in range(len(gt)):
            positions[i] = np.where(order[i] == gt[i])[0][0]
        return positions

    results = {}

    # 6. Evaluate Baselines
    print("\n" + "=" * 70)
    print("5. EVALUATING BASELINES ON TEST HOLDOUT")
    print("=" * 70)

    # 5.1 Popularity
    pos_pop = rank_from_scores(test_data["pop_raw"], test_gt)
    results["Popularity Baseline"] = compute_ranking_metrics(pos_pop)
    print(f"Popularity Baseline: HR@10={results['Popularity Baseline']['HR@10']:.4f}, NDCG@10={results['Popularity Baseline']['NDCG@10']:.4f}")

    # 5.2 Standalone SBERT (Content-Based)
    pos_sb = rank_from_scores(test_data["sbert_raw"], test_gt)
    results["SBERT Alone (CB)"] = compute_ranking_metrics(pos_sb)
    print(f"SBERT Alone (CB):    HR@10={results['SBERT Alone (CB)']['HR@10']:.4f}, NDCG@10={results['SBERT Alone (CB)']['NDCG@10']:.4f}")

    # 5.3 ALS (CF Baseline)
    pos_als = rank_from_scores(test_data["als_raw"], test_gt)
    results["ALS (CF Baseline)"] = compute_ranking_metrics(pos_als)
    print(f"ALS (CF Baseline):   HR@10={results['ALS (CF Baseline)']['HR@10']:.4f}, NDCG@10={results['ALS (CF Baseline)']['NDCG@10']:.4f}")

    # 5.4 BPR (CF Baseline)
    pos_bpr = rank_from_scores(test_data["bpr_raw"], test_gt)
    results["BPR (CF Baseline)"] = compute_ranking_metrics(pos_bpr)
    print(f"BPR (CF Baseline):   HR@10={results['BPR (CF Baseline)']['HR@10']:.4f}, NDCG@10={results['BPR (CF Baseline)']['NDCG@10']:.4f}")

    # 5.5 LightGCN (Graph CF)
    pos_gcn = rank_from_scores(test_data["gcn_raw"], test_gt)
    results["LightGCN (Graph CF)"] = compute_ranking_metrics(pos_gcn)
    print(f"LightGCN (Graph CF): HR@10={results['LightGCN (Graph CF)']['HR@10']:.4f}, NDCG@10={results['LightGCN (Graph CF)']['NDCG@10']:.4f}")

    # 5.6 Late Fusion (Tuned on Validation)
    print("\n   Tuning Late Fusion (ALS + SBERT) alpha on VALIDATION...")
    best_a, best_val_ndcg = 0.8, -1.0
    for a in [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        val_lf = a * val_data["z_als"] + (1 - a) * val_data["z_txt"]
        m = compute_ranking_metrics(rank_from_scores(val_lf, val_gt))
        if m["NDCG@10"] > best_val_ndcg:
            best_val_ndcg = m["NDCG@10"]
            best_a = a
    test_lf = best_a * test_data["z_als"] + (1 - best_a) * test_data["z_txt"]
    results[f"Late Fusion (alpha={best_a})"] = compute_ranking_metrics(rank_from_scores(test_lf, test_gt))
    print(f"Late Fusion (best a={best_a}): HR@10={results[f'Late Fusion (alpha={best_a})']['HR@10']:.4f}, NDCG@10={results[f'Late Fusion (alpha={best_a})']['NDCG@10']:.4f}")

    # 7. Train & Evaluate Proposed Model: CF-Anchored Residual Gated Hybrid (RGH)
    print("\n" + "=" * 70)
    print("6. TRAINING PROPOSED CF-ANCHORED RESIDUAL GATED HYBRID (RGH)")
    print("=" * 70)
    rgh_model = ResidualGatedHybrid(in_features=6, hidden_dim=64)
    train_rgh(rgh_model, val_data["feats"], val_data["z_als"], val_data["z_gcn"], val_data["z_txt"], val_gt,
              epochs=15, lr=0.005, batch_size=64)
    
    # Save trained RGH checkpoint
    torch.save(rgh_model.state_dict(), A / "residual_gated_hybrid.pt")
    
    # Test RGH
    with torch.no_grad():
        rgh_test_scores = rgh_model(
            torch.from_numpy(test_data["feats"]).float(),
            torch.from_numpy(test_data["z_als"]).float(),
            torch.from_numpy(test_data["z_gcn"]).float(),
            torch.from_numpy(test_data["z_txt"]).float()
        ).numpy()
    pos_rgh = rank_from_scores(rgh_test_scores, test_gt)
    results["Proposed RGH (Residual Gated Hybrid)"] = compute_ranking_metrics(pos_rgh)
    print(f"Proposed RGH:        HR@10={results['Proposed RGH (Residual Gated Hybrid)']['HR@10']:.4f}, NDCG@10={results['Proposed RGH (Residual Gated Hybrid)']['NDCG@10']:.4f}, MRR={results['Proposed RGH (Residual Gated Hybrid)']['MRR']:.4f}")

    # 8. Train & Evaluate Two-Stage LightGBM LambdaRank
    print("\n" + "=" * 70)
    print("7. TRAINING PROPOSED TWO-STAGE LAMBDARANK (LIGHTGBM)")
    print("=" * 70)
    # Prepare tabular ranking pairs for LightGBM
    # Each validation user is a query with 100 items; positive has label=1, negatives have label=0
    N_val = len(val_gt)
    X_val = val_data["feats"].reshape(N_val * 100, -1)
    y_val = np.zeros((N_val, 100), dtype=np.int32)
    for i, g in enumerate(val_gt):
        y_val[i, g] = 1
    y_val = y_val.reshape(-1)
    q_val = [100] * N_val

    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        eval_at=[5, 10, 20],
        n_estimators=100,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        random_state=args.seed,
        n_jobs=-1
    )
    ranker.fit(X_val, y_val, group=q_val)
    
    # Score test holdout
    N_test = len(test_gt)
    X_test = test_data["feats"].reshape(N_test * 100, -1)
    lgb_scores = ranker.predict(X_test).reshape(N_test, 100)
    pos_lgb = rank_from_scores(lgb_scores, test_gt)
    results["Proposed Two-Stage LambdaRank (LightGBM)"] = compute_ranking_metrics(pos_lgb)
    print(f"Proposed LambdaRank: HR@10={results['Proposed Two-Stage LambdaRank (LightGBM)']['HR@10']:.4f}, NDCG@10={results['Proposed Two-Stage LambdaRank (LightGBM)']['NDCG@10']:.4f}, MRR={results['Proposed Two-Stage LambdaRank (LightGBM)']['MRR']:.4f}")

    # 9. Ensemble Model
    print("\n" + "=" * 70)
    print("8. ENSEMBLE HYBRID (RGH + LambdaRank + LightGCN)")
    print("=" * 70)
    ens_test_scores = 0.40 * rgh_test_scores + 0.40 * lgb_scores + 0.20 * test_data["z_gcn"]
    pos_ens = rank_from_scores(ens_test_scores, test_gt)
    results["Proposed Stacked Ensemble"] = compute_ranking_metrics(pos_ens)
    print(f"Stacked Ensemble:    HR@10={results['Proposed Stacked Ensemble']['HR@10']:.4f}, NDCG@10={results['Proposed Stacked Ensemble']['NDCG@10']:.4f}, MRR={results['Proposed Stacked Ensemble']['MRR']:.4f}")

    # 10. Summary and Comparison Table
    print("\n" + "=" * 80)
    print(f"{'MODEL':<42} | {'HR@5':<7} | {'HR@10':<7} | {'HR@20':<7} | {'NDCG@10':<8} | {'MRR':<7}")
    print("-" * 80)
    for model_name, m in results.items():
        print(f"{model_name:<42} | {m['HR@5']:<7.4f} | {m['HR@10']:<7.4f} | {m['HR@20']:<7.4f} | {m['NDCG@10']:<8.4f} | {m['MRR']:<7.4f}")
    print("=" * 80)

    # Save to JSON
    summary_report = {
        "protocol": "Strict Leave-One-Out (1 Positive from Test split + 99 Negatives), leak-free history from Train, hyperparams tuned on Validation",
        "n_val_users": args.val_users,
        "n_test_users": args.test_users,
        "catalog_size": CATALOG_SIZE,
        "elapsed_seconds": round(time.time() - t0, 2),
        "results": results
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2)
    print(f"\nWrote empirical results to: {out_path}")

if __name__ == "__main__":
    main()
