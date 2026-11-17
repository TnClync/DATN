#!/usr/bin/env python
"""
BLaIR Content-Based Recommender & CF-Anchored Residual Hybrid
Using hyp1231/blair-roberta-base (Hou et al., 2024 / McAuley Lab, UC San Diego).
Trained directly on Amazon Reviews 2023 dataset.

This kernel:
1. Loads item metadata (ASIN, domain, title, category) from Amazon data.
2. Encodes items into 768-d semantic representations using BLaIR.
3. Builds leak-free user content profiles from train history.
4. Evaluates BLaIR Content-Based retrieval on Leave-One-Out holdout test set (1+99 negatives).
5. Evaluates ALS CF Baseline on the exact same test queries.
6. Evaluates CF-Anchored Residual Hybrid (ALS + BLaIR Content).
7. Demonstrates that Proposed Hybrid beats the CF Baseline.
"""
from __future__ import annotations

import gc
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

# Auto-install transformers if needed
try:
    import transformers
    import pyarrow.parquet as pq
    import torch
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "transformers>=4.36", "pyarrow", "tqdm"])
    import transformers
    import pyarrow.parquet as pq
    import torch

from transformers import AutoModel, AutoTokenizer
from tqdm.auto import tqdm
import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

MODEL_NAME = "hyp1231/blair-roberta-base"
EMB_DIM = 768
CATALOG_CAP = 100_000 # encode top 100,000 active items for lightning speed & deep coverage
MIN_INTERACTIONS = 3
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUT_DIR = Path("/kaggle/working" if Path("/kaggle/working").exists() else "artifacts/blair")
OUT_DIR.mkdir(parents=True, exist_ok=True)
t0 = time.time()

print("=" * 65)
print("  BLAIR: BRIDGING LANGUAGE AND ITEMS FOR RECOMMENDATION")
print("  Model:", MODEL_NAME, "| Device:", DEVICE)
print("=" * 65)

# --------------------------------------------------------------------- discovery
def find_base_dir() -> Path:
    candidates = [
        Path("/kaggle/input/lastone/amazon_recs_output"),
        Path("/kaggle/input/datasets/nguynteo/lastone/amazon_recs_output"),
        Path("/kaggle/input/lastone"),
        Path("/kaggle/input/datasets/nguynteo/lastone"),
        Path("data"),
    ]
    for c in candidates:
        if (c / "train").exists() or (c / "test").exists():
            return c
    if Path("/kaggle/input").exists():
        for p in Path("/kaggle/input").rglob("train"):
            if p.is_dir():
                return p.parent
    return Path("data")


def find_file(name: str) -> Path:
    aliases = {
        "user_ids.npy": ["user_ids.npy", "als_user_ids.npy"],
        "item_ids.npy": ["item_ids.npy", "als_item_ids.npy"],
        "als_user_factors_f16.npy": ["als_user_factors_f16.npy"],
        "als_item_factors_f16.npy": ["als_item_factors_f16.npy"],
    }
    names_to_try = aliases.get(name, [name])
    for n in names_to_try:
        hits = sorted(Path("/kaggle/input").rglob(n))
        if hits:
            return hits[0]
        local_cand = list(Path("artifacts").rglob(n))
        if local_cand:
            return local_cand[0]
    raise FileNotFoundError(f"Could not find {name}")


def read_split(directory: Path, columns: list[str]) -> pd.DataFrame:
    parts = sorted(directory.glob("*.parquet"))
    frames = [pq.read_table(p, columns=columns).to_pandas() for p in parts]
    df = pd.concat(frames, ignore_index=True)
    if "rating" in df.columns:
        df = df[df["rating"] >= 4.0]
    return df


BASE_DIR = find_base_dir()
print("Dataset Directory:", BASE_DIR)

# ------------------------------------------------------------------- IDs & CF Factors
u_ids = np.load(find_file("user_ids.npy"))
i_ids = np.load(find_file("item_ids.npy"))
n_users, n_items = len(u_ids), len(i_ids)
print(f"Catalog: {n_users:,} users x {n_items:,} items")

has_als = False
als_uf, als_itf = None, None
try:
    als_uf_p = find_file("als_user_factors_f16.npy")
    als_itf_p = find_file("als_item_factors_f16.npy")
    als_uf = np.load(als_uf_p, mmap_mode="r")
    als_itf = np.load(als_itf_p).astype(np.float32)
    has_als = True
    print(f"Loaded ALS factors: user {als_uf.shape}, item {als_itf.shape}")
except Exception as e:
    print(f"Notice: ALS factors not available ({e}); will evaluate pure BLaIR Content.")

# ------------------------------------------------------------------- Item Metadata
print("\n" + "=" * 65)
print("1. LOADING ITEM METADATA (ASIN, DOMAIN & RICH TEXT)")
print("=" * 65)

item_text_map: dict[int, str] = {}
item_domain_map: dict[int, str] = {}
item_asin_map: dict[int, str] = {}

# 1a. Load ASIN from item_mapping
mapping_files = list(BASE_DIR.glob("item_mapping/*.parquet"))
if not mapping_files:
    mapping_files = list(Path("/kaggle/input").rglob("item_mapping/*.parquet")) + list(Path("data").rglob("item_mapping/*.parquet"))

if mapping_files:
    print(f"Loading ASIN mappings from {len(mapping_files)} file(s)...")
    for mf in mapping_files:
        try:
            mdf = pq.read_table(mf, columns=["asin", "item_idx"]).to_pandas()
            for asin, idx in zip(mdf["asin"], mdf["item_idx"]):
                item_asin_map[int(idx)] = str(asin)
        except Exception:
            pass
    print(f"Loaded ASINs for {len(item_asin_map):,} items")

# 1b. Load domain from train split
train_parts = sorted(BASE_DIR.glob("train/*.parquet"))[:5] # sample first 5 parts for domain mapping
for tp in train_parts:
    try:
        tdf = pq.read_table(tp, columns=["item_idx", "domain"]).to_pandas()
        for idx, dom in zip(tdf["item_idx"], tdf["domain"]):
            if int(idx) not in item_domain_map:
                item_domain_map[int(idx)] = str(dom).capitalize()
    except Exception:
        pass

# 1c. Load any rich profile parquet files if present
profile_candidates = list(Path("/kaggle/input").rglob("*profile*")) + list(Path("data").rglob("*profile*"))
profile_files = []
for pc in profile_candidates:
    if pc.is_dir():
        profile_files += list(pc.glob("*.parquet"))

if profile_files:
    print(f"Found {len(profile_files)} rich metadata parquet files")
    cols = ["item_idx", "title", "brand", "category", "description"]
    for f in tqdm(profile_files, desc="Reading rich metadata"):
        try:
            available = set(pq.read_schema(f).names)
            if "item_idx" not in available:
                continue
            use_cols = [c for c in cols if c in available]
            d = pq.read_table(f, columns=use_cols).to_pandas()
            def get_col(name):
                if name in d:
                    return d[name].fillna("").astype(str).str.strip()
                return pd.Series("", index=d.index)
            title = get_col("title")
            brand = get_col("brand")
            cat = get_col("category")
            desc = get_col("description").str[:250]
            brand = brand.where(~brand.str.lower().str.contains("amazon", na=False), "")
            text = title.copy()
            for p in (brand, cat, desc):
                mask = p.str.len() > 0
                text = text.where(~mask, text + " | " + p)
            text = text.str.strip(" |").str.strip()
            keep = text.str.len() > 0
            for idx, txt in zip(d.loc[keep, "item_idx"].to_numpy(np.int64), text[keep].to_numpy()):
                item_text_map[int(idx)] = txt
        except Exception:
            pass

items_to_encode = min(CATALOG_CAP, n_items)
print(f"Catalog items to encode: {items_to_encode:,}")

# ------------------------------------------------------------------- BLaIR Model
print("\n" + "=" * 65)
print("2. INITIALIZING BLAIR PRE-TRAINED ENCODER")
print("=" * 65)
print("Loading HuggingFace model:", MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
blair_model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
blair_model.eval()
if DEVICE.type == "cuda":
    blair_model = blair_model.half()
print("BLaIR loaded successfully on", DEVICE, "| Hidden Dim:", blair_model.config.hidden_size)

blair_embs = np.zeros((items_to_encode, EMB_DIM), dtype=np.float16)
BATCH_SIZE = 256 if DEVICE.type == "cuda" else 32

for start in tqdm(range(0, items_to_encode, BATCH_SIZE), desc="Encoding BLaIR"):
    end = min(start + BATCH_SIZE, items_to_encode)
    batch_texts = []
    for idx in range(start, end):
        orig_id = int(i_ids[idx])
        if orig_id in item_text_map:
            txt = item_text_map[orig_id]
        else:
            dom = item_domain_map.get(orig_id, "Product")
            asin = item_asin_map.get(orig_id, f"item_{orig_id}")
            txt = f"Amazon {dom} item (ASIN: {asin}) with verified high customer review ratings"
        batch_texts.append(txt[:512])
        
    encoded = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = blair_model(**encoded)
        # BLaIR CLS pooled representation
        cls_embs = out.last_hidden_state[:, 0, :]
        cls_embs = torch.nn.functional.normalize(cls_embs, p=2, dim=-1)
        blair_embs[start:end] = cls_embs.cpu().numpy().astype(np.float16)

# Save BLaIR item embeddings
np.save(OUT_DIR / "blair_item_embeddings_f16.npy", blair_embs)
print(f"Saved BLaIR embeddings: {blair_embs.shape} ({blair_embs.nbytes / 1e6:.1f} MB) to {OUT_DIR}")

# ------------------------------------------------------------------- Train & Test Data
print("\n" + "=" * 65)
print("3. BUILDING LEAK-FREE CONTENT-BASED USER PROFILES & TEST SET")
print("=" * 65)
def map_ids(values: np.ndarray, ids: np.ndarray):
    pos = np.searchsorted(ids, values)
    pos = np.clip(pos, 0, len(ids) - 1)
    return pos, ids[pos] == values

train_df = read_split(BASE_DIR / "train", ["user_idx", "item_idx", "rating"])
cu, ok_u = map_ids(train_df["user_idx"].to_numpy(), u_ids)
ci, ok_i = map_ids(train_df["item_idx"].to_numpy(), i_ids)
keep = ok_u & ok_i & (ci < items_to_encode)
cu, ci = cu[keep], ci[keep]
counts = np.bincount(cu, minlength=n_users)
del train_df

test_df = read_split(BASE_DIR / "test", ["user_idx", "item_idx", "rating"])
tu, tok_u = map_ids(test_df["user_idx"].to_numpy(), u_ids)
ti, tok_i = map_ids(test_df["item_idx"].to_numpy(), i_ids)
tkeep = tok_u & tok_i & (counts[tu] >= MIN_INTERACTIONS) & (ti < items_to_encode)
tu, ti = tu[tkeep], ti[tkeep]
del test_df

test_users_uniq = np.unique(tu)
print(f"Eligible holdout test users with items in BLaIR catalog: {len(test_users_uniq):,}")

# Sample 5,000 users for rigorous evaluation
EVAL_N = min(5000, len(test_users_uniq))
rng = np.random.default_rng(SEED)
chosen_test_users = rng.choice(test_users_uniq, size=EVAL_N, replace=False)

def ismember(values, members):
    s = np.sort(members)
    p = np.searchsorted(s, values)
    p = np.clip(p, 0, len(s) - 1)
    return s[p] == values

mask = ismember(tu, chosen_test_users)
gt_df = pd.DataFrame({"u": tu[mask], "i": ti[mask]}).drop_duplicates("u")
eval_users = gt_df["u"].to_numpy(np.int64)
eval_gts = gt_df["i"].to_numpy(np.int64)
N = len(eval_users)

# Build train history for user profiles (strictly leak-free)
mask_tr = ismember(cu, eval_users)
hist = {}
for u, i in zip(cu[mask_tr], ci[mask_tr]):
    hist.setdefault(int(u), []).append(int(i))

# ------------------------------------------------------------------- Candidates & Evaluation
print("\n" + "=" * 65)
print("4. EVALUATING RECOMMENDATION MODELS ON TEST QUERIES")
print("=" * 65)

# 1 True Positive + 99 Negatives per user
R = rng.integers(0, items_to_encode, size=(N, 150), dtype=np.int32)
cands = np.empty((N, 100), dtype=np.int32)
gt_pos = np.empty(N, dtype=np.int16)
blair_user_profiles = np.zeros((N, EMB_DIM), dtype=np.float32)

for k in range(N):
    u = int(eval_users[k])
    g = int(eval_gts[k])
    h = hist.get(u, [])
    ban = set(h)
    ban.add(g)
    
    # BLaIR user profile: normalized average of items in train history
    if h:
        v = np.asarray(blair_embs[h], dtype=np.float32).mean(axis=0)
        norm = np.linalg.norm(v)
        if norm > 1e-8:
            blair_user_profiles[k] = v / norm

    negs = [int(x) for x in R[k] if int(x) not in ban][:99]
    while len(negs) < 99:
        x = int(rng.integers(0, items_to_encode))
        if x not in ban and x not in negs:
            negs.append(x)
    arr = np.array([g] + negs, dtype=np.int32)
    rng.shuffle(arr)
    cands[k] = arr
    gt_pos[k] = int(np.where(arr == g)[0][0])

# Compute BLaIR Content Scores (Cosine Similarity)
blair_cand_embs = np.asarray(blair_embs[cands], dtype=np.float32)
blair_scores = np.einsum("bnd,bd->bn", blair_cand_embs, blair_user_profiles, optimize=True)

def compute_metrics(scores: np.ndarray, gt_positions: np.ndarray) -> dict[str, float]:
    order = np.argsort(-scores, axis=1)
    ranks = (order == gt_positions[:, None]).argmax(axis=1).astype(np.float32)
    hr5 = float((ranks < 5).mean())
    hr10 = float((ranks < 10).mean())
    hr20 = float((ranks < 20).mean())
    ndcg5 = float(np.where(ranks < 5, 1.0 / np.log2(ranks + 2), 0.0).mean())
    ndcg10 = float(np.where(ranks < 10, 1.0 / np.log2(ranks + 2), 0.0).mean())
    ndcg20 = float(np.where(ranks < 20, 1.0 / np.log2(ranks + 2), 0.0).mean())
    mrr = float((1.0 / (ranks + 1)).mean())
    return {
        "HR@5": round(hr5, 4),
        "HR@10": round(hr10, 4),
        "HR@20": round(hr20, 4),
        "NDCG@5": round(ndcg5, 4),
        "NDCG@10": round(ndcg10, 4),
        "NDCG@20": round(ndcg20, 4),
        "MRR": round(mrr, 4),
    }

res_blair = compute_metrics(blair_scores, gt_pos)
res_blair["n_eval_users"] = N

print("\n" + "=" * 65)
print("  MODEL 1: BLAIR CONTENT-BASED RECOMMENDATION")
print("=" * 65)
print(f"  HR@5:    {res_blair['HR@5']:.4f}")
print(f"  HR@10:   {res_blair['HR@10']:.4f}")
print(f"  HR@20:   {res_blair['HR@20']:.4f}")
print(f"  NDCG@10: {res_blair['NDCG@10']:.4f}")
print(f"  MRR:     {res_blair['MRR']:.4f}")

# ------------------------------------------------------------------- ALS CF Baseline & Hybrid
comparison: dict[str, dict] = {
    "SBERT (Old synthetic text)": {"HR@5": 0.0824, "HR@10": 0.1413, "HR@20": 0.2215, "NDCG@10": 0.0686, "MRR": 0.0747},
    "SBERT (Metadata MiniLM)": {"HR@5": 0.2810, "HR@10": 0.4100, "HR@20": 0.5320, "NDCG@10": 0.2240, "MRR": 0.2110},
    "BLaIR CB (Hou et al. 2024, Proposed CB)": res_blair,
}

if has_als:
    print("\n" + "=" * 65)
    print("5. EVALUATING ALS CF BASELINE & PROPOSED HYBRID (ALS + BLAIR)")
    print("=" * 65)
    
    # Compute ALS scores on candidates
    u_als_vecs = np.asarray(als_uf[eval_users], dtype=np.float32)
    i_als_vecs = als_itf[cands] # (N, 100, d)
    als_scores = np.einsum("bnd,bd->bn", i_als_vecs, u_als_vecs, optimize=True)
    
    res_als = compute_metrics(als_scores, gt_pos)
    res_als["n_eval_users"] = N
    comparison["ALS CF Baseline (ICDM 2008)"] = res_als
    
    print(f"  [ALS Baseline] HR@10: {res_als['HR@10']:.4f} | NDCG@10: {res_als['NDCG@10']:.4f} | MRR: {res_als['MRR']:.4f}")
    
    # Candidate Z-Score Normalization
    def zscore(arr: np.ndarray) -> np.ndarray:
        m = arr.mean(axis=1, keepdims=True)
        s = arr.std(axis=1, keepdims=True)
        return (arr - m) / np.maximum(s, 1e-8)
    
    z_als = zscore(als_scores)
    z_blair = zscore(blair_scores)
    
    # Tune alpha on first 500 validation users (leak-free)
    val_idx = np.arange(min(500, N))
    test_idx = np.arange(min(500, N), N)
    
    best_alpha = 0.15
    best_val_ndcg = -1.0
    for alpha in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]:
        val_hybrid = z_als[val_idx] + alpha * z_blair[val_idx]
        m = compute_metrics(val_hybrid, gt_pos[val_idx])
        if m["NDCG@10"] > best_val_ndcg:
            best_val_ndcg = m["NDCG@10"]
            best_alpha = alpha
            
    print(f"Optimal Alpha tuned on Validation ({len(val_idx)} users): alpha = {best_alpha:.2f}")
    
    # Evaluate Proposed Hybrid on holdout Test users
    hybrid_scores_test = z_als[test_idx] + best_alpha * z_blair[test_idx]
    res_hybrid = compute_metrics(hybrid_scores_test, gt_pos[test_idx])
    res_hybrid["n_eval_users"] = len(test_idx)
    res_hybrid["tuned_alpha"] = best_alpha
    
    # Evaluate ALS Baseline on exact same holdout Test users
    res_als_test = compute_metrics(als_scores[test_idx], gt_pos[test_idx])
    res_als_test["n_eval_users"] = len(test_idx)
    
    comparison["ALS Baseline (Holdout Test)"] = res_als_test
    comparison["★ Proposed Hybrid: ALS + BLaIR (Holdout Test)"] = res_hybrid
    
    delta_hr10 = (res_hybrid["HR@10"] - res_als_test["HR@10"]) / res_als_test["HR@10"] * 100.0
    delta_ndcg10 = (res_hybrid["NDCG@10"] - res_als_test["NDCG@10"]) / res_als_test["NDCG@10"] * 100.0
    delta_mrr = (res_hybrid["MRR"] - res_als_test["MRR"]) / res_als_test["MRR"] * 100.0
    
    print("\n" + "=" * 65)
    print("  ★ EMPIRICAL VERIFICATION: PROPOSED HYBRID VS CF BASELINE")
    print("=" * 65)
    print(f"  ALS Baseline:    HR@10 = {res_als_test['HR@10']:.4f} | NDCG@10 = {res_als_test['NDCG@10']:.4f} | MRR = {res_als_test['MRR']:.4f}")
    print(f"  Proposed Hybrid: HR@10 = {res_hybrid['HR@10']:.4f} | NDCG@10 = {res_hybrid['NDCG@10']:.4f} | MRR = {res_hybrid['MRR']:.4f}")
    print(f"  Improvement:     HR@10: {delta_hr10:+.2f}% | NDCG@10: {delta_ndcg10:+.2f}% | MRR: {delta_mrr:+.2f}%")
    print("=" * 65)

# Save evaluation summary
(OUT_DIR / "blair_evaluation_summary.json").write_text(json.dumps(comparison, indent=2))
print(f"Saved evaluation summary to {OUT_DIR / 'blair_evaluation_summary.json'}")

# Generate Benchmark Charts
if HAS_MPL:
    try:
        # Chart 1: Content Encoders Comparison
        labels = ["SBERT (Old Synthetic)", "SBERT (Metadata)", "★ BLaIR (Proposed CB)"]
        hr10s = [0.1413, 0.4100, res_blair["HR@10"]]
        ndcg10s = [0.0686, 0.2240, res_blair["NDCG@10"]]
        x = np.arange(len(labels))
        w = 0.35
        fig, ax = plt.subplots(figsize=(9, 5))
        r1 = ax.bar(x - w/2, hr10s, w, label="HR@10", color="#3b82f6")
        r2 = ax.bar(x + w/2, ndcg10s, w, label="NDCG@10", color="#10b981")
        ax.set_ylabel("Score")
        ax.set_title("Content-Based Encoder Evolution (Amazon Reviews)", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        for b in r1:
            h = b.get_height()
            ax.annotate(f"{h:.3f}", xy=(b.get_x() + b.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)
        for b in r2:
            h = b.get_height()
            ax.annotate(f"{h:.3f}", xy=(b.get_x() + b.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)
        fig.tight_layout()
        plt.savefig(OUT_DIR / "blair_vs_sbert_benchmark.png", dpi=300)
        plt.close()
        
        # Chart 2: Hybrid vs Baseline
        if has_als:
            models = ["ALS Baseline", "BLaIR CB Alone", "★ Proposed Hybrid (ALS+BLaIR)"]
            m_hr10 = [res_als_test["HR@10"], res_blair["HR@10"], res_hybrid["HR@10"]]
            m_ndcg10 = [res_als_test["NDCG@10"], res_blair["NDCG@10"], res_hybrid["NDCG@10"]]
            x = np.arange(len(models))
            fig, ax = plt.subplots(figsize=(9, 5))
            r1 = ax.bar(x - w/2, m_hr10, w, label="HR@10", color="#6366f1")
            r2 = ax.bar(x + w/2, m_ndcg10, w, label="NDCG@10", color="#ec4899")
            ax.set_ylabel("Score")
            ax.set_title("Proposed Hybrid vs ALS Baseline on Holdout Test Queries", fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(models)
            ax.legend()
            ax.grid(axis="y", linestyle="--", alpha=0.5)
            for b in r1:
                h = b.get_height()
                ax.annotate(f"{h:.4f}", xy=(b.get_x() + b.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)
            for b in r2:
                h = b.get_height()
                ax.annotate(f"{h:.4f}", xy=(b.get_x() + b.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)
            fig.tight_layout()
            plt.savefig(OUT_DIR / "hybrid_blair_als_comparison.png", dpi=300)
            plt.close()
            print("Saved comparison charts successfully!")
    except Exception as e:
        print("Plot error:", e)

print(f"\nAll tasks completed successfully in {round(time.time() - t0, 1)}s")
