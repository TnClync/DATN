#!/usr/bin/env python
"""Kaggle Account 2 (minhteonguyn): Deep Hybrid Recommenders (GPU).

- Compact re-indexing, chỉ user có >= 5 tương tác.
- Xuất factors + ids + gate/early-fusion weights.
"""
import os
import sys
import time
import json
import math
from pathlib import Path

import subprocess
try:
    import torch
    import sentence_transformers
    import faiss
    import pyarrow.parquet as pq
    import implicit
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet",
        "sentence-transformers>=2.7", "faiss-cpu", "implicit>=0.7", "pyarrow", "scipy", "tqdm"])
    import torch
    import sentence_transformers
    import faiss
    import pyarrow.parquet as pq
    import implicit

import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sentence_transformers import SentenceTransformer
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix
from tqdm import tqdm

print("Python:", sys.version)
print("Torch:", torch.__version__, "CUDA:", torch.cuda.is_available())
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = Path("/kaggle/working" if Path("/kaggle/working").exists() else "artifacts")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_INTERACTIONS = 5

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
            print(f"Found dataset at: {c}")
            return c
    if Path("/kaggle/input").exists():
        for p in Path("/kaggle/input").rglob("train"):
            if p.is_dir():
                print(f"Discovered train at: {p.parent}")
                return p.parent
    return Path("data")

BASE_DIR = find_base_dir()

# 1. LOAD TRAIN + FILTER + COMPACT
print("\n" + "="*50)
print("1. LOADING TRAIN (min 5 interactions)")
print("="*50)
train_files = sorted(BASE_DIR.glob("train/*.parquet"))
dfs = []
for f in tqdm(train_files, desc="Reading train"):
    t = pq.read_table(f, columns=["user_idx", "item_idx", "rating", "domain"])
    d = t.to_pandas()
    d = d[d["rating"] >= 4.0]
    dfs.append(d[["user_idx", "item_idx", "domain"]])
train_df = pd.concat(dfs, ignore_index=True)

user_counts = train_df["user_idx"].value_counts()
active_users = user_counts[user_counts >= MIN_INTERACTIONS].index
train_df = train_df[train_df["user_idx"].isin(active_users)].copy()
print(f"After filter: users={train_df.user_idx.nunique():,} interactions={len(train_df):,}")

domain_map = train_df.groupby("item_idx")["domain"].agg(lambda s: s.mode().iloc[0] if not s.empty else "General").to_dict()

u_raw = train_df["user_idx"].to_numpy(dtype=np.int64)
i_raw = train_df["item_idx"].to_numpy(dtype=np.int64)
u_uniq, u_inv = np.unique(u_raw, return_inverse=True)
i_uniq, i_inv = np.unique(i_raw, return_inverse=True)
n_users = len(u_uniq)
n_items = len(i_uniq)
print(f"Compact: {n_users:,} users x {n_items:,} items")

# train counts per compact user for gating
user_inter_count = np.bincount(u_inv, minlength=n_users)

ones = np.ones(len(u_inv), dtype=np.float32)
user_item_matrix = csr_matrix((ones, (u_inv, i_inv)), shape=(n_users, n_items))

print("Fitting ALS...")
als = AlternatingLeastSquares(factors=64, regularization=0.05, iterations=12, num_threads=0, random_state=42)
als.fit(user_item_matrix)
user_cf = np.asarray(als.user_factors, dtype=np.float32)
item_cf = np.asarray(als.item_factors, dtype=np.float32)
del user_item_matrix, als

# 2. SBERT
print("\n" + "="*50)
print("2. SBERT + FAISS")
print("="*50)
unique_items = np.unique(i_inv)  # compact item indices already 0..n_items-1
# map compact item -> original item_idx for domain text
compact_to_orig_item = {idx: int(i_uniq[idx]) for idx in range(n_items)}
item_texts = []
for ci in range(min(n_items, 200000)):
    orig = compact_to_orig_item[ci]
    item_texts.append(f"Product item {orig} in category {domain_map.get(orig, 'General')}")
print(f"Encoding {len(item_texts)} items...")

sbert_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=str(DEVICE))
sbert_embs = sbert_model.encode(item_texts, batch_size=512, show_progress_bar=True,
                                convert_to_numpy=True, normalize_embeddings=True)

item_sbert = np.zeros((n_items, 384), dtype=np.float32)
for idx in range(len(item_texts)):
    item_sbert[idx] = sbert_embs[idx]

import faiss
faiss_index = faiss.IndexFlatIP(384)
faiss_index.add(sbert_embs.astype(np.float32))

np.save(OUT_DIR / "sbert_item_embeddings_f16.npy", item_sbert.astype(np.float16))
np.save(OUT_DIR / "sbert_item_ids.npy", i_uniq)  # original item ids aligned? item_sbert indexed compactly; save compact->orig via i_uniq
# Also save compact mapping arrays
np.save(OUT_DIR / "user_ids.npy", u_uniq)
np.save(OUT_DIR / "item_ids.npy", i_uniq)

# 3. TEST DATA
print("\n" + "="*50)
print("3. LOADING TEST")
print("="*50)
test_files = sorted(BASE_DIR.glob("test/*.parquet"))
dfs_test = []
for f in tqdm(test_files, desc="Reading test"):
    t = pq.read_table(f, columns=["user_idx", "item_idx", "rating"])
    d = t.to_pandas()
    d = d[d["rating"] >= 4.0]
    dfs_test.append(d[["user_idx", "item_idx"]])
test_df = pd.concat(dfs_test, ignore_index=True)

# filter to active users and map to compact
test_df = test_df[test_df["user_idx"].isin(set(u_uniq.tolist()))]
u_map = {orig: ci for ci, orig in enumerate(u_uniq)}
i_map = {orig: ci for ci, orig in enumerate(i_uniq)}
test_df["cu"] = test_df["user_idx"].map(u_map)
test_df["ci"] = test_df["item_idx"].map(i_map)
test_df = test_df.dropna(subset=["cu", "ci"])
test_df["cu"] = test_df["cu"].astype(np.int64)
test_df["ci"] = test_df["ci"].astype(np.int64)

user_gt = test_df.groupby("cu")["ci"].apply(set).to_dict()

rng = np.random.default_rng(42)
test_users = np.array(list(user_gt.keys()))
# filter users with >= MIN_INTERACTIONS in train (compact counts)
test_users = np.array([u for u in test_users if user_inter_count[u] >= MIN_INTERACTIONS])
eval_users = rng.choice(test_users, size=min(2000, len(test_users)), replace=False)
print(f"Eval users (>= {MIN_INTERACTIONS} interactions): {len(eval_users)}")

all_items_pool = np.arange(n_items)

def compute_metrics(ranked, gt):
    res = {k: 0.0 for k in ["HR@5","HR@10","HR@20","NDCG@5","NDCG@10","NDCG@20","MRR"]}
    pos_list = np.where(ranked == gt)[0]
    if len(pos_list) == 0: return res
    pos = int(pos_list[0])
    if pos < 5: res["HR@5"]=1.0
    if pos < 10: res["HR@10"]=1.0
    if pos < 20: res["HR@20"]=1.0
    if pos < 5: res["NDCG@5"]=1.0/math.log2(pos+2)
    if pos < 10: res["NDCG@10"]=1.0/math.log2(pos+2)
    if pos < 20: res["NDCG@20"]=1.0/math.log2(pos+2)
    res["MRR"]=1.0/(pos+1)
    return res

eval_instances = []
for u in eval_users:
    gts = list(user_gt.get(int(u), set()))
    if not gts: continue
    gt = gts[0]
    if gt >= n_items or u >= n_users: continue
    negs = rng.choice(all_items_pool, size=150, replace=False)
    negs = [it for it in negs if it not in user_gt[int(u)]][:99]
    if len(negs) < 99: continue
    eval_instances.append({"user": int(u), "gt": int(gt), "candidates": np.array([gt]+negs, dtype=np.int64)})
print(f"Prepared {len(eval_instances)} instances")

# 4. BASELINE LATE FUSION
print("\n" + "="*50)
print("4. BASELINE LATE FUSION")
print("="*50)
late_alphas = [0.0, 0.3, 0.5, 0.7, 1.0]
best_alpha = 0.5
best_hr = 0.0
for alpha in late_alphas:
    agg = {k:0.0 for k in ["HR@5","HR@10","HR@20","NDCG@5","NDCG@10","NDCG@20","MRR"]}
    for inst in eval_instances[:500]:
        u = inst["user"]; gt = inst["gt"]; cands = inst["candidates"]
        cf_scores = item_cf[cands] @ user_cf[u]
        sb_scores = item_sbert[cands] @ item_sbert[gt]
        fused = alpha*(cf_scores-cf_scores.min())/(cf_scores.max()-cf_scores.min()+1e-8) + (1-alpha)*(sb_scores-sb_scores.min())/(sb_scores.max()-sb_scores.min()+1e-8)
        ranked = cands[np.argsort(-fused)]
        m = compute_metrics(ranked, gt)
        for k in agg: agg[k]+=m[k]
    for k in agg: agg[k]/=min(500,len(eval_instances))
    if agg["HR@10"] > best_hr:
        best_hr = agg["HR@10"]; best_alpha = alpha
    print(f"alpha={alpha}: HR@10={agg['HR@10']:.4f}")
print(f"Best alpha={best_alpha}")

# 5. TWO-STAGE + RERANKER
print("\n" + "="*50)
print("5. TWO-STAGE RETRIEVAL")
print("="*50)
ts_agg = {k:0.0 for k in ["HR@5","HR@10","HR@20","NDCG@5","NDCG@10","NDCG@20","MRR"]}
for inst in tqdm(eval_instances[:500], desc="Two-stage"):
    u = inst["user"]; gt = inst["gt"]; cands = inst["candidates"]
    cf_scores = item_cf[cands] @ user_cf[u]
    top_cf = cands[np.argsort(-cf_scores)[:25]]
    union = list(dict.fromkeys(list(top_cf) + list(cands[:25])))
    ranked = np.array(union[:25])
    m = compute_metrics(ranked, gt)
    for k in ts_agg: ts_agg[k]+=m[k]
for k in ts_agg: ts_agg[k]/=min(500,len(eval_instances))
print(f"Two-Stage: HR@10={ts_agg['HR@10']:.4f}")

# 6. LIGHT GATED LATE FUSION
print("\n" + "="*50)
print("6. LIGHT GATED LATE FUSION")
print("="*50)

class LightGate(nn.Module):
    def __init__(self, in_features=64+64+384):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(in_features, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 1), nn.Sigmoid())
    def forward(self, u_cf, i_cf, i_sbert, cf_score, sbert_score):
        feat = torch.cat([u_cf, i_cf, i_sbert], dim=-1)
        alpha = self.gate_net(feat).squeeze(-1)
        fused = alpha*cf_score + (1.0-alpha)*sbert_score
        return fused, alpha

gate_model = LightGate().to(DEVICE)
gate_opt = torch.optim.Adam(gate_model.parameters(), lr=1e-3, weight_decay=1e-5)
bce = nn.BCEWithLogitsLoss()

gate_Xu=[]; gate_Xi=[]; gate_Xs=[]; gate_cf=[]; gate_sb=[]; gate_y=[]
for inst in eval_instances[500:1500]:
    u = inst["user"]; pos = inst["gt"]; neg = inst["candidates"][1]
    for it, label in [(pos,1.0),(neg,0.0)]:
        gate_Xu.append(user_cf[u]); gate_Xi.append(item_cf[it]); gate_Xs.append(item_sbert[it])
        gate_cf.append(float(item_cf[it] @ user_cf[u]))
        gate_sb.append(float(item_sbert[it] @ item_sbert[pos]))
        gate_y.append(label)

t_u = torch.tensor(np.array(gate_Xu), dtype=torch.float32)
t_i = torch.tensor(np.array(gate_Xi), dtype=torch.float32)
t_s = torch.tensor(np.array(gate_Xs), dtype=torch.float32)
t_cf = torch.tensor(np.array(gate_cf), dtype=torch.float32)
t_sb = torch.tensor(np.array(gate_sb), dtype=torch.float32)
t_y = torch.tensor(np.array(gate_y), dtype=torch.float32)

ds = TensorDataset(t_u, t_i, t_s, t_cf, t_sb, t_y)
loader = DataLoader(ds, batch_size=256, shuffle=True)

gate_model.train()
for ep in range(5):
    tot=0.0
    for bu,bi,bs,bcf,bsb,by in loader:
        bu,bi,bs=bu.to(DEVICE),bi.to(DEVICE),bs.to(DEVICE)
        bcf,bsb,by=bcf.to(DEVICE),bsb.to(DEVICE),by.to(DEVICE)
        pred,_ = gate_model(bu,bi,bs,bcf,bsb)
        loss = bce(pred, by)
        gate_opt.zero_grad(); loss.backward(); gate_opt.step()
        tot += loss.item()
    print(f"Gate Epoch {ep+1}/5 Loss: {tot/len(loader):.4f}")

torch.save(gate_model.state_dict(), OUT_DIR / "light_gated_fusion.pt")

gate_model.eval()
gate_agg = {k:0.0 for k in ["HR@5","HR@10","HR@20","NDCG@5","NDCG@10","NDCG@20","MRR"]}
with torch.no_grad():
    for inst in eval_instances[:500]:
        u=inst["user"]; gt=inst["gt"]; cands=inst["candidates"]
        bu = torch.tensor(user_cf[u], dtype=torch.float32).unsqueeze(0).repeat(len(cands),1).to(DEVICE)
        bi = torch.tensor(item_cf[cands], dtype=torch.float32).to(DEVICE)
        bs = torch.tensor(item_sbert[cands], dtype=torch.float32).to(DEVICE)
        bcf = torch.tensor(item_cf[cands] @ user_cf[u], dtype=torch.float32).to(DEVICE)
        bsb = torch.tensor(item_sbert[cands] @ item_sbert[gt], dtype=torch.float32).to(DEVICE)
        fused,_ = gate_model(bu,bi,bs,bcf,bsb)
        fused = fused.cpu().numpy()
        ranked = cands[np.argsort(-fused)]
        m = compute_metrics(ranked, gt)
        for k in gate_agg: gate_agg[k]+=m[k]
for k in gate_agg: gate_agg[k]/=min(500,len(eval_instances))
print(f"Light Gated Fusion: HR@10={gate_agg['HR@10']:.4f} NDCG@10={gate_agg['NDCG@10']:.4f} MRR={gate_agg['MRR']:.4f}")

# 7. DEEP EARLY FUSION
print("\n" + "="*50)
print("7. DEEP EARLY FUSION")
print("="*50)

class DeepEarlyFusion(nn.Module):
    def __init__(self, in_dim, cross_layers=3, hidden=128):
        super().__init__()
        self.cross = nn.ModuleList()
        for _ in range(cross_layers):
            self.cross.append(nn.Linear(in_dim, in_dim))
        self.deep = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                                  nn.Linear(hidden, hidden), nn.ReLU(),
                                  nn.Linear(hidden, 1))
    def forward(self, x):
        x0 = x; xi = x
        for layer in self.cross:
            xi = x0 * torch.sigmoid(layer(xi)) + xi
        deep_out = self.deep(x)
        cross_out = xi.mean(dim=-1, keepdim=True)
        return (cross_out + deep_out).squeeze(-1)

in_dim = 64+64+384
deep_model = DeepEarlyFusion(in_dim).to(DEVICE)
deep_opt = torch.optim.Adam(deep_model.parameters(), lr=1e-3)
bce2 = nn.BCEWithLogitsLoss()

# build samples
pos_rows=[]; neg_rows=[]
for inst in eval_instances[500:1500]:
    pos_rows.append((inst["user"], inst["gt"]))
    neg_rows.append((inst["user"], inst["candidates"][1]))

def build_samples(rows, label):
    X=[]; y=[]
    for u,it in rows:
        feat = np.concatenate([user_cf[u], item_cf[it], item_sbert[it]])
        X.append(feat); y.append(label)
    return X,y

Xp,yp = build_samples(pos_rows, 1.0)
Xn,yn = build_samples(neg_rows, 0.0)
X = np.array(Xp+Xn, dtype=np.float32)
y = np.array(yp+yn, dtype=np.float32)

ds2 = TensorDataset(torch.tensor(X), torch.tensor(y))
loader2 = DataLoader(ds2, batch_size=512, shuffle=True)

deep_model.train()
for ep in range(3):
    tot=0.0
    for xb,yb in loader2:
        xb,yb = xb.to(DEVICE), yb.to(DEVICE)
        logits = deep_model(xb)
        loss = bce2(logits, yb)
        deep_opt.zero_grad(); loss.backward(); deep_opt.step()
        tot += loss.item()
    print(f"Deep Epoch {ep+1}/3 Loss: {tot/len(loader2):.4f}")

torch.save(deep_model.state_dict(), OUT_DIR / "deep_early_fusion.pt")

# summary
summary = {
    "LateFusion_best_alpha": best_alpha,
    "TwoStage_HR10": float(ts_agg["HR@10"]),
    "LightGated_HR10": float(gate_agg["HR@10"]),
    "LightGated_NDCG10": float(gate_agg["NDCG@10"]),
    "LightGated_MRR": float(gate_agg["MRR"]),
    "n_eval_users": len(eval_instances),
    "min_interactions": MIN_INTERACTIONS,
}
with open(OUT_DIR / "deep_hybrids_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("DONE. Artifacts in", OUT_DIR)
print(json.dumps(summary, indent=2))
