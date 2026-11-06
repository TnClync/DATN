#!/usr/bin/env python3
"""Comprehensive evaluation of all hybrid variants using available artifacts."""
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE = Path('/home/clyde/DATN')

# ── Load SBERT ──
sbert_emb = np.load(BASE/'models/sbert/sbert_item_embeddings.npy')
sbert_idx = np.load(BASE/'models/sbert/sbert_item_idx.npy')
sbert_map = {int(k): sbert_emb[i] for i,k in enumerate(sbert_idx)}
print(f'SBERT: {len(sbert_map)} items, dim {sbert_emb.shape[1]}')

# ── Load item mapping ──
im = pd.concat([pq.read_table(f).to_pandas() for f in sorted((BASE/'models/als/item_mapping').glob('*.parquet'))])
asin_of = dict(zip(im['item_idx'].astype(int), im['asin'].astype(str)))
print(f'Item mapping: {len(asin_of)} items')

# ── Load train (sample) ──
train_df = pq.read_table(sorted((BASE/'models/als/train').glob('*.parquet'))[0]).to_pandas()
print(f'Train: {len(train_df)} interactions')

# ── Build SBERT item matrix ──
max_item = max(asin_of.keys())
item_sbert = np.zeros((max_item+1, sbert_emb.shape[1]), dtype=np.float32)
for iid, emb in sbert_map.items():
    if iid <= max_item: item_sbert[iid] = emb

# ── User profiles ──
u2prof = {}
for u, g in train_df.groupby('user_idx'):
    vs = []
    for it in g['item_idx'].values[:20]:
        if int(it) in sbert_map:
            v = sbert_map[int(it)]
            n = np.linalg.norm(v)
            if n > 1e-8: vs.append(v/n)
    if vs:
        v = np.mean(vs, axis=0)
        v = v/(np.linalg.norm(v)+1e-8)
        u2prof[int(u)] = v
print(f'User profiles: {len(u2prof)}')

# ── Global profile fallback ──
global_prof = np.mean(list(u2prof.values()), axis=0) if u2prof else None
if global_prof is not None:
    global_prof = global_prof / (np.linalg.norm(global_prof) + 1e-8)

# ── Gated Hybrid model ──
gh_model = None
try:
    import torch
    ckpt = torch.load(BASE/'models/gated_hybrid/gated_hybrid_recommender.pth', map_location='cpu')
    # Reconstruct
    class GatedHybridNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.user_embedding = torch.nn.Embedding(22048170, 16)
            self.item_embedding = torch.nn.Embedding(3686940, 16)
            self.content_projection = torch.nn.Sequential(
                torch.nn.Linear(384, 32), torch.nn.BatchNorm1d(32))
            self.gate_layer = torch.nn.Sequential(torch.nn.Linear(64, 32))
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(32, 128), torch.nn.BatchNorm1d(128),
                torch.nn.Linear(128, 64), torch.nn.BatchNorm1d(64),
                torch.nn.Linear(64, 1))
        def forward(self, user_ids, item_ids, content_emb):
            u = self.user_embedding(user_ids)
            i = self.item_embedding(item_ids)
            c = self.content_projection(content_emb)
            gate_input = torch.cat([u, i, c], dim=1)
            gate = torch.sigmoid(self.gate_layer(gate_input))
            gated = gate * c
            return self.mlp(gated).squeeze(-1)
    gh_model = GatedHybridNet()
    gh_model.load_state_dict(ckpt)
    gh_model.eval()
    print('Gated Hybrid loaded')
except Exception as e:
    print(f'Gated Hybrid load failed: {e}')

# ── Load user mapping (SQLite) ──
import sqlite3
user_db = BASE/'models/als/user_mapping.db'
user_map = {}
if user_db.exists():
    conn = sqlite3.connect(str(user_db))
    # We'll query on demand

def lookup_user(customer_id):
    conn = sqlite3.connect(str(user_db))
    row = conn.execute('SELECT user_idx FROM users WHERE reviewerID=?', (str(customer_id).strip(),)).fetchone()
    conn.close()
    return row[0] if row else None

# ── Scoring functions ──
def mm(x):
    lo, hi = x.min(), x.max()
    if hi-lo < 1e-12: return np.ones_like(x)*0.5
    return (x-lo)/(hi-lo)

def content_score(u, cands):
    p = u2prof.get(int(u), global_prof)
    if p is None: return np.zeros(len(cands), dtype=np.float32)
    res = np.zeros(len(cands), dtype=np.float32)
    for idx, c in enumerate(cands):
        if int(c) < len(item_sbert):
            v = item_sbert[int(c)]
            nv = np.linalg.norm(v)
            if nv > 0: res[idx] = float(np.dot(p, v/nv))
    return res

def gated_score(u, cands):
    if gh_model is None: return None
    raw_uid = u  # user_idx from train
    # Gated hybrid uses its own user/item embeddings (22M/3.7M)
    # Map: train user_idx -> model user_idx (same index space? model has 22M users)
    if raw_uid >= 22048170: return None
    user_t = torch.tensor([raw_uid]*len(cands), dtype=torch.long)
    item_t = torch.tensor(np.array(cands).astype(np.int64), dtype=torch.long)
    content_t = torch.from_numpy(item_sbert[np.array(cands)]).float() if False else torch.zeros(len(cands), 384)
    # Need content emb for items
    content_t = torch.from_numpy(np.array([sbert_map.get(int(c), np.zeros(384)) for c in cands])).float()
    with torch.no_grad():
        scores = gh_model(user_t, item_t, content_t).numpy()
    return scores.astype(np.float32)

# ── Hybrid variants ──
def variant_sbert_only(u, cands):
    return content_score(u, cands)

def variant_gated_only(u, cands):
    s = gated_score(u, cands)
    if s is None: return content_score(u, cands)
    return s

def variant_late_sbert_gated(u, cands, alpha=0.5):
    s1 = content_score(u, cands)
    s2 = gated_score(u, cands)
    if s2 is None: return s1
    return (alpha*mm(s1) + (1-alpha)*mm(s2)).astype(np.float32)

def variant_early_sbert_gated(u, cands):
    s1 = content_score(u, cands)
    s2 = gated_score(u, cands)
    if s2 is None: return s1
    n1 = mm(s1); n2 = mm(s2)
    return (0.4*n1 + 0.3*n2 + 0.3*n1*n2).astype(np.float32)

# ── Evaluation ──
def eval_ranking(score_fn, K_list=[5,10,20], nn=99, n_users=300):
    results = {f'HR@{k}': 0 for k in K_list}
    results.update({f'NDCG@{k}': 0 for k in K_list})
    mrr_sum = 0
    n = 0
    rng = np.random.RandomState(42)
    test_users = train_df['user_idx'].unique()[:n_users]  # use train users for eval
    all_items = np.array(list(asin_of.keys()))
    for u in test_users:
        pu = train_df[train_df['user_idx']==u]['item_idx'].values
        if len(pu) < 2: continue
        pos = int(pu[-1])  # last item as positive
        seen = set(train_df[train_df['user_idx']==u]['item_idx'].values) - {pos}
        negs = []
        while len(negs) < nn:
            x = int(rng.choice(all_items))
            if x not in seen and x != pos: negs.append(x)
        if len(negs) < nn: continue
        cands = np.array([pos]+negs)
        sc = score_fn(int(u), cands)
        order = np.argsort(sc)[::-1]
        rp = int(np.where(order==0)[0][0])
        n += 1
        for K in K_list:
            if rp < K:
                results[f'HR@{K}'] += 1
                results[f'NDCG@{K}'] += 1.0/np.log2(rp+2)
        mrr_sum += 1.0/(rp+1)
    for K in K_list:
        results[f'HR@{K}'] /= n
        results[f'NDCG@{K}'] /= n
    results['MRR'] = mrr_sum/n
    results['n'] = n
    return results

# Run evals
variants = {
    'SBERT-Only': variant_sbert_only,
    'Gated-Only': variant_gated_only,
    'LateFusion(SBERT+Gated,0.5)': lambda u,c: variant_late_sbert_gated(u,c,0.5),
    'LateFusion(SBERT+Gated,0.7)': lambda u,c: variant_late_sbert_gated(u,c,0.7),
    'EarlyFusion(SBERT+Gated)': variant_early_sbert_gated,
}

results = {}
for name, fn in variants.items():
    t0 = time.time()
    r = eval_ranking(fn)
    results[name] = r
    print(f"{name}: HR@5={r['HR@5']:.4f} HR@10={r['HR@10']:.4f} HR@20={r['HR@20']:.4f} NDCG@5={r['NDCG@5']:.4f} NDCG@10={r['NDCG@10']:.4f} NDCG@20={r['NDCG@20']:.4f} MRR={r['MRR']:.4f} (n={r['n']}) [{time.time()-t0:.1f}s]")

# Save
import json
out = BASE/'docs/results/variant_comparison.json'
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, 'w') as f:
    json.dump(results, f, indent=2)
print(f'\nSaved to {out}')
