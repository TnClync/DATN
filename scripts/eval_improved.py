#!/usr/bin/env python3
"""Improved evaluation: fix SBERT alignment, add HR@5/10/20, NDCG@5/10/20, MRR."""
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

BASE = Path('/home/clyde/DATN')

# Load SBERT
sbert_emb = np.load(BASE/'models/sbert/sbert_item_embeddings.npy')
sbert_idx = np.load(BASE/'models/sbert/sbert_item_idx.npy')
sbert_map = {int(k): sbert_emb[i] for i,k in enumerate(sbert_idx)}
print(f'SBERT: {len(sbert_map)} items, dim {sbert_emb.shape[1]}')

# Load item mapping
im = pd.concat([pq.read_table(f).to_pandas() for f in sorted((BASE/'models/als/item_mapping').glob('*.parquet'))])
asin_of = dict(zip(im['item_idx'].astype(int), im['asin'].astype(str)))
print(f'Item mapping: {len(asin_of)} items')

# Load train
train_df = pq.read_table(sorted((BASE/'models/als/train').glob('*.parquet'))[0]).to_pandas()
print(f'Train: {len(train_df)} interactions')

# Build user profiles (weighted by rating)
u2prof = {}
for u, g in train_df.groupby('user_idx'):
    vs = []
    weights = []
    for _, row in g.iterrows():
        it = int(row['item_idx'])
        if it in sbert_map:
            v = sbert_map[it]
            n = np.linalg.norm(v)
            if n > 1e-8:
                vs.append(v/n)
                weights.append(float(row['rating']))
    if vs:
        vs = np.array(vs)
        w = np.array(weights)
        w = w / w.sum()
        v = np.average(vs, axis=0, weights=w)
        v = v/(np.linalg.norm(v)+1e-8)
        u2prof[int(u)] = v
print(f'User profiles: {len(u2prof)}')

# Item SBERT matrix (by item_idx)
max_item = max(asin_of.keys())
item_sbert = np.zeros((max_item+1, sbert_emb.shape[1]), dtype=np.float32)
for iid, emb in sbert_map.items():
    if iid <= max_item: item_sbert[iid] = emb

# Global profile
global_prof = np.mean(list(u2prof.values()), axis=0) if u2prof else None
if global_prof is not None:
    global_prof = global_prof / (np.linalg.norm(global_prof) + 1e-8)

# FAISS index
import faiss
valid_items = np.array([i for i in asin_of.keys() if i < len(item_sbert)])
valid_embs = item_sbert[valid_items].astype(np.float32)
norms = np.linalg.norm(valid_embs, axis=1, keepdims=True)
norms[norms == 0] = 1
valid_embs_norm = valid_embs / norms
faiss_index = faiss.IndexFlatIP(valid_embs_norm.shape[1])
faiss_index.add(valid_embs_norm)
faiss_items = valid_items

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

def mm(x):
    lo, hi = x.min(), x.max()
    if hi-lo < 1e-12: return np.ones_like(x)*0.5
    return (x-lo)/(hi-lo)

# Load Gated Hybrid
import torch
gh_model = None
try:
    ckpt = torch.load(BASE/'models/gated_hybrid/gated_hybrid_recommender.pth', map_location='cpu')
    class GatedHybridNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.user_embedding = torch.nn.Embedding(22048170, 16)
            self.item_embedding = torch.nn.Embedding(3686940, 16)
            self.content_projection = torch.nn.Sequential(torch.nn.Linear(384, 32), torch.nn.BatchNorm1d(32))
            self.gate_layer = torch.nn.Sequential(torch.nn.Linear(64, 32))
            self.mlp = torch.nn.Sequential(torch.nn.Linear(32, 128), torch.nn.BatchNorm1d(128), torch.nn.Linear(128, 64), torch.nn.BatchNorm1d(64), torch.nn.Linear(64, 1))
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

def gated_score(u, cands):
    if gh_model is None: return None
    raw_uid = int(u)
    if raw_uid >= 22048170: return None
    user_t = torch.tensor([raw_uid]*len(cands), dtype=torch.long)
    item_t = torch.tensor(np.array(cands).astype(np.int64), dtype=torch.long)
    content_t = torch.from_numpy(np.array([sbert_map.get(int(c), np.zeros(384)) for c in cands])).float()
    with torch.no_grad():
        scores = gh_model(user_t, item_t, content_t).numpy()
    return scores.astype(np.float32)

# Variants
def variant_sbert(u, cands):
    return content_score(u, cands)

def variant_gated(u, cands):
    s = gated_score(u, cands)
    return s if s is not None else content_score(u, cands)

def variant_late_05(u, cands):
    s1 = content_score(u, cands)
    s2 = gated_score(u, cands)
    if s2 is None: return s1
    return (0.5*mm(s1) + 0.5*mm(s2)).astype(np.float32)

def variant_late_07(u, cands):
    s1 = content_score(u, cands)
    s2 = gated_score(u, cands)
    if s2 is None: return s1
    return (0.7*mm(s1) + 0.3*mm(s2)).astype(np.float32)

def variant_early(u, cands):
    s1 = content_score(u, cands)
    s2 = gated_score(u, cands)
    if s2 is None: return s1
    n1 = mm(s1); n2 = mm(s2)
    return (0.4*n1 + 0.3*n2 + 0.3*n1*n2).astype(np.float32)

# FAISS search for SBERT-only
def faiss_search(u, top_k):
    p = u2prof.get(int(u), global_prof)
    if p is None: return np.array([], dtype=np.int64)
    p_norm = (p / (np.linalg.norm(p) + 1e-8)).reshape(1, -1).astype(np.float32)
    k_search = min(top_k*3, faiss_index.ntotal)
    D, I = faiss_index.search(p_norm, k_search)
    return np.array([int(sbert_idx[i]) for i in I[0]])

# Eval with HR@5/10/20, NDCG@5/10/20, MRR
def eval_ranking(score_fn, K_list=[5,10,20], nn=99, n_users=500, use_faiss=False):
    results = {f'HR@{k}': 0 for k in K_list}
    results.update({f'NDCG@{k}': 0 for k in K_list})
    mrr_sum = 0; n = 0
    rng = np.random.RandomState(42)
    test_users = list(u2prof.keys())[:n_users]
    all_items = np.array(list(asin_of.keys()))
    for u in test_users:
        pu = train_df[train_df['user_idx']==u]['item_idx'].values
        if len(pu) < 2: continue
        pos = int(pu[-1])
        seen = set(train_df[train_df['user_idx']==u]['item_idx'].values) - {pos}
        negs = []
        while len(negs) < nn:
            x = int(rng.choice(all_items))
            if x not in seen and x != pos: negs.append(x)
        if len(negs) < nn: continue
        cands = np.array([pos]+negs)
        if use_faiss:
            faiss_cands = faiss_search(u, 500)
            cands = np.array([i for i in faiss_cands if i not in seen][:nn+1])
            if len(cands) < 2: continue
            if pos not in cands:
                cands = np.concatenate([[pos], cands[:nn]])
        sc = score_fn(int(u), cands)
        order = np.argsort(sc)[::-1]
        rp = int(np.where(order==0)[0][0]) if 0 in cands else 999
        # Find position of positive (pos is at index 0 in cands)
        pos_in_cands = 0
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
    'SBERT-Only': variant_sbert,
    'Gated-Only': variant_gated,
    'LateFusion(0.5)': lambda u,c: variant_late_05(u,c),
    'LateFusion(0.7)': lambda u,c: variant_late_07(u,c),
    'EarlyFusion': variant_early,
}

results = {}
for name, fn in variants.items():
    t0 = time.time()
    r = eval_ranking(fn)
    results[name] = r
    print(f"{name}: HR@5={r['HR@5']:.4f} HR@10={r['HR@10']:.4f} HR@20={r['HR@20']:.4f} NDCG@5={r['NDCG@5']:.4f} NDCG@10={r['NDCG@10']:.4f} NDCG@20={r['NDCG@20']:.4f} MRR={r['MRR']:.4f} (n={r['n']}) [{time.time()-t0:.1f}s]")

# Save
import json
out = BASE/'docs/results/variant_comparison_v2.json'
with open(out, 'w') as f:
    json.dump(results, f, indent=2)
print(f'\nSaved to {out}')
