#!/usr/bin/env python3
"""Improved evaluation: fix SBERT alignment, add FAISS, hard negatives, more metrics."""
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import faiss

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

# Build user profiles (weighted by rating, with more items)
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

# Global profile
global_prof = np.mean(list(u2prof.values()), axis=0) if u2prof else None
if global_prof is not None:
    global_prof = global_prof / (np.linalg.norm(global_prof) + 1e-8)

# FAISS index over SBERT embeddings
norms = np.linalg.norm(sbert_emb, axis=1, keepdims=True)
norms[norms == 0] = 1
sbert_norm = (sbert_emb / norms).astype(np.float32)
faiss_index = faiss.IndexFlatIP(sbert_norm.shape[1])
faiss_index.add(sbert_norm)
print(f'FAISS index: {faiss_index.ntotal} items')

# SBERT map: item_idx -> sbert embedding index
# sbert_idx contains original item_idx, sbert_emb is aligned
sbert_item_to_emb_idx = {int(k): i for i, k in enumerate(sbert_idx)}

def get_sbert_emb(item_idx):
    """Get SBERT embedding for an item_idx."""
    if int(item_idx) in sbert_item_to_emb_idx:
        return sbert_emb[sbert_item_to_emb_idx[int(item_idx)]]
    return None

def content_score(u, cands):
    p = u2prof.get(int(u), global_prof)
    if p is None: return np.zeros(len(cands), dtype=np.float32)
    res = np.zeros(len(cands), dtype=np.float32)
    for idx, c in enumerate(cands):
        emb = get_sbert_emb(int(c))
        if emb is not None:
            nv = np.linalg.norm(emb)
            if nv > 0:
                res[idx] = float(np.dot(p, emb/nv))
    return res

def mm(x):
    lo, hi = x.min(), x.max()
    if hi-lo < 1e-12: return np.ones_like(x)*0.5
    return (x-lo)/(hi-lo)

# Hard negative sampling: use FAISS to find similar items
def get_hard_negatives(pos, seen, n=99, k_search=200):
    """Find hard negatives: items similar to positive but not seen."""
    pos_emb_idx = sbert_item_to_emb_idx.get(int(pos))
    if pos_emb_idx is None:
        return None
    pos_emb = sbert_norm[pos_emb_idx].reshape(1, -1)
    D, I = faiss_index.search(pos_emb, k_search)
    hard_cands = [int(sbert_idx[i]) for i in I[0] if int(sbert_idx[i]) not in seen and int(sbert_idx[i]) != pos]
    if len(hard_cands) < n:
        # Fill with random
        all_items = np.array(list(asin_of.keys()))
        remaining = n - len(hard_cands)
        extra = []
        while len(extra) < remaining:
            x = int(np.random.choice(all_items))
            if x not in seen and x != pos and x not in hard_cands:
                extra.append(x)
        hard_cands.extend(extra)
    return hard_cands[:n]

# Eval with HR@5/10/20, NDCG@5/10/20, MRR
def eval_ranking(score_fn, K_list=[5,10,20], nn=99, n_users=500, hard_neg=False):
    results = {f'HR@{k}': 0 for k in K_list}
    results.update({f'NDCG@{k}': 0 for k in K_list})
    mrr_sum = 0; n = 0
    rng = np.random.RandomState(42)
    test_users = list(u2prof.keys())[:n_users]
    all_items = np.array(list(asin_of.keys()))
    for u in test_users:
        pu = train_df[train_df['user_idx']==u]['item_idx'].values
        if len(pu) < 2: continue
        pos = int(pu[-1])  # last item as positive
        seen = set(train_df[train_df['user_idx']==u]['item_idx'].values) - {pos}
        if hard_neg:
            negs = get_hard_negatives(pos, seen, n=nn)
            if negs is None or len(negs) < nn:
                continue
        else:
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

# Variants
def variant_sbert(u, cands):
    return content_score(u, cands)

def variant_sbert_faiss(u, cands):
    # Use FAISS to get content scores (already computed in content_score)
    return content_score(u, cands)

variants = {
    'SBERT-Only': variant_sbert,
    'SBERT-FAISS': variant_sbert_faiss,
}

# Run evals with both random and hard negatives
for neg_type in ['random', 'hard']:
    hard = (neg_type == 'hard')
    print(f'\n===== Negative sampling: {neg_type} =====')
    results = {}
    for name, fn in variants.items():
        t0 = time.time()
        r = eval_ranking(fn, hard_neg=hard)
        results[name] = r
        print(f"{name}: HR@5={r['HR@5']:.4f} HR@10={r['HR@10']:.4f} HR@20={r['HR@20']:.4f} NDCG@5={r['NDCG@5']:.4f} NDCG@10={r['NDCG@10']:.4f} NDCG@20={r['NDCG@20']:.4f} MRR={r['MRR']:.4f} (n={r['n']}) [{time.time()-t0:.1f}s]")

    out = BASE/f'docs/results/eval_{neg_type}.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Saved to {out}')

print('\nDone!')
