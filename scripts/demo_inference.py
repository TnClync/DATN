#!/usr/bin/env python3
"""CLI Demo Inference for Hybrid Recommender (Late Fusion: SBERT + Popularity)."""
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
import pandas as pd
import pyarrow.parquet as pq

BASE = Path(__file__).resolve().parent.parent

sbert_emb = np.load(BASE/'models/sbert/sbert_item_embeddings.npy')
sbert_idx = np.load(BASE/'models/sbert/sbert_item_idx.npy')
sbert_map = {int(k): sbert_emb[i] for i,k in enumerate(sbert_idx)}
im = pd.concat([pq.read_table(f).to_pandas() for f in sorted((BASE/'models/als/item_mapping').glob('*.parquet'))])
asin_of = dict(zip(im['item_idx'].astype(int), im['asin'].astype(str)))
train_df = pq.read_table(sorted((BASE/'models/als/train').glob('*.parquet'))[0]).to_pandas()
ic = train_df['item_idx'].value_counts().to_dict()
pop_arr = np.zeros(len(asin_of), dtype=np.float32)
m = max(ic.values()) if ic else 1
for k,v in ic.items():
    if int(k) < len(pop_arr): pop_arr[int(k)] = np.log1p(v)/np.log1p(m)
u2prof = {}
for u, g in tqdm(train_df.groupby('user_idx'), desc="Building profiles"):
    vs = []
    for it in g['item_idx'].values[:20]:
        if int(it) in sbert_map:
            v = sbert_map[int(it)]; n = np.linalg.norm(v)
            if n > 1e-8: vs.append(v/n)
    if vs:
        v = np.mean(vs, axis=0); v = v/(np.linalg.norm(v)+1e-8)
        u2prof[int(u)] = v
max_item = max(asin_of.keys())
item_sbert = np.zeros((max_item+1, sbert_emb.shape[1]), dtype=np.float32)
for iid, emb in sbert_map.items():
    if iid <= max_item: item_sbert[iid] = emb

def mm(x):
    lo, hi = x.min(), x.max()
    if hi-lo < 1e-12: return np.ones_like(x)*0.5
    return (x-lo)/(hi-lo)

def recommend(user_idx, top_k, alpha):
    seen = set(train_df[train_df['user_idx']==user_idx]['item_idx'].values)
    top_pop = sorted(ic.items(), key=lambda x:-x[1])[:500]
    cands = np.array([i for i,_ in top_pop if i not in seen])
    if len(cands) == 0: return []
    p = u2prof.get(int(user_idx))
    s_c = np.zeros(len(cands), dtype=np.float32)
    if p is not None:
        for idx, c in enumerate(cands):
            if int(c) < len(item_sbert):
                v = item_sbert[int(c)]; nv = np.linalg.norm(v)
                if nv > 0: s_c[idx] = float(np.dot(p, v/nv))
    s_p = np.array([pop_arr[int(c)] if int(c) < len(pop_arr) else 0.0 for c in cands], dtype=np.float32)
    scores = (alpha*mm(s_c) + (1-alpha)*mm(s_p)).astype(np.float32)
    order = np.argsort(scores)[::-1][:top_k]
    top = cands[order]
    return [{'rank': r+1, 'item_idx': int(top[r]), 'asin': asin_of.get(int(top[r]), 'N/A'),
             'score': round(float(scores[order[r]]), 4)} for r in range(len(top))]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--user_idx', type=int, default=3)
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=0.6)
    args = parser.parse_args()
    res = recommend(args.user_idx, args.top_k, args.alpha)
    print(f'\nTop-{args.top_k} for user {args.user_idx} (LateFusion alpha={args.alpha}):')
    print(f'{"Rank":>4} {"ItemIdx":>8} {"ASIN":>12} {"Score":>8}')
    for r in res:
        print(f'{r["rank"]:>4} {r["item_idx"]:>8} {r["asin"]:>12} {r["score"]:>8}')

if __name__ == '__main__':
    main()
