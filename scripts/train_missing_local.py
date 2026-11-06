#!/usr/bin/env python3
"""Train missing hybrid models locally (CPU) using frozen features."""
import json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import pyarrow.parquet as pq

BASE = Path('/home/clyde/DATN')
device = torch.device('cpu')

# Load SBERT
sbert_emb = np.load(BASE/'models/sbert/sbert_item_embeddings.npy')
sbert_idx = np.load(BASE/'models/sbert/sbert_item_idx.npy')
sbert_map = {int(k): sbert_emb[i] for i,k in enumerate(sbert_idx)}

# Load train (sample for CPU speed)
train_df = pq.read_table(sorted((BASE/'models/als/train').glob('*.parquet'))[0]).to_pandas()
# Use subset for speed
train_df = train_df.sample(n=min(10000, len(train_df)), random_state=42).reset_index(drop=True)
print(f'Train sample: {len(train_df)} interactions')

# Build user profiles
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

# MF features: zeros (state corrupted)
mf_dim = 64
n_users = int(train_df['user_idx'].max()) + 1
n_items = int(train_df['item_idx'].max()) + 1
mf_ue = np.zeros((n_users, mf_dim), dtype=np.float32)
mf_ie = np.zeros((n_items, mf_dim), dtype=np.float32)

max_item = max(sbert_map.keys()) if sbert_map else 0
item_sbert = np.zeros((max(max_item, n_items-1)+1, sbert_emb.shape[1]), dtype=np.float32)
for iid, emb in sbert_map.items():
    if iid < len(item_sbert): item_sbert[iid] = emb

def make_feat(u, i):
    u = int(u); i = int(i)
    cb_u = u2prof.get(u, np.zeros(384))
    cb_i = item_sbert[i] if i < len(item_sbert) else np.zeros(384)
    return np.concatenate([mf_ue[u % n_users], mf_ie[i % n_items], cb_u, cb_i]).astype(np.float32)

# Dataset
u2set = {int(u): set(g['item_idx'].values) for u, g in train_df.groupby('user_idx')}
class BPRDS(Dataset):
    def __init__(self, df, u2set, n_items, neg=2):
        self.u = df['user_idx'].values.astype(np.int64)
        self.i = df['item_idx'].values.astype(np.int64)
        self.u2set = u2set; self.n_items = n_items; self.neg = neg
    def __len__(self): return len(self.u)
    def __getitem__(self, idx):
        u = int(self.u[idx]); i = int(self.i[idx])
        neg = []; seen = self.u2set.get(u, set())
        while len(neg) < self.neg:
            j = np.random.randint(0, self.n_items)
            if j not in seen and j != i: neg.append(j)
        return u, i, np.array(neg, dtype=np.int64)

ds = BPRDS(train_df, u2set, n_items, neg=2)
dl = DataLoader(ds, batch_size=512, shuffle=True)

# Models
class AutoInt(nn.Module):
    def __init__(self, in_dim=896, n_heads=4, n_layers=2, deep_hidden=[128,64], drop=0.2):
        super().__init__()
        self.proj = nn.Linear(in_dim, 128)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.MultiheadAttention(128, n_heads, dropout=drop, batch_first=True))
        self.norm = nn.LayerNorm(128)
        deep = []; d = 128
        for h in deep_hidden:
            deep.append(nn.Linear(d,h)); deep.append(nn.ReLU()); deep.append(nn.Dropout(drop)); d=h
        self.deep = nn.Sequential(*deep)
        self.out = nn.Linear(128 + deep_hidden[-1], 1)
    def forward(self, x):
        h = self.proj(x).unsqueeze(1)
        for attn in self.layers:
            h2,_ = attn(h,h,h)
            h = self.norm(h + h2)
        h = h.squeeze(1)
        deep_out = self.deep(h)
        return self.out(torch.cat([h, deep_out], dim=1)).squeeze()

class NeuMF(nn.Module):
    def __init__(self, in_dim=896, mlp_hidden=[256,128,64], drop=0.2):
        super().__init__()
        self.gmf = nn.Sequential(nn.Linear(128,64), nn.ReLU())
        mlp = []; d = in_dim
        for h in mlp_hidden:
            mlp.append(nn.Linear(d,h)); mlp.append(nn.ReLU()); mlp.append(nn.Dropout(drop)); d=h
        self.mlp = nn.Sequential(*mlp)
        self.out = nn.Linear(64 + mlp_hidden[-1], 1)
    def forward(self, x):
        gmf_out = self.gmf(x[:,:128])
        mlp_out = self.mlp(x)
        return self.out(torch.cat([gmf_out, mlp_out], dim=1)).squeeze()

# Train
for name, ModelClass in [('AutoInt', AutoInt), ('NeuMF', NeuMF)]:
    print(f'\n===== Training {name} =====')
    model = ModelClass()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    t0 = time.time()
    for ep in range(2):
        model.train(); tl=0; n=0
        for u, i, neg in dl:
            B = u.shape[0]; Nn = neg.shape[1]
            Xpos = np.array([make_feat(int(uu), int(ii)) for uu, ii in zip(u.numpy(), i.numpy())])
            neg_flat = neg.numpy().reshape(-1)
            u_rep = np.repeat(u.numpy(), Nn)
            Xneg = np.array([make_feat(int(uu), int(ii)) for uu, ii in zip(u_rep, neg_flat)])
            xt = torch.from_numpy(Xpos).float()
            xn = torch.from_numpy(Xneg).float()
            pos = model(xt); negs = model(xn)
            loss = -torch.log(torch.sigmoid(pos.repeat_interleave(Nn) - negs) + 1e-12).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item() * B; n += B
        print(f'  {name} ep{ep+1} loss={tl/n:.4f} {time.time()-t0:.1f}s', flush=True)
    torch.save(model.state_dict(), BASE/f'models/{name.lower()}_local.pt')
    print(f'  {name} saved to models/{name.lower()}_local.pt')

print('\nDone!')
