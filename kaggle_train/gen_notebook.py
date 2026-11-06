import json, sys, os

MODEL_NAME = sys.argv[1]
MODEL_CODE = sys.argv[2]

cells = []
def code(s): cells.append({"cell_type": "code", "metadata": {}, "id": __import__('uuid').uuid4().hex, "source": [s], "outputs": [], "execution_count": None})
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "id": __import__('uuid').uuid4().hex, "source": [s]})

md(f"# {MODEL_NAME} Hybrid — Train on LastOne + LastReview (2x T4)\n\nUses pre-trained MF-bias + SBERT from kernel data sources.")
code("""!pip install -q sentence-transformers pyarrow pandas tqdm faiss-cpu
import os, json, time, glob
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import faiss
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
""")

# Load artifacts from kernel data sources
code("""
def find_file(name):
    for root,_,files in os.walk('/kaggle/input'):
        if name in files: return os.path.join(root,name)
    return None

# Load MF-bias
mf_state_path = find_file('mf_bias_state.pt')
mf_uids_path = find_file('mf_user_ids.npy')
mf_iids_path = find_file('mf_item_ids.npy')
print('MF state:', mf_state_path)
ckpt = torch.load(mf_state_path, map_location='cpu')
ue_key = [k for k in ckpt if 'user' in k.lower() and 'emb' in k.lower()][0]
ie_key = [k for k in ckpt if 'item' in k.lower() and 'emb' in k.lower()][0]
mf_ue_big = ckpt[ue_key].numpy() if torch.is_tensor(ckpt[ue_key]) else ckpt[ue_key]
mf_ie_big = ckpt[ie_key].numpy() if torch.is_tensor(ckpt[ie_key]) else ckpt[ie_key]
mf_uids = np.load(mf_uids_path, allow_pickle=True)
mf_iids = np.load(mf_iids_path, allow_pickle=True)
raw2mf_u = {int(raw): idx for idx, raw in enumerate(mf_uids)}
raw2mf_i = {int(raw): idx for idx, raw in enumerate(mf_iids)}
print('MF big:', mf_ue_big.shape, mf_ie_big.shape)

# Load SBERT
sbert_emb_path = find_file('sbert_item_embeddings.npy')
sbert_idx_path = find_file('sbert_item_idx.npy')
sbert_emb = np.load(sbert_emb_path)
sbert_idx = np.load(sbert_idx_path)
sbert_map = {int(k): sbert_emb[i] for i, k in enumerate(sbert_idx)}
print('SBERT:', sbert_emb.shape, 'items:', len(sbert_map))
""")

# Load data
code("""
base_lo = Path('/kaggle/input/datasets/nguynteo/lastone/amazon_recs_output')
dfs = [pq.read_table(f).to_pandas() for f in sorted(base_lo.glob('train/*.parquet'))]
train_df = pd.concat(dfs, ignore_index=True)
test_dfs = [pq.read_table(f).to_pandas() for f in sorted(base_lo.glob('test/*.parquet'))]
test_df = pd.concat(test_dfs, ignore_index=True)
print('train:', len(train_df), 'test:', len(test_df))

users_raw = train_df['user_idx'].to_numpy()
items_raw = train_df['item_idx'].to_numpy()
u_uniq, u_inv = np.unique(users_raw, return_inverse=True)
i_uniq, i_inv = np.unique(items_raw, return_inverse=True)
U = len(u_uniq); I = len(i_uniq)
train_df['u'] = u_inv
train_df['i'] = i_inv
# Test mapping
test_u_map = {uid: idx for idx, uid in enumerate(u_uniq)}
test_i_map = {iid: idx for idx, iid in enumerate(i_uniq)}
test_df['u'] = test_df['user_idx'].map(test_u_map)
test_df['i'] = test_df['item_idx'].map(test_i_map)
test_df = test_df.dropna(subset=['u','i'])
test_df['u'] = test_df['u'].astype(int)
test_df['i'] = test_df['i'].astype(int)
print('U,I', U, I)
""")

# Build features
code("""
# Build compact MF embeddings aligned with hybrid indices
mf_ue = np.zeros((U, mf_ue_big.shape[1]), dtype=np.float32)
mf_ie = np.zeros((I, mf_ie_big.shape[1]), dtype=np.float32)
for r in range(U):
    mi = raw2mf_u.get(int(u_uniq[r]))
    if mi is not None and mi < mf_ue_big.shape[0]:
        mf_ue[r] = mf_ue_big[mi]
for r in range(I):
    mi = raw2mf_i.get(int(i_uniq[r]))
    if mi is not None and mi < mf_ie_big.shape[0]:
        mf_ie[r] = mf_ie_big[mi]
print('MF compact:', mf_ue.shape, mf_ie.shape)

# Build SBERT content embeddings aligned with hybrid indices
i2row = {int(x): i for i, x in enumerate(i_uniq)}
emb_c = np.zeros((I, sbert_emb.shape[1]), dtype=np.float32)
for orig, e in sbert_map.items():
    if orig in i2row:
        emb_c[i2row[orig]] = e
print('SBERT emb_c:', emb_c.shape, (emb_c.sum(1)!=0).sum(), 'items')

# User profiles
user_prof_all = np.zeros((U, emb_c.shape[1]), dtype=np.float32)
for u, its in train_df.groupby('u')['i']:
    vs = [emb_c[i] for i in its if np.linalg.norm(emb_c[i]) > 0]
    if vs:
        v = np.mean(vs, axis=0)
        v /= (np.linalg.norm(v) + 1e-8)
        user_prof_all[u] = v
print('User profiles:', (user_prof_all.sum(1) != 0).sum())

def make_feat_batch(us, is_):
    mf_u = mf_ue[us]; mf_i = mf_ie[is_]
    cb_u = user_prof_all[us]; cb_i = emb_c[is_]
    return np.concatenate([mf_u, mf_i, cb_u, cb_i], axis=1).astype(np.float32)
print('Feature builder ready (896-dim)')
""")

code("""
# Dataset with hard negatives
import faiss
valid_items = np.array([i for i in range(I) if np.linalg.norm(emb_c[i]) > 0])
valid_embs = emb_c[valid_items].astype(np.float32)
norms = np.linalg.norm(valid_embs, axis=1, keepdims=True)
norms[norms == 0] = 1
valid_embs_norm = valid_embs / norms
faiss_index = faiss.IndexFlatIP(valid_embs_norm.shape[1])
faiss_index.add(valid_embs_norm)

def get_hard_negatives(pos, seen, n=99, k_search=200):
    if np.linalg.norm(emb_c[pos]) == 0:
        return None
    pe = (emb_c[pos] / np.linalg.norm(emb_c[pos])).reshape(1, -1).astype(np.float32)
    D, II = faiss_index.search(pe, k_search)
    hard_cands = [int(valid_items[x]) for x in II[0] if int(valid_items[x]) not in seen and int(valid_items[x]) != pos]
    if len(hard_cands) < n:
        all_items = np.arange(I)
        extra = []
        while len(extra) < n - len(hard_cands):
            x = int(np.random.choice(all_items))
            if x not in seen and x != pos and x not in hard_cands:
                extra.append(x)
        hard_cands.extend(extra)
    return hard_cands[:n]

class BPRDS(Dataset):
    def __init__(self, tr_df, u2set, n_items, neg=4, hard=True):
        self.u = tr_df['u'].values.astype(np.int64)
        self.i = tr_df['i'].values.astype(np.int64)
        self.u2set = u2set; self.n_items = n_items; self.neg = neg; self.hard = hard
    def __len__(self): return len(self.u)
    def __getitem__(self, idx):
        u = int(self.u[idx]); i = int(self.i[idx])
        neg = []; seen = self.u2set.get(u, set())
        if self.hard:
            negs = get_hard_negatives(i, seen, n=self.neg)
            if negs: neg = negs
            else:
                while len(neg) < self.neg:
                    j = np.random.randint(0, self.n_items)
                    if j not in seen and j != i: neg.append(j)
        else:
            while len(neg) < self.neg:
                j = np.random.randint(0, self.n_items)
                if j not in seen and j != i: neg.append(j)
        return u, i, np.array(neg, dtype=np.int64)

u2items_tr_set = {int(u): set(g['i'].values) for u, g in tr.groupby('u')}
ds = BPRDS(tr, u2items_tr_set, I, neg=4, hard=True)
dl = DataLoader(ds, batch_size=2048, shuffle=True, num_workers=4, pin_memory=True)
dev = torch.device('cuda')
print(f'Dataset: {len(ds):,} samples (hard negatives)')
""")

# Model code
code(MODEL_CODE)

# Training
code(f"""
model = {MODEL_CLASS}()
model = model.to(dev)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
t0 = time.time()
best_val_hr10 = 0
patience = 3
no_improve = 0
history = []

# Val eval function
def eval_split(split_df, K_list=[5,10,20], nn=99, n_users=300):
    model.eval()
    h = {{f'HR@{{k}}': 0 for k in K_list}}
    nd = {{f'NDCG@{{k}}': 0 for k in K_list}}
    mrr_sum = 0; n = 0
    rng = np.random.default_rng(888)
    su = rng.choice(split_df['u'].unique(), size=min(n_users, len(split_df['u'].unique())), replace=False)
    for u in tqdm(su, desc='Eval', leave=False):
        pu = split_df[split_df['u']==int(u)]['i'].values
        if len(pu) == 0: continue
        pos = int(pu[0]); seen = set(u2items_tr.get(int(u), set())) | {{pos}}
        negs = []
        while len(negs) < nn:
            x = int(rng.integers(0, I))
            if x != pos and x not in seen: negs.append(x)
        if len(negs) < nn: continue
        cands = np.array([pos] + negs)
        X = make_feat_batch(np.array([int(u)] * len(cands)), cands)
        xt = torch.from_numpy(X).to(dev)
        with torch.no_grad():
            if hasattr(model, 'module'): sc = model.module(xt).cpu().numpy()
            else: sc = model(xt).cpu().numpy()
        order = np.argsort(sc)[::-1]; rp = int(np.where(order == 0)[0][0]); n += 1
        for K in K_list:
            if rp < K:
                h[f'HR@{{K}}'] += 1; nd[f'NDCG@{{K}}'] += 1.0 / np.log2(rp + 2)
        mrr_sum += 1.0 / (rp + 1)
    for K in K_list:
        h[f'HR@{{K}}'] /= n; nd[f'NDCG@{{K}}'] /= n
    return {{'HR@5': h['HR@5'], 'HR@10': h['HR@10'], 'HR@20': h['HR@20'],
            'NDCG@5': nd['NDCG@5'], 'NDCG@10': nd['NDCG@10'], 'NDCG@20': nd['NDCG@20'],
            'MRR': mrr_sum / n, 'n': n}}

for ep in range(15):
    model.train(); tl = 0; n = 0
    pbar = tqdm(dl, desc=f'Epoch {{ep+1}}/15')
    for u, i, neg in pbar:
        u = u.numpy() if torch.is_tensor(u) else np.array(u)
        i = i.numpy() if torch.is_tensor(i) else np.array(i)
        neg = neg.numpy() if torch.is_tensor(neg) else np.array(neg)
        B = u.shape[0]; Nn = neg.shape[1]
        Xpos = make_feat_batch(u, i)
        neg_flat = neg.reshape(-1)
        u_rep = np.repeat(u, Nn)
        Xneg = make_feat_batch(u_rep, neg_flat)
        xt = torch.from_numpy(Xpos).to(dev); xn = torch.from_numpy(Xneg).to(dev)
        if hasattr(model, 'module'):
            pos = model.module(xt); negs = model.module(xn)
        else:
            pos = model(xt); negs = model(xn)
        loss = -torch.log(torch.sigmoid(pos.repeat_interleave(Nn) - negs) + 1e-12).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        tl += loss.item() * B; n += B
        pbar.set_postfix({{'loss': f'{{tl/n:.4f}}'}})
    sched.step()
    train_loss = tl / n

    val_metrics = eval_split(val_df, n_users=300)
    val_hr10 = val_metrics['HR@10']
    print(f'Epoch {{ep+1}}/15: loss={{train_loss:.4f}} val_HR@10={{val_hr10:.4f}} val_NDCG@10={{val_metrics["NDCG@10"]:.4f}} val_MRR={{val_metrics["MRR"]:.4f}} time={{time.time()-t0:.1f}}s', flush=True)
    history.append({{'epoch': ep+1, 'train_loss': train_loss, 'val': val_metrics}})

    if val_hr10 > best_val_hr10:
        best_val_hr10 = val_hr10
        no_improve = 0
        sd = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        torch.save(sd, '/kaggle/working/best_model.pt')
        print(f'  -> New best! Saved best_model.pt')
    else:
        no_improve += 1
        if no_improve >= 3:
            print(f'  Early stopping at epoch {{ep+1}}')
            break

# Test eval
print('\\n===== TEST SET EVALUATION =====')
test_metrics = eval_split(te, n_users=1000)
print(json.dumps(test_metrics, indent=2))
torch.save(model.module.state_dict() if hasattr(model, 'module') else model.state_dict(), '/kaggle/working/final_model.pt')
open('/kaggle/working/eval_results.json', 'w').write(json.dumps({{'history': history, 'test': test_metrics}}, indent=2))
print('SAVED: best_model.pt, final_model.pt, eval_results.json')
""")

code("""# Save model
sd = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
torch.save(sd, '/kaggle/working/MODEL_NAME_model.pt')
print('Model saved')
""")

nb = {"nbformat": 4, "nbformat_minor": 5, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}, "cells": cells}
outpath = f'/tmp/opencode/comprehensive_template.ipynb'
open(outpath, 'w').write(json.dumps(nb))
print(f'created {outpath}')
