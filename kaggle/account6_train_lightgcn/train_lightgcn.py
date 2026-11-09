#!/usr/bin/env python
"""Kaggle: Train LightGCN only (graph-based CF). Saves graph_user/item factors.

Run this BEFORE the graph_* per-model kernels.
"""
import os, sys, time, json, math, gc, subprocess
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from pathlib import Path
try:
    import torch, faiss, pyarrow.parquet as pq, implicit, sentence_transformers
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet",
        "sentence-transformers>=2.7", "faiss-cpu", "implicit>=0.7", "pyarrow", "scipy", "tqdm"])
    import torch, faiss, pyarrow.parquet as pq, implicit, sentence_transformers
import numpy as np
import pandas as pd
import torch.nn as nn
from scipy.sparse import csr_matrix, coo_matrix
from tqdm import tqdm

_T0 = time.time()

def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')} +{time.time() - _T0:7.1f}s] {msg}", flush=True)

def mem_info() -> str:
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return f"{rss / 1e9:.1f} GB" if rss > 1e9 else f"{rss / 1e6:.1f} MB"
    except Exception:
        return "n/a"

def gpu_mem() -> str:
    if not torch.cuda.is_available():
        return "CPU"
    return " | ".join(f"GPU{i} alloc={torch.cuda.memory_allocated(i)/1e9:.1f}G" for i in range(torch.cuda.device_count()))

MIN_INTERACTIONS = 3
CATALOG = 200_000
SEED = 42
GRAPH_DIM = 32
N_EPOCHS = 50
OUT_DIR = Path("/kaggle/working" if Path("/kaggle/working").exists() else "artifacts")
OUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log(f"DEVICE={DEVICE} | GRAPH_DIM={GRAPH_DIM} N_EPOCHS={N_EPOCHS} | {gpu_mem()} | RAM {mem_info()}")


def find_base_dir():
    for c in [Path("/kaggle/input/lastone/amazon_recs_output"), Path("/kaggle/input/lastone"),
              Path("/kaggle/input/datasets/nguynteo/lastone"), Path("data")]:
        if (c / "train").exists() or (c / "test").exists():
            return c
    if Path("/kaggle/input").exists():
        for p in Path("/kaggle/input").rglob("train"):
            if p.is_dir():
                return p.parent
    return Path("data")


BASE_DIR = find_base_dir()

# ---- Load train ----
train_files = sorted(BASE_DIR.glob("train/*.parquet"))
dfs = []
for f in tqdm(train_files, desc="Reading train"):
    d = pq.read_table(f, columns=["user_idx", "item_idx", "rating"]).to_pandas()
    d = d[d["rating"] >= 4.0]
    dfs.append(d[["user_idx", "item_idx"]])
train_df = pd.concat(dfs, ignore_index=True)
user_counts = train_df["user_idx"].value_counts()
active = user_counts[user_counts >= MIN_INTERACTIONS].index
train_df = train_df[train_df["user_idx"].isin(active)]
u_raw = train_df["user_idx"].to_numpy(np.int64)
i_raw = train_df["item_idx"].to_numpy(np.int64)
u_uniq, u_inv = np.unique(u_raw, return_inverse=True)
i_uniq, i_inv = np.unique(i_raw, return_inverse=True)
n_users, n_items = len(u_uniq), len(i_uniq)
log(f"[STEP 1] Loaded train: {n_users:,} users x {n_items:,} items, {len(train_df):,} interactions | RAM {mem_info()}")

# ---- Build graph ----
src = u_inv.astype(np.int64)
dst = (n_users + i_inv).astype(np.int64)
rows = np.concatenate([src, dst])
cols = np.concatenate([dst, src])
vals = np.ones(len(rows), dtype=np.float32)
N_nodes = n_users + n_items
adj = coo_matrix((vals, (rows, cols)), shape=(N_nodes, N_nodes)).tocsr()
deg = np.asarray(adj.sum(axis=1)).flatten()
deg_inv_sqrt = np.power(deg, -0.5)
deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.0
D = coo_matrix((deg_inv_sqrt, (np.arange(N_nodes), np.arange(N_nodes))), shape=(N_nodes, N_nodes)).tocsr()
adj_norm = (D @ adj @ D).tocsr()
adj_coo = adj_norm.tocoo()
indices = torch.tensor(np.vstack([adj_coo.row, adj_coo.col]), dtype=torch.long, device=DEVICE)
values = torch.tensor(adj_coo.data, dtype=torch.float32, device=DEVICE)
adj_sparse = torch.sparse_coo_tensor(indices, values, (N_nodes, N_nodes)).coalesce()

pos_edges = list(zip(u_inv.tolist(), i_inv.tolist()))
pos_set = set(pos_edges)
n_edges = len(pos_edges)
rng = np.random.default_rng(SEED)
log(f"[STEP 2] Graph built: {N_nodes:,} nodes, {adj_sparse._nnz():,} nonzero | {gpu_mem()} | RAM {mem_info()}")

# ---- LightGCN ----
torch.manual_seed(SEED)
np.random.seed(SEED)
E0 = nn.Embedding(N_nodes, GRAPH_DIM)
nn.init.xavier_uniform_(E0.weight)
E0 = E0.to(DEVICE)
optimizer = torch.optim.Adam(E0.parameters(), lr=0.01)
E0.train()
batch_size = 4096
epoch_edges = min(500_000, n_edges)
log(f"[STEP 3] Training LightGCN: N_EPOCHS={N_EPOCHS} epoch_edges={epoch_edges:,} batch_size={batch_size}")
t_train = time.time()
for epoch in range(N_EPOCHS):
    perm = rng.choice(n_edges, size=epoch_edges, replace=False)
    total_loss = 0.0
    n_batches = 0
    for start in range(0, epoch_edges, batch_size):
        idx = perm[start:start + batch_size]
        if len(idx) == 0:
            continue
        u_batch = np.array([pos_edges[i][0] for i in idx], dtype=np.int64)
        i_batch = np.array([pos_edges[i][1] for i in idx], dtype=np.int64)
        neg_items = rng.integers(0, n_items, size=len(idx))
        for j in range(len(idx)):
            while (u_batch[j], neg_items[j]) in pos_set:
                neg_items[j] = rng.integers(0, n_items)
        u_t = torch.tensor(u_batch, dtype=torch.long, device=DEVICE)
        pos_i_t = torch.tensor(n_users + i_batch, dtype=torch.long, device=DEVICE)
        neg_i_t = torch.tensor(n_users + neg_items, dtype=torch.long, device=DEVICE)
        E = E0.weight
        E_list = [E]
        E_cur = E
        for _ in range(2):
            E_cur = torch.sparse.mm(adj_sparse, E_cur)
            E_list.append(E_cur)
        E_final = torch.stack(E_list).mean(dim=0)
        u_emb = E_final[u_t]
        pos_emb = E_final[pos_i_t]
        neg_emb = E_final[neg_i_t]
        pos_score = (u_emb * pos_emb).sum(dim=1)
        neg_score = (u_emb * neg_emb).sum(dim=1)
        loss = -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-8).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    avg = total_loss / max(1, n_batches)
    if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == N_EPOCHS - 1:
        log(f"[STEP 3] Epoch {epoch+1:>3}/{N_EPOCHS} loss={avg:.4f} | {time.time()-t_train:.1f}s elapsed | {gpu_mem()}")

log(f"[STEP 3] Training finished in {time.time()-t_train:.1f}s")
with torch.no_grad():
    E = E0.weight
    E_list = [E]
    E_cur = E
    for _ in range(2):
        E_cur = torch.sparse.mm(adj_sparse, E_cur)
        E_list.append(E_cur)
    E_final = torch.stack(E_list).mean(dim=0).cpu().numpy()
graph_user = E_final[:n_users].astype(np.float32)
graph_item = E_final[n_users:].astype(np.float32)

log("[STEP 4] Saving artifacts...")
np.save(OUT_DIR / "graph_user_factors_f16.npy", graph_user.astype(np.float16))
np.save(OUT_DIR / "graph_item_factors_f16.npy", graph_item.astype(np.float16))
np.save(OUT_DIR / "user_ids.npy", u_uniq)
np.save(OUT_DIR / "item_ids.npy", i_uniq)
log(f"[ARTIFACT SAVED] graph_user {graph_user.shape}, graph_item {graph_item.shape} -> {OUT_DIR}")
log(f"DONE in {time.time()-_T0:.1f}s | RAM {mem_info()}")
print("DONE", flush=True)
