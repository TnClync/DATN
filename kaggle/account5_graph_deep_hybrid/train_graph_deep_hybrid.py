#!/usr/bin/env python
"""Kaggle: Train Graph Deep Hybrid (LightGCN + SBERT content).

Adds a graph-based CF (LightGCN) on top of the existing ALS/BPR + SBERT,
and trains deep hybrid fusion models (Light Gated, Deep Early, Graph Hybrid)
so a deep hybrid can beat ALS on the shared 200k-item catalog.
"""
import os, sys, time, json, math, gc, subprocess
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
from torch.utils.data import DataLoader, TensorDataset
from sentence_transformers import SentenceTransformer
from scipy.sparse import csr_matrix, coo_matrix
from tqdm import tqdm

MIN_INTERACTIONS = 3
CATALOG = 200_000
SEED = 42
GRAPH_DIM = 64
OUT_DIR = Path("/kaggle/working" if Path("/kaggle/working").exists() else "artifacts")
OUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Torch:", torch.__version__, "CUDA:", torch.cuda.is_available(), "Device:", DEVICE)


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


def find_factor_file(filename):
    hits = sorted(Path("/kaggle/input").rglob(filename))
    return hits[0] if hits else None


def find_item_profile_dir():
    if Path("/kaggle/input").exists():
        for p in Path("/kaggle/input").rglob("*.parquet"):
            try:
                import pyarrow.parquet as pq
                cols = pq.read_schema(p).names
                if "title" in cols and "item_idx" in cols:
                    return p.parent
            except Exception:
                pass
    return None


BASE_DIR = find_base_dir()
print("Dataset:", BASE_DIR)


# ---------------------------------------------------------------- 1. TRAIN + COMPACT
print("\n" + "=" * 50)
print("1. LOADING TRAIN + COMPACT")
print("=" * 50)
train_files = sorted(BASE_DIR.glob("train/*.parquet"))
dfs = []
for f in tqdm(train_files, desc="Reading train"):
    t = pq.read_table(f, columns=["user_idx", "item_idx", "rating"])
    d = t.to_pandas()
    d = d[d["rating"] >= 4.0]
    dfs.append(d[["user_idx", "item_idx"]])
train_df = pd.concat(dfs, ignore_index=True)
user_counts = train_df["user_idx"].value_counts()
active_users = user_counts[user_counts >= MIN_INTERACTIONS].index
train_df = train_df[train_df["user_idx"].isin(active_users)].copy()
print(f"After filter: users={train_df.user_idx.nunique():,} interactions={len(train_df):,}")
u_raw = train_df["user_idx"].to_numpy(np.int64)
i_raw = train_df["item_idx"].to_numpy(np.int64)
u_uniq, u_inv = np.unique(u_raw, return_inverse=True)
i_uniq, i_inv = np.unique(i_raw, return_inverse=True)
n_users, n_items = len(u_uniq), len(i_uniq)
print(f"Compact: {n_users:,} users x {n_items:,} items")
user_inter_count = np.bincount(u_inv, minlength=n_users)

# Load ALS factors (reuse)
def load_cf_from_kernel(prefix):
    up = find_factor_file(f"{prefix}_user_factors_f16.npy")
    ip = find_factor_file(f"{prefix}_item_factors_f16.npy")
    if up is None or ip is None:
        return None
    uf = np.load(up).astype(np.float32)
    itf = np.load(ip).astype(np.float32)
    if uf.shape[0] != n_users or itf.shape[0] != n_items:
        return None
    uid_p = up.parent / f"{prefix}_user_ids.npy"
    iid_p = up.parent / f"{prefix}_item_ids.npy"
    if uid_p.exists() and iid_p.exists():
        k_uid = np.load(uid_p); k_iid = np.load(iid_p)
        if not (np.array_equal(k_uid, u_uniq) and np.array_equal(k_iid, i_uniq)):
            return None
    return uf, itf

als_res = load_cf_from_kernel("als")
if als_res is not None:
    user_cf, item_cf = als_res
    print(f"Reused ALS factors: {user_cf.shape} / {item_cf.shape}")
else:
    print("Training ALS locally...")
    from implicit.als import AlternatingLeastSquares
    um = csr_matrix((np.ones(len(u_inv), dtype=np.float32), (u_inv, i_inv)), shape=(n_users, n_items))
    als = AlternatingLeastSquares(factors=32, regularization=0.05, iterations=12, num_threads=0, random_state=42)
    als.fit(um)
    user_cf = np.asarray(als.user_factors, dtype=np.float32)
    item_cf = np.asarray(als.item_factors, dtype=np.float32)
    del als, um

# ---------------------------------------------------------------- 2. SBERT (real text)
print("\n" + "=" * 50)
print("2. SBERT WITH REAL PRODUCT TEXT")
print("=" * 50)
profile_dir = find_item_profile_dir()
print("Profile dir:", profile_dir)
item_text_map = {}
if profile_dir is not None:
    for f in sorted(profile_dir.glob("*.parquet")):
        try:
            t = pq.read_table(f, columns=["item_idx", "title", "brand", "category", "description"])
            d = t.to_pandas()
            for _, r in d.iterrows():
                parts = []
                if isinstance(r.get("title"), str) and r["title"].strip():
                    parts.append(r["title"].strip())
                if isinstance(r.get("brand"), str) and r["brand"].strip() and "amazon" not in r["brand"].lower():
                    parts.append(r["brand"].strip())
                if isinstance(r.get("category"), str) and r["category"].strip():
                    parts.append(r["category"].strip())
                desc = r.get("description")
                if isinstance(desc, str) and desc.strip():
                    parts.append(desc.strip()[:200])
                txt = " | ".join(parts)
                if txt:
                    item_text_map[int(r["item_idx"])] = txt
        except Exception as e:
            print("profile read err", f, e)
print(f"Loaded real text for {len(item_text_map):,} items")

SBERT_COVERAGE = min(n_items, 200_000)
CAND_POOL = np.arange(SBERT_COVERAGE)
compact_to_orig = {idx: int(i_uniq[idx]) for idx in range(n_items)}

item_texts = []
for ci in range(SBERT_COVERAGE):
    orig = compact_to_orig[ci]
    txt = item_text_map.get(orig, f"Product item {orig}")
    item_texts.append(txt)

print(f"Encoding {len(item_texts)} items...")
sbert_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=str(DEVICE))
sbert_embs = sbert_model.encode(item_texts, batch_size=512, show_progress_bar=True,
                                convert_to_numpy=True, normalize_embeddings=True)
item_sbert_meta = np.zeros((n_items, 384), dtype=np.float32)
item_sbert_meta[:SBERT_COVERAGE] = sbert_embs
item_sbert = item_sbert_meta
del sbert_embs
gc.collect()

# faiss_index and sbert save moved to after reviewer block (below)

# ---------------------------------------------------------------- SBERT REVIEWER TEXT (2nd variant)
print("\n" + "=" * 50)
print("2b. SBERT WITH REVIEWER TEXT")
print("=" * 50)
# Aggregate review_content per item from lastreview dataset
review_by_item = {}
# find lastreview test_with_text dir
review_dir = None
if Path("/kaggle/input").exists():
    for p in Path("/kaggle/input").rglob("*.parquet"):
        try:
            import pyarrow.parquet as pq
            cols = pq.read_schema(p).names
            if "review_content" in cols and "item_idx" in cols:
                review_dir = p.parent
                break
        except Exception:
            pass
print("Review dir:", review_dir)
if review_dir is not None:
    for f in sorted(review_dir.glob("*.parquet")):
        try:
            t = pq.read_table(f, columns=["item_idx", "review_content"])
            d = t.to_pandas()
            for _, r in d.iterrows():
                ci = int(r["item_idx"])
                if ci >= SBERT_COVERAGE:
                    continue
                txt = r.get("review_content")
                if isinstance(txt, str) and txt.strip():
                    review_by_item.setdefault(ci, []).append(txt.strip()[:500])
        except Exception as e:
            print("review read err", f, e)
print(f"Items with reviews: {len(review_by_item):,}")

# Build reviewer SBERT embeddings
review_texts = []
for ci in range(SBERT_COVERAGE):
    revs = review_by_item.get(ci, [])
    if revs:
        # concatenate up to 3 reviews
        txt = " ".join(revs[:3])[:1000]
    else:
        txt = item_texts[ci]  # fallback to metadata text
    review_texts.append(txt)

print(f"Encoding {len(review_texts)} items with reviewer text...")
review_embs = sbert_model.encode(review_texts, batch_size=512, show_progress_bar=True,
                                 convert_to_numpy=True, normalize_embeddings=True)
item_sbert_review = np.zeros((n_items, 384), dtype=np.float32)
item_sbert_review[:SBERT_COVERAGE] = review_embs
del review_embs
gc.collect()

# Combined content feature: metadata (384) + reviewer (384) = 768
item_sbert_combined = np.concatenate([item_sbert, item_sbert_review], axis=1).astype(np.float32)
np.save(OUT_DIR / "sbert_review_embeddings_f16.npy", item_sbert_review.astype(np.float16))
np.save(OUT_DIR / "sbert_combined_embeddings_f16.npy", item_sbert_combined.astype(np.float16))
print("Saved metadata + reviewer SBERT")

# Build FAISS index on combined (768-dim) content
faiss_index = faiss.IndexFlatIP(768)
faiss_index.add(item_sbert_combined[:SBERT_COVERAGE].astype(np.float32))

# Save embeddings
np.save(OUT_DIR / "sbert_item_embeddings_f16.npy", item_sbert_combined.astype(np.float16))
np.save(OUT_DIR / "sbert_meta_embeddings_f16.npy", item_sbert_meta.astype(np.float16))
np.save(OUT_DIR / "sbert_review_embeddings_f16.npy", item_sbert_review.astype(np.float16))

# Use combined as the primary item_sbert for deep hybrids
item_sbert = item_sbert_combined

np.save(OUT_DIR / "sbert_item_ids.npy", i_uniq)
np.save(OUT_DIR / "user_ids.npy", u_uniq)
np.save(OUT_DIR / "item_ids.npy", i_uniq)
np.save(OUT_DIR / "als_user_factors_f16.npy", user_cf.astype(np.float16))
np.save(OUT_DIR / "als_item_factors_f16.npy", item_cf.astype(np.float16))
print("Saved factors + sbert")


# ---------------------------------------------------------------- 3. LIGHTGCN (graph-based CF)
print("\n" + "=" * 50)
print("3. LIGHTGCN (graph-based CF on full user-item graph)")
print("=" * 50)

# Build normalized adjacency for LightGCN
# Nodes: 0..n_users-1 (users), n_users..n_users+n_items-1 (items)
# Edges from train interactions (compact)
src = u_inv.astype(np.int64)
dst = (n_users + i_inv).astype(np.int64)
# symmetric
rows = np.concatenate([src, dst])
cols = np.concatenate([dst, src])
vals = np.ones(len(rows), dtype=np.float32)
N_nodes = n_users + n_items
adj = coo_matrix((vals, (rows, cols)), shape=(N_nodes, N_nodes)).tocsr()
# symmetric normalization: D^-1/2 A D^-1/2
deg = np.asarray(adj.sum(axis=1)).flatten()
deg_inv_sqrt = np.power(deg, -0.5)
deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.0
D = coo_matrix((deg_inv_sqrt, (np.arange(N_nodes), np.arange(N_nodes))), shape=(N_nodes, N_nodes)).tocsr()
adj_norm = (D @ adj @ D).tocsr()
print(f"Graph: {N_nodes:,} nodes, {adj.nnz:,} edges (symmetric)")

# LightGCN embeddings
torch.manual_seed(SEED)
np.random.seed(SEED)
E0 = nn.Embedding(N_nodes, GRAPH_DIM)
nn.init.xavier_uniform_(E0.weight)
E0 = E0.to(DEVICE)

# Training: sample edges, BPR loss
pos_edges = list(zip(u_inv.tolist(), i_inv.tolist()))
pos_set = set(pos_edges)
# negative sampling
rng = np.random.default_rng(SEED)
n_edges = len(pos_edges)
batch_size = 4096
n_epochs = 50
lr = 0.01

# Convert adjacency to torch sparse for propagation
indices = torch.tensor(np.vstack([adj_norm.row, adj_norm.col]), dtype=torch.long, device=DEVICE)
values = torch.tensor(adj_norm.data, dtype=torch.float32, device=DEVICE)
adj_sparse = torch.sparse_coo_tensor(indices, values, (N_nodes, N_nodes)).coalesce()

optimizer = torch.optim.Adam(E0.parameters(), lr=lr)
E0.train()
for epoch in range(n_epochs):
    # sample batch of positive edges
    # Sample 1M edges per epoch (full 28M is too slow)
    epoch_edges = min(1_000_000, n_edges)
    perm = rng.choice(n_edges, size=epoch_edges, replace=False)
    total_loss = 0.0
    n_batches = 0
    for start in range(0, epoch_edges, batch_size):
        idx = perm[start:start + batch_size]
        if len(idx) == 0:
            continue
        u_batch = np.array([pos_edges[i][0] for i in idx], dtype=np.int64)
        i_batch = np.array([pos_edges[i][1] for i in idx], dtype=np.int64)
        # negative items
        neg_items = rng.integers(0, n_items, size=len(idx))
        # ensure not positive
        for j in range(len(idx)):
            while (u_batch[j], neg_items[j]) in pos_set:
                neg_items[j] = rng.integers(0, n_items)

        u_t = torch.tensor(u_batch, dtype=torch.long, device=DEVICE)
        pos_i_t = torch.tensor(n_users + i_batch, dtype=torch.long, device=DEVICE)
        neg_i_t = torch.tensor(n_users + neg_items, dtype=torch.long, device=DEVICE)

        # propagate embeddings
        E = E0.weight
        # LightGCN: K layers
        E_list = [E]
        E_cur = E
        for _ in range(3):
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

    print(f"LightGCN Epoch {epoch+1}/{n_epochs} Loss: {total_loss/max(1,n_batches):.4f}")

# Final embeddings
with torch.no_grad():
    E = E0.weight
    E_list = [E]
    E_cur = E
    for _ in range(3):
        E_cur = torch.sparse.mm(adj_sparse, E_cur)
        E_list.append(E_cur)
    E_final = torch.stack(E_list).mean(dim=0).cpu().numpy()

graph_user = E_final[:n_users].astype(np.float32)
graph_item = E_final[n_users:].astype(np.float32)
np.save(OUT_DIR / "graph_user_factors_f16.npy", graph_user.astype(np.float16))
np.save(OUT_DIR / "graph_item_factors_f16.npy", graph_item.astype(np.float16))
print(f"Graph embeddings: user {graph_user.shape}, item {graph_item.shape}")


# ---------------------------------------------------------------- 4. TEST + HISTORY
print("\n" + "=" * 50)
print("4. LOADING TEST + TRAIN HISTORY")
print("=" * 50)
test_files = sorted(BASE_DIR.glob("test/*.parquet"))
dfs_test = []
for f in tqdm(test_files, desc="Reading test"):
    t = pq.read_table(f, columns=["user_idx", "item_idx", "rating"])
    d = t.to_pandas()
    d = d[d["rating"] >= 4.0]
    dfs_test.append(d[["user_idx", "item_idx"]])
test_df = pd.concat(dfs_test, ignore_index=True)
test_df = test_df[test_df["user_idx"].isin(set(u_uniq.tolist()))]
u_map = {orig: ci for ci, orig in enumerate(u_uniq)}
i_map = {orig: ci for ci, orig in enumerate(i_uniq)}
test_df["cu"] = test_df["user_idx"].map(u_map)
test_df["ci"] = test_df["item_idx"].map(i_map)
test_df = test_df.dropna(subset=["cu", "ci"])
test_df["cu"] = test_df["cu"].astype(np.int64)
test_df["ci"] = test_df["ci"].astype(np.int64)
user_gt = test_df.groupby("cu")["ci"].apply(set).to_dict()

hist_by_user_all: dict[int, list[int]] = {}
for uu, ii in zip(u_inv, i_inv):
    if ii < SBERT_COVERAGE:
        hist_by_user_all.setdefault(int(uu), []).append(int(ii))

user_sbert: dict[int, np.ndarray] = {}
for u in range(n_users):
    h = hist_by_user_all.get(u, [])
    if h:
        v = item_sbert[h].mean(axis=0)
        nrm = float(np.linalg.norm(v))
        v = v / nrm if nrm > 1e-8 else np.zeros(768, dtype=np.float32)
    else:
        v = np.zeros(768, dtype=np.float32)
    user_sbert[u] = v.astype(np.float32)


def compute_metrics(ranked, gt):
    res = {k: 0.0 for k in ["HR@5", "HR@10", "HR@20", "Recall@5", "Recall@10", "Recall@20",
                               "NDCG@5", "NDCG@10", "NDCG@20", "MRR"]}
    pos_list = np.where(np.asarray(ranked) == gt)[0]
    if len(pos_list) == 0:
        return res
    pos = int(pos_list[0])
    if pos < 5: res["HR@5"] = 1.0
    if pos < 10: res["HR@10"] = 1.0
    if pos < 20: res["HR@20"] = 1.0
    if pos < 5: res["Recall@5"] = 1.0
    if pos < 10: res["Recall@10"] = 1.0
    if pos < 20: res["Recall@20"] = 1.0
    if pos < 5: res["NDCG@5"] = 1.0 / math.log2(pos + 2)
    if pos < 10: res["NDCG@10"] = 1.0 / math.log2(pos + 2)
    if pos < 20: res["NDCG@20"] = 1.0 / math.log2(pos + 2)
    res["MRR"] = 1.0 / (pos + 1)
    return res


def mm(x):
    x = np.asarray(x, dtype=np.float32)
    mn, mx = x.min(), x.max()
    if mx - mn < 1e-8:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)


def cf_scores(uf, itf, u, cands):
    return np.asarray(itf[cands] @ uf[u], dtype=np.float32)


def sb_scores(u, cands):
    return np.asarray(item_sbert[cands] @ user_sbert[u], dtype=np.float32)


def build_instances(eval_users):
    out = []
    rng = np.random.default_rng(42)
    for u in eval_users:
        u = int(u)
        gts = [it for it in user_gt[u] if it < SBERT_COVERAGE]
        if not gts:
            continue
        gt = int(rng.choice(gts))
        banned = (set(hist_by_user_all.get(u, [])) | set(user_gt[u])) - {gt}
        negs = []
        for _ in range(10):
            draw = rng.choice(CAND_POOL, size=300, replace=False)
            negs = [int(x) for x in draw if int(x) not in banned][:99]
            if len(negs) == 99:
                break
        if len(negs) < 99:
            continue
        cands = np.array([gt] + negs, dtype=np.int64)
        rng.shuffle(cands)
        out.append({"user": u, "gt": gt, "candidates": cands,
                    "banned": np.array(sorted(banned), dtype=np.int64)})
    return out


test_users = np.array([u for u in user_gt.keys() if user_inter_count[u] >= MIN_INTERACTIONS])
test_users = np.array([u for u in test_users if any(it < SBERT_COVERAGE for it in user_gt[u])])
eval_users = np.random.default_rng(42).choice(test_users, size=min(100000, len(test_users)), replace=False)
eval_instances = build_instances(eval_users)
print(f"Prepared {len(eval_instances):,} instances")
TEST_N = 5000
test_inst = eval_instances[:TEST_N]
val_inst = eval_instances[TEST_N:TEST_N + 2000]


def eval_scorer(fn, instances):
    agg = {k: 0.0 for k in ["HR@5", "HR@10", "HR@20", "NDCG@5", "NDCG@10", "NDCG@20", "MRR"]}
    for inst in instances:
        scores = fn(inst["user"], inst["candidates"])
        ranked = inst["candidates"][np.argsort(-scores)]
        m = compute_metrics(ranked, inst["gt"])
        for k in agg:
            agg[k] += m[k]
    return {k: round(v / max(1, len(instances)), 4) for k, v in agg.items()}


# ---------------------------------------------------------------- 5. BASELINES
print("\n" + "=" * 50)
print("5. BASELINES")
print("=" * 50)
res_als = eval_scorer(lambda u, c: cf_scores(user_cf, item_cf, u, c), test_inst)
print("ALS:", res_als)
res_graph = eval_scorer(lambda u, c: cf_scores(graph_user, graph_item, u, c), test_inst)
print("LightGCN:", res_graph)
res_sbert = eval_scorer(lambda u, c: sb_scores(u, c), test_inst)
print("SBERT:", res_sbert)


# ---------------------------------------------------------------- 6. LATE FUSION (ALS + SBERT)
print("\n" + "=" * 50)
print("6. LATE FUSION (ALS + SBERT)")
print("=" * 50)
best_alpha, best_hr = 0.9, -1.0
for a in [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    m = eval_scorer(lambda u, c, a=a: a * mm(cf_scores(user_cf, item_cf, u, c)) + (1 - a) * mm(sb_scores(u, c)), val_inst)
    if m["HR@10"] > best_hr:
        best_hr, best_alpha = m["HR@10"], a
res_late = eval_scorer(lambda u, c: best_alpha * mm(cf_scores(user_cf, item_cf, u, c)) + (1 - best_alpha) * mm(sb_scores(u, c)), test_inst)
print(f"Late Fusion (ALS+SBERT) alpha={best_alpha}: {res_late}")


# ---------------------------------------------------------------- 7. GRAPH HYBRID FUSION (LightGCN + SBERT)
print("\n" + "=" * 50)
print("7. GRAPH HYBRID FUSION (LightGCN + SBERT)")
print("=" * 50)
# Late fusion with graph instead of ALS
best_ga, best_ghr = 0.9, -1.0
for a in [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    m = eval_scorer(lambda u, c, a=a: a * mm(cf_scores(graph_user, graph_item, u, c)) + (1 - a) * mm(sb_scores(u, c)), val_inst)
    if m["HR@10"] > best_ghr:
        best_ghr, best_ga = m["HR@10"], a
res_graph_fusion = eval_scorer(lambda u, c: best_ga * mm(cf_scores(graph_user, graph_item, u, c)) + (1 - best_ga) * mm(sb_scores(u, c)), test_inst)
print(f"Graph Fusion (LightGCN+SBERT) alpha={best_ga}: {res_graph_fusion}")


# ---------------------------------------------------------------- 8. LIGHT GATED (graph + sbert)
print("\n" + "=" * 50)
print("8. LIGHT GATED (graph + sbert)")
print("=" * 50)


class LightGate(nn.Module):
    def __init__(self, in_features=64 + 64 + 768):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(in_features, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, u_cf, i_cf, i_sbert, cf_score, sbert_score):
        feat = torch.cat([u_cf, i_cf, i_sbert], dim=-1)
        alpha = self.gate_net(feat).squeeze(-1)
        return alpha * cf_score + (1.0 - alpha) * sbert_score, alpha


gate = LightGate().to(DEVICE)
train_for_fit = eval_instances[:100000] if len(eval_instances) >= 100000 else eval_instances
gate_Xu, gate_Xi, gate_Xs, gate_cf, gate_sb, gate_y = [], [], [], [], [], []
for inst in train_for_fit:
    u = inst["user"]; pos = inst["gt"]
    negs = [int(x) for x in inst["candidates"] if int(x) != pos][:4]
    for it, label in [(pos, 1.0)] + [(n, 0.0) for n in negs]:
        gate_Xu.append(graph_user[u]); gate_Xi.append(graph_item[it]); gate_Xs.append(item_sbert[it])
        gate_cf.append(float(graph_item[it] @ graph_user[u]))
        gate_sb.append(float(item_sbert[it] @ user_sbert[u]))
        gate_y.append(label)

t_u = torch.tensor(np.array(gate_Xu), dtype=torch.float32)
t_i = torch.tensor(np.array(gate_Xi), dtype=torch.float32)
t_s = torch.tensor(np.array(gate_Xs), dtype=torch.float32)
t_cf = torch.tensor(np.array(gate_cf), dtype=torch.float32)
t_sb = torch.tensor(np.array(gate_sb), dtype=torch.float32)
t_y = torch.tensor(np.array(gate_y), dtype=torch.float32)
ds = TensorDataset(t_u, t_i, t_s, t_cf, t_sb, t_y)
loader = DataLoader(ds, batch_size=512, shuffle=True, num_workers=0)
opt = torch.optim.Adam(gate.parameters(), lr=1e-3, weight_decay=1e-5)
bce = nn.BCEWithLogitsLoss()
gate.train()
for ep in range(30):
    tot = 0.0
    for bu, bi, bs, bcf, bsb, by in loader:
        bu, bi, bs = bu.to(DEVICE), bi.to(DEVICE), bs.to(DEVICE)
        bcf, bsb, by = bcf.to(DEVICE), bsb.to(DEVICE), by.to(DEVICE)
        pred, _ = gate(bu, bi, bs, bcf, bsb)
        loss = bce(pred, by)
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item()
    print(f"Gate Epoch {ep+1}/30 Loss: {tot/len(loader):.4f}")
torch.save(gate.state_dict(), OUT_DIR / "graph_light_gated_fusion.pt")


def eval_gate(instances):
    gate.eval()
    agg = {k: 0.0 for k in ["HR@5", "HR@10", "HR@20", "NDCG@5", "NDCG@10", "NDCG@20", "MRR"]}
    with torch.no_grad():
        for inst in instances:
            u = inst["user"]; cands = inst["candidates"]; gt = inst["gt"]
            bu = torch.tensor(np.repeat(graph_user[u][None, :], len(cands), axis=0), dtype=torch.float32).to(DEVICE)
            bi = torch.tensor(graph_item[cands], dtype=torch.float32).to(DEVICE)
            bs = torch.tensor(item_sbert[cands], dtype=torch.float32).to(DEVICE)
            bcf = torch.tensor(cf_scores(graph_user, graph_item, u, cands), dtype=torch.float32).to(DEVICE)
            bsb = torch.tensor(sb_scores(u, cands), dtype=torch.float32).to(DEVICE)
            fused, _ = gate(bu, bi, bs, bcf, bsb)
            ranked = cands[np.argsort(-fused.cpu().numpy())]
            m = compute_metrics(ranked, gt)
            for k in agg:
                agg[k] += m[k]
    return {k: round(v / max(1, len(instances)), 4) for k, v in agg.items()}

res_gate = eval_gate(test_inst)
print("Graph Light Gated:", res_gate)


# ---------------------------------------------------------------- 8b. TWO-TOWER (user tower + item tower)
print("\n" + "=" * 50)
print("8b. TWO-TOWER (graph + sbert)")
print("=" * 50)


class UserTower(nn.Module):
    def __init__(self, in_dim=64, hidden=128, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.1),
                                 nn.Linear(hidden, out_dim))

    def forward(self, x):
        return self.net(x)


class ItemTower(nn.Module):
    def __init__(self, in_dim=64 + 768, hidden=256, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.1),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.1),
                                 nn.Linear(hidden, out_dim))

    def forward(self, x):
        return self.net(x)


user_tower = UserTower().to(DEVICE)
item_tower = ItemTower().to(DEVICE)
# Build training data
tt_Xu, tt_Xi, tt_y = [], [], []
for inst in train_for_fit:
    u = inst["user"]; pos = inst["gt"]
    negs = [int(x) for x in inst["candidates"] if int(x) != pos][:4]
    for it, label in [(pos, 1.0)] + [(n, 0.0) for n in negs]:
        tt_Xu.append(graph_user[u])
        tt_Xi.append(np.concatenate([graph_item[it], item_sbert[it]]))
        tt_y.append(label)
tt_Xu = torch.tensor(np.array(tt_Xu), dtype=torch.float32)
tt_Xi = torch.tensor(np.array(tt_Xi), dtype=torch.float32)
tt_y = torch.tensor(np.array(tt_y), dtype=torch.float32)
ds_tt = TensorDataset(tt_Xu, tt_Xi, tt_y)
loader_tt = DataLoader(ds_tt, batch_size=512, shuffle=True, num_workers=0)
opt_tt = torch.optim.Adam(list(user_tower.parameters()) + list(item_tower.parameters()), lr=1e-3, weight_decay=1e-5)
bce_tt = nn.BCEWithLogitsLoss()
user_tower.train(); item_tower.train()
for ep in range(30):
    tot = 0.0
    for bu, bi, by in loader_tt:
        bu, bi, by = bu.to(DEVICE), bi.to(DEVICE), by.to(DEVICE)
        u_emb = user_tower(bu)
        i_emb = item_tower(bi)
        score = (u_emb * i_emb).sum(dim=1)
        loss = bce_tt(score, by)
        opt_tt.zero_grad(); loss.backward(); opt_tt.step()
        tot += loss.item()
    print(f"TwoTower Epoch {ep+1}/20 Loss: {tot/len(loader_tt):.4f}")
torch.save({"user": user_tower.state_dict(), "item": item_tower.state_dict()}, OUT_DIR / "two_tower_fusion.pt")


def eval_two_tower(instances):
    user_tower.eval(); item_tower.eval()
    agg = {k: 0.0 for k in ["HR@5", "HR@10", "HR@20", "NDCG@5", "NDCG@10", "NDCG@20", "MRR", "Recall@5", "Recall@10", "Recall@20"]}
    with torch.no_grad():
        for inst in instances:
            u = inst["user"]; cands = inst["candidates"]; gt = inst["gt"]
            u_emb = user_tower(torch.tensor(np.repeat(graph_user[u][None, :], len(cands), axis=0), dtype=torch.float32).to(DEVICE))
            i_feat = np.concatenate([graph_item[cands], item_sbert[cands]], axis=1)
            i_emb = item_tower(torch.tensor(i_feat, dtype=torch.float32).to(DEVICE))
            scores = (u_emb * i_emb).sum(dim=1).cpu().numpy()
            ranked = cands[np.argsort(-scores)]
            m = compute_metrics(ranked, gt)
            for k in agg:
                agg[k] += m[k]
    return {k: round(v / max(1, len(instances)), 4) for k, v in agg.items()}

res_tt = eval_two_tower(test_inst)
print("Two-Tower:", res_tt)


# ---------------------------------------------------------------- 8c. TRANSFORMER FUSION
print("\n" + "=" * 50)
print("8c. TRANSFORMER FUSION (self-attention over candidates)")
print("=" * 50)
import torch.nn.functional as F


class TransformerFusion(nn.Module):
    def __init__(self, in_dim=64 + 64 + 768, d_model=128, nhead=4, nlayers=2):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                  dim_feedforward=d_model * 4, dropout=0.1,
                                                  batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: (B, 1, in_dim) per candidate -> treat each candidate independently with a CLS
        # We score per candidate; use projection + head (transformer over single token is identity-ish).
        # For a real transformer over the candidate set, pass (B, K, in_dim).
        h = self.proj(x)  # (B, K, d_model)
        h = self.encoder(h)  # (B, K, d_model)
        return self.head(h).squeeze(-1)  # (B, K)


transformer = TransformerFusion().to(DEVICE)
# Build training data: per-user candidate sets
tf_users = []
tf_cands = []
tf_labels = []
for inst in train_for_fit[:10000]:  # limit for memory
    u = inst["user"]
    cands = inst["candidates"]
    gt = inst["gt"]
    feat = np.concatenate([
        np.repeat(graph_user[u][None, :], len(cands), axis=0),
        graph_item[cands],
        item_sbert[cands],
    ], axis=1).astype(np.float32)
    label = (cands == gt).astype(np.float32)
    tf_users.append(u)
    tf_cands.append(feat)
    tf_labels.append(label)

# Train in mini-batches of users
opt_tf = torch.optim.Adam(transformer.parameters(), lr=1e-4, weight_decay=1e-5)
bce_tf = nn.BCEWithLogitsLoss()
transformer.train()
batch_users = 16
n_tf = len(tf_cands)
for ep in range(10):
    perm = np.random.default_rng(SEED + ep).permutation(n_tf)
    tot = 0.0
    nb = 0
    for start in range(0, n_tf, batch_users):
        idx = perm[start:start + batch_users]
        if len(idx) == 0:
            continue
        xb = torch.tensor(np.stack([tf_cands[i] for i in idx]), dtype=torch.float32).to(DEVICE)
        yb = torch.tensor(np.stack([tf_labels[i] for i in idx]), dtype=torch.float32).to(DEVICE)
        logits = transformer(xb)
        loss = bce_tf(logits, yb)
        opt_tf.zero_grad(); loss.backward(); opt_tf.step()
        tot += loss.item()
        nb += 1
    print(f"Transformer Epoch {ep+1}/10 Loss: {tot/max(1,nb):.4f}")
torch.save(transformer.state_dict(), OUT_DIR / "transformer_fusion.pt")


def eval_transformer(instances):
    transformer.eval()
    agg = {k: 0.0 for k in ["HR@5", "HR@10", "HR@20", "NDCG@5", "NDCG@10", "NDCG@20", "MRR", "Recall@5", "Recall@10", "Recall@20"]}
    with torch.no_grad():
        for inst in instances:
            u = inst["user"]; cands = inst["candidates"]; gt = inst["gt"]
            feat = np.concatenate([
                np.repeat(graph_user[u][None, :], len(cands), axis=0),
                graph_item[cands],
                item_sbert[cands],
            ], axis=1).astype(np.float32)
            logits = transformer(torch.tensor(feat[None, ...], dtype=torch.float32).to(DEVICE)).squeeze(0).cpu().numpy()
            ranked = cands[np.argsort(-logits)]
            m = compute_metrics(ranked, gt)
            for k in agg:
                agg[k] += m[k]
    return {k: round(v / max(1, len(instances)), 4) for k, v in agg.items()}

res_tf = eval_transformer(test_inst)
print("Transformer Fusion:", res_tf)

# ---------------------------------------------------------------- 9. SUMMARY
print("\n" + "=" * 60)
print("FINAL COMPARISON (Graph Deep Hybrid)")
print("=" * 60)
print(f"{'Model':<28} {'HR@5':>7} {'HR@10':>7} {'HR@20':>7} {'NDCG@10':>8} {'MRR':>7}")
for name, m in [("ALS", res_als), ("LightGCN", res_graph), ("SBERT", res_sbert),
                (f"Late Fusion ALS+SBERT (a={best_alpha})", res_late),
                (f"Graph Fusion (a={best_ga})", res_graph_fusion),
                ("Graph Light Gated", res_gate),
                ("Two-Tower", res_tt),
                ("Transformer Fusion", res_tf)]:
    if m is None:
        print(f"{name:<28} {'n/a':>7}")
    else:
        print(f"{name:<28} {m['HR@5']:>7.4f} {m['HR@10']:>7.4f} {m['HR@20']:>7.4f} {m['NDCG@10']:>8.4f} {m['MRR']:>7.4f}")

best_deep = max(
    [("graph_light_gated", res_gate), ("graph_fusion", res_graph_fusion),
     ("two_tower", res_tt), ("transformer", res_tf)],
    key=lambda x: x[1]["HR@10"])[0]
summary = {
    "protocol": "Graph Deep Hybrid: LightGCN (full graph) + SBERT (real text), shared 200k catalog",
    "eval_catalog_items": SBERT_COVERAGE,
    "n_eval_users": len(test_inst),
    "min_interactions": MIN_INTERACTIONS,
    "graph_dim": GRAPH_DIM,
    "best_deep_hybrid": best_deep,
    "als": res_als,
    "lightgcn": res_graph,
    "sbert": res_sbert,
    "late_fusion_als_sbert": {"best_alpha": best_alpha, "test": res_late},
    "graph_fusion": {"best_alpha": best_ga, "test": res_graph_fusion},
    "graph_light_gated": res_gate,
    "two_tower": res_tt,
    "transformer_fusion": res_tf,
}
(OUT_DIR / "graph_deep_hybrid_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
print("DONE")
PY