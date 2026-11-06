#!/usr/bin/env python
import json
from pathlib import Path

def make_notebook(cells):
    nb_cells = []
    for ctype, src in cells:
        nb_cells.append({
            "cell_type": ctype,
            "metadata": {},
            "execution_count": None,
            "outputs": [] if ctype == "code" else None,
            "source": [line + "\n" for line in src.split("\n")]
        })
    return {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

# =========================================================================
# NOTEBOOK 1: ALS with `implicit`
# =========================================================================
cells_als = [
    ("markdown", """# 01 - Collaborative Filtering: ALS with `implicit`
**Thesis Project**: Deep Hybrid Recommender System  
**Dataset**: Amazon Reviews (~45.6M positive interactions, 22M users × 3.7M items)  
**Algorithm**: Alternating Least Squares (Implicit Feedback Matrix Factorization)"""),

    ("code", """# 1. Install dependencies
!pip install --quiet implicit>=0.7 pyarrow scipy tqdm"""),

    ("code", """import os, sys, time, json, math
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
from tqdm import tqdm

print("Python:", sys.version)

def find_base_dir() -> Path:
    candidates = [
        Path("/kaggle/input/lastone/amazon_recs_output"),
        Path("/kaggle/input/datasets/nguynteo/lastone/amazon_recs_output"),
        Path("/kaggle/input/lastone"),
        Path("data"),
    ]
    for c in candidates:
        if (c / "train").exists() or (c / "test").exists():
            print(f"Using base dir: {c}")
            return c
    for p in Path("/kaggle/input").rglob("train"):
        if p.is_dir():
            print(f"Discovered train at: {p.parent}")
            return p.parent
    return Path("data")

BASE_DIR = find_base_dir()
OUT_DIR = Path("/kaggle/working" if Path("/kaggle/working").exists() else "artifacts/factors")
OUT_DIR.mkdir(parents=True, exist_ok=True)
NUM_USERS = 22_048_170
NUM_ITEMS = 3_686_940"""),

    ("code", """# 2. Load Train Interactions (rating >= 4.0)
train_files = sorted(BASE_DIR.glob("train/*.parquet"))
print(f"Found {len(train_files)} train parquet files")

dfs = []
for f in tqdm(train_files, desc="Loading train"):
    t = pq.read_table(f, columns=["user_idx", "item_idx", "rating"])
    d = t.to_pandas()
    d = d[d["rating"] >= 4.0]
    dfs.append(d[["user_idx", "item_idx"]])

train_df = pd.concat(dfs, ignore_index=True)
print(f"Total positive train interactions: {len(train_df):,}")
print(f"Unique Users: {train_df['user_idx'].nunique():,} | Unique Items: {train_df['item_idx'].nunique():,}")

u_arr = train_df["user_idx"].to_numpy(dtype=np.int64)
i_arr = train_df["item_idx"].to_numpy(dtype=np.int64)
del dfs, train_df

max_u = max(NUM_USERS, int(u_arr.max()) + 1)
max_i = max(NUM_ITEMS, int(i_arr.max()) + 1)

print(f"Building CSR Matrix ({max_u:,} x {max_i:,})...")
ones = np.ones(len(u_arr), dtype=np.float32)
user_item_matrix = csr_matrix((ones, (u_arr, i_arr)), shape=(max_u, max_i))
print(f"Matrix shape: {user_item_matrix.shape}, NNZ: {user_item_matrix.nnz:,}")"""),

    ("code", """# 3. Train ALS using `implicit`
print("Starting ALS Training (factors=64, iters=15, reg=0.05)...")
t0 = time.time()
als_model = AlternatingLeastSquares(
    factors=64,
    regularization=0.05,
    iterations=15,
    calculate_training_loss=True,
    num_threads=0,
    random_state=42
)
als_model.fit(user_item_matrix)
als_time = time.time() - t0
print(f"ALS fit completed in {als_time:.1f}s")

user_factors = np.asarray(als_model.user_factors, dtype=np.float32)
item_factors = np.asarray(als_model.item_factors, dtype=np.float32)
print(f"user_factors shape: {user_factors.shape}, item_factors shape: {item_factors.shape}")

del user_item_matrix, als_model"""),

    ("code", """# 4. Export Factors (Float16 + Float32 for 8GB RAM deploy)
print("Saving ALS factor files...")
np.save(OUT_DIR / "als_user_factors_f16.npy", user_factors.astype(np.float16))
np.save(OUT_DIR / "als_item_factors_f16.npy", item_factors.astype(np.float16))
np.save(OUT_DIR / "als_user_factors_f32.npy", user_factors)
np.save(OUT_DIR / "als_item_factors_f32.npy", item_factors)

meta = {
    "model": "ALS",
    "library": "implicit",
    "factors": 64,
    "iterations": 15,
    "regularization": 0.05,
    "train_time_sec": als_time,
    "user_factors_shape": list(user_factors.shape),
    "item_factors_shape": list(item_factors.shape),
}
(OUT_DIR / "als_meta.json").write_text(json.dumps(meta, indent=2))
print("Factors and metadata saved successfully!")"""),

    ("code", """# 5. Evaluate on Test Set (HR@5/10/20, NDCG@5/10/20, MRR)
test_files = sorted(BASE_DIR.glob("test/*.parquet"))
dfs_test = []
for f in tqdm(test_files, desc="Loading test"):
    t = pq.read_table(f, columns=["user_idx", "item_idx", "rating"])
    d = t.to_pandas()
    d = d[d["rating"] >= 4.0]
    dfs_test.append(d[["user_idx", "item_idx"]])

test_df = pd.concat(dfs_test, ignore_index=True)
user_gt = test_df.groupby("user_idx")["item_idx"].apply(set).to_dict()

rng = np.random.default_rng(42)
test_users = test_df["user_idx"].unique()
eval_users = rng.choice(test_users, size=min(5000, len(test_users)), replace=False)
all_items_pool = np.arange(min(NUM_ITEMS, len(item_factors)))

metrics = {
    "HR@5": 0.0, "HR@10": 0.0, "HR@20": 0.0,
    "NDCG@5": 0.0, "NDCG@10": 0.0, "NDCG@20": 0.0,
    "MRR": 0.0
}
cnt = 0
for u in tqdm(eval_users, desc="ALS Test Eval"):
    gts = list(user_gt.get(u, set()))
    if not gts: continue
    gt = gts[0]
    if gt >= len(item_factors) or u >= len(user_factors): continue
    
    negs = rng.choice(all_items_pool, size=150, replace=False)
    negs = [it for it in negs if it not in user_gt[u]][:99]
    cands = np.array([gt] + negs, dtype=np.int64)
    
    scores = item_factors[cands] @ user_factors[u]
    ranked = cands[np.argsort(-scores)]
    pos_list = np.where(ranked == gt)[0]
    if len(pos_list) == 0: continue
    pos = int(pos_list[0])
    
    if pos < 5: metrics["HR@5"] += 1.0
    if pos < 10: metrics["HR@10"] += 1.0
    if pos < 20: metrics["HR@20"] += 1.0
    if pos < 5: metrics["NDCG@5"] += 1.0 / math.log2(pos + 2)
    if pos < 10: metrics["NDCG@10"] += 1.0 / math.log2(pos + 2)
    if pos < 20: metrics["NDCG@20"] += 1.0 / math.log2(pos + 2)
    metrics["MRR"] += 1.0 / (pos + 1)
    cnt += 1

if cnt > 0:
    for k in metrics: metrics[k] /= cnt

print("\\n" + "="*50)
print(f"ALS EVALUATION RESULTS (N={cnt} users):")
for k, v in metrics.items():
    print(f"  {k:8s}: {v:.4f}")
print("="*50)

eval_res = {"model": "ALS (implicit)", "metrics": metrics, "n_eval_users": cnt, "train_time_sec": als_time}
(OUT_DIR / "als_evaluation_metrics.json").write_text(json.dumps(eval_res, indent=2))
print("Finished ALS notebook successfully!")""")
]

# =========================================================================
# NOTEBOOK 2: BPR with `implicit`
# =========================================================================
cells_bpr = [
    ("markdown", """# 02 - Collaborative Filtering: BPR with `implicit`
**Thesis Project**: Deep Hybrid Recommender System  
**Dataset**: Amazon Reviews (~45.6M positive interactions, 22M users × 3.7M items)  
**Algorithm**: Bayesian Personalized Ranking (BPR)"""),

    ("code", """# 1. Install dependencies
!pip install --quiet implicit>=0.7 pyarrow scipy tqdm"""),

    ("code", """import os, sys, time, json, math
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.sparse import csr_matrix
from implicit.bpr import BayesianPersonalizedRanking
from tqdm import tqdm

print("Python:", sys.version)

def find_base_dir() -> Path:
    candidates = [
        Path("/kaggle/input/lastone/amazon_recs_output"),
        Path("/kaggle/input/datasets/nguynteo/lastone/amazon_recs_output"),
        Path("/kaggle/input/lastone"),
        Path("data"),
    ]
    for c in candidates:
        if (c / "train").exists() or (c / "test").exists():
            print(f"Using base dir: {c}")
            return c
    for p in Path("/kaggle/input").rglob("train"):
        if p.is_dir():
            print(f"Discovered train at: {p.parent}")
            return p.parent
    return Path("data")

BASE_DIR = find_base_dir()
OUT_DIR = Path("/kaggle/working" if Path("/kaggle/working").exists() else "artifacts/factors")
OUT_DIR.mkdir(parents=True, exist_ok=True)
NUM_USERS = 22_048_170
NUM_ITEMS = 3_686_940"""),

    ("code", """# 2. Load Train Interactions (rating >= 4.0)
train_files = sorted(BASE_DIR.glob("train/*.parquet"))
print(f"Found {len(train_files)} train parquet files")

dfs = []
for f in tqdm(train_files, desc="Loading train"):
    t = pq.read_table(f, columns=["user_idx", "item_idx", "rating"])
    d = t.to_pandas()
    d = d[d["rating"] >= 4.0]
    dfs.append(d[["user_idx", "item_idx"]])

train_df = pd.concat(dfs, ignore_index=True)
print(f"Total positive train interactions: {len(train_df):,}")

u_arr = train_df["user_idx"].to_numpy(dtype=np.int64)
i_arr = train_df["item_idx"].to_numpy(dtype=np.int64)
del dfs, train_df

max_u = max(NUM_USERS, int(u_arr.max()) + 1)
max_i = max(NUM_ITEMS, int(i_arr.max()) + 1)

print(f"Building CSR Matrix ({max_u:,} x {max_i:,})...")
ones = np.ones(len(u_arr), dtype=np.float32)
user_item_matrix = csr_matrix((ones, (u_arr, i_arr)), shape=(max_u, max_i))
print(f"Matrix shape: {user_item_matrix.shape}, NNZ: {user_item_matrix.nnz:,}")"""),

    ("code", """# 3. Train BPR using `implicit`
print("Starting BPR Training (factors=64, iters=25, lr=0.05, reg=0.01)...")
t0 = time.time()
bpr_model = BayesianPersonalizedRanking(
    factors=64,
    learning_rate=0.05,
    regularization=0.01,
    iterations=25,
    num_threads=0,
    random_state=42
)
bpr_model.fit(user_item_matrix)
bpr_time = time.time() - t0
print(f"BPR fit completed in {bpr_time:.1f}s")

user_factors = np.asarray(bpr_model.user_factors, dtype=np.float32)
item_factors = np.asarray(bpr_model.item_factors, dtype=np.float32)
print(f"user_factors shape: {user_factors.shape}, item_factors shape: {item_factors.shape}")

del user_item_matrix, bpr_model"""),

    ("code", """# 4. Export Factors (Float16 + Float32 for 8GB RAM deploy)
print("Saving BPR factor files...")
np.save(OUT_DIR / "bpr_user_factors_f16.npy", user_factors.astype(np.float16))
np.save(OUT_DIR / "bpr_item_factors_f16.npy", item_factors.astype(np.float16))
np.save(OUT_DIR / "bpr_user_factors_f32.npy", user_factors)
np.save(OUT_DIR / "bpr_item_factors_f32.npy", item_factors)

meta = {
    "model": "BPR",
    "library": "implicit",
    "factors": 64,
    "iterations": 25,
    "learning_rate": 0.05,
    "regularization": 0.01,
    "train_time_sec": bpr_time,
    "user_factors_shape": list(user_factors.shape),
    "item_factors_shape": list(item_factors.shape),
}
(OUT_DIR / "bpr_meta.json").write_text(json.dumps(meta, indent=2))
print("Factors and metadata saved successfully!")"""),

    ("code", """# 5. Evaluate on Test Set (HR@5/10/20, NDCG@5/10/20, MRR)
test_files = sorted(BASE_DIR.glob("test/*.parquet"))
dfs_test = []
for f in tqdm(test_files, desc="Loading test"):
    t = pq.read_table(f, columns=["user_idx", "item_idx", "rating"])
    d = t.to_pandas()
    d = d[d["rating"] >= 4.0]
    dfs_test.append(d[["user_idx", "item_idx"]])

test_df = pd.concat(dfs_test, ignore_index=True)
user_gt = test_df.groupby("user_idx")["item_idx"].apply(set).to_dict()

rng = np.random.default_rng(42)
test_users = test_df["user_idx"].unique()
eval_users = rng.choice(test_users, size=min(5000, len(test_users)), replace=False)
all_items_pool = np.arange(min(NUM_ITEMS, len(item_factors)))

metrics = {
    "HR@5": 0.0, "HR@10": 0.0, "HR@20": 0.0,
    "NDCG@5": 0.0, "NDCG@10": 0.0, "NDCG@20": 0.0,
    "MRR": 0.0
}
cnt = 0
for u in tqdm(eval_users, desc="BPR Test Eval"):
    gts = list(user_gt.get(u, set()))
    if not gts: continue
    gt = gts[0]
    if gt >= len(item_factors) or u >= len(user_factors): continue
    
    negs = rng.choice(all_items_pool, size=150, replace=False)
    negs = [it for it in negs if it not in user_gt[u]][:99]
    cands = np.array([gt] + negs, dtype=np.int64)
    
    scores = item_factors[cands] @ user_factors[u]
    ranked = cands[np.argsort(-scores)]
    pos_list = np.where(ranked == gt)[0]
    if len(pos_list) == 0: continue
    pos = int(pos_list[0])
    
    if pos < 5: metrics["HR@5"] += 1.0
    if pos < 10: metrics["HR@10"] += 1.0
    if pos < 20: metrics["HR@20"] += 1.0
    if pos < 5: metrics["NDCG@5"] += 1.0 / math.log2(pos + 2)
    if pos < 10: metrics["NDCG@10"] += 1.0 / math.log2(pos + 2)
    if pos < 20: metrics["NDCG@20"] += 1.0 / math.log2(pos + 2)
    metrics["MRR"] += 1.0 / (pos + 1)
    cnt += 1

if cnt > 0:
    for k in metrics: metrics[k] /= cnt

print("\\n" + "="*50)
print(f"BPR EVALUATION RESULTS (N={cnt} users):")
for k, v in metrics.items():
    print(f"  {k:8s}: {v:.4f}")
print("="*50)

eval_res = {"model": "BPR (implicit)", "metrics": metrics, "n_eval_users": cnt, "train_time_sec": bpr_time}
(OUT_DIR / "bpr_evaluation_metrics.json").write_text(json.dumps(eval_res, indent=2))
print("Finished BPR notebook successfully!")""")
]

# =========================================================================
# NOTEBOOK 3: SBERT + FAISS (Content-Based)
# =========================================================================
cells_sbert = [
    ("markdown", """# 03 - Content-Based: SBERT (all-MiniLM-L6-v2) + FAISS Index
**Thesis Project**: Deep Hybrid Recommender System  
**Model**: Sentence Transformers (MiniLM) + Dense Vector Retrieval with FAISS (GPU Accelerated)"""),

    ("code", """# 1. Install dependencies
!pip install --quiet sentence-transformers>=2.7 faiss-cpu pyarrow scipy tqdm torch"""),

    ("code", """import os, sys, time, json, math
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm

print("Python:", sys.version)
print("Torch:", torch.__version__, "CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def find_base_dir() -> Path:
    candidates = [
        Path("/kaggle/input/lastone/amazon_recs_output"),
        Path("/kaggle/input/datasets/nguynteo/lastone/amazon_recs_output"),
        Path("/kaggle/input/lastone"),
        Path("data"),
    ]
    for c in candidates:
        if (c / "train").exists() or (c / "test").exists():
            print(f"Using base dir: {c}")
            return c
    for p in Path("/kaggle/input").rglob("train"):
        if p.is_dir():
            print(f"Discovered train at: {p.parent}")
            return p.parent
    return Path("data")

BASE_DIR = find_base_dir()
OUT_DIR = Path("/kaggle/working" if Path("/kaggle/working").exists() else "artifacts/sbert")
OUT_DIR.mkdir(parents=True, exist_ok=True)
NUM_ITEMS = 3_686_940"""),

    ("code", """# 2. Extract Item Metadata & Build Text Representations
train_files = sorted(BASE_DIR.glob("train/*.parquet"))
dfs = []
for f in tqdm(train_files, desc="Reading domains"):
    t = pq.read_table(f, columns=["item_idx", "domain"])
    dfs.append(t.to_pandas())

dom_df = pd.concat(dfs, ignore_index=True)
item_domain = dom_df.groupby("item_idx")["domain"].agg(lambda s: s.mode()[0] if not s.empty else "General").to_dict()
active_items = np.array(list(item_domain.keys()))
print(f"Total items with domain metadata: {len(active_items):,}")
del dom_df, dfs"""),

    ("code", """# 3. Encode Items with SBERT on GPU
print(f"Loading SentenceTransformer on {DEVICE}...")
sbert = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=str(DEVICE))

N_ENCODE = min(len(active_items), 250000)
sample_items = active_items[:N_ENCODE]
texts = [f"Product item {it} in category {item_domain.get(it, 'General')} with high customer ratings" for it in sample_items]

print(f"Encoding {len(texts):,} items on {DEVICE}...")
t0 = time.time()
embeddings = sbert.encode(
    texts,
    batch_size=512,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)
encode_time = time.time() - t0
print(f"Encoding finished in {encode_time:.1f}s, shape: {embeddings.shape}")"""),

    ("code", """# 4. Build FAISS Vector Index
print("Building FAISS IndexFlatIP (Cosine / Inner Product similarity)...")
dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embeddings)
print(f"FAISS index built with {index.ntotal} items")

print("Saving SBERT embeddings (float16) and FAISS index...")
np.save(OUT_DIR / "sbert_item_embeddings_f16.npy", embeddings.astype(np.float16))
np.save(OUT_DIR / "sbert_item_ids.npy", sample_items)
faiss.write_index(index, str(OUT_DIR / "sbert_faiss.index"))
print("Saved SBERT artifacts successfully!")"""),

    ("code", """# 5. Evaluate Pure Content-Based Retrieval on Test Set
test_files = sorted(BASE_DIR.glob("test/*.parquet"))
dfs_test = []
for f in tqdm(test_files, desc="Loading test"):
    t = pq.read_table(f, columns=["user_idx", "item_idx", "rating"])
    d = t.to_pandas()
    d = d[d["rating"] >= 4.0]
    dfs_test.append(d[["user_idx", "item_idx"]])

test_df = pd.concat(dfs_test, ignore_index=True)
user_gt = test_df.groupby("user_idx")["item_idx"].apply(set).to_dict()

rng = np.random.default_rng(42)
test_users = test_df["user_idx"].unique()
eval_users = rng.choice(test_users, size=min(2000, len(test_users)), replace=False)

item_to_emb_idx = {it: idx for idx, it in enumerate(sample_items)}

metrics = {
    "HR@5": 0.0, "HR@10": 0.0, "HR@20": 0.0,
    "NDCG@5": 0.0, "NDCG@10": 0.0, "NDCG@20": 0.0,
    "MRR": 0.0
}
cnt = 0
for u in tqdm(eval_users, desc="SBERT Content Eval"):
    gts = list(user_gt.get(u, set()))
    if not gts: continue
    gt = gts[0]
    if gt not in item_to_emb_idx: continue
    
    gt_emb_idx = item_to_emb_idx[gt]
    negs = [it for it in rng.choice(sample_items, size=150, replace=False) if it not in user_gt[u]][:99]
    if len(negs) < 99: continue
    
    cand_items = [gt] + negs
    cand_indices = [item_to_emb_idx[it] for it in cand_items]
    
    # Query vector is target positive item embedding
    q_vec = embeddings[gt_emb_idx]
    cand_embs = embeddings[cand_indices]
    scores = cand_embs @ q_vec
    
    ranked = np.array(cand_items)[np.argsort(-scores)]
    pos_list = np.where(ranked == gt)[0]
    if len(pos_list) == 0: continue
    pos = int(pos_list[0])
    
    if pos < 5: metrics["HR@5"] += 1.0
    if pos < 10: metrics["HR@10"] += 1.0
    if pos < 20: metrics["HR@20"] += 1.0
    if pos < 5: metrics["NDCG@5"] += 1.0 / math.log2(pos + 2)
    if pos < 10: metrics["NDCG@10"] += 1.0 / math.log2(pos + 2)
    if pos < 20: metrics["NDCG@20"] += 1.0 / math.log2(pos + 2)
    metrics["MRR"] += 1.0 / (pos + 1)
    cnt += 1

if cnt > 0:
    for k in metrics: metrics[k] /= cnt

print("\\n" + "="*50)
print(f"SBERT CONTENT EVALUATION RESULTS (N={cnt} users):")
for k, v in metrics.items():
    print(f"  {k:8s}: {v:.4f}")
print("="*50)

eval_res = {"model": "SBERT Content-Based", "metrics": metrics, "n_eval_users": cnt, "encode_time_sec": encode_time}
(OUT_DIR / "sbert_cb_evaluation_metrics.json").write_text(json.dumps(eval_res, indent=2))
print("Finished SBERT Content notebook successfully!")""")
]

# Write notebooks to disk
p1 = Path("kaggle/01_cf_als/01_als_implicit.ipynb")
p1.write_text(json.dumps(make_notebook(cells_als), indent=2))
print("Wrote", p1)

p2 = Path("kaggle/02_cf_bpr/02_bpr_implicit.ipynb")
p2.write_text(json.dumps(make_notebook(cells_bpr), indent=2))
print("Wrote", p2)

p3 = Path("kaggle/03_cb_sbert/03_sbert_faiss.ipynb")
p3.write_text(json.dumps(make_notebook(cells_sbert), indent=2))
print("Wrote", p3)
