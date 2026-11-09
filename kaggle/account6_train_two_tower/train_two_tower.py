#!/usr/bin/env python
"""Kaggle: Train Two-Tower (graph + sbert) (single model)."""
import os, sys, time, json, math, gc, subprocess
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass
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
from tqdm import tqdm

MIN_INTERACTIONS = 3
CATALOG = 2464371
SEED = 42
OUT_DIR = Path("/kaggle/working" if Path("/kaggle/working").exists() else "artifacts")
OUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EVAL_BATCH_USERS = 128   # eval: score this many users in ONE item-tower forward pass

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
    return " | ".join(f"GPU{i} alloc={torch.cuda.memory_allocated(i)/1e9:.1f}G res={torch.cuda.memory_reserved(i)/1e9:.1f}G" for i in range(torch.cuda.device_count()))

log(f"CONFIG: DEVICE={DEVICE} EVAL_BATCH_USERS={EVAL_BATCH_USERS} | {gpu_mem()}")


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


def find_file(name):
    hits = sorted(Path("/kaggle/input").rglob(name))
    pref = [p for p in hits if "train-lightgcn" in str(p) or "graph-deep-hybrid" in str(p) or "deep-hybrid-recommender" in str(p)]
    return (pref[0] if pref else hits[0]) if hits else None

def find_sbert_file(dim=768):
    """Locate the 768-d combined SBERT artifact (older kernels also wrote 384-d copies)."""
    hits = sorted(Path("/kaggle/input").rglob("sbert_item_embeddings_f16.npy"))
    hits.sort(key=lambda p: 0 if "train-sbert" in str(p) else 1)
    for p in hits:
        try:
            a = np.load(p, mmap_mode="r")
            if a.ndim == 2 and a.shape[1] == dim:
                return p
        except Exception:
            continue
    return None


BASE_DIR = find_base_dir()

# ---- Load factors + SBERT + graph embeddings ----
graph_user = np.load(find_file("graph_user_factors_f16.npy")).astype(np.float32)
graph_item = np.load(find_file("graph_item_factors_f16.npy")).astype(np.float32)
sbert_path = find_sbert_file()
assert sbert_path is not None, "768-d sbert_item_embeddings_f16.npy not found - run the SBERT kernel first"
item_sbert = np.asarray(np.load(sbert_path, mmap_mode="r")[:CATALOG], dtype=np.float32)
print(f"SBERT item embeddings: {item_sbert.shape} | {sbert_path}")
u_ids = np.load(find_file("user_ids.npy"))
i_ids = np.load(find_file("item_ids.npy"))
n_users, n_items = len(u_ids), len(i_ids)
SBERT_COVERAGE = n_items

# ---- Train + test (same as other kernels) ----
train_files = sorted(BASE_DIR.glob("train/*.parquet"))
dfs = []
for f in train_files:
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

test_files = sorted(BASE_DIR.glob("test/*.parquet"))
dfs_test = []
for f in test_files:
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

hist_by_user_all = {}
for uu, ii in zip(u_inv, i_inv):
    if ii < SBERT_COVERAGE:
        hist_by_user_all.setdefault(int(uu), []).append(int(ii))
user_sbert = {}
for u in range(len(u_uniq)):
    h = hist_by_user_all.get(u, [])
    if h:
        v = item_sbert[h].mean(axis=0)
        nrm = float(np.linalg.norm(v))
        v = v / nrm if nrm > 1e-8 else np.zeros(768, dtype=np.float32)
    else:
        v = np.zeros(768, dtype=np.float32)
    user_sbert[u] = v.astype(np.float32)
user_inter_count = np.bincount(u_inv, minlength=len(u_uniq))


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


def build_instances(eval_users):
    out = []
    rng = np.random.default_rng(42)
    CAND_POOL = np.arange(SBERT_COVERAGE)
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
eval_users = np.random.default_rng(42).choice(test_users, size=min(50000, len(test_users)), replace=False)
eval_instances = build_instances(eval_users)
log(f"Prepared {len(eval_instances):,} instances | RAM {mem_info()}")
TEST_N = 5000
test_inst = eval_instances[:TEST_N]
train_for_fit = eval_instances[TEST_N:TEST_N + 100000]


# ---- Two-Tower ----
print("\n[STEP 6/8] TRAIN: TWO-TOWER (graph + sbert)")
class UserTower(nn.Module):
    def __init__(self, in_dim=32, hidden=128, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.1),
                                 nn.Linear(hidden, out_dim))
    def forward(self, x):
        return self.net(x)

class ItemTower(nn.Module):
    def __init__(self, in_dim=32 + 768, hidden=256, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.1),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.1),
                                 nn.Linear(hidden, out_dim))
    def forward(self, x):
        return self.net(x)

user_tower = UserTower().to(DEVICE)
item_tower = ItemTower().to(DEVICE)
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
_t_train = time.time()
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
    if (ep+1)%5==0 or ep==0 or ep==29: log(f"[TRAIN] TwoTower Epoch {ep+1}/30 Loss: {tot/len(loader_tt):.4f} | {time.time()-_t_train:.1f}s | {gpu_mem()}")
log(f"[TRAIN] Training done in {time.time()-_t_train:.1f}s")
torch.save({"user": user_tower.state_dict(), "item": item_tower.state_dict()}, OUT_DIR / "two_tower_fusion.pt")

# Eval — batched: B users go through the item tower in ONE forward pass
user_tower.eval(); item_tower.eval()


def eval_two_tower(instances, batch_users=EVAL_BATCH_USERS):
    keys = ["HR@5", "HR@10", "HR@20", "Recall@5", "Recall@10", "Recall@20",
            "NDCG@5", "NDCG@10", "NDCG@20", "MRR"]
    agg = {k: 0.0 for k in keys}
    n = len(instances)
    n_batches = (n + batch_users - 1) // batch_users
    t0 = time.time()
    with torch.no_grad():
        for bi, start in enumerate(range(0, n, batch_users)):
            batch = instances[start:start + batch_users]
            b = len(batch)
            c = len(batch[0]["candidates"])
            users = np.array([inst["user"] for inst in batch], dtype=np.int64)
            cands = np.stack([inst["candidates"] for inst in batch])           # (B, C)

            u_emb = user_tower(torch.from_numpy(graph_user[users]).to(DEVICE))  # (B, Du)
            i_feat = np.concatenate([graph_item[cands], item_sbert[cands]],
                                    axis=2).astype(np.float32)                 # (B, C, 32+768)
            i_emb = item_tower(torch.from_numpy(i_feat.reshape(b * c, -1)).to(DEVICE))
            i_emb = i_emb.reshape(b, c, -1)                                    # (B, C, Di)
            scores = torch.einsum("bcd,bd->bc", i_emb, u_emb).cpu().numpy()
            for row in range(b):
                ranked = cands[row][np.argsort(-scores[row])]
                m = compute_metrics(ranked, batch[row]["gt"])
                for k in keys:
                    agg[k] += m[k]
            if bi % 10 == 0 or bi == n_batches - 1:
                print(f"  eval batch {bi + 1}/{n_batches} "
                      f"({start + b:,}/{n:,} users) {time.time() - t0:.1f}s", flush=True)
    print(f"  eval done: {n:,} users in {time.time() - t0:.1f}s "
          f"({n / max(time.time() - t0, 1e-9):,.0f} users/s)", flush=True)
    return {k: round(v / max(1, n), 4) for k, v in agg.items()}


res_tt = eval_two_tower(test_inst)
log(f"[RESULT] Two-Tower: {res_tt}")

summary = {"model": "two_tower", "n_eval_users": len(test_inst), "two_tower": res_tt}
(OUT_DIR / "two_tower_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
log(f"DONE in {time.time()-_T0:.1f}s | RAM {mem_info()} | {gpu_mem()}")
print("[ARTIFACT SAVED]", flush=True)
