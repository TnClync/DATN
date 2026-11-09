#!/usr/bin/env python
"""Kaggle Account 4: Train Deep Hybrids V2 (real metadata text, more data/epochs).

Goal: make a deep hybrid actually beat ALS on the shared 200k-item catalog.
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
from scipy.sparse import csr_matrix
from tqdm import tqdm

MIN_INTERACTIONS = 3
CATALOG = 2464371
SEED = 42
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
    # mounted from nguynteo/item-profile
    for c in [Path("/kaggle/input/item-profile/Item_profile"),
              Path("/kaggle/input/itemprofile/Item_profile"),
              Path("/kaggle/input/item_profile/Item_profile")]:
        if c.exists():
            return c
    if Path("/kaggle/input").exists():
        for p in Path("/kaggle/input").rglob("*.parquet"):
            if "item" in str(p).lower() and "profile" in str(p).lower():
                return p.parent
        # fallback: any dir with item_idx+title
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


# ---------------------------------------------------------------- 1. TRAIN
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

# Load ALS/BPR factors from CF kernel
def load_cf_from_kernel(prefix):
    up = find_factor_file(f"{prefix}_user_factors_f16.npy")
    ip = find_factor_file(f"{prefix}_item_factors_f16.npy")
    if up is None or ip is None:
        return None
    uf = np.load(up).astype(np.float32)
    itf = np.load(ip).astype(np.float32)
    if uf.shape[0] != n_users or itf.shape[0] != n_items:
        print(f"[{prefix}] shape mismatch {uf.shape}/{itf.shape} vs ({n_users},{n_items})")
        return None
    uid_p = up.parent / f"{prefix}_user_ids.npy"
    iid_p = up.parent / f"{prefix}_item_ids.npy"
    if uid_p.exists() and iid_p.exists():
        k_uid = np.load(uid_p); k_iid = np.load(iid_p)
        if not (np.array_equal(k_uid, u_uniq) and np.array_equal(k_iid, i_uniq)):
            print(f"[{prefix}] id mapping differs -> fallback")
            return None
    return uf, itf

als_res = load_cf_from_kernel("als")
if als_res is not None:
    user_cf, item_cf = als_res
    print(f"Reused ALS factors: user {user_cf.shape}, item {item_cf.shape}")
else:
    print("Training ALS locally...")
    from implicit.als import AlternatingLeastSquares
    um = csr_matrix((np.ones(len(u_inv), dtype=np.float32), (u_inv, i_inv)), shape=(n_users, n_items))
    als = AlternatingLeastSquares(factors=32, regularization=0.05, iterations=12, num_threads=0, random_state=42)
    als.fit(um)
    user_cf = np.asarray(als.user_factors, dtype=np.float32)
    item_cf = np.asarray(als.item_factors, dtype=np.float32)
    del als, um

bpr_res = load_cf_from_kernel("bpr")
if bpr_res is not None:
    user_bpr, item_bpr = bpr_res
    print(f"Reused BPR factors: user {user_bpr.shape}, item {item_bpr.shape}")
else:
    user_bpr, item_bpr = None, None
    print("BPR not available")


# ---------------------------------------------------------------- 2. SBERT (real text)
print("\n" + "=" * 50)
print("2. SBERT WITH REAL PRODUCT TEXT")
print("=" * 50)
# item_idx -> text from item-profile
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
                    parts.append(desc.strip()[:300])
                txt = " | ".join(parts)
                if txt:
                    item_text_map[int(r["item_idx"])] = txt
        except Exception as e:
            print("profile read err", f, e)
print(f"Loaded real text for {len(item_text_map):,} items")

SBERT_COVERAGE = n_items
CAND_POOL = np.arange(SBERT_COVERAGE)
compact_to_orig = {idx: int(i_uniq[idx]) for idx in range(n_items)}

item_texts = []
for ci in range(SBERT_COVERAGE):
    orig = compact_to_orig[ci]
    txt = item_text_map.get(orig)
    if not txt:
        txt = f"Product item {orig}"
    item_texts.append(txt)

print(f"Encoding {len(item_texts)} items with real text...")
sbert_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=str(DEVICE))
sbert_embs = sbert_model.encode(item_texts, batch_size=512, show_progress_bar=True,
                                convert_to_numpy=True, normalize_embeddings=True)
item_sbert = np.zeros((n_items, 384), dtype=np.float32)
item_sbert[:SBERT_COVERAGE] = sbert_embs
del sbert_embs
gc.collect()

faiss_index = faiss.IndexFlatIP(384)
faiss_index.add(item_sbert[:SBERT_COVERAGE].astype(np.float32))

np.save(OUT_DIR / "sbert_item_embeddings_f16.npy", item_sbert.astype(np.float16))
np.save(OUT_DIR / "sbert_item_ids.npy", i_uniq)
np.save(OUT_DIR / "user_ids.npy", u_uniq)
np.save(OUT_DIR / "item_ids.npy", i_uniq)
np.save(OUT_DIR / "als_user_factors_f16.npy", user_cf.astype(np.float16))
np.save(OUT_DIR / "als_item_factors_f16.npy", item_cf.astype(np.float16))
if user_bpr is not None:
    np.save(OUT_DIR / "bpr_user_factors_f16.npy", user_bpr.astype(np.float16))
    np.save(OUT_DIR / "bpr_item_factors_f16.npy", item_bpr.astype(np.float16))
print("Saved factors + sbert")


# ---------------------------------------------------------------- 3. TEST + HISTORY
print("\n" + "=" * 50)
print("3. LOADING TEST + TRAIN HISTORY")
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

# train history for all users
u_inv_arr = u_inv
i_inv_arr = i_inv
hist_by_user_all: dict[int, list[int]] = {}
for uu, ii in zip(u_inv_arr, i_inv_arr):
    if ii < SBERT_COVERAGE:
        hist_by_user_all.setdefault(int(uu), []).append(int(ii))

user_sbert: dict[int, np.ndarray] = {}
for u in range(n_users):
    h = hist_by_user_all.get(u, [])
    if h:
        v = item_sbert[h].mean(axis=0)
        nrm = float(np.linalg.norm(v))
        v = v / nrm if nrm > 1e-8 else np.zeros(384, dtype=np.float32)
    else:
        v = np.zeros(384, dtype=np.float32)
    user_sbert[u] = v.astype(np.float32)


def compute_metrics(ranked, gt):
    res = {k: 0.0 for k in ["HR@5", "HR@10", "HR@20", "Recall@5", "Recall@10", "Recall@20", "NDCG@5", "NDCG@10", "NDCG@20", "MRR"]}
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
print(f"Prepared {len(eval_instances):,} evaluation instances")
TEST_N = 5000
test_inst = eval_instances[:TEST_N]
val_inst = eval_instances[TEST_N:TEST_N + 1000]


def eval_scorer(fn, instances):
    agg = {k: 0.0 for k in ["HR@5", "HR@10", "HR@20", "NDCG@5", "NDCG@10", "NDCG@20", "MRR"]}
    for inst in instances:
        scores = fn(inst["user"], inst["candidates"])
        ranked = inst["candidates"][np.argsort(-scores)]
        m = compute_metrics(ranked, inst["gt"])
        for k in agg:
            agg[k] += m[k]
    return {k: round(v / max(1, len(instances)), 4) for k, v in agg.items()}


# ---------------------------------------------------------------- 4. BASELINES
print("\n" + "=" * 50)
print("4. BASELINES")
print("=" * 50)
res_als = eval_scorer(lambda u, c: cf_scores(user_cf, item_cf, u, c), test_inst)
print("ALS:", res_als)
res_bpr = None
if user_bpr is not None:
    res_bpr = eval_scorer(lambda u, c: cf_scores(user_bpr, item_bpr, u, c), test_inst)
    print("BPR:", res_bpr)
res_sbert = eval_scorer(lambda u, c: sb_scores(u, c), test_inst)
print("SBERT:", res_sbert)


# ---------------------------------------------------------------- 5. LATE FUSION
print("\n" + "=" * 50)
print("5. LATE FUSION")
print("=" * 50)
late_alphas = [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
best_alpha, best_hr = 0.9, -1.0
for a in late_alphas:
    m = eval_scorer(lambda u, c, a=a: a * mm(cf_scores(user_cf, item_cf, u, c)) + (1 - a) * mm(sb_scores(u, c)), val_inst)
    print(f"alpha={a}: val HR@10={m['HR@10']}")
    if m["HR@10"] > best_hr:
        best_hr, best_alpha = m["HR@10"], a
res_late = eval_scorer(lambda u, c: best_alpha * mm(cf_scores(user_cf, item_cf, u, c)) + (1 - best_alpha) * mm(sb_scores(u, c)), test_inst)
print(f"Best alpha={best_alpha} -> {res_late}")


# ---------------------------------------------------------------- 6. TWO-STAGE (linear rerank, shared catalog)
print("\n" + "=" * 50)
print("6. TWO-STAGE RETRIEVAL")
print("=" * 50)


def topk_full(mat, vec, k, banned):
    s = mat @ vec
    if len(banned):
        b = banned[(banned >= 0) & (banned < len(s))]
        s[b] = -np.inf
    k = min(k, len(s) - 1)
    idx = np.argpartition(-s, k)[:k]
    return idx[np.argsort(-s[idx])]


ts_agg = {k: 0.0 for k in ["HR@5", "HR@10", "HR@20", "NDCG@5", "NDCG@10", "NDCG@20", "MRR"]}
ts_n = 0
for inst in tqdm(test_inst, desc="Two-stage"):
    u = inst["user"]
    banned = inst["banned"]
    cf_top = topk_full(item_cf[:SBERT_COVERAGE], user_cf[u], 50, banned)
    sb_top = topk_full(item_sbert[:SBERT_COVERAGE], user_sbert[u], 50, banned)
    union = np.array(list(dict.fromkeys(list(cf_top) + list(sb_top))), dtype=np.int64)
    if len(union) == 0:
        continue
    cfu = item_cf[union] @ user_cf[u]
    sbu = item_sbert[union] @ user_sbert[u]
    fused = best_alpha * mm(cfu) + (1 - best_alpha) * mm(sbu)
    ranked = union[np.argsort(-fused)]
    m = compute_metrics(ranked, inst["gt"])
    for k in ts_agg:
        ts_agg[k] += m[k]
    ts_n += 1
res_two = {k: round(v / max(1, ts_n), 4) for k, v in ts_agg.items()}
print("Two-Stage:", res_two)


# ---------------------------------------------------------------- 7. LIGHT GATED
print("\n" + "=" * 50)
print("7. LIGHT GATED LATE FUSION (more data + epochs)")
print("=" * 50)


class LightGate(nn.Module):
    def __init__(self, in_features=32 + 32 + 384):
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
# Build training data from 20k instances
train_for_fit = eval_instances[:100000] if len(eval_instances) >= 100000 else eval_instances
gate_Xu, gate_Xi, gate_Xs, gate_cf, gate_sb, gate_y = [], [], [], [], [], []
for inst in train_for_fit:
    u = inst["user"]; pos = inst["gt"]
    negs = [int(x) for x in inst["candidates"] if int(x) != pos][:4]
    for it, label in [(pos, 1.0)] + [(n, 0.0) for n in negs]:
        gate_Xu.append(user_cf[u]); gate_Xi.append(item_cf[it]); gate_Xs.append(item_sbert[it])
        gate_cf.append(float(item_cf[it] @ user_cf[u]))
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
torch.save(gate.state_dict(), OUT_DIR / "light_gated_fusion.pt")

def eval_gate(instances):
    gate.eval()
    agg = {k: 0.0 for k in ["HR@5", "HR@10", "HR@20", "NDCG@5", "NDCG@10", "NDCG@20", "MRR"]}
    with torch.no_grad():
        for inst in instances:
            u = inst["user"]; cands = inst["candidates"]; gt = inst["gt"]
            bu = torch.tensor(np.repeat(user_cf[u][None, :], len(cands), axis=0), dtype=torch.float32).to(DEVICE)
            bi = torch.tensor(item_cf[cands], dtype=torch.float32).to(DEVICE)
            bs = torch.tensor(item_sbert[cands], dtype=torch.float32).to(DEVICE)
            bcf = torch.tensor(cf_scores(user_cf, item_cf, u, cands), dtype=torch.float32).to(DEVICE)
            bsb = torch.tensor(sb_scores(u, cands), dtype=torch.float32).to(DEVICE)
            fused, _ = gate(bu, bi, bs, bcf, bsb)
            ranked = cands[np.argsort(-fused.cpu().numpy())]
            m = compute_metrics(ranked, gt)
            for k in agg:
                agg[k] += m[k]
    return {k: round(v / max(1, len(instances)), 4) for k, v in agg.items()}

res_gate = eval_gate(test_inst)
print("Light Gated:", res_gate)


# ---------------------------------------------------------------- 8. DEEP EARLY FUSION
print("\n" + "=" * 50)
print("8. DEEP EARLY FUSION (more data + epochs)")
print("=" * 50)


class DeepEarlyFusion(nn.Module):
    def __init__(self, in_dim, cross_layers=4, hidden=256):
        super().__init__()
        self.cross = nn.ModuleList([nn.Linear(in_dim, in_dim) for _ in range(cross_layers)])
        self.deep = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                                  nn.Linear(hidden, hidden), nn.ReLU(),
                                  nn.Linear(hidden, hidden // 2), nn.ReLU(),
                                  nn.Linear(hidden // 2, 1))

    def forward(self, x):
        x0 = x; xi = x
        for layer in self.cross:
            xi = x0 * torch.sigmoid(layer(xi)) + xi
        deep_out = self.deep(x)
        cross_out = xi.mean(dim=-1, keepdim=True)
        return (cross_out + deep_out).squeeze(-1)


in_dim = 32 + 32 + 384
deep = DeepEarlyFusion(in_dim).to(DEVICE)
X, y = [], []
for inst in train_for_fit:
    u = inst["user"]; pos = inst["gt"]
    negs = [int(x) for x in inst["candidates"] if int(x) != pos][:4]
    for it, label in [(pos, 1.0)] + [(n, 0.0) for n in negs]:
        X.append(np.concatenate([user_cf[u], item_cf[it], item_sbert[it]]))
        y.append(label)
X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)
ds2 = TensorDataset(torch.tensor(X), torch.tensor(y))
loader2 = DataLoader(ds2, batch_size=512, shuffle=True, num_workers=0)
opt2 = torch.optim.Adam(deep.parameters(), lr=1e-3, weight_decay=1e-5)
bce2 = nn.BCEWithLogitsLoss()
deep.train()
for ep in range(30):
    tot = 0.0
    for xb, yb in loader2:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        loss = bce2(deep(xb), yb)
        opt2.zero_grad(); loss.backward(); opt2.step()
        tot += loss.item()
    print(f"Deep Epoch {ep+1}/30 Loss: {tot/len(loader2):.4f}")
torch.save(deep.state_dict(), OUT_DIR / "deep_early_fusion.pt")

def eval_deep(instances):
    deep.eval()
    agg = {k: 0.0 for k in ["HR@5", "HR@10", "HR@20", "NDCG@5", "NDCG@10", "NDCG@20", "MRR"]}
    with torch.no_grad():
        for inst in instances:
            u = inst["user"]; cands = inst["candidates"]; gt = inst["gt"]
            feats = np.concatenate([
                np.repeat(user_cf[u][None, :], len(cands), axis=0),
                item_cf[cands],
                item_sbert[cands],
            ], axis=1)
            logits = deep(torch.tensor(feats, dtype=torch.float32).to(DEVICE)).cpu().numpy()
            ranked = cands[np.argsort(-logits)]
            m = compute_metrics(ranked, gt)
            for k in agg:
                agg[k] += m[k]
    return {k: round(v / max(1, len(instances)), 4) for k, v in agg.items()}

res_deep = eval_deep(test_inst)
print("Deep Early:", res_deep)


# ---------------------------------------------------------------- 9. SUMMARY
print("\n" + "=" * 60)
print("FINAL COMPARISON (V2, real metadata text)")
print("=" * 60)
print(f"{'Model':<22} {'HR@5':>7} {'HR@10':>7} {'HR@20':>7} {'NDCG@10':>8} {'MRR':>7}")
for name, m in [("ALS", res_als), ("BPR", res_bpr), ("SBERT", res_sbert),
                (f"Late Fusion (a={best_alpha})", res_late),
                ("Two-Stage", res_two),
                ("Light Gated", res_gate),
                ("Deep Early", res_deep)]:
    if m is None:
        print(f"{name:<22} {'n/a':>7}")
    else:
        print(f"{name:<22} {m['HR@5']:>7.4f} {m['HR@10']:>7.4f} {m['HR@20']:>7.4f} {m['NDCG@10']:>8.4f} {m['MRR']:>7.4f}")

summary = {
    "protocol": "V2: real product text SBERT, more training data/epochs, shared 200k catalog",
    "eval_catalog_items": SBERT_COVERAGE,
    "n_eval_users": len(test_inst),
    "min_interactions": MIN_INTERACTIONS,
    "best_deep_hybrid": "light_gated" if res_gate["HR@10"] >= res_deep["HR@10"] else "deep_early",
    "als": res_als,
    "bpr": res_bpr,
    "sbert": res_sbert,
    "late_fusion": {"best_alpha": best_alpha, "test": res_late},
    "two_stage": res_two,
    "light_gated": res_gate,
    "deep_early_fusion": res_deep,
}
(OUT_DIR / "deep_hybrids_v2_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
print("DONE")
