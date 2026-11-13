#!/usr/bin/env python
"""Kaggle: Train Deep Early Fusion V3 (Anchored-Listwise DCN).

Khác bản Deep Early Fusion cũ (BCE + raw embeddings chưa chuẩn hoá):
  1. Feature vector hợp nhất **embeddings (L2-norm) + per-user z-scores nhánh**
     (z_cf, z_bpr, z_txt) + popularity/history scalars — fusion ngay từ input.
  2. Loss **listwise sampled softmax** trên nhóm candidate (1 positive + 40
     hard negatives), thay vì pointwise BCE 1 positive + 1 negative.
  3. Cross layers có residual, deep MLP rộng hơn, dropout, checkpoint theo
     **NDCG@10(validation)**; test chỉ báo cáo một lần.

Single model per kernel. Output:
  - early_fusion_v3.pt            (state dict của model tốt nhất)
  - early_fusion_v3_summary.json  (metrics + baselines)
"""
import os, sys, time, json, math, gc, subprocess
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass
try:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
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
# tqdm removed for cleaner Kaggle logs

MIN_INTERACTIONS = 3
CATALOG = 2464371
SEED = 42
OUT_DIR = Path("/kaggle/working" if Path("/kaggle/working").exists() else "artifacts")
OUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EVAL_BATCH_USERS = 128

N_TRAIN_INSTANCES = 60000
N_NEG_PER_INSTANCE = 40            # listwise group = 1 pos + 40 negatives
TRAIN_BATCH_GROUPS = 64            # groups per optimizer step
MAX_EPOCHS = 10
LR = 1e-3
WEIGHT_DECAY = 1e-4

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

log(f"CONFIG: MIN_INTERACTIONS={MIN_INTERACTIONS} DEVICE={DEVICE} | {gpu_mem()} | RAM {mem_info()}")


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


def find_sbert_file(dim=768):
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
log(f"[STEP 1] Base dir: {BASE_DIR}")


def load(name):
    up = find_factor_file(f"{name}_user_factors_f16.npy")
    ip = find_factor_file(f"{name}_item_factors_f16.npy")
    if up is None or ip is None:
        return None
    return np.load(up).astype(np.float32), np.load(ip).astype(np.float32)


als = load("als")
assert als is not None
user_cf, item_cf = als
bpr = load("bpr")
if bpr is not None:
    user_bpr, item_bpr = bpr
    HAS_BPR = True
    log("BPR factors loaded")
else:
    log("BPR factors not found - using 0-vector BPR features (masked)")
    user_bpr = np.zeros_like(user_cf)
    item_bpr = np.zeros_like(item_cf)
    HAS_BPR = False

# ---- Compact indices (same protocol as the SBERT kernel) ----
train_files = sorted(BASE_DIR.glob("train/*.parquet"))
log(f"[STEP 2] Reading {len(train_files)} train parquet file(s)...")
dfs = []
for i, f in enumerate(train_files, 1):
    log(f"[STEP 2]   train {i}/{len(train_files)}: {f.name}")
    d = pq.read_table(f, columns=["user_idx", "item_idx", "rating"]).to_pandas()
    d = d[d["rating"] >= 4.0]
    dfs.append(d[["user_idx", "item_idx"]])
    log(f"[STEP 2]   -> kept {len(dfs[-1]):,} positive rows")
train_df = pd.concat(dfs, ignore_index=True)
user_counts = train_df["user_idx"].value_counts()
active = user_counts[user_counts >= MIN_INTERACTIONS].index
train_df = train_df[train_df["user_idx"].isin(active)]
u_raw = train_df["user_idx"].to_numpy(np.int64)
i_raw = train_df["item_idx"].to_numpy(np.int64)
u_uniq, u_inv = np.unique(u_raw, return_inverse=True)
i_uniq, i_inv = np.unique(i_raw, return_inverse=True)
n_users, n_items = len(u_uniq), len(i_uniq)
SBERT_COVERAGE = min(n_items, CATALOG)
item_pop = np.bincount(i_inv, minlength=n_items).astype(np.float32)
user_hist_len = np.bincount(u_inv, minlength=n_users).astype(np.float32)
POP_MAX = float(np.log1p(item_pop.max())) if item_pop.size else 1.0
HIST_MAX = float(np.log1p(user_hist_len.max())) if user_hist_len.size else 1.0
log(f"[STEP 2] Compact: users={n_users:,} items={n_items:,} SBERT_COVERAGE={SBERT_COVERAGE:,} | RAM {mem_info()}")

# ---- SBERT item embeddings: 768-d (metadata | reviewer) ----
sbert_path = find_sbert_file()
assert sbert_path is not None, "sbert_item_embeddings_f16.npy not found - run the SBERT kernel first"
sbert_rows = np.load(sbert_path, mmap_mode="r").shape[0]
assert sbert_rows >= n_items, f"SBERT rows {sbert_rows:,} < items {n_items:,} - index mismatch"
item_sbert = np.asarray(np.load(sbert_path, mmap_mode="r")[:n_items], dtype=np.float32)
assert item_sbert.shape[1] == 768, f"expected 768-d SBERT, got {item_sbert.shape}"
log(f"[STEP 2] SBERT item embeddings: {item_sbert.shape} | RAM {mem_info()}")

np.save(OUT_DIR / "user_ids.npy", u_uniq)
np.save(OUT_DIR / "item_ids.npy", i_uniq)

# ---- Test + history ----
test_files = sorted(BASE_DIR.glob("test/*.parquet"))
log(f"[STEP 3] Reading {len(test_files)} test parquet file(s)...")
dfs_test = []
for i, f in enumerate(test_files, 1):
    log(f"[STEP 3]   test {i}/{len(test_files)}: {f.name}")
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

log(f"[STEP 3] Building user content profiles (mean train history) ... | RAM {mem_info()}")
user_sbert = np.zeros((n_users, 768), dtype=np.float32)
for u in range(n_users):
    h = hist_by_user_all.get(u)
    if h:
        v = item_sbert[h].mean(axis=0)
        n = float(np.linalg.norm(v))
        user_sbert[u] = v / n if n > 1e-8 else v
user_sbert = user_sbert.astype(np.float32)
log(f"[STEP 3] User content profiles: {user_sbert.shape}")


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
    rng = np.random.default_rng(SEED)
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
        out.append({"user": u, "gt": gt, "candidates": cands})
    return out


CAND_POOL = np.arange(SBERT_COVERAGE)
user_inter_count = np.bincount(u_inv, minlength=len(u_uniq))
test_users = np.array([u for u in user_gt.keys() if user_inter_count[u] >= MIN_INTERACTIONS])
test_users = np.array([u for u in test_users if any(it < SBERT_COVERAGE for it in user_gt[u])])
eval_users = np.random.default_rng(SEED).choice(test_users, size=min(200000, len(test_users)), replace=False)
eval_instances = build_instances(eval_users)
TEST_N = 5000
test_inst = eval_instances[:TEST_N]
val_inst = eval_instances[TEST_N:TEST_N + 2000]
train_inst = eval_instances[TEST_N + 2000:TEST_N + 2000 + N_TRAIN_INSTANCES]
log(f"[STEP 4] Prepared {len(eval_instances):,} instances | test={len(test_inst)} val={len(val_inst)} train={len(train_inst)}")


# ---- L2 row normalisation (on the fly, no extra full copies) ----
def l2n(x):
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-8)


def zscore(v):
    v = np.asarray(v, dtype=np.float32)
    m, s = float(v.mean()), float(v.std())
    if s < 1e-8:
        return np.zeros_like(v)
    return (v - m) / (s + 1e-8)


def pop_feat(items):
    return np.log1p(item_pop[items]) / POP_MAX


def hist_feat_user(u_idx):
    return np.log1p(user_hist_len[u_idx]) / HIST_MAX


CF_DIM = int(user_cf.shape[1])
BPR_DIM = int(user_bpr.shape[1]) if HAS_BPR else CF_DIM
SBERT_DIM = int(item_sbert.shape[1])
IN_DIM = CF_DIM*2 + BPR_DIM*2 + SBERT_DIM*2 + 3 + 1 + 1

def make_group(instances, want_mask=False):
    B = len(instances)
    C = len(instances[0]["candidates"]) if instances else 0
    out = np.zeros((B, C, IN_DIM), dtype=np.float32)
    o = 0
    ua_s = o; o += CF_DIM
    ia_s = o; o += CF_DIM
    ub_s = o; o += BPR_DIM
    ib_s = o; o += BPR_DIM
    ut_s = o; o += SBERT_DIM
    it_s = o; o += SBERT_DIM
    z_cf_i, z_bpr_i, z_txt_i = o, o+1, o+2; o += 3
    pop_i, hist_i = o, o+1; o += 2
    assert o == IN_DIM
    for b, inst in enumerate(instances):
        u = int(inst["user"])
        c = inst["candidates"]
        als_s = item_cf[c] @ user_cf[u]
        if HAS_BPR:
            bpr_s = item_bpr[c] @ user_bpr[u]
        else:
            bpr_s = als_s
        txt_s = item_sbert[c] @ user_sbert[u]
        z_cf = zscore(als_s)
        z_bpr = zscore(bpr_s)
        z_txt = zscore(txt_s)
        ua = l2n(user_cf[u][None, :])[0]
        ia = l2n(item_cf[c])
        ub = l2n(user_bpr[u][None, :])[0] if HAS_BPR else ua
        ib = l2n(item_bpr[c]) if HAS_BPR else ia
        ut = l2n(user_sbert[u][None, :])[0]
        it = l2n(item_sbert[c])
        out[b, :, ua_s:ia_s] = ua
        out[b, :, ia_s:ub_s] = ia
        out[b, :, ub_s:ib_s] = ub
        out[b, :, ib_s:ut_s] = ib
        out[b, :, ut_s:it_s] = ut
        out[b, :, it_s:z_cf_i] = it
        out[b, :, z_cf_i] = z_cf
        out[b, :, z_bpr_i] = z_bpr
        out[b, :, z_txt_i] = z_txt
        out[b, :, pop_i] = pop_feat(c)
        out[b, :, hist_i] = hist_feat_user(np.array([u], dtype=np.int64))[0]
    return out


# ---- Model ----
class EarlyFusionV3(nn.Module):
    def __init__(self, in_dim=IN_DIM, cross_layers=3, hidden=512, d1=256, d2=128):
        super().__init__()
        self.cross = nn.ModuleList([nn.Linear(in_dim, in_dim, bias=False) for _ in range(cross_layers)])
        for layer in self.cross:
            nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        self.deep = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, d1), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(d1, d2), nn.ReLU(),
            nn.Linear(d2, 1),
        )

    def forward(self, x):                                  # x: (N, IN_DIM)
        x0 = x
        xi = x
        for layer in self.cross:
            xi = x0 * torch.tanh(layer(xi)) + xi
        out = self.deep(x) + xi.mean(dim=-1, keepdim=True)
        return out.squeeze(-1)


model = EarlyFusionV3().to(DEVICE)

# ===========================================================================
# PER-USER RANK EXPORT + STATISTICAL TESTING (P0)
#
# Một kernel duy nhất chấm TẤT CẢ model trên CÙNG tập instance (cùng user, cùng
# 1 positive, cùng 99 negative, cùng seed) => rank per-user ghép cặp được.
# Xuất:
#   per_user_ranks.npz        : users, gt, rank_<model> (test 5k + ext 20k)
#   per_user_stats.json       : metrics + bootstrap CI95 + paired Wilcoxon/McNemar
#
# Nhờ cùng candidate nên CI và paired test là hợp lệ; đây là phần báo cáo còn
# thiếu (mục 7.2 hứa bootstrap + Wilcoxon nhưng bảng 8.1 chỉ có điểm ước lượng).
# ===========================================================================
hits = sorted(Path("/kaggle/input").rglob("early_fusion_v3.pt"))
assert hits, "early_fusion_v3.pt not found - add the training kernel as a kernel source"
CKPT = hits[0]
log(f"[EVAL] checkpoint: {CKPT} ({CKPT.stat().st_size/1e6:.1f} MB)")
model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
model.eval()

# ---- tập test mở rộng: các instance CHƯA từng dùng để train/tune ----
EXTRA_TEST_N = 15000
ext_start = TEST_N + 2000 + N_TRAIN_INSTANCES          # sau test + val + train
extra_inst = eval_instances[ext_start:ext_start + EXTRA_TEST_N]
test_ext_inst = test_inst + extra_inst
log(f"[EVAL] test={len(test_inst)} | extended test={len(test_ext_inst)} "
    f"(thêm {len(extra_inst)} user chưa dùng) | val={len(val_inst)}")

MODELS = ["als", "bpr", "sbert", "als_bpr", "late_fusion_v3", "early_fusion_v3"]
W_LF = (0.35, 0.35, 0.30)     # trọng số Late Fusion V3 đã tune trên validation


def scores_branches(inst):
    """Điểm từng nhánh + 2 fusion tuyến tính, tất cả trên cùng candidate."""
    u = int(inst["user"])
    c = inst["candidates"]
    s_als = item_cf[c] @ user_cf[u]
    s_bpr = (item_bpr[c] @ user_bpr[u]) if HAS_BPR else s_als
    s_txt = item_sbert[c] @ user_sbert[u]
    z_als, z_bpr, z_txt = zscore(s_als), zscore(s_bpr), zscore(s_txt)
    return {
        "als": s_als,
        "bpr": s_bpr,
        "sbert": s_txt,
        "als_bpr": 0.5 * z_als + 0.5 * z_bpr,
        "late_fusion_v3": W_LF[0] * z_als + W_LF[1] * z_bpr + W_LF[2] * z_txt,
    }


@torch.no_grad()
def scores_early_fusion(instances, batch_users=EVAL_BATCH_USERS):
    out = []
    for start in range(0, len(instances), batch_users):
        batch = instances[start:start + batch_users]
        c = len(batch[0]["candidates"])
        feats = make_group(batch)
        xb = torch.from_numpy(feats.reshape(len(batch) * c, IN_DIM)).to(DEVICE)
        logits = model(xb).cpu().numpy().reshape(len(batch), c)
        out.append(logits)
    return np.concatenate(out, axis=0) if out else np.zeros((0, 0), dtype=np.float32)


def collect_ranks(instances):
    """Trả về dict model -> rank (0-based) của positive trong 100 candidate."""
    n = len(instances)
    sc = {m: np.zeros((n, len(instances[0]["candidates"])), dtype=np.float32) for m in MODELS}
    for i, inst in enumerate(instances):
        for m, v in scores_branches(inst).items():
            sc[m][i] = v
    sc["early_fusion_v3"] = scores_early_fusion(instances)

    ranks = {}
    for m in MODELS:
        s = sc[m]
        order = np.argsort(-s, axis=1)          # order[i] = vị trí trong candidate
        pos = np.empty(n, dtype=np.int16)
        for i, inst in enumerate(instances):
            c = inst["candidates"]
            pos[i] = int(np.where(c[order[i]] == inst["gt"])[0][0])
        ranks[m] = pos
        del s, order
    users = np.array([int(x["user"]) for x in instances], dtype=np.int64)
    gts = np.array([int(x["gt"]) for x in instances], dtype=np.int64)
    return ranks, users, gts


def per_user_ndcg(ranks, k=10):
    r = ranks.astype(np.float64)
    return np.where(r < k, 1.0 / np.log2(r + 2.0), 0.0)


def per_user_mrr(ranks):
    return 1.0 / (ranks.astype(np.float64) + 1.0)


def point_metrics(ranks):
    out = {}
    for k in (5, 10, 20):
        out[f"HR@{k}"] = float(np.mean(ranks < k))
        out[f"Recall@{k}"] = out[f"HR@{k}"]          # 1 positive/user => trùng
        out[f"NDCG@{k}"] = float(np.mean(per_user_ndcg(ranks, k)))
    out["MRR"] = float(np.mean(per_user_mrr(ranks)))
    return {k: round(v, 4) for k, v in out.items()}


def bootstrap_ci(ranks, metric="HR@10", n_boot=2000, seed=SEED):
    """CI95 theo user (resample có hoàn lại)."""
    rng = np.random.default_rng(seed)
    n = len(ranks)
    if metric.startswith("HR@"):
        k = int(metric.split("@")[1]); vals = (ranks < k).astype(np.float64)
    elif metric.startswith("NDCG@"):
        k = int(metric.split("@")[1]); vals = per_user_ndcg(ranks, k)
    elif metric == "MRR":
        vals = per_user_mrr(ranks)
    else:
        raise ValueError(metric)
    boot = np.empty(n_boot, dtype=np.float64)
    for s in range(0, n_boot, 250):                      # chunk để tiết kiệm RAM
        e = min(s + 250, n_boot)
        idx = rng.integers(0, n, size=(e - s, n))
        boot[s:e] = vals[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


def paired_tests(ranks_a, ranks_b, name_a, name_b):
    """Wilcoxon signed-rank trên NDCG@10 per-user + McNemar exact trên HR@10."""
    from scipy.stats import wilcoxon, binomtest
    a, b = per_user_ndcg(ranks_a, 10), per_user_ndcg(ranks_b, 10)
    diff = a - b
    if np.allclose(diff, 0):
        p_w = 1.0
    else:
        try:
            p_w = float(wilcoxon(a, b, zero_method="wilcox", alternative="two-sided").pvalue)
        except Exception:
            p_w = float("nan")
    ha, hb = ranks_a < 10, ranks_b < 10
    n10 = int(np.sum(ha & ~hb))
    n01 = int(np.sum(~ha & hb))
    p_m = float(binomtest(n10, n10 + n01, 0.5).pvalue) if (n10 + n01) > 0 else 1.0
    return {
        "a": name_a, "b": name_b,
        "ndcg10_a": round(float(a.mean()), 4), "ndcg10_b": round(float(b.mean()), 4),
        "delta_ndcg10": round(float(a.mean() - b.mean()), 4),
        "wilcoxon_p": p_w,
        "mcnemar_hr10_a_only": n10, "mcnemar_hr10_b_only": n01,
        "mcnemar_p": p_m,
        "wins": int(np.sum(ranks_a < ranks_b)),
        "losses": int(np.sum(ranks_a > ranks_b)),
        "ties": int(np.sum(ranks_a == ranks_b)),
    }


def evaluate_split(instances, tag):
    ranks, users, gts = collect_ranks(instances)
    res = {"tag": tag, "n_users": len(instances), "metrics": {}, "ci95": {}}
    for m in MODELS:
        res["metrics"][m] = point_metrics(ranks[m])
        res["ci95"][m] = {k: bootstrap_ci(ranks[m], k) for k in ("HR@10", "NDCG@10", "MRR")}
        log(f"[{tag}] {m:16s} HR@10={res['metrics'][m]['HR@10']:.4f} "
            f"NDCG@10={res['metrics'][m]['NDCG@10']:.4f} MRR={res['metrics'][m]['MRR']:.4f} "
            f"CI95(HR@10)={res['ci95'][m]['HR@10']}")
    pairs = [("early_fusion_v3", "als"), ("early_fusion_v3", "als_bpr"),
             ("early_fusion_v3", "late_fusion_v3"), ("late_fusion_v3", "als"),
             ("als", "bpr"), ("als", "sbert")]
    res["paired_tests"] = [paired_tests(ranks[a], ranks[b], a, b) for a, b in pairs]
    for t in res["paired_tests"]:
        log(f"[{tag}] PAIRED {t['a']} vs {t['b']}: dNDCG10={t['delta_ndcg10']:+.4f} "
            f"wilcoxon_p={t['wilcoxon_p']:.3g} mcnemar_p={t['mcnemar_p']:.3g} "
            f"W/L/T={t['wins']}/{t['losses']}/{t['ties']}")
    return res, ranks, users, gts


print("\n" + "=" * 70)
print("[EVAL] PER-USER RANK EXPORT — 6 model, cùng candidate, cùng seed")
print("=" * 70)

res_test, ranks_test, users_test, gts_test = evaluate_split(test_inst, "test_5000")
res_ext, ranks_ext, users_ext, gts_ext = evaluate_split(test_ext_inst, "test_extended")

np.savez_compressed(
    OUT_DIR / "per_user_ranks.npz",
    models=np.array(MODELS),
    users_test=users_test, gt_test=gts_test,
    users_ext=users_ext, gt_ext=gts_ext,
    **{f"rank_test_{m}": ranks_test[m] for m in MODELS},
    **{f"rank_ext_{m}": ranks_ext[m] for m in MODELS},
)

payload = {
    "protocol": "1 positive + 99 random negatives, shared SBERT-covered catalog, leak-free SBERT user profiles",
    "min_interactions": MIN_INTERACTIONS,
    "seed": SEED,
    "late_fusion_weights": {"w_als": W_LF[0], "w_bpr": W_LF[1], "w_txt": W_LF[2]},
    "note": "Recall@K == HR@K because |relevant| = 1 per user; identical candidates across models make pairing valid",
    "splits": {"test_5000": res_test, "test_extended": res_ext},
}
(OUT_DIR / "per_user_stats.json").write_text(json.dumps(payload, indent=2))
print(json.dumps(payload, indent=2))
log(f"DONE in {time.time()-_T0:.1f}s | RAM {mem_info()}")
print("[ARTIFACT SAVED]", flush=True)
