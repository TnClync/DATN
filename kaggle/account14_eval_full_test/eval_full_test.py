#!/usr/bin/env python
"""Kaggle: Evaluate 6 models on the FULL eligible test set (1+99 protocol).

Same protocol as eval-per-user-datn but scores ALL eligible test users
(min_interactions>=3, gt in SBERT coverage) instead of 5k/20k samples.
Outputs per-model metrics + bootstrap CI95 and (for a sample) paired tests,
plus compressed per-user ranks for the full set.

Single kernel. Output: full_test_stats.json, full_test_ranks.npz
"""
import os, sys, time, json, math, gc, subprocess
from pathlib import Path

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

try:
    import torch, faiss, pyarrow.parquet as pq, implicit, sentence_transformers, scipy
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet",
        "sentence-transformers>=2.7", "faiss-cpu", "implicit>=0.7", "pyarrow", "scipy", "tqdm"])
    import torch, faiss, pyarrow.parquet as pq, implicit, sentence_transformers, scipy

import numpy as np
import pandas as pd
import torch.nn as nn

MIN_INTERACTIONS = 3
CATALOG = 2464371
SEED = 42
OUT_DIR = Path("/kaggle/working" if Path("/kaggle/working").exists() else "artifacts/eval_full_test")
OUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EVAL_BATCH_USERS = 256   # users per forward batch
CHUNK_USERS = 4096       # users per build+score chunk (controls RAM)

_T0 = time.time()
def log(msg): print(f"[{time.strftime('%H:%M:%S')} +{time.time()-_T0:7.1f}s] {msg}", flush=True)

def mem_info():
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return f"{rss/1e9:.1f}GB" if rss > 1e9 else f"{rss/1e6:.1f}MB"
    except Exception:
        return "n/a"

log(f"CONFIG: DEVICE={DEVICE} | RAM {mem_info()}")

def find_base_dir():
    for c in [Path("/kaggle/input/lastone/amazon_recs_output"), Path("/kaggle/input/lastone"),
              Path("/kaggle/input/datasets/nguynteo/lastone"), Path("data")]:
        if (c / "train").exists() or (c / "test").exists():
            return c
    if Path("/kaggle/input").exists():
        for p in Path("/kaggle/input").rglob("train"):
            if p.is_dir(): return p.parent
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
            if a.ndim == 2 and a.shape[1] == dim: return p
        except Exception: continue
    return None

BASE_DIR = find_base_dir()
log(f"[STEP 1] Base dir: {BASE_DIR}")

def load(name):
    up = find_factor_file(f"{name}_user_factors_f16.npy")
    ip = find_factor_file(f"{name}_item_factors_f16.npy")
    if up is None or ip is None: return None
    return np.load(up).astype(np.float32), np.load(ip).astype(np.float32)

als = load("als"); assert als is not None
user_cf, item_cf = als
bpr = load("bpr")
if bpr is not None:
    user_bpr, item_bpr = bpr; HAS_BPR = True
    log("BPR factors loaded")
else:
    user_bpr = np.zeros_like(user_cf); item_bpr = np.zeros_like(item_cf); HAS_BPR = False
    log("BPR factors not found - masked")

train_files = sorted(BASE_DIR.glob("train/*.parquet"))
log(f"[STEP 2] Reading {len(train_files)} train parquet files...")
dfs = []
for i, f in enumerate(train_files, 1):
    d = pq.read_table(f, columns=["user_idx","item_idx","rating"]).to_pandas()
    d = d[d["rating"] >= 4.0]
    dfs.append(d[["user_idx","item_idx"]])
train_df = pd.concat(dfs, ignore_index=True)
active = train_df["user_idx"].value_counts()
active = active[active >= MIN_INTERACTIONS].index
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
log(f"[STEP 2] users={n_users:,} items={n_items:,} SBERT_COVERAGE={SBERT_COVERAGE:,}")

sbert_path = find_sbert_file()
assert sbert_path is not None, "sbert_item_embeddings_f16.npy not found"
sbert_rows = np.load(sbert_path, mmap_mode="r").shape[0]
assert sbert_rows >= n_items, f"SBERT rows {sbert_rows} < items {n_items}"
item_sbert = np.asarray(np.load(sbert_path, mmap_mode="r")[:n_items], dtype=np.float32)
assert item_sbert.shape[1] == 768
log(f"[STEP 2] SBERT items: {item_sbert.shape}")

test_files = sorted(BASE_DIR.glob("test/*.parquet"))
log(f"[STEP 3] Reading {len(test_files)} test parquet files...")
dfs_test = []
for f in test_files:
    d = pq.read_table(f, columns=["user_idx","item_idx","rating"]).to_pandas()
    d = d[d["rating"] >= 4.0]
    dfs_test.append(d[["user_idx","item_idx"]])
test_df = pd.concat(dfs_test, ignore_index=True)
test_df = test_df[test_df["user_idx"].isin(set(u_uniq.tolist()))]
u_map = {orig: ci for ci, orig in enumerate(u_uniq)}
i_map = {orig: ci for ci, orig in enumerate(i_uniq)}
test_df["cu"] = test_df["user_idx"].map(u_map)
test_df["ci"] = test_df["item_idx"].map(i_map)
test_df = test_df.dropna(subset=["cu","ci"])
test_df["cu"] = test_df["cu"].astype(np.int64)
test_df["ci"] = test_df["ci"].astype(np.int64)
user_gt = test_df.groupby("cu")["ci"].apply(lambda x: list(set(int(v) for v in x))).to_dict()

hist_by_user_all = {}
for uu, ii in zip(u_inv, i_inv):
    ii = int(ii)
    if ii < SBERT_COVERAGE:
        hist_by_user_all.setdefault(int(uu), []).append(ii)

user_inter_count = np.bincount(u_inv, minlength=n_users)
test_users = np.array([u for u in user_gt.keys() if user_inter_count[u] >= MIN_INTERACTIONS], dtype=np.int64)
test_users = np.array([u for u in test_users if any(it < SBERT_COVERAGE for it in user_gt[u])], dtype=np.int64)
rng_split = np.random.default_rng(SEED)
rng_split.shuffle(test_users)
log(f"[STEP 3] eligible full-test users: {len(test_users):,}")

# user sbert profiles for all eligible test users
log("[STEP 3] building user SBERT profiles...")
user_sbert = np.zeros((n_users, 768), dtype=np.float32)
need = set(int(u) for u in test_users)
filled = 0
for u in need:
    h = hist_by_user_all.get(u)
    if h:
        v = item_sbert[h].mean(axis=0)
        n = float(np.linalg.norm(v))
        user_sbert[u] = v / n if n > 1e-8 else v
        filled += 1
log(f"[STEP 3] profiles filled for {filled:,}/{len(need):,} users | RAM {mem_info()}")

def l2n(x):
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.where(n < 1e-8, 1.0, n)
def zscore(v):
    v = np.asarray(v, dtype=np.float32)
    m, s = float(v.mean()), float(v.std())
    if s < 1e-8: return np.zeros_like(v)
    return (v - m) / (s + 1e-8)
def pop_feat(items): return (np.log1p(item_pop[items]) / POP_MAX).astype(np.float32)
def hist_feat(u): return np.float32(np.log1p(user_hist_len[u]) / HIST_MAX)

CF_DIM = int(user_cf.shape[1]); BPR_DIM = int(user_bpr.shape[1]) if HAS_BPR else CF_DIM; SBERT_DIM = 768
IN_DIM = CF_DIM*2 + BPR_DIM*2 + SBERT_DIM*2 + 3 + 1 + 1

class EarlyFusionV3(nn.Module):
    def __init__(self, in_dim=IN_DIM, cross_layers=3, hidden=512, d1=256, d2=128):
        super().__init__()
        self.cross = nn.ModuleList([nn.Linear(in_dim, in_dim, bias=False) for _ in range(cross_layers)])
        self.deep = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, d1), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(d1, d2), nn.ReLU(), nn.Linear(d2, 1),
        )
    def forward(self, x):
        x0 = x; xi = x
        for layer in self.cross:
            xi = x0 * torch.tanh(layer(xi)) + xi
        out = self.deep(x) + xi.mean(dim=-1, keepdim=True)
        return out.squeeze(-1)

model = EarlyFusionV3().to(DEVICE)
hits = sorted(Path("/kaggle/input").rglob("early_fusion_v3.pt"))
assert hits, "early_fusion_v3.pt not found"
sd = torch.load(hits[0], map_location=DEVICE)
model.load_state_dict(sd.get("state_dict", sd))
model.eval()
log(f"[STEP 4] EF V3 model loaded: {hits[0]}")

CAND_POOL = np.arange(SBERT_COVERAGE, dtype=np.int64)
MODELS = ["als", "bpr", "sbert", "als_bpr", "late_fusion_v3", "early_fusion_v3"]
W_LF = (0.35, 0.35, 0.30)

def score_chunk(users_chunk, seed_base):
    rng = np.random.default_rng(seed_base)
    inst_u=[]; inst_gt=[]; inst_c=[]
    for u in users_chunk:
        u=int(u); g=[it for it in user_gt[u] if it < SBERT_COVERAGE]
        if not g: continue
        gt=int(rng.choice(g))
        banned=(set(hist_by_user_all.get(u,[])) | set(user_gt[u])) - {gt}
        negs=[]
        for _ in range(20):
            draw=rng.choice(CAND_POOL,size=400,replace=False)
            negs=[int(x) for x in draw if int(x) not in banned][:99]
            if len(negs)==99: break
        if len(negs)<99: continue
        c=np.array([gt]+negs,dtype=np.int64); rng.shuffle(c)
        inst_u.append(u); inst_gt.append(gt); inst_c.append(c)
    B=len(inst_u)
    if B==0: return None
    U=np.array(inst_u,dtype=np.int64); GT=np.array(inst_gt,dtype=np.int64); C=np.stack(inst_c)
    K=C.shape[1]
    ranks={m: np.empty(B,dtype=np.int16) for m in MODELS}
    ua=user_cf[U]; ia=item_cf[C]; s_cf=np.einsum('bd,bkd->bk',ua,ia)
    ub=user_bpr[U]; ib=item_bpr[C]; s_bpr=np.einsum('bd,bkd->bk',ub,ib)
    ut=user_sbert[U]; it=np.asarray(item_sbert[C],dtype=np.float32); s_txt=np.einsum('bd,bkd->bk',ut,it)
    z_cf=np.stack([zscore(s_cf[i]) for i in range(B)])
    z_bpr=np.stack([zscore(s_bpr[i]) for i in range(B)])
    z_txt=np.stack([zscore(s_txt[i]) for i in range(B)])
    def rfs(S):
        order=np.argsort(-S,axis=1); pos=np.empty(B,dtype=np.int16)
        for i in range(B): pos[i]=int(np.where(C[i,order[i]]==GT[i])[0][0])
        return pos
    ranks["als"]=rfs(z_cf); ranks["bpr"]=rfs(z_bpr); ranks["sbert"]=rfs(z_txt)
    ranks["als_bpr"]=rfs(0.5*z_cf+0.5*z_bpr)
    ranks["late_fusion_v3"]=rfs(W_LF[0]*z_cf+W_LF[1]*z_bpr+W_LF[2]*z_txt)
    feats=np.zeros((B,K,IN_DIM),dtype=np.float32)
    o=0
    feats[:,:,o:o+CF_DIM]=l2n(ua)[:,None,:]; o+=CF_DIM
    feats[:,:,o:o+CF_DIM]=l2n(ia.reshape(-1,CF_DIM)).reshape(B,K,CF_DIM); o+=CF_DIM
    feats[:,:,o:o+BPR_DIM]=l2n(ub)[:,None,:]; o+=BPR_DIM
    feats[:,:,o:o+BPR_DIM]=l2n(ib.reshape(-1,BPR_DIM)).reshape(B,K,BPR_DIM); o+=BPR_DIM
    feats[:,:,o:o+SBERT_DIM]=l2n(ut)[:,None,:]; o+=SBERT_DIM
    feats[:,:,o:o+SBERT_DIM]=l2n(it.reshape(-1,SBERT_DIM)).reshape(B,K,SBERT_DIM); o+=SBERT_DIM
    feats[:,:,o]=z_cf; o+=1; feats[:,:,o]=z_bpr; o+=1; feats[:,:,o]=z_txt; o+=1
    feats[:,:,o]=pop_feat(C); o+=1; feats[:,:,o]=hist_feat(U)[:,None]; o+=1
    s_ef=np.empty((B,K),dtype=np.float32)
    with torch.no_grad():
        flat=feats.reshape(B*K,IN_DIM)
        step=65536
        for st in range(0,flat.shape[0],step):
            xb=torch.from_numpy(flat[st:st+step]).to(DEVICE)
            s_ef.reshape(-1)[st:st+step]=model(xb).cpu().numpy()
    ranks["early_fusion_v3"]=rfs(s_ef)
    return ranks

all_r={m: [] for m in MODELS}
total_valid=0
for st in range(0,len(test_users),CHUNK_USERS):
    chunk=test_users[st:st+CHUNK_USERS]
    res=score_chunk(chunk, SEED + st//CHUNK_USERS)
    if res is not None:
        for m in MODELS: all_r[m].append(res[m])
        total_valid += len(res[MODELS[0]])
    if (st//CHUNK_USERS) % 5 == 0:
        log(f"progress {min(st+CHUNK_USERS,len(test_users)):,}/{len(test_users):,} | valid={total_valid:,} | RAM {mem_info()}")
    gc.collect()

for m in MODELS:
    all_r[m]=np.concatenate(all_r[m]) if all_r[m] else np.array([],dtype=np.int16)
n=len(all_r["early_fusion_v3"])
log(f"[DONE] valid users: {n:,}")

def metrics(arr):
    arr=np.asarray(arr,dtype=np.int64)
    hr=lambda k: float((arr<k).mean())
    ndcg=lambda k: float(np.where(arr<k,1.0/np.log2(arr+2),0.0).mean())
    mrr=float((1.0/(arr+1)).mean())
    return {"HR@5":hr(5),"HR@10":hr(10),"HR@20":hr(20),"NDCG@5":ndcg(5),"NDCG@10":ndcg(10),"NDCG@20":ndcg(20),"MRR":mrr}
def ci95(arr,B=2000,seed=12345):
    rng=np.random.default_rng(seed); arr=np.asarray(arr); n=len(arr)
    if n==0: return {}
    stats=np.empty((B,3),dtype=np.float64)
    for s in range(0,B,200):
        e=min(s+200,B); idx=rng.integers(0,n,size=(e-s,n)); samp=arr[idx]
        stats[s:e,0]=(samp<10).mean(axis=1)
        stats[s:e,1]=np.where(samp<10,1.0/np.log2(samp+2),0.0).mean(axis=1)
        stats[s:e,2]=(1.0/(samp+1)).mean(axis=1)
    lo=np.percentile(stats,2.5,axis=0); hi=np.percentile(stats,97.5,axis=0)
    return {"HR@10":[round(float(lo[0]),4),round(float(hi[0]),4)],"NDCG@10":[round(float(lo[1]),4),round(float(hi[1]),4)],"MRR":[round(float(lo[2]),4),round(float(hi[2]),4)]}

summary={"n_users":int(n),"protocol":"1 positive + 99 random negatives, SBERT_COVERAGE pool, seed=42, ALL eligible test users","metrics":{},"ci95":{}}
for m in MODELS:
    summary["metrics"][m]={k:round(v,4) for k,v in metrics(all_r[m]).items()}
    summary["ci95"][m]=ci95(all_r[m])
    log(f"{m}: HR@10={summary['metrics'][m]['HR@10']:.4f} CI95={summary['ci95'][m].get('HR@10')} NDCG@10={summary['metrics'][m]['NDCG@10']:.4f}")

# paired tests on a deterministic 20k subset (same as extended sample logic) for tractability & comparability
rng_sub=np.random.default_rng(SEED)
sub_n=min(20000,n)
sub_idx=rng_sub.choice(n,size=sub_n,replace=False) if n>sub_n else np.arange(n)
def per_user_ndcg(r,k=10): r=r.astype(np.float64); return np.where(r<k,1.0/np.log2(r+2.0),0.0)
def paired(a,b,na,nb):
    from scipy.stats import wilcoxon, binomtest
    ra,rb=all_r[a][sub_idx],all_r[b][sub_idx]
    da,db=per_user_ndcg(ra,10),per_user_ndcg(rb,10); diff=da-db
    try: p_w=float(wilcoxon(da,db,zero_method="wilcox",alternative="two-sided").pvalue) if not np.allclose(diff,0) else 1.0
    except Exception: p_w=float('nan')
    ha,hb=ra<10,rb<10; n10=int(np.sum(ha&~hb)); n01=int(np.sum(~ha&hb))
    p_m=float(binomtest(n10,n10+n01,0.5).pvalue) if (n10+n01)>0 else 1.0
    return {"a":na,"b":nb,"n":int(sub_n),"delta_ndcg10":round(float(da.mean()-db.mean()),4),"wilcoxon_p":p_w,
            "mcnemar_a_only":n10,"mcnemar_b_only":n01,"mcnemar_p":p_m,
            "wins":int(np.sum(ra<rb)),"losses":int(np.sum(ra>rb)),"ties":int(np.sum(ra==rb))}
pairs=[("early_fusion_v3","als"),("early_fusion_v3","als_bpr"),("early_fusion_v3","late_fusion_v3"),("late_fusion_v3","als"),("als","bpr"),("als","sbert")]
summary["paired_tests_subsample"]=[paired(a,b,a,b) for a,b in pairs]
for t in summary["paired_tests_subsample"]:
    log(f"PAIRED({t['n']}) {t['a']} vs {t['b']}: dNDCG={t['delta_ndcg10']:+.4f} wilcoxon={t['wilcoxon_p']:.3g} mcnemar={t['mcnemar_p']:.3g} W/L/T={t['wins']}/{t['losses']}/{t['ties']}")

(OUT_DIR/"full_test_stats.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False))
np.savez_compressed(OUT_DIR/"full_test_ranks.npz", n_users=n, models=np.array(MODELS), **{f"rank_{m}":all_r[m] for m in MODELS})
log(f"saved full_test_stats.json, full_test_ranks.npz | total {time.time()-_T0:.1f}s")
print(json.dumps(summary,indent=2,ensure_ascii=False))
print("[ARTIFACT SAVED]", flush=True)
