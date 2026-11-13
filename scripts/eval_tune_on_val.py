#!/usr/bin/env python
"""Evaluate baselines + deep heads with NO test-tuning leak.

- Tune late-fusion alpha and stacked-ensemble weights on VALIDATION users.
- Report final metrics on the 20% sampled TEST users (seeded), disjoint role.
- Protocol: 1 positive + 99 random negatives from the shared 2,464,371-item
  catalog, leak-free SBERT profiles (history from train only).

Usage:
  .venv/bin/python scripts/eval_tune_on_val.py \
      --artifacts artifacts/eval_inputs \
      --data data --out artifacts/results/eval_tune_on_val.json
"""
from __future__ import annotations

import argparse, json, math, os, time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn

CATALOG = 2464371
SEED = 12345
BATCH = 1024
DEEP_BATCH = 256
TEST_SAMPLE_FRAC = 0.20
TUNE_N = 10000  # cap validation users used for tuning (for speed)


def read_split(directory: Path, columns):
    parts = sorted(directory.glob("*.parquet"))
    frames = [pq.read_table(p, columns=columns).to_pandas() for p in parts]
    df = pd.concat(frames, ignore_index=True)
    if "rating" in df.columns:
        df = df[df["rating"] >= 4.0]
    return df


def map_ids(values, ids):
    pos = np.searchsorted(ids, values)
    pos = np.clip(pos, 0, len(ids) - 1)
    return pos, ids[pos] == values


def mm_rows(x: np.ndarray) -> np.ndarray:
    lo = x.min(axis=1, keepdims=True)
    hi = x.max(axis=1, keepdims=True)
    return (x - lo) / np.where(hi > lo, hi - lo, 1.0)


def metrics_from_positions(pos: np.ndarray, valid: np.ndarray | None = None):
    if valid is not None:
        pos = pos[valid]
    n = len(pos)
    if n == 0:
        return {"HR@5": 0, "HR@10": 0, "HR@20": 0, "NDCG@5": 0, "NDCG@10": 0,
                "NDCG@20": 0, "MRR": 0, "Recall@10": 0, "Recall@20": 0, "n_eval_users": 0}
    hr5 = float((pos < 5).mean()); hr10 = float((pos < 10).mean()); hr20 = float((pos < 20).mean())
    ndcg = lambda k: float(np.mean(np.where(pos < k, 1.0 / np.log2(pos + 2.0), 0.0)))
    return {
        "HR@5": round(hr5, 4), "HR@10": round(hr10, 4), "HR@20": round(hr20, 4),
        "Recall@5": round(hr5, 4), "Recall@10": round(hr10, 4), "Recall@20": round(hr20, 4),
        "NDCG@5": round(ndcg(5), 4), "NDCG@10": round(ndcg(10), 4), "NDCG@20": round(ndcg(20), 4),
        "MRR": round(float(np.mean(1.0 / (pos + 1.0))), 4),
        "n_eval_users": int(n),
    }


def positions_from_scores(scores: np.ndarray, gt_pos: np.ndarray) -> np.ndarray:
    return (scores > scores[np.arange(len(scores)), gt_pos][:, None]).sum(axis=1).astype(np.int32)


def build_instances(split_df, active_set, item_set, rng, label):
    df = split_df[split_df["user_idx"].isin(active_set) & split_df["item_idx"].isin(item_set)].copy()
    if len(df) == 0:
        raise RuntimeError(f"no eligible rows in {label}")
    # one positive per user (val/test have 1; be defensive)
    df = df.drop_duplicates(subset=["user_idx"], keep="last")
    users = df["user_idx"].to_numpy(np.int64)
    gts = df["item_idx"].to_numpy(np.int64)
    N = len(users)
    print(f"[{label}] eligible users: {N:,}")
    cands = np.empty((N, 100), dtype=np.int64)
    gt_pos = np.empty(N, dtype=np.int32)
    # banned = train history handled outside? no: we must not put train items as negs except none; but easier ban = train+gt
    return df, users, gts, cands, gt_pos


class LightGate(nn.Module):
    def __init__(self, in_features, hidden1=64, hidden2=None):
        super().__init__()
        layers = [nn.Linear(in_features, hidden1), nn.ReLU(), nn.Dropout(0.1)]
        if hidden2:
            layers += [nn.Linear(hidden1, hidden2), nn.ReLU(), nn.Dropout(0.1)]
        layers += [nn.Linear(hidden2 or hidden1, 1), nn.Sigmoid()]
        self.gate_net = nn.Sequential(*layers)
    def forward(self, u_cf, i_cf, i_sbert, cf_score, sbert_score):
        alpha = self.gate_net(torch.cat([u_cf, i_cf, i_sbert], dim=-1)).squeeze(-1)
        return alpha * cf_score + (1.0 - alpha) * sbert_score, alpha


class DeepEarlyFusion(nn.Module):
    def __init__(self, in_dim, cross_layers=3, hidden=128):
        super().__init__()
        self.cross = nn.ModuleList([nn.Linear(in_dim, in_dim) for _ in range(cross_layers)])
        self.deep = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                                  nn.Linear(hidden, hidden), nn.ReLU(),
                                  nn.Linear(hidden, 1))
    def forward(self, x):
        x0, xi = x, x
        for layer in self.cross:
            xi = x0 * torch.sigmoid(layer(xi)) + xi
        return (xi.mean(dim=-1, keepdim=True) + self.deep(xi)).squeeze(-1)


class UserTower(nn.Module):
    def __init__(self, in_dim=32, hidden=128, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.1),
                                 nn.Linear(hidden, out_dim))
    def forward(self, x): return self.net(x)


class ItemTower(nn.Module):
    def __init__(self, in_dim=32 + 768, hidden=256, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.1),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.1),
                                 nn.Linear(hidden, out_dim))
    def forward(self, x): return self.net(x)


class TransformerFusion(nn.Module):
    def __init__(self, in_dim=832, d_model=128, nhead=4, nlayers=2):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                         dim_feedforward=d_model * 4, dropout=0.1,
                                         batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc, num_layers=nlayers)
        self.head = nn.Linear(d_model, 1)
    def forward(self, x):
        h = self.encoder(self.proj(x))
        return self.head(h).squeeze(-1)


def state_dict(path):
    sd = torch.load(path, map_location="cpu")
    return sd.get("state_dict", sd) if isinstance(sd, dict) and "state_dict" in sd else sd


def gate_arch(sd):
    h1 = int(sd["gate_net.0.weight"].shape[0])
    in_f = int(sd["gate_net.0.weight"].shape[1])
    h2 = int(sd["gate_net.3.weight"].shape[0]) if "gate_net.6.weight" in sd else None
    return dict(in_features=in_f, hidden1=h1, hidden2=h2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts/eval_inputs")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="artifacts/results/eval_tune_on_val.json")
    ap.add_argument("--test-sample-frac", type=float, default=TEST_SAMPLE_FRAC)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--tune-n", type=int, default=TUNE_N)
    args = ap.parse_args()

    A = Path(args.artifacts); D = Path(args.data); out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("Loading artifacts from", A)
    user_ids = np.load(A/"user_ids.npy")
    item_ids = np.load(A/"item_ids.npy")
    als_uf = np.load(A/"als_user_factors_f16.npy", mmap_mode="r")
    als_itf = np.load(A/"als_item_factors_f16.npy", mmap_mode="r")
    bpr_uf = np.load(A/"bpr_user_factors_f16.npy", mmap_mode="r")
    bpr_itf = np.load(A/"bpr_item_factors_f16.npy", mmap_mode="r")
    graph_uf = np.load(A/"graph_user_factors_f16.npy", mmap_mode="r")
    graph_itf = np.load(A/"graph_item_factors_f16.npy", mmap_mode="r")
    sb768 = np.load(A/"sbert_item_embeddings_f16.npy", mmap_mode="r")  # 768
    sb384p = A/"sbert_item_embeddings_384_f16.npy"
    sb384 = np.load(sb384p, mmap_mode="r") if sb384p.exists() else None
    assert sb768.shape == (CATALOG, 768)
    print("SBERT 768:", sb768.shape, "384:", None if sb384 is None else sb384.shape)

    # heads
    lg_sd = state_dict(A/"light_gated_fusion.pt")
    lg = LightGate(**gate_arch(lg_sd)); lg.load_state_dict(lg_sd); lg.eval()
    lg_sdim = gate_arch(lg_sd)["in_features"] - 64
    de_sd = state_dict(A/"deep_early_fusion.pt")
    de_in = int(de_sd["cross.0.weight"].shape[1])
    de = DeepEarlyFusion(de_in); de.load_state_dict(de_sd); de.eval()
    de_sdim = de_in - 64
    gg_sd = state_dict(A/"graph_light_gated_fusion.pt")
    gg = LightGate(**gate_arch(gg_sd)); gg.load_state_dict(gg_sd); gg.eval()
    ut = UserTower(); it = ItemTower()
    tt = torch.load(A/"two_tower_fusion.pt", map_location="cpu")
    ut.load_state_dict(tt["user"]); it.load_state_dict(tt["item"]); ut.eval(); it.eval()
    tr = TransformerFusion(); tr.load_state_dict(torch.load(A/"transformer_fusion.pt", map_location="cpu")); tr.eval()
    print("heads loaded: light_gated sdim", lg_sdim, "deep_early sdim", de_sdim)

    # splits
    print("Reading splits...")
    train = read_split(D/"train", ["user_idx","item_idx","rating"])
    val = read_split(D/"validation", ["user_idx","item_idx","rating"])
    test = read_split(D/"test", ["user_idx","item_idx","rating"])
    active = set(train.groupby("user_idx").size()[lambda s: s>=3].index.to_numpy(np.int64))
    item_set = set(item_ids.tolist())

    def prep(split, label, sample_frac=None):
        df = split[split["user_idx"].isin(active) & split["item_idx"].isin(item_set)].copy()
        df = df.drop_duplicates("user_idx", keep="last")
        if sample_frac is not None and 0 < sample_frac < 1:
            rng = np.random.default_rng(args.seed)
            take = rng.choice(len(df), size=max(1, int(len(df)*sample_frac)), replace=False)
            df = df.iloc[np.sort(take)].copy()
        u = df["user_idx"].to_numpy(np.int64); g = df["item_idx"].to_numpy(np.int64)
        cu, oku = map_ids(u, user_ids); ci, oki = map_ids(g, item_ids)
        keep = oku & oki
        cu, ci = cu[keep], ci[keep]
        print(f"[{label}] users={len(cu):,} (from eligible {len(df):,})")
        return cu, ci

    cu_val, ci_val = prep(val, "val", sample_frac=None)
    cu_test, ci_test = prep(test, "test", sample_frac=args.test_sample_frac)

    # tune subset from val
    rng = np.random.default_rng(args.seed)
    tune_n = min(args.tune_n, len(cu_val))
    tune_idx = np.sort(rng.choice(len(cu_val), size=tune_n, replace=False))
    print(f"tune on val subset: {tune_n} users; report on test: {len(cu_test)} users")

    # train history for bans and user sbert profiles (need compact indices)
    print("Building train history for candidates/profiles...")
    # history per compact user: list of compact item indices (all train items, not just <CATALOG except for sbert)
    tr_u, oku = map_ids(train["user_idx"].to_numpy(np.int64), user_ids)
    tr_i, oki = map_ids(train["item_idx"].to_numpy(np.int64), item_ids)
    ok = oku & oki
    tr_u, tr_i = tr_u[ok], tr_i[ok]
    hist = {}
    for u,i in zip(tr_u, tr_i):
        hist.setdefault(int(u), []).append(int(i))
    print("users with history:", len(hist))

    def make_cands(cu, ci, label):
        N = len(cu)
        cands = np.empty((N,100), dtype=np.int64)
        gt_pos = np.empty(N, dtype=np.int32)
        R = rng.integers(0, CATALOG, size=(N, 140))
        for k in range(N):
            u = int(cu[k]); g = int(ci[k])
            ban = set(hist.get(u, [])); ban.add(g)
            negs=[]; 
            for x in R[k]:
                x=int(x)
                if x==g or x in ban: continue
                negs.append(x)
                if len(negs)==99: break
            # fallback if rare
            while len(negs)<99:
                x=int(rng.integers(0,CATALOG))
                if x==g or x in ban or x in negs: continue
                negs.append(x)
            arr = np.array([g]+negs, dtype=np.int64)
            rng.shuffle(arr)
            cands[k]=arr
            gt_pos[k]=int(np.where(arr==g)[0][0])
            if (k+1)%100000==0: print(f"  {label} cands {k+1}/{N}")
        return cands, gt_pos

    print("Building VAL candidates...")
    val_cands, val_gt = make_cands(cu_val, ci_val, "val")
    print("Building TEST candidates...")
    test_cands, test_gt = make_cands(cu_test, ci_test, "test")

    # user sbert profiles (768 and 384) from history
    def user_profiles(cu, emb):
        dim = emb.shape[1]
        U = np.zeros((len(cu), dim), dtype=np.float32)
        for k,u in enumerate(cu):
            h = [i for i in hist.get(int(u), []) if i < emb.shape[0]]
            if h:
                v = np.asarray(emb[h], dtype=np.float32).mean(0)
                n = np.linalg.norm(v); U[k]=v/n if n>1e-8 else 0
        return U
    print("Building user SBERT profiles...")
    u768_val = user_profiles(cu_val, sb768); u768_test = user_profiles(cu_test, sb768)
    u384_val = user_profiles(cu_val, sb384) if sb384 is not None else None
    u384_test = user_profiles(cu_test, sb384) if sb384 is not None else None

    def sbert_rows(emb, idx2d):
        flat = idx2d.reshape(-1)
        return np.asarray(emb[flat], dtype=np.float32).reshape(idx2d.shape[0], idx2d.shape[1], emb.shape[1])

    def cf_score(uf,itf,cu,cands):
        U=np.asarray(uf[cu],dtype=np.float32)
        I=np.asarray(itf[cands],dtype=np.float32)
        return np.einsum('bnd,bd->bn', I, U, optimize=True)

    def run_eval(cu,cands,gt,idx,name,need384=False):
        # returns scores (N,100) for idx subset? easier: compute positions for idx
        s = np.empty((len(idx),100), dtype=np.float32)
        rows = idx
        for off in range(0,len(rows),DEEP_BATCH):
            sl = rows[off:off+DEEP_BATCH]
            c = cands[sl]; u = cu[sl]; b=len(sl)
            if name=='als':
                s[off:off+b]=cf_score(als_uf,als_itf,u,c)
            elif name=='bpr':
                s[off:off+b]=cf_score(bpr_uf,bpr_itf,u,c)
            elif name=='sbert':
                usb = u768_val if cu is cu_val else u768_test
                isb = sbert_rows(sb768,c)
                s[off:off+b]=np.einsum('bnd,bd->bn', isb, usb[sl], optimize=True)
            elif name=='lightgcn':
                s[off:off+b]=cf_score(graph_uf,graph_itf,u,c)
            elif name=='late_fusion':
                usb = u768_val if cu is cu_val else u768_test
                cfv=mm_rows(cf_score(als_uf,als_itf,u,c))
                sbv=mm_rows(np.einsum('bnd,bd->bn', sbert_rows(sb768,c), usb[sl], optimize=True))
                s[off:off+b]=name_alpha*cfv+(1-name_alpha)*sbv
            elif name=='light_gated':
                usb_s = (u384_val if cu is cu_val else u384_test) if lg_sdim==384 else (u768_val if cu is cu_val else u768_test)
                emb_s = sb384 if lg_sdim==384 else sb768
                if emb_s is None: return None
                ucf=np.asarray(als_uf[u],dtype=np.float32); icf=np.asarray(als_itf[c],dtype=np.float32)
                isb=sbert_rows(emb_s,c); cfv=np.einsum('bnd,bd->bn',icf,ucf,optimize=True)
                sbv=np.einsum('bnd,bd->bn',isb,usb_s[sl],optimize=True)
                with torch.no_grad():
                    fused,_=lg(torch.from_numpy(np.repeat(ucf[:,None,:],100,1)), torch.from_numpy(icf), torch.from_numpy(isb), torch.from_numpy(cfv), torch.from_numpy(sbv))
                s[off:off+b]=fused.numpy()
            elif name=='deep_early':
                emb_s = sb384 if de_sdim==384 else sb768
                usb_s = (u384_val if cu is cu_val else u384_test) if de_sdim==384 else (u768_val if cu is cu_val else u768_test)
                if emb_s is None: return None
                ucf=np.asarray(als_uf[u],dtype=np.float32); icf=np.asarray(als_itf[c],dtype=np.float32)
                isb=sbert_rows(emb_s,c)
                feats=np.concatenate([np.repeat(ucf[:,None,:],100,1), icf, isb], axis=2)
                with torch.no_grad():
                    s[off:off+b]=de(torch.from_numpy(feats.reshape(b*100,-1))).numpy().reshape(b,100)
            elif name=='graph_light_gated':
                usb=u768_val if cu is cu_val else u768_test
                ucf=np.asarray(graph_uf[u],dtype=np.float32); icf=np.asarray(graph_itf[c],dtype=np.float32)
                isb=sbert_rows(sb768,c); cfv=np.einsum('bnd,bd->bn',icf,ucf,optimize=True)
                sbv=np.einsum('bnd,bd->bn',isb,usb[sl],optimize=True)
                with torch.no_grad():
                    fused,_=gg(torch.from_numpy(np.repeat(ucf[:,None,:],100,1)), torch.from_numpy(icf), torch.from_numpy(isb), torch.from_numpy(cfv), torch.from_numpy(sbv))
                s[off:off+b]=fused.numpy()
            elif name=='two_tower':
                usb=u768_val if cu is cu_val else u768_test
                uin=np.asarray(graph_uf[u],dtype=np.float32)
                iin=np.concatenate([np.asarray(graph_itf[c],dtype=np.float32), sbert_rows(sb768,c)], axis=2)
                with torch.no_grad():
                    ue=ut(torch.from_numpy(uin)).numpy()
                    ie=it(torch.from_numpy(iin.reshape(b*100,-1))).numpy().reshape(b,100,-1)
                    s[off:off+b]=np.einsum('bd,bnd->bn', ue, ie, optimize=True)
            elif name=='transformer':
                ucf=np.asarray(graph_uf[u],dtype=np.float32); icf=np.asarray(graph_itf[c],dtype=np.float32)
                isb=sbert_rows(sb768,c)
                feats=np.concatenate([np.repeat(ucf[:,None,:],100,1), icf, isb], axis=2).astype(np.float32)
                with torch.no_grad():
                    s[off:off+b]=tr(torch.from_numpy(feats)).numpy()
            else:
                raise ValueError(name)
        return s

    # We need scores for all val (for tuning subset metrics) and all test (for report). Compute and cache mm-normalized per model.
    model_names = ['als','bpr','sbert','lightgcn','light_gated','deep_early','graph_light_gated','two_tower','transformer']
    # tune alpha for late fusion on VAL tune subset
    name_alpha = 0.8
    def eval_lf(alpha, subset_idx):
        nonlocal name_alpha; name_alpha=alpha
        s = run_eval(cu_val,val_cands,val_gt, subset_idx, 'late_fusion')
        return metrics_from_positions(positions_from_scores(s, val_gt[subset_idx]))

    print("\nTuning late-fusion alpha on VAL...")
    best_a,best_m=0.8,-1
    for a in [0.0,0.2,0.4,0.5,0.6,0.7,0.8,1.0]:
        m=eval_lf(a,tune_idx); print(f" alpha={a} val HR@10={m['HR@10']:.4f} NDCG@10={m['NDCG@10']:.4f}")
        if m['HR@10']>best_m: best_m,best_a=m['HR@10'],a
    print("best alpha", best_a, "val HR@10", round(best_m,4))
    name_alpha=best_a

    def compute_scores(cu,cands,gt,which,lf_alpha):
        nonlocal name_alpha; name_alpha=lf_alpha
        idx_all = np.arange(len(cu))
        scores = {}
        for nm in model_names + ['late_fusion']:
            print(f" scoring {which}:{nm} ...", flush=True)
            s = run_eval(cu,cands,gt, idx_all, nm)
            if s is None:
                print(f"  skip {nm} (missing dep)"); continue
            scores[nm]=mm_rows(s).astype(np.float16)
        return scores

    print("\nScoring VAL models...")
    val_scores = compute_scores(cu_val,val_cands,val_gt,'val',best_a)
    print("Scoring TEST models...")
    test_scores = compute_scores(cu_test,test_cands,test_gt,'test',best_a)

    base = [k for k in ['als','bpr','sbert','lightgcn','late_fusion','light_gated','deep_early','graph_light_gated','two_tower','transformer'] if k in val_scores]
    print("available bases", base)

    # tune ensemble on VAL tune subset
    def blend(sc, idx, w):
        out=np.zeros((len(idx),100),dtype=np.float32)
        for n,ww in w.items(): out += ww*np.asarray(sc[n][idx],dtype=np.float32)
        return out
    def met(sc,idx,w,gt): return metrics_from_positions(positions_from_scores(blend(sc,idx,w), gt[idx]))

    counts={}
    if len(base)>=2:
        best=-1
        for _ in range(40):
            champ=None
            for n in base:
                trc=dict(counts); trc[n]=trc.get(n,0)+1
                tot=sum(trc.values()); w={k:v/tot for k,v in trc.items()}
                m=met(val_scores,tune_idx,w,val_gt)
                if champ is None or m['NDCG@10']>champ[0]: champ=(m['NDCG@10'],n,w,m)
            if champ[0] <= best + 1e-6: break
            best=champ[0]; counts[champ[1]]=counts.get(champ[1],0)+1
        tot=sum(counts.values()) if counts else 1
        ens_w={k:round(v/tot,6) for k,v in counts.items()}
    else:
        ens_w={base[0]:1.0} if base else {}
    print("ensemble weights (tuned on val)", ens_w)

    # report on TEST (full sampled)
    all_test_idx = np.arange(len(cu_test))
    results={}
    for n in base:
        results[n]=metrics_from_positions(positions_from_scores(np.asarray(test_scores[n],dtype=np.float32), test_gt))
        print(n, results[n])
    # per-model val (for reference, tuned set & full val)
    val_results={n: metrics_from_positions(positions_from_scores(np.asarray(val_scores[n],dtype=np.float32), val_gt)) for n in base}
    ens_test = met(test_scores, all_test_idx, ens_w, test_gt) if ens_w else None
    if ens_test: print('ensemble_test', ens_test)

    summary={
        'protocol': 'Tune alpha/ensemble on VALIDATION; report on 20% TEST sample. 1+99 neg, shared 2.46M catalog, train-only history.',
        'tune_split':'validation','report_split':'test_20pct','seed':args.seed,
        'n_val_eligible': int(len(cu_val)), 'n_tune': int(tune_n), 'n_test_report': int(len(cu_test)),
        'best_alpha_late_fusion': best_a,
        'ensemble_weights_val_tuned': ens_w,
        'test': results,
        'test_ensemble': ens_test,
        'val_full_reference': val_results,
        'elapsed_sec': round(time.time()-t0,1),
    }
    out.write_text(json.dumps(summary, indent=2))
    print('Wrote', out)

if __name__=='__main__':
    main()
