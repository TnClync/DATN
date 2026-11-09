#!/usr/bin/env python
"""Build a small simulation cache (sampled users + histories + positives)."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FACTORS = ROOT / "artifacts" / "factors"
OUT = ROOT / "artifacts" / "simulation"
OUT.mkdir(parents=True, exist_ok=True)
CATALOG = 200_000
MIN_INT = 3
N_SAMPLE = 6000
SEED = 42


def read_split(name):
    frames = []
    for f in sorted((DATA / name).glob("*.parquet")):
        d = pq.read_table(f, columns=["user_idx", "item_idx", "rating"]).to_pandas()
        d = d[d["rating"] >= 4.0]
        frames.append(d[["user_idx", "item_idx"]])
    return pd.concat(frames, ignore_index=True)


u_ids = np.load(FACTORS / "als_user_ids.npy")
i_ids = np.load(FACTORS / "als_item_ids.npy")


def to_compact_user(arr):
    pos = np.searchsorted(u_ids, arr)
    pos = np.clip(pos, 0, len(u_ids) - 1)
    ok = u_ids[pos] == arr
    out = np.where(ok, pos, -1)
    return out


def to_compact_item(arr):
    pos = np.searchsorted(i_ids, arr)
    pos = np.clip(pos, 0, len(i_ids) - 1)
    ok = i_ids[pos] == arr
    out = np.where(ok, pos, -1)
    return out


train = read_split("train")
test = read_split("test")

# compact mapping
train["cu"] = to_compact_user(train["user_idx"].to_numpy(np.int64))
train["ci"] = to_compact_item(train["item_idx"].to_numpy(np.int64))
train = train[(train["cu"] >= 0) & (train["ci"] >= 0)]
train = train[train["ci"] < CATALOG]

test["cu"] = to_compact_user(test["user_idx"].to_numpy(np.int64))
test["ci"] = to_compact_item(test["item_idx"].to_numpy(np.int64))
test = test[(test["cu"] >= 0) & (test["ci"] >= 0)]
test = test[test["ci"] < CATALOG]

# active users
counts = train["cu"].value_counts()
active = counts[counts >= MIN_INT].index
train = train[train["cu"].isin(active)]

# test positives by user
pos_by_user = test.groupby("cu")["ci"].apply(lambda s: sorted(set(int(x) for x in s)))
elig = [int(u) for u in pos_by_user.index if u in set(active.tolist())]
print("eligible users:", len(elig), flush=True)

rng = np.random.default_rng(SEED)
chosen = np.sort(rng.choice(elig, size=min(N_SAMPLE, len(elig)), replace=False))

# train history for chosen users
train_chosen = train[train["cu"].isin(chosen)]
hist_by_user = train_chosen.groupby("cu")["ci"].apply(lambda s: sorted(set(int(x) for x in s))).to_dict()

hist_items_list = [hist_by_user.get(int(u), []) for u in chosen]
pos_items_list = [pos_by_user.get(int(u), []) for u in chosen]

hist_offsets = [0]
for h in hist_items_list:
    hist_offsets.append(hist_offsets[-1] + len(h))
pos_offsets = [0]
for p in pos_items_list:
    pos_offsets.append(pos_offsets[-1] + len(p))

hist_flat = np.concatenate([np.asarray(h, dtype=np.int32) for h in hist_items_list]) if hist_items_list else np.array([], dtype=np.int32)
pos_flat = np.concatenate([np.asarray(p, dtype=np.int32) for p in pos_items_list]) if pos_items_list else np.array([], dtype=np.int32)

np.savez_compressed(
    OUT / "sim_eval_cache.npz",
    user_rows=np.asarray(chosen, dtype=np.int32),
    hist_offsets=np.asarray(hist_offsets, dtype=np.int64),
    hist_items=hist_flat,
    pos_offsets=np.asarray(pos_offsets, dtype=np.int64),
    pos_items=pos_flat,
)
print("saved", OUT / "sim_eval_cache.npz", "users", len(chosen), "hist", len(hist_flat), "pos", len(pos_flat), flush=True)
EOF