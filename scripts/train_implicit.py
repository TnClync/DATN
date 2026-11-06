"""Train implicit-feedback baseline + recommended models on CPU (<=4GB RAM, 6 cores)."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "6")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from src.spark_session import get_spark

BASE = Path(__file__).resolve().parents[1]
TRAIN = BASE / "data/train"
OUT_BASE = BASE / "models/baseline"
OUT_REC = BASE / "models/recommended"
OUT_BASE.mkdir(parents=True, exist_ok=True)
OUT_REC.mkdir(parents=True, exist_ok=True)

# 1) Popularity via Spark aggregation (laptop-friendly)
spark, tmp = get_spark("DATN-train", cores=6, memory_gb=4)
df = spark.read.parquet(str(TRAIN))
pop = df.groupBy("item_idx").count().orderBy("count", ascending=False)
pop_items = np.array([r.item_idx for r in pop.collect()], dtype=np.int64)
np.save(OUT_BASE / "popularity.npy", pop_items)
print("Popularity saved:", pop_items.shape)

# 2) Implicit ALS on CPU (sampled to fit 4GB)
import implicit
from scipy.sparse import csr_matrix

# Read a manageable subset of interactions
pdf = pd.concat([pq.read_table(f).to_pandas() for f in sorted(TRAIN.glob("*.parquet"))])
# downsample users to keep matrix small
max_users = 50000
users = pdf["user_idx"].unique()
if len(users) > max_users:
    keep = set(np.random.choice(users, max_users, replace=False))
    pdf = pdf[pdf["user_idx"].isin(keep)]

user_ids = np.sort(pdf["user_idx"].unique())
item_ids = np.sort(pdf["item_idx"].unique())
u_map = {u: i for i, u in enumerate(user_ids)}
i_map = {it: i for i, it in enumerate(item_ids)}

rows = pdf["user_idx"].map(u_map).to_numpy()
cols = pdf["item_idx"].map(i_map).to_numpy()
data = np.ones(len(pdf), dtype=np.float32)
mat = csr_matrix((data, (rows, cols)), shape=(len(user_ids), len(item_ids)))

model = implicit.als.AlternatingLeastSquares(factors=32, iterations=5, use_gpu=False, num_threads=6)
model.fit(mat)
np.savez(
    OUT_REC / "implicit_als.npz",
    user_factors=model.user_factors.astype(np.float32),
    item_factors=model.item_factors.astype(np.float32),
    user_ids=user_ids.astype(np.int64),
    item_ids=item_ids.astype(np.int64),
)
print("Implicit ALS saved.")
spark.stop()
print("Done. Spark temp dir:", tmp)
