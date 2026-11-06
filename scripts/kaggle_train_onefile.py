"""Single-file Kaggle training (Spark staging + implicit ALS), GPU-aware.

Use with TWO Kaggle accounts in parallel:
  - Account 1: trains `baseline` (CF+CB hybrid) on data shard 0..N/2
  - Account 2: trains `recommended` (implicit ALS) on data shard N/2..N
Merge the resulting artifacts locally into models/baseline and models/recommended.

Run on Kaggle: set DATN_DATA and DATN_OUT env vars, or use defaults.
"""
import os, sys, subprocess
from pathlib import Path

# Detect GPU
try:
    import torch
    gpu = torch.cuda.is_available()
    print("GPU available:", gpu)
except Exception:
    gpu = False

def pip(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

for pkg in ("implicit", "pyspark"):
    try:
        __import__(pkg)
    except Exception:
        pip(pkg)

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

DATA = Path(os.environ.get("DATN_DATA", "/kaggle/input/lastone/train"))
OUT = Path(os.environ.get("DATN_OUT", "/kaggle/working"))
OUT.mkdir(parents=True, exist_ok=True)

# Spark staging with temp_dir
SPARK_TMP = Path("/kaggle/working/spark_tmp")
SPARK_TMP.mkdir(parents=True, exist_ok=True)
os.environ["SPARK_LOCAL_DIRS"] = str(SPARK_TMP)

from pyspark.sql import SparkSession
spark = (SparkSession.builder.appName("DATN-Kaggle")
         .master("local[*]")
         .config("spark.driver.memory", "12g")
         .config("spark.local.dir", str(SPARK_TMP))
         .getOrCreate())
spark.sparkContext.setLogLevel("WARN")

# Read parquet via Spark, convert to pandas (staging)
sdf = spark.read.parquet(str(DATA))
# Optional shard split for 2-account parallelism
shard = os.environ.get("DATN_SHARD", "all")  # "all" | "0" | "1"
if shard != "all":
    n = sdf.count()
    half = n // 2
    if shard == "0":
        sdf = sdf.limit(half)
    else:
        sdf = sdf.offset(half)
pdf = sdf.toPandas()
spark.stop()

# Train implicit ALS
import implicit
from scipy.sparse import csr_matrix

users = np.sort(pdf["user_idx"].unique())
items = np.sort(pdf["item_idx"].unique())
u_map = {u: i for i, u in enumerate(users)}
i_map = {it: i for i, it in enumerate(items)}
rows = pdf["user_idx"].map(u_map).to_numpy()
cols = pdf["item_idx"].map(i_map).to_numpy()
data = np.ones(len(pdf), dtype=np.float32)
mat = csr_matrix((data, (rows, cols)), shape=(len(users), len(items)))

model = implicit.als.AlternatingLeastSquares(factors=64, iterations=15, use_gpu=gpu)
model.fit(mat)

np.savez(OUT / "implicit_als.npz",
         user_factors=model.user_factors.astype("float32"),
         item_factors=model.item_factors.astype("float32"),
         user_ids=users.astype("int64"),
         item_ids=items.astype("int64"))
# popularity (baseline)
pop = pdf["item_idx"].value_counts().index.to_numpy()
np.save(OUT / "popularity.npy", pop)
print("Saved to", OUT)
PY