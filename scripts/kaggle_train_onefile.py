# Single-file Kaggle training script (GPU-aware)
import os, sys, subprocess
import numpy as np

# Detect GPU
try:
    import torch
    gpu = torch.cuda.is_available()
    print("GPU available:", gpu, torch.cuda.get_device_name(0) if gpu else "CPU")
except Exception:
    gpu = False

# Install deps if missing (Kaggle)
def pip(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

try:
    import implicit
except Exception:
    pip("implicit")

import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path

# Expect data at /kaggle/input/lastone/... or ./data
DATA = Path(os.environ.get("DATN_DATA", "./data/train"))
OUT = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("./models/recommended")
OUT.mkdir(parents=True, exist_ok=True)

pdf = pd.concat([pq.read_table(f).to_pandas() for f in sorted(DATA.glob("*.parquet"))])
users = np.sort(pdf["user_idx"].unique())
items = np.sort(pdf["item_idx"].unique())
u_map = {u: i for i, u in enumerate(users)}
i_map = {it: i for i, it in enumerate(items)}
rows = pdf["user_idx"].map(u_map).to_numpy()
cols = pdf["item_idx"].map(i_map).to_numpy()
data = np.ones(len(pdf), dtype=np.float32)
from scipy.sparse import csr_matrix
mat = csr_matrix((data, (rows, cols)), shape=(len(users), len(items)))

import implicit
model = implicit.als.AlternatingLeastSquares(factors=64, iterations=15, use_gpu=gpu)
model.fit(mat)

np.savez(OUT / "implicit_als.npz",
         user_factors=model.user_factors.astype("float32"),
         item_factors=model.item_factors.astype("float32"),
         user_ids=users.astype("int64"),
         item_ids=items.astype("int64"))
print("Saved to", OUT / "implicit_als.npz")
