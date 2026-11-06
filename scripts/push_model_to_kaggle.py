"""Push a trained model folder to Kaggle as a dataset version (single command)."""
import os, subprocess, sys
from pathlib import Path

MODEL_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("models/recommended")
DATASET_SLUG = os.environ.get("KAGGLE_DATASET", "tnclync/datn-recommended-model")

# Create dataset metadata if missing
meta = MODEL_DIR / "dataset-metadata.json"
if not meta.exists():
    import json
    meta.write_text(json.dumps({
        "title": "DATN Recommended Model",
        "id": DATASET_SLUG,
        "licenses": [{"name": "MIT"}],
    }, indent=2))

# Create or version
try:
    subprocess.check_call(["kaggle", "datasets", "create", "-p", str(MODEL_DIR), "--dir-mode", "zip"])
except subprocess.CalledProcessError:
    subprocess.check_call(["kaggle", "datasets", "version", "-p", str(MODEL_DIR), "-m", "update", "--dir-mode", "zip"])
print("Pushed", MODEL_DIR, "to", DATASET_SLUG)
