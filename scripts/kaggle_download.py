import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
NOTEBOOK = "nguynteo/mapping"
OUTPUT_DIR = PROJECT_DIR / "data" / "mapping"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

cmd = [
    sys.executable, "-m", "kaggle",
    "kernels", "output",
    NOTEBOOK,
    "-p", str(OUTPUT_DIR),
    "-o",
]

print("Đang tải output Kaggle...", flush=True)

try:
    subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print(f"Tải xong: {OUTPUT_DIR}", flush=True)

except subprocess.CalledProcessError:
    print(
        "Tải thất bại. Kiểm tra API key, quyền truy cập "
        "và trạng thái notebook.",
        flush=True,
    )