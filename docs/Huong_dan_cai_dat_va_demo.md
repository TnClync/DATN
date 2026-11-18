# HƯỚNG DẪN CÀI ĐẶT VÀ CHẠY DEMO — Deep Hybrid Recommender

> Tài liệu hướng dẫn từ A-Z: clone repo → tải model → chạy demo inference.  
> Áp dụng cho cả **Windows / macOS / Linux**.

---

## 1. Yêu cầu hệ thống

| Thành phần | Yêu cầu tối thiểu | Khuyến nghị |
|---|---|---|
| Python | 3.10+ | 3.10 (đã test) |
| RAM | 8 GB | 16 GB |
| Disk trống | 5 GB (chỉ model + code) | 10 GB (đầy đủ artifact) |
| GPU | Không bắt buộc | NVIDIA (CUDA) nếu train |
| OS | Windows 10 / macOS 12 / Ubuntu 20.04 | Ubuntu 22.04 / Windows 11 |

---

## 2. Cài đặt từ đầu (Fresh machine)

### Bước 1 — Clone repository
```bash
git clone https://github.com/TnClync/DATN.git
cd DATN
```

### Bước 2 — Tạo virtual environment
**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

### Bước 3 — Cài dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> Nếu chưa có `requirements.txt`, cài thủ công:
> ```bash
> pip install numpy pandas scipy scikit-learn torch faiss-cpu implicit \
>             sentence-transformers huggingface_hub fastapi uvicorn \
>             python-docx matplotlib pyarrow
> ```

### Bước 4 — Tải artifact từ Hugging Face (chỉ cần 1 lần)
```bash
python scripts/download_demo_artifacts.py
```
- Tổng dung lượng: **~3.9 GB** (EF V3 model 36MB + SBERT embeddings 1.9GB + ALS/BPR factors 328MB + item metadata DB 725MB + user history DB 343MB + FAISS index 293MB + IDs/popularity).
- Thời gian: 10–30 phút tuỳ tốc độ mạng.
- Artifact sẽ nằm tại `artifacts/` (cùng cấp với `src/`).

> **Nếu Hugging Face bị chặn (firewall):** tải thủ công từ https://huggingface.co/datasets/Clync/datn-deep-hybrid-recommender-data → giải nén vào `artifacts/`.

### Bước 5 — Kiểm tra cài đặt
```bash
python -c "import numpy, torch, faiss, sentence_transformers, huggingface_hub; print('OK')"
python -c "from src.serving.pipeline import RecommenderPipeline; p=RecommenderPipeline(); p.load(); print('Pipeline available:', p._available)"
```

---

## 3. Chạy Demo

### 3.1 Demo CLI (inference nhanh)
```bash
python scripts/run_demo.py --customer-id 12345 --top-k 10
```
**Tham số:**
| Tham số | Mặc định | Mô tả |
|---|---|---|
| `--customer-id` | 0 | User index (0 … 2.559.403) |
| `--top-k` | 10 | Số gợi ý trả về |
| `--model` | als | Model: `als`, `bpr`, `late_fusion`, `early_fusion` |

**Ví dụ:**
```bash
# Gợi ý top-20 cho user 100
python scripts/run_demo.py --customer-id 100 --top-k 20

# Dùng BPR
python scripts/run_demo.py --customer-id 100 --model bpr --top-k 10
```

### 3.2 Web UI (FastAPI)
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```
- Trang chủ: http://localhost:8000
- Swagger API docs: http://localhost:8000/docs
- Endpoints:
  - `GET /health` — kiểm tra tình trạng
  - `GET /models` — danh sách model
  - `POST /recommend` — gợi ý (customer_id, model, top_k, alpha)
  - `GET /compare` — bảng so sánh model
  - `GET /customer/{id}` — thông tin customer

### 3.3 Chạy từ Docker (tuỳ chọn)
```bash
# Build
docker build -t datn-recommender .

# Run (mount artifacts)
docker run -p 8000:8000 -v /path/to/artifacts:/app/artifacts datn-recommender
```

---

## 4. Cấu trúc thư mục sau khi cài đặt

```
DATN/
├── src/
│   ├── api/main.py              # FastAPI backend
│   ├── config.py                # Cấu hình đường dẫn (env override)
│   ├── serving/pipeline.py      # Luồng suy luận
│   └── models/                  # Lớp mô hình
├── scripts/
│   ├── download_demo_artifacts.py  # Tải artifact từ HF
│   ├── run_demo.py                 # Demo CLI
│   ├── make_diagrams.py            # Sinh sơ đồ
│   └── md_to_docx.py               # Chuyển MD → DOCX
├── artifacts/                   # Model weights, embeddings, DBs (tải từ HF)
│   ├── early_fusion_v3/         # EF V3 model + IDs
│   ├── sbert/                   # SBERT embeddings + index IDs
│   ├── factors/                 # ALS/BPR factors + IDs
│   ├── faiss/                   # FAISS index
│   ├── item_metadata.db         # SQLite metadata
│   ├── user_history.db          # SQLite user history
│   └── item_popularity.npy      # Item popularity
├── docs/                        # Báo cáo, sơ đồ, kịch bản
│   ├── Bao_cao_day_du.md/.docx  # Báo cáo đầy đủ
│   ├── Tom_gon_do_an.md/.docx   # Báo cáo tóm gọn
│   ├── Kich_ban_bao_ve.md/.docx # Kịch bản bảo vệ
│   ├── Co_so_nen.md             # Cơ sở lý thuyết
│   └── figures/                 # 6 sơ đồ PNG
├── data/                        # Train/test parquet (từ Kaggle)
├── models/                      # Trọng số model khác
└── requirements.txt
```

---

## 5. Biến môi trường (tùy chọn)

| Biến | Mặc định | Mô tả |
|---|---|---|
| `DATN_ROOT` | thư mục gốc project | Thư mục gốc |
| `DATN_DATA_DIR` | `$DATN_ROOT/data` | Thư mục dữ liệu train/test |
| `DATN_ARTIFACTS_DIR` | `$DATN_ROOT/artifacts` | Thư mục artifact (model, embeddings, DB) |
| `HF_TOKEN` | (unset) | Hugging Face token (nếu repo private) |
| `TMPDIR` | `/tmp` | Thư mục tạm (nên set `/home/clyde/DATN/.scratch` trên máy này) |

**Ví dụ chạy với đường dẫn tùy chỉnh:**
```bash
DATN_ARTIFACTS_DIR=/mnt/data/datn_artifacts python scripts/run_demo.py --customer-id 100
```

---

## 6. Khắc phục sự cố

| Sự cố | Nguyên nhân | Cách xử lý |
|---|---|---|
| `ModuleNotFoundError: No module named 'torch'` | Chưa cài torch | `pip install torch` |
| `No space left on device` | Disk đầy | Dùng `TMPDIR=/home/clyde/DATN/.scratch` |
| `FileNotFoundError: artifacts/...` | Chưa tải artifact | Chạy `python scripts/download_demo_artifacts.py` |
| `Pipeline not available` | Thiếu factor/embedding | Kiểm tra `artifacts/factors/`, `artifacts/sbert/` có file `.npy` |
| Tải HF chậm | Mạng chậm | Dùng `--token` hoặc tải thủ công từ web |
| `CUDA out of memory` | GPU không đủ VRAM | Chạy CPU: `CUDA_VISIBLE_DEVICES="" python ...` |
| Port 8000 đang dùng | Process khác | Đổi port: `--port 8001` |

---

## 7. Liên kết

| Nguồn | URL |
|---|---|
| GitHub | https://github.com/TnClync/DATN |
| Hugging Face (artifact) | https://huggingface.co/datasets/Clync/datn-deep-hybrid-recommender-data |
| Kaggle (kernels) | https://www.kaggle.com/minhteonguyn |
| Báo cáo đầy đủ | `docs/Bao_cao_day_du.docx` |
| Báo cáo tóm tắt | `docs/Tom_gon_do_an.docx` |
| Kịch bản bảo vệ | `docs/Kich_ban_bao_ve.docx` |
| Cơ sở lý thuyết | `docs/Co_so_nen.md` |

---
*Hướng dẫn này áp dụng cho phiên bản mới nhất trên `main` branch. Cập nhật: 18/11/2026.*