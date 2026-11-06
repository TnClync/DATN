# Deep Hybrid Recommender System

Hệ thống gợi ý lai sâu (Deep Hybrid) cho dữ liệu Amazon Reviews (~45.6M tương tác implicit, 22M users × 3.7M items). Train trên Kaggle (GPU/CPU), deploy local 8GB RAM.

## Kiến trúc

- **Baseline**: Late Fusion (ALS/BPR từ `implicit` + SBERT + FAISS)
- **1. Two-Stage Retrieval + Cross-Encoder Reranker** (bge-reranker-v2-m3)
- **2. Light Gated Late Fusion** (học alpha động)
- **3. Deep Early Fusion** (Gated / Cross Network nhỏ)

## Cấu trúc

```
src/
  config.py            # đường dẫn & hằng số
  data.py              # load parquet, build CSR matrix
  metrics.py           # HR@K, NDCG@K, MRR
  models/              # cf, sbert, late_fusion, two_stage, gated_fusion, early_fusion
  serving/pipeline.py  # inference 8GB-friendly
  api/main.py          # FastAPI
scripts/
  01_train_cf.py
  02_build_sbert.py
  03_baseline_late_fusion_eval.py
  04_two_stage_rerank_eval.py
  05_train_gated_fusion.py
  06_train_deep_early_fusion.py
app/frontend/          # React UI (Vite)
tests/
```

## Chạy trên Kaggle

```bash
# 1. Train CF
python scripts/01_train_cf.py --model als --factors 64 --iters 15
python scripts/01_train_cf.py --model bpr --factors 64 --iters 30

# 2. Build SBERT + FAISS
python scripts/02_build_sbert.py

# 3-6. Eval / train fusion
python scripts/03_baseline_late_fusion_eval.py --users 500
python scripts/04_two_stage_rerank_eval.py --users 200
python scripts/05_train_gated_fusion.py --epochs 5
python scripts/06_train_deep_early_fusion.py --epochs 3
```

## Deploy local (8GB)

```bash
pip install -r requirements.txt  # python 3.11
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
# frontend: cd app/frontend && npm run build
```

Mở http://localhost:8000

## Metrics

HR@5/10/20, NDCG@5/10/20, MRR (src/metrics.py).

## Tối ưu bộ nhớ

- Factors float16 + `mmap_mode="r"` cho user factors.
- Two-stage retrieval (không score toàn bộ 3.7M items).
- Cross-encoder chỉ rerank top candidates.
