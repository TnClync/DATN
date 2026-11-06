# Rubric: Tiêu chuẩn chọn Model — Đồ án Hybrid Recommender

## 1. Tiêu chí đánh giá tổng quan

| Tiêu chí | Trọng số | Mô tả |
|---|---|---|
| **Accuracy (RMSE/MAE)** | 25% | Độ chính xác dự đoán rating |
| **Ranking (HR@K/NDCG@K/MRR)** | 35% | Khả năng xếp hạng đúng items |
| **Efficiency (inference time)** | 15% | Tốc độ gợi ý trên CPU |
| **Scalability** | 10% | Khả năng scale với data lớn |
| **Interpretability** | 10% | Dễ giải thích, debug |
| **Resource usage** | 5% | Memory, disk footprint |

## 2. Ngưỡng chấp nhận (Threshold)

### 2.1 Rating Prediction
| Metric | Xuất sắc | Tốt | Chấp nhận | Kém |
|---|---|---|---|---|
| RMSE | < 0.3 | 0.3–0.5 | 0.5–1.0 | > 1.0 |
| MAE | < 0.2 | 0.2–0.4 | 0.4–0.8 | > 0.8 |

### 2.2 Ranking (Top-K)
| Metric | Xuất sắc | Tốt | Chấp nhận | Kém |
|---|---|---|---|---|
| HR@5 | > 0.5 | 0.3–0.5 | 0.1–0.3 | < 0.1 |
| HR@10 | > 0.6 | 0.4–0.6 | 0.2–0.4 | < 0.2 |
| HR@20 | > 0.7 | 0.5–0.7 | 0.3–0.5 | < 0.3 |
| NDCG@5 | > 0.4 | 0.2–0.4 | 0.1–0.2 | < 0.1 |
| NDCG@10 | > 0.5 | 0.3–0.5 | 0.15–0.3 | < 0.15 |
| NDCG@20 | > 0.6 | 0.4–0.6 | 0.2–0.4 | < 0.2 |
| MRR | > 0.4 | 0.2–0.4 | 0.1–0.2 | < 0.1 |

### 2.3 Efficiency
| Metric | Xuất sắc | Tốt | Chấp nhận | Kém |
|---|---|---|---|---|
| Inference time (CPU) | < 0.1s | 0.1–0.5s | 0.5–2s | > 2s |
| Model size | < 100MB | 100MB–1GB | 1GB–5GB | > 5GB |
| Load time | < 5s | 5–30s | 30–120s | > 120s |

---

## 3. So sánh các model đã train

### 3.1 Collaborative Filtering (CF)

| Model | RMSE | MAE | HR@10 | NDCG@10 | MRR | Size | Đánh giá |
|---|---|---|---|---|---|---|---|
| **MF-bias** | 0.3727 | 0.2869 | 0.6470 | 0.4368 | 0.3705 | 209MB | ✅ Tốt (rating) |
| **ALS (f=32)** | 3.8359 | 3.8165 | 0.9725 | 0.7710 | 0.7062 | ~1MB | ⚠️ Scale khác |
| **ALS (f=64)** | 3.8331 | 3.8134 | 1.0000 | 0.9252 | 0.8990 | ~1MB | ⚠️ Scale khác |
| **ALS Best** | 1282051 | 2333.93 | 0.1644 | 0.0834 | 0.0590 | 24MB | ❌ Kém |
| **ALS Last** | 4.7383 | 4.7186 | 0.0534 | 0.0367 | 0.0316 | — | ❌ Kém |
| **BPR-MF** | — | — | 0.0184 | 0.0104 | — | lost | ❌ Kém |

### 3.2 Content-Based (CB)

| Model | HR@10 | NDCG@10 | MRR | Size | Đánh giá |
|---|---|---|---|---|---|
| **SBERT Content** | 0.0000 | 0.0000 | 0.0000 | 622MB | ❌ Lỗi index alignment |
| **SBERT + FAISS** | — | — | — | 622MB | ✅ Tìm kiếm nhanh (424K items) |

### 3.3 Hybrid — Linear

| Model | HR@5 | HR@10 | HR@20 | NDCG@5 | NDCG@10 | NDCG@20 | MRR | Đánh giá |
|---|---|---|---|---|---|---|---|---|
| **Popularity** | 0.7316 | 0.6560 | — | 0.6023 | 0.4680 | — | 0.4094 | ✅ Xuất sắc (baseline) |
| **LateFusion (α=0.5)** | — | 0.6515 | — | — | 0.4324 | — | 0.3633 | ✅ Tốt |
| **LateFusion (α=0.7)** | — | 0.6515 | — | — | 0.4321 | — | 0.3631 | ✅ Tốt |
| **EarlyFusion** | — | 0.6470 | — | — | 0.4346 | — | 0.3678 | ✅ Tốt |

### 3.4 Hybrid — Deep

| Model | File | Size | RMSE | MAE | HR@5 | HR@10 | HR@20 | NDCG@5 | NDCG@10 | NDCG@20 | MRR | Đánh giá |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Gated Hybrid** | `gated_hybrid_recommender.pth` | 1.6GB | — | — | 0.2251 | 0.3298 | 0.3927 | 0.1485 | 0.1815 | 0.1974 | 0.1500 | ⚠️ Chấp nhận |
| **DCN-v2** | `dcnv2_state.pt` | 2.7GB | 0.4445 | 0.2536 | 0.1500 | 0.2632 | — | 0.0832 | 0.1193 | — | 0.1018 | ⚠️ Chấp nhận |
| **TwoTower** | `twotower_state.pt` | 2.7GB | 0.4385 | 0.2491 | 0.1026 | 0.1974 | — | 0.0632 | 0.0936 | — | 0.0902 | ⚠️ Chấp nhận |
| **ResidualCascade** | `residual_cascade_state.pt` | 2.8GB | 0.2647* | — | 0.1000 | 0.1632 | — | 0.0655 | 0.0863 | — | 0.0893 | ⚠️ Chấp nhận |
| **AutoInt** | `autoint_state.pt` | 2.7GB | 0.4629 | 0.2455 | 0.1263 | 0.2289 | — | 0.0760 | 0.1087 | — | 0.1005 | ⚠️ Chấp nhận |
| **Deep Fusion MLP** | `deep_fusion_mlp.pt` | 1.1MB | — | — | — | — | — | — | — | — | — | ⚠️ Chưa eval |

*ResidualCascade: train RMSE only

### 3.5 Hybrid — Deep (chưa train, chỉ có eval metrics)

| Model | HR@5 | HR@10 | HR@20 | NDCG@5 | NDCG@10 | NDCG@20 | MRR | Delta vs Baseline |
|---|---|---|---|---|---|---|---|---|
| EarlyFusion (Baseline) | 0.1132 | 0.2026 | — | 0.0739 | 0.1029 | — | 0.1008 | Baseline |
| AutoInt (Self-Attention) | 0.1263 | 0.2289 | — | 0.0760 | 0.1087 | — | 0.1005 | +16% |
| DCN-v2 (Deep & Cross) | 0.1500 | 0.2632 | — | 0.0832 | 0.1193 | — | 0.1018 | +33% |
| NeuMF-Hybrid | 0.1237 | 0.1974 | — | 0.0787 | 0.1022 | — | 0.1008 | -2.6% |
| Two-Tower Contrastive | 0.1026 | 0.1974 | — | 0.0632 | 0.0936 | — | 0.0902 | -2.6% |
| Multi-Affinity Bilinear | 0.0868 | 0.1868 | — | 0.0559 | 0.0876 | — | 0.0829 | -5% |
| Residual Cascade | 0.1000 | 0.1632 | — | 0.0655 | 0.0863 | — | 0.0893 | -17% |
| Late Fusion (α=0.65) | 0.0421 | 0.0895 | — | 0.0255 | 0.0409 | — | 0.0503 | -56% |
| Pure CF (BPR-MF) | 0.0474 | 0.0895 | — | 0.0282 | 0.0413 | — | 0.0500 | — |
| Pure CB (SBERT) | 0.0605 | 0.1132 | — | 0.0409 | 0.0575 | — | 0.0651 | — |

---

## 4. Hybrid Variants (Local eval, SBERT + Gated, 191 users)

| Model | HR@5 | HR@10 | HR@20 | NDCG@5 | NDCG@10 | NDCG@20 | MRR |
|---|---|---|---|---|---|---|---|
| **SBERT-Only** | 0.5026 | 0.5602 | 0.5654 | 0.4575 | 0.4757 | 0.4770 | 0.4541 |
| **LateFusion (0.5)** | 0.5183 | 0.6073 | 0.6859 | 0.4153 | 0.4448 | 0.4645 | 0.4046 |
| **LateFusion (0.7)** | 0.5183 | 0.5864 | 0.6754 | 0.4697 | 0.4906 | 0.5135 | 0.4743 |
| **EarlyFusion** | 0.4974 | 0.6073 | 0.6806 | 0.4082 | 0.4434 | 0.4622 | 0.4041 |
| **Gated-Only** | 0.2251 | 0.3298 | 0.3927 | 0.1485 | 0.1815 | 0.1974 | 0.1500 |

---

## 5. Khuyến nghị theo rubric

### 5.1 Tiêu chí đánh giá

| Tiêu chí | Trọng số | Mô tả |
|---|---|---|
| Accuracy (RMSE/MAE) | 25% | Độ chính xác dự đoán rating |
| Ranking (HR@K/NDCG@K/MRR) | 35% | Khả năng xếp hạng đúng items |
| Efficiency (inference time) | 15% | Tốc độ gợi ý trên CPU |
| Scalability | 10% | Khả năng scale với data lớn |
| Interpretability | 10% | Dễ giải thích, debug |
| Resource usage | 5% | Memory, disk footprint |

### 5.2 Ngưỡng chấp nhận

| Metric | Xuất sắc | Tốt | Chấp nhận | Kém |
|---|---|---|---|---|
| RMSE | < 0.3 | 0.3–0.5 | 0.5–1.0 | > 1.0 |
| MAE | < 0.2 | 0.2–0.4 | 0.4–0.8 | > 0.8 |
| HR@5 | > 0.5 | 0.3–0.5 | 0.1–0.3 | < 0.1 |
| HR@10 | > 0.6 | 0.4–0.6 | 0.2–0.4 | < 0.2 |
| HR@20 | > 0.7 | 0.5–0.7 | 0.3–0.5 | < 0.3 |
| NDCG@5 | > 0.4 | 0.2–0.4 | 0.1–0.2 | < 0.1 |
| NDCG@10 | > 0.5 | 0.3–0.5 | 0.15–0.3 | < 0.15 |
| NDCG@20 | > 0.6 | 0.4–0.6 | 0.2–0.4 | < 0.2 |
| MRR | > 0.4 | 0.2–0.4 | 0.1–0.2 | < 0.1 |

### 5.3 Đánh giá theo rubric

| Model | RMSE | MAE | HR@10 | NDCG@10 | MRR | Size | Đánh giá |
|---|---|---|---|---|---|---|---|
| MF-bias | 0.3727 | 0.2869 | 0.6470 | 0.4368 | 0.3705 | 209MB | ✅ Tốt (rating) |
| ALS (f=32) | 3.8359 | 3.8165 | 0.9725 | 0.7710 | 0.7062 | ~1MB | ⚠️ Scale khác |
| DCN-v2 | 0.4445 | 0.2536 | 0.2632 | 0.1193 | 0.1018 | 2.7GB | ⚠️ Chấp nhận |
| TwoTower | 0.4385 | 0.2491 | 0.1974 | 0.0936 | 0.0902 | 2.7GB | ⚠️ Chấp nhận |
| ResidualCascade | 0.2647* | — | 0.1632 | 0.0863 | 0.0893 | 2.8GB | ⚠️ Chấp nhận |
| AutoInt | 0.4629 | 0.2455 | 0.2289 | 0.1087 | 0.1005 | 2.7GB | ⚠️ Chấp nhận |
| LateFusion (α=0.5) | — | — | 0.6515 | 0.4324 | 0.3633 | — | ✅ Tốt |
| LateFusion (α=0.7) | — | — | 0.6515 | 0.4321 | 0.3631 | — | ✅ Tốt |
| EarlyFusion | — | — | 0.6470 | 0.4346 | 0.3678 | — | ✅ Tốt |
| Popularity | — | — | 0.6560 | 0.4680 | 0.4094 | — | ✅ Xuất sắc (baseline) |
| SBERT-Only | — | — | 0.5602 | 0.4757 | 0.4541 | 622MB | ✅ Tốt (local) |
| SBERT Content | — | — | 0.0000 | 0.0000 | 0.0000 | 622MB | ❌ Lỗi index |
| BPR-MF | — | — | 0.0184 | 0.0104 | — | lost | ❌ Kém |
| LateFusion(0.7) local | — | — | 0.5864 | 0.4906 | 0.4743 | — | ✅ Tốt (local) |

### 5.4 Kết luận theo rubric

| Vai trò | Model | Lý do |
|---|---|---|
| **Baseline hybrid** | EarlyFusion (HR@10=0.6470) | Hybrid baseline hợp lý |
| **Recommended deep** | DCN-v2 (HR@10=0.2632, +33%) | Cross-features phi tuyến, tốt nhất |
| **Best linear hybrid** | LateFusion (HR@10=0.6515) | Blend CF+CB, dễ giải thích |
| **Rating prediction** | MF-bias (RMSE 0.3727) | Tốt nhất cho rating |
| **Content-only** | SBERT + FAISS | Tìm kiếm nhanh trên 424K items |
| **Best local hybrid** | LateFusion(0.7) (HR@10=0.5864, NDCG@10=0.4906, MRR=0.4743) | Eval trên 191 users |

### 5.5 Để cải thiện lên "xuất sắc"

1. **Tăng eval users**: từ 380 → 2000+ để có kết quả ổn định
2. **Tune hyperparameters**: learning rate, embedding dim, dropout, epochs
3. **Hard negative sampling**: thay vì random, dùng items tương tự positive
4. **Feature engineering**: thêm user/item features (category, brand, price)
5. **Ensemble**: kết hợp nhiều model (stacking/blending)
6. **Fix SBERT index alignment**: để SBERT-Content không còn scored 0.0
7. **Train Gated Hybrid + Deep Fusion MLP** với ranking eval
8. **Larger embedding dim**: từ 32 → 64 hoặc 128 cho deep models

---

## 6. Web Inference (FastAPI)

```bash
uv run uvicorn api:app --host 0.0.0.0 --port 8000
```

- Web UI: http://localhost:8000
- Swagger: http://localhost:8000/docs
- Models: gated_hybrid, dcnv2, twotower, residual_cascade, late_fusion
- Customer ID → user_idx mapping (SQLite, 22M users)
- Pre-load: lazy (load on first request)
- FAISS: tìm kiếm trên 424K SBERT items

### API Endpoints
| Method | Path | Mô tả |
|---|---|---|
| GET | `/` | Trang chủ (HTML) |
| GET | `/health` | Health check |
| GET | `/models` | List available models |
| POST | `/recommend` | Gợi ý (customer_id, model, top_k, alpha) |
| GET | `/compare` | Bảng so sánh 14 model |
| GET | `/customer/{id}` | Thông tin customer

### CLI Demo
```bash
uv run python scripts/demo_inference.py --user_idx 3 --top_k 10
```

### File JSON
- `docs/results/all_trained_models_eval.json` — 19 models với đầy đủ metrics
- `docs/results/all_models_comparison.json` — so sánh tổng hợp
- `docs/results/variant_comparison.json` — Local eval (5 variants, HR@5/10/20, NDCG@5/10/20, MRR)
- `docs/results/latest_eval_summary.json` — Tổng hợp eval mới nhất

---

## 7. Cấu trúc thư mục

```
DATN/
├── api.py                          # FastAPI backend (primary web interface)
├── src/models/                     # Modular model classes
│   ├── base.py                     # BaseRecommender interface
│   ├── gated_hybrid.py             # Gated Hybrid (1.6GB)
│   ├── dcnv2.py                    # DCN-v2 inference
│   ├── twotower.py                 # TwoTower inference
│   ├── residual_cascade.py         # ResidualCascade inference
│   ├── late_fusion.py              # LateFusion (SBERT + FAISS)
│   └── registry.py                 # Model registry
├── scripts/
│   ├── demo_inference.py           # CLI demo
│   ├── eval_all_variants.py        # Eval script
│   ├── generate_report.py          # Report generator
│   └── train_missing_local.py      # Local training
├── models/                         # Trained models
├── docs/                           # Documentation + comparison
├── static/                         # Web frontend (HTML)
└── data/                           # Demo user profiles
```

## 8. Datasets
- `nguynteo/lastone` — interactions (user_idx, item_idx, rating, ...)
- `nguynteo/lastreview` — review content (review_content per item_idx)

## 9. Kaggle kernels
### Thành công (✅)
- `mf-bias-2gpu` — MF-bias (RMSE 0.3727)
- `sbert-content-2gpu-v2` — SBERT (424361 × 384)
- `fusion-eval-2gpu` — Fusion eval (Pop/MF/SBERT/LateFusion/EarlyFusion)
- `gatehybrid` — Gated Hybrid (1.6GB)
- `dcn-v2-hybrid-recommender-gpu-v5-fixed-oom` — DCN-v2 (RMSE 0.4445)
- `twotower-retrieval-gpu-v4-fixed-oom` — TwoTower (RMSE 0.4385)
- `residualcascade-hybrid-recommender-gpu-v4` — ResidualCascade (RMSE 0.2647)
- `autoint-self-attention-hybrid-2gpu` — AutoInt (RMSE 0.4629)
- `als-best-2gpu` — ALS Best (HR@10 0.1644)
- `als-torch-2gpu-v3` — ALS Torch (HR@10 0.9725)
- `bpr-mf-final` — BPR-MF (HR@10 0.0184)

### Chưa thành công (❌)
- `multiaffinity-*` — OOM (kernel died)
- `neuhybrid-sbert-*` — OOM / ERROR
- `hybrid-frozen-features-2gpu` — CANCELLED
- `train-missing-hybrids-2gpu` — Quota exceeded
- `eval-all-models-v3-2gpu` — RUNNING (đang chờ)