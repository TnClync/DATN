# Đồ án tốt nghiệp: Hệ thống gợi ý lai ghép (Hybrid Recommender System)

## Tổng quan
Hệ thống gợi ý sản phẩm lai ghép kết hợp Collaborative Filtering (CF) và Content-Based (CB) trên dữ liệu Amazon (LastOne + LastReview).

## Models đã train

### Collaborative Filtering (CF)
| Model | File | Metrics |
|---|---|---|
| MF-bias | `models/mf_bias/` | RMSE 0.3727, MAE 0.2869 |
| ALS (PyTorch) | `models/als_torch/` | RMSE 3.8359, HR@10 0.9725 |
| BPR-MF | N/A (lost) | HR@10 0.0184 |

### Content-Based (CB)
| Model | File | Artifact |
|---|---|---|
| SBERT (all-MiniLM-L6-v2) | `models/sbert/` | 424361 items × 384 dim |

### Hybrid (Deep)
| Model | File | Size | Metrics |
|---|---|---|---|
| Gated Hybrid | `models/gated_hybrid/` | 1.6 GB | user/item emb + content gate + MLP |
| DCN-v2 | `models/dcnv2/` | 2.7 GB | RMSE 0.4445, MAE 0.2536, HR@10 0.2632 |
| TwoTower | `models/twotower/` | 2.7 GB | RMSE 0.4385, MAE 0.2491, HR@10 0.1974 |
| ResidualCascade | `models/residual_cascade/` | 2.8 GB | train RMSE 0.2647, HR@10 0.1632 |
| AutoInt | `models/autoint/` | 2.7 GB | RMSE 0.4629, MAE 0.2455, HR@10 0.2289 |
| Deep Fusion MLP | `models/deep_fusion/` | 1.1 MB | Frozen features MLP |

### Hybrid (Linear)
| Model | HR@10 | NDCG@10 | MRR |
|---|---|---|---|
| LateFusion (α=0.5) | 0.6515 | 0.4324 | 0.3633 |
| LateFusion (α=0.7) | 0.6515 | 0.4321 | 0.3631 |
| EarlyFusion | 0.6470 | 0.4346 | 0.3678 |

## Cấu trúc thư mục
```
DATN/
├── api.py                          # FastAPI backend (primary web interface)
├── src/
│   ├── models/                     # Modular model classes (BaseRecommender subclasses)
│   │   ├── baseline (PopularityRecommender)
│   │   └── recommended (ImplicitALSRecommender)
│   └── spark_session.py            # Spark session with temp_dir (laptop-friendly)
├── scripts/
│   ├── demo_inference.py           # CLI demo
│   ├── train_implicit.py           # CPU training (<=4GB RAM, 6 cores)
│   ├── kaggle_train_onefile.py     # Single-file Kaggle GPU training
│   └── push_model_to_kaggle.py     # Push model to Kaggle
├── demo/studio.tsx                 # React/TSX demo template
├── models/                         # Trained models (baseline/, recommended/, ...)
├── docs/                           # Documentation + comparison + report
└── data/                           # Train/test/validation parquet
```

## Baseline vs Recommended
- **Baseline**: `PopularityRecommender` (global popularity, implicit feedback) — fast, CPU-only.
- **Recommended**: `ImplicitALSRecommender` (implicit ALS) — better personalization, CPU/GPU.

Use model name `baseline` or `recommended` in `/recommend` endpoint.

## Chạy
```bash
# API server (web UI tại http://localhost:8000)
uv run uvicorn api:app --host 0.0.0.0 --port 8000

# Tests
uv run pytest tests/ -q

# CLI demo
uv run python scripts/demo_inference.py --user_idx 3 --top_k 10
```

## API Endpoints
| Method | Path | Mô tả |
|---|---|---|
| GET | `/` | Trang chủ (HTML) |
| GET | `/health` | Health check |
| GET | `/models` | List available models |
| POST | `/recommend` | Gợi ý (customer_id, model, top_k, alpha) |
| GET | `/compare` | Bảng so sánh 14 model |
| GET | `/customer/{id}` | Thông tin customer |

## Datasets
- `nguynteo/lastone` — interactions (user_idx, item_idx, rating, ...)
- `nguynteo/lastreview` — review content (review_content per item_idx)

## Báo cáo
- `docs/Đồ án.docx` — Báo cáo đầy đủ
- `docs/model_comparison.md` — So sánh chi tiết các model
- `docs/results/all_trained_models_eval.json` — Metrics JSON