# Đồ Án Tốt Nghiệp: Xây Dựng Mô Hình Lai Ghép Cho Hệ Thống Khuyến Nghị Sản Phẩm (Amazon)

Hệ thống khuyến nghị sản phẩm lai ghép (Hybrid Recommender System) kết hợp:

1. **Collaborative Filtering (CF)**: điểm phổ biến của sản phẩm từ lịch sử tương tác
   (có thể nâng cấp lên ALS / BPR-MF khi có latent factors).
2. **Content-Based Filtering (CB)**: TF-IDF trên mô tả, thương hiệu, ngành hàng.
3. **Hybrid Fusion**: `hybrid_score = α·CF + (1−α)·CB`, xử lý cold-start và
   tăng độ chính xác Top-K.

---

## 📂 Cấu Trúc Thư Mục

```text
DATN/
├── app.py                          # Web demo Streamlit (tự chạy khi nhập user)
├── main.py                         # CLI demo nhanh
├── config.py                       # Cấu hình đường dẫn & tham số
├── pyproject.toml                  # Dependencies (uv / pip)
├── .gitignore
│
├── data/
│   └── demo_user_profile.json      # 5 hồ sơ người dùng mẫu
│
├── docs/                           # Báo cáo & theo dõi tiến độ
│   ├── Đồ án.docx
│   └── Đồ án progress.xlsx
│
├── models/
│   ├── als/                        # ALS data: mappings, train/test/validation
│   ├── bpr_mf/                     # BPR-MF checkpoint & dữ liệu train
│   ├── content_based/              # TF-IDF model + item vectors
│   ├── hybrid/                     # Cấu hình hybrid (hybrid_config.json)
│   └── serving/                    # ⚡ Cache tối ưu RAM (sinh ra, không commit)
│
├── outputs/                        # Kết quả demo / inference
│   ├── jev_electronics_recommendations.csv
│   ├── jev_scored_products.csv
│   └── bpr_mf/                     # Top-K xuất từ BPR-MF
│
├── scripts/
│   ├── build_serving_cache.py      # ⭐ Build cache 1 lần cho app
│   ├── kaggle_download.py          # Tải output từ Kaggle
│   ├── rating_distribution.py      # EDA phân bố đánh giá
│   └── test_content_similar.py     # Kiểm tra cosine similarity TF-IDF
│
└── src/
    ├── __init__.py
    ├── spark_session.py            # Khởi tạo Spark Session
    ├── recommender_cf.py           # CF engine (ALS qua Spark)
    ├── split_interaction.py        # Chia train/validation/test
    ├── mapping.py                  # Mapping reviewerID / ASIN → index
    ├── train.py                    # Train ALS
    ├── Inference.py                # Inference qua JevOS
    ├── MostPopRec.py               # Baseline popularity
    └── models/
        ├── __init__.py
        ├── hybrid_recommender.py   # ⭐ API gợi ý lai ghép
        ├── serving_store.py        # ⭐ Cache + truy vấn tiết kiệm RAM
        └── train_content_tfidf.py  # Train TF-IDF content-based
```

> `models/als`, `models/content_based`, `models/bpr_mf` và `models/serving`
> được đưa vào `.gitignore` vì dung lượng lớn.

---

## 🚀 Hướng Dẫn Chạy

### 0. Cài dependencies

```bash
uv sync
# hoặc
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Các thư viện chính (khai báo trong `pyproject.toml`):

- `numpy`, `pandas`, `pyarrow` — xử lý dữ liệu
- `pyspark` — train ALS / TF-IDF
- `torch` — BPR-MF & Hybrid
- `streamlit` — web demo
- `requests` — gọi JevOS (Inference)
- `kaggle` — tải output từ Kaggle

### 1. Build serving cache (chỉ một lần)

Cache giúp app mở nhanh và không ngốn RAM. Cần chạy lại khi dữ liệu model thay đổi.

```bash
python scripts/build_serving_cache.py
# build lại từ đầu:
python scripts/build_serving_cache.py --force
```

Kết quả nằm trong `models/serving/` (khoảng 3.5GB, chủ yếu là TF-IDF
memory-map; metadata dùng SQLite để tra cứu Top-K mà không nạp vào RAM).

| File | Vai trò | Cách dùng |
|---|---|---|
| `catalog_index.npz` | popularity, domain, item norm | nạp vào RAM (~88MB) |
| `catalog_meta.sqlite` | asin/title/brand/category/price | chỉ query Top-K |
| `tfidf_*.{f32,i32}` + `tfidf_indptr.npy` | TF-IDF CSR | memory-map |
| `users_sorted.s20` + `users_idx.i32` | reviewerID → user_idx | binary search trên memmap |
| `seen_indptr.npy` + `seen_items.i32` | item đã tương tác | memory-map |


### 2. Chạy web demo

```bash
streamlit run app.py
```

Mở `http://localhost:8501`.

**Điểm mới:** không cần bấm nút “Tạo gợi ý”. Khi bạn chọn tài khoản mẫu hoặc
nhập `reviewerID` (rồi nhấn Enter), hệ thống **tự động chạy gợi ý**. Kết quả
cũng tự cập nhật khi đổi Top-K, ngành hàng hoặc α.

### 3. Chạy CLI

```bash
# Hybrid (mặc định)
python main.py --model hybrid --user A1IPEJDMQNXJYF --top_k 5

# Tùy chỉnh trọng số và ngành hàng
python main.py --model hybrid --user A1IPEJDMQNXJYF --alpha 0.7 --domain Electronics
```

---

## ⚡ Vì Sao App Không Còn Lag?

Bản cũ mỗi lần mở app đều:

- đọc **3.7M item vectors** (1.2GB Parquet) vào pandas,
- đọc **45.6M interactions** (260MB Parquet) vào pandas và tạo `set` cho từng user,
- chuyển từng vector TF-IDF 200.000 chiều thành dense array bằng vòng lặp Python.

Trên laptop 8GB RAM, cách đó ngốn nhiều GB và gây swap → treo máy.

Bản mới:

- **Lazy load**: chỉ nạp model sau khi có `reviewerID`.
- **Serving cache**: TF-IDF lưu dạng CSR và **memory-map**; chỉ đọc đúng vài
  trang cần thiết.
- **Candidate pool**: chỉ chấm Content-Based cho ~500 ứng viên phổ biến nhất
  thay vì toàn bộ 3.7M sản phẩm.
- **Tra cứu user**: binary search trên mảng `reviewerID` đã sort (log N), không
  quét file 226MB mỗi lần.
- **Cache kết quả** của Streamlit: đổi tham số nhỏ không tính lại từ đầu.

---

## 🎯 Quy Trình Train Trên Kaggle / Colab

1. Train mô hình Hybrid trên Kaggle Notebook (GPU T4) hoặc Colab.
2. Xuất checkpoint / trọng số về máy.
3. Đặt file vào đúng thư mục:
   - Hybrid weights: `models/hybrid/hybrid_model.pt`
   - ALS factors: `models/als/userFactors/`, `models/als/itemFactors/`
4. Nếu dữ liệu thay đổi, build lại cache:
   ```bash
   python scripts/build_serving_cache.py --force
   ```
5. Mở `streamlit run app.py` — hệ thống tự nhận diện và chạy.
