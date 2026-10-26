# Hybrid Model Storage & Deployment Guide (Hướng Dẫn Nạp Model Lai Ghép)

Thư mục này được thiết kế để chứa các thành phần và trọng số của **Mô hình Lai ghép (Hybrid Recommender Model)** sau khi bạn huấn luyện xong trên **Kaggle** hoặc **Google Colab**.

---

## 1. Định dạng file hỗ trợ
Hệ thống demo tự động phát hiện các định dạng file sau trong thư mục `models/hybrid/`:

### Trường hợp A: Huấn luyện bằng PyTorch (Deep Hybrid / NCF / BPR + Side Info)
- **Tên file quy ước**: `hybrid_model.pt` hoặc `hybrid_best.pt`
- **Nội dung**: File `torch.save(model.state_dict(), ...)` hoặc full checkpoint gồm `model_state_dict`, `user_mapping`, `item_mapping`.

### Trường hợp B: Huấn luyện mô hình Lai ghép bằng PySpark / Scikit-Learn
- **Spark Pipeline / Parquet Factors**: Thư mục `factors/` hoặc `user_factors.parquet`, `item_factors.parquet`.
- **Pickle / Joblib weights**: `hybrid_model.pkl` hoặc `hybrid_weights.json` (chứa các trọng số tối ưu $\alpha, \beta, \gamma$).

---

## 2. Cách tải model từ Kaggle / Colab về thư mục này

### Nếu train trên Kaggle:
Dùng API Kaggle hoặc script `scripts/kaggle_download.py` để kéo file output về:
```bash
python scripts/kaggle_download.py
```
Hoặc tải trực tiếp từ giao diện Kaggle Notebook -> Đặt file `.pt` vào `models/hybrid/hybrid_model.pt`.

### Nếu train trên Google Colab:
Tải file checkpoint từ Google Drive về máy local, sau đó copy vào:
```bash
cp /duong_dan_file_tai_ve/model.pt /home/clyde/DATN/models/hybrid/hybrid_model.pt
```

---

## 3. Cấu hình nhanh qua `hybrid_config.json`
Bạn có thể điều chỉnh trọng số kết hợp hoặc metadata tại file `hybrid_config.json`:
```json
{
  "model_name": "Hybrid-CF-CB-Recommender",
  "version": "1.0",
  "weights_file": "hybrid_model.pt",
  "cf_weight": 0.6,
  "cb_weight": 0.4,
  "status": "ready"
}
```

---

## 4. Chạy Demo
Sau khi thả file model vào đây, ứng dụng Demo Streamlit ([`app.py`](file:///home/clyde/DATN/app.py)) sẽ tự động phát hiện model đã sẵn sàng và chuyển trạng thái từ **Chế độ Giả lập / Ensemble** sang **Chế độ Mô hình Hybrid Thực tế**!
