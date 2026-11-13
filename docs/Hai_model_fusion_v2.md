# Hai mô hình fusion V2 (so sánh với baseline ALS)

> **Mục tiêu:** có **một early fusion** và **một late fusion** vượt được baseline ALS (HR@10 = 0.6845, NDCG@10 = 0.4882) trên cùng protocol đánh giá (1 positive + 99 random negatives, shared catalog SBERT-covered, leak-free history).
>
> **Trạng thái:** chưa chạy kernel — chỉ có code đã viết + kiểm tra tĩnh; kết quả thực tế sẽ có sau khi chạy trên Kaggle GPU.

---

## 1. Late Fusion V2 — `kaggle/account9_train_late_fusion_v2/`

### Khác bản Late Fusion cũ (0.6133 HR@10, thua ALS)

| Điểm yếu cũ | Sửa |
|---|---|
| Chỉ 2 nhánh ALS + SBERT | Thêm nhánh **BPR** (ensemble ALS+BPR đã đạt 0.7242 HR@10) |
| `min-max` toàn cục làm lệch thang đo | **z-score per-user** trên tập candidate |
| Trọng số cố định/thô (chỉ alpha) | **Trọng số simplex tune trên validation**, test đánh giá một lần |

### Công thức

```text
z_cf  = z(u_als · i_als)          (trên 100 candidates của instance)
z_bpr = z(u_bpr · i_bpr)
z_txt = z(u_txt · i_txt)          u_txt = mean(embedding train history)
score(u,i) = w_als·z_cf + w_bpr·z_bpr + w_txt·z_txt
```

- Grid simplex `step = 0.05`, chọn theo **HR@10(validation)**, tie-break NDCG@10(val).
- Lưới gồm cả điểm `(1.0, 0.0, 0.0)` ⇒ về mặt lý thuyết **không nên tệ hơn ALS trên validation**; trên test, với BPR, kỳ vọng ≥ ALS (do thành quả ensemble cũ).
- Output: `late_fusion_v2_summary.json` (có cả metrics ALS/BPR/SBERT và ALS+BPR 0.5/0.5 để tham chiếu).

### Kỳ vọng

- HR@10 ≥ 0.6845 (ALS), khả năng đạt ~0.71–0.73 (xấp xỉ ensemble ALS+BPR).

---

## 2. Deep Early Fusion V2 — `kaggle/account9_train_early_fusion_v2/`

### Khác bản Deep Early Fusion cũ (0.5700 HR@10, thua ALS)

| Điểm yếu cũ | Sửa |
|---|---|
| Embeddings thô, chưa chuẩn hoá | L2-normalized user/item (ALS, **BPR**, SBERT) + SBERT user profile |
| Chỉ 3 scalars thông tin hóa | Thêm **z_cf, z_bpr, z_txt** + popularity + history length |
| BCELoss pointwise (1 pos + 1 neg) | **Listwise softmax** trên nhóm 1 pos + 40 negatives (cùng phân phối đánh giá) |
| Cross với `sigmoid`, không residual | DCN-V2 cross với `tanh` + **residual cross readout** (`xi.mean(-1)`) |
| Checkpoint tùy tiện | Checkpoint theo **NDCG@10(validation)** + early stop (patience 3) |

### Kiến trúc

```text
Input z(x) = [u_als||i_als||u_bpr||i_bpr||u_txt||i_txt   (L2-norm, 1664-d)
              z_cf, z_bpr, z_txt                          (per-user, 3-d)
              log1p(pop)/log1p(maxpop), log1p(hist)/log1p(maxhist)  (2-d)]   → 1669-d
        ↓
DCN cross (3 layers, tanh, no bias)  +  Deep MLP 1669→512→256→ReLU→128→1
        ↓
score = deep(x) + mean(x_cross)      (residual cross readout)
```

### Training

- Listwise CE: nhóm `[pos, 40 negatives]`, target = 0; 60k instance; batch 64 groups; AdamW lr 1e-3, wd 1e-4, grad-clip 1.0, tối đa 10 epochs, early stop theo val NDCG@10.
- Output: `early_fusion_v2.pt` + `early_fusion_v2_summary.json`.

### Rủi ro / trung thực

- Không thể cam kết chắc 100% vượt ALS trước khi chạy; nhưng định tuyến (z_cf/z_bpr/z_txt + listwise) khiến model có thể tái hiện và cải thiện ALS. Nếu không vượt trên test → báo thật, điều chỉnh `a_cf` anchor hoặc chuyển sang RGH (`account8_train_rgh`).

---

## 3. Quy trình chấp nhận

1. Chạy `account9_train_late_fusion_v2` (không cần GPU mạnh, chạy nhanh).
2. Chạy `account9_train_early_fusion_v2` (Kaggle GPU).
3. So sánh trên **cùng eval_instances seed 42, TEST_N=5000, VAL=2000**.
4. Cả hai phải ≥ baseline ALS trên ít nhất HR@10 hoặc NDCG@10 (chấp nhận được nếu metric còn lại không kém đáng kể).
5. Nếu đạt → cập nhật bảng kết quả báo cáo; nếu một model không đạt → điều chỉnh và chạy lại, **không sửa protocol để làm số đẹp**.

## 4. Các file liên quan

- `kaggle/account9_train_late_fusion_v2/train_late_fusion_v2.py` (+ `.ipynb`, `kernel-metadata.json`)
- `kaggle/account9_train_early_fusion_v2/train_early_fusion_v2.py` (+ `.ipynb`, `kernel-metadata.json`)
- Notebook được sinh bằng: `python3 kaggle/make_kernel_notebooks.py account9_train_late_fusion_v2 account9_train_early_fusion_v2`

## 5. Lưu ý chạy Kaggle

- Cả hai kernel cần kernel-sources: `nguynteo/train-sbert-metadata-reviewer-datn` (SBERT embeddings) và `nguynteo/cf-als-bpr-implicit-datn` (ALS + BPR factors) — cùng cách với các kernel account6 khác.
- Không cần chạy RGH trước; hai kernel này độc lập.
- Chạy xong: mở `early_fusion_v2_summary.json` / `late_fusion_v2_summary.json` để lấy kết quả test.
