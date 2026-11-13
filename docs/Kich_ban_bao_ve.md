# KỊCH BẢN BẢO VỆ ĐỒ ÁN TỐT NGHIỆP

## Deep Hybrid Recommender cho hệ thống khuyến nghị sản phẩm Amazon

> Tài liệu dùng cho buổi bảo vệ trước hội đồng (12–15 phút trình bày + 10–15 phút hỏi đáp).  
> Số liệu lấy từ các kernel Kaggle đã COMPLETE; mọi con số đều truy vết được về `summary.json`.

---

# 0. THÔNG ĐIỆP CỐT LÕI (nếu chỉ nhớ 3 câu)

1. **Vấn đề:** ALS — baseline CF mạnh nhất — chỉ đạt Recall@10 = 0.6752 trên catalog 2,46 triệu sản phẩm; content-only (SBERT) chỉ 0.6096. Cả hai đều chưa đủ.
2. **Giải pháp:** hybrid **học cách kết hợp** (Early Fusion V3: DCN cross + deep MLP + listwise loss) thay vì cộng trọng số thủ công; kết quả **Recall@10 = 0.9004**, tức **+33,3% tương đối** so với ALS.
3. **Giá trị học thuật:** chứng minh bằng thực nghiệm rằng nút thắt của content branch là **chất lượng dữ liệu metadata** (rebuild v4 đưa SBERT từ 0.4362 lên 0.6096), và hybrid chỉ thắng khi **neo CF + chuẩn hoá per-user + listwise loss** — không phải "deep hơn là tốt hơn".

---

# 1. CẤU TRÚC TRÌNH BÀY (8 slide, 12 phút)

| Slide | Nội dung | Thời lượng |
|---|---|---|
| 1 | Đặt vấn đề: bài toán, baseline, khoảng trống | 1,5 phút |
| 2 | Dữ liệu & protocol đánh giá (1+99, leak-free, tune val) | 1,5 phút |
| 3 | Kiến trúc: Early Fusion V3, Late Fusion V3, RGH | 3 phút |
| 4 | Kết quả chính: bảng Recall/NDCG/MRR | 2,5 phút |
| 5 | Recall@K và ý nghĩa triển khai | 1 phút |
| 6 | Ablation & phân tích lỗi | 1,5 phút |
| 7 | Giới hạn (trung thực) | 0,5 phút |
| 8 | Kết luận & hướng phát triển | 0,5 phút |

---

# 2. SCRIPT LỜI NÓI TỪNG SLIDE

## Slide 1 — Đặt vấn đề

> "Kính thưa hội đồng. Đề tài của em là xây dựng hệ khuyến nghị lai ghép cho sản phẩm Amazon.  
> Bài toán: với mỗi user, xếp hạng sản phẩm phù hợp từ catalog **2.464.371 sản phẩm**.  
> Baseline mạnh nhất là ALS — lọc cộng tác trên implicit feedback — đạt **Recall@10 = 0.6752**.  
> Nhánh nội dung SBERT ban đầu chỉ đạt **0.4362** vì cơ sở dữ liệu metadata bị thiếu trường.  
> Câu hỏi nghiên cứu: **liệu một mô hình lai ghép có học được cách kết hợp CF và content để vượt ALS một cách có ý nghĩa thống kê hay không?**"

## Slide 2 — Dữ liệu & protocol

> "Dữ liệu: `nguynteo/lastone`, lọc positive `rating ≥ 4`, user có tối thiểu 3 tương tác.  
> Metadata được rebuild từ `nguynteo/finaldata` qua asin: **2.464.371 item**, phủ 99,97% product_text và 99,95% title.  
> Protocol đánh giá: **1 positive + 99 negative** lấy từ catalog chung, cùng user/candidate/seed cho mọi mô hình;  
> **tune trên validation, test chỉ chạy một lần** để chống leakage; SBERT user profile chỉ dùng lịch sử train."

## Slide 3 — Kiến trúc

> "Em đề xuất ba hướng, cùng một triết lý: **neo vào CF, học phần dư bằng gate**.  
> **Early Fusion V3** nhận embedding ALS, BPR, SBERT đã chuẩn hoá L2, cộng z-score per-user và đặc trưng popularity/history;  
> qua **3 lớp DCN cross** để học tương tác bậc hữu hạn, rồi MLP 512–256–128; huấn luyện bằng **listwise softmax** trên 1 positive + 40 negative — khớp trực tiếp với bài toán xếp hạng.  
> **Late Fusion V3** là baseline trung gian: học trọng số tuyến tính trên simplex, chọn bằng validation.  
> **RGH** là kiến trúc đề xuất về mặt lý thuyết: score = a_cf·z_cf + a_gcn·z_gcn + a_txt·z_txt + g(h_u,h_i)·Δ;  
> trong đó gate khởi tạo âm để ban đầu gần như chỉ dùng CF, rồi mở dần khi dữ liệu đủ."

## Slide 4 — Kết quả chính

> "Đây là kết quả trên cùng một protocol, N=5000 user:  
> ALS 0.6752; BPR 0.6340; SBERT 0.6096; ensemble ALS+BPR 0.7370;  
> Late Fusion V3 **0.7736**; Early Fusion V3 **0.9004**.  
> So với ALS, Early Fusion tăng **33,3% tương đối**; so với ensemble mạnh nhất, tăng **22,2%**.  
> Chênh lệch 0.2252 có **z ≈ 28**, tức xác suất do nhiễu gần như bằng 0 — không phải may mắn."

## Slide 5 — Recall@K

> "Em báo cáo thêm Recall@5/10/20. Trong protocol 1 positive/user, Recall@K **trùng** HR@K về mặt toán học — đây là hệ quả của thiết kế, không phải trùng lặp số liệu.  
> Với K=20, Early Fusion phủ **95,8%** user; Late Fusion 84,0%; ALS 77,7%.  
> Nếu giao diện hiển thị 20 sản phẩm, mô hình đề xuất đáp ứng gần như toàn bộ nhu cầu tìm kiếm."

## Slide 6 — Ablation

> "Em kiểm tra 6 giả thuyết: bỏ content, bỏ graph, bỏ gate, thay listwise bằng BCE, bỏ z-score, và so với ensemble.  
> Kết quả cho thấy: (i) bỏ content → giảm ít, xác nhận content là bổ trợ; (ii) bỏ listwise → NDCG giảm rõ;  
> (iii) gate khởi tạo âm giúp mô hình không "phá" thứ hạng CF ở giai đoạn đầu; (iv) hybrid > weighted average ở mọi cấu hình."

## Slide 7 — Giới hạn

> "Em xin nêu thẳng ba giới hạn:  
> Một, metric sampled 1+99 **không thay thế** full-catalog — theo Krichene & Rendle 2020, absolute Recall có thể thấp hơn; em đã có kế hoạch full-catalog.  
> Hai, split theo user, **không temporal-strict** — chưa khẳng định được khả năng dự đoán tương lai.  
> Ba, khoảng **6,13% positive out-of-catalog** do cold item — đã loại khỏi catalog đánh giá và ghi rõ."

## Slide 8 — Kết luận

> "Kết luận: mô hình lai ghép có học **vượt baseline CF mạnh nhất** trên cùng protocol, với mức tăng lớn và ổn định.  
> Đóng góp chính: (i) kiến trúc fusion có kiểm soát; (ii) pipeline rebuild metadata chứng minh bằng số; (iii) protocol đánh giá nghiêm, tái lập được.  
> Hướng phát triển: full-catalog evaluation, temporal split, và triển khai two-stage retrieval–rerank."

---

# 3. BẢNG SỐ LIỆU "PHẢI NHỚ"

## 3.1 Protocol 1+99, N=5000 (số liệu chốt cho phần fusion)

| Model | Recall@5 | Recall@10 | Recall@20 | NDCG@10 | MRR |
|---|---:|---:|---:|---:|---:|
| ALS | 0.5740 | 0.6752 | 0.7770 | 0.4925 | 0.4476 |
| BPR | 0.5326 | 0.6340 | 0.7314 | 0.4386 | 0.3899 |
| SBERT (content-only) | 0.4728 | 0.6096 | 0.7348 | 0.3928 | 0.3411 |
| ALS+BPR (50/50) | 0.6532 | 0.7370 | 0.8092 | 0.5499 | 0.4996 |
| Late Fusion V3 | 0.6940 | 0.7736 | 0.8404 | 0.5846 | 0.5327 |
| **Early Fusion V3** | **0.8238** | **0.9004** | **0.9580** | **0.7305** | **0.6819** |

## 3.2 Các mức tăng tương đối (dùng khi bị hỏi "tăng bao nhiêu?")

| So sánh | Recall@10 | Tương đối |
|---|---:|---:|
| Early Fusion vs ALS | +0.2252 | **+33,3%** |
| Early Fusion vs ALS+BPR | +0.1634 | **+22,2%** |
| Early Fusion vs Late Fusion | +0.1268 | **+16,4%** |
| Late Fusion vs ALS | +0.0984 | +14,6% |
| SBERT mới vs SBERT cũ (0.4362) | +0.1734 | **+39,8%** |
| NDCG@10 Early Fusion vs ALS | +0.2380 | +48,3% |
| MRR Early Fusion vs ALS | +0.2343 | +52,3% |

## 3.3 Kiểm định thống kê (nói khi bị hỏi "có chắc không?") — số chốt

Kernel `eval-per-user-datn` chấm lại **6 model trong một lần chạy** trên đúng 5.000 user / cùng 99 negative / cùng seed, và **tái lập đúng từng con số** của các kernel huấn luyện (ALS 0.6752, EF 0.9004...). Sau đó bootstrap 2.000 lần theo user + paired test:

| Model | Recall@10 | CI95 | NDCG@10 | CI95 |
|---|---:|---|---:|---|
| ALS | 0.6752 | [0.6624, 0.6878] | 0.4925 | [0.4815, 0.5040] |
| ALS+BPR | 0.7370 | [0.7250, 0.7484] | 0.5499 | [0.5393, 0.5612] |
| Late Fusion V3 | 0.7736 | [0.7618, 0.7850] | 0.5846 | [0.5739, 0.5952] |
| **Early Fusion V3** | **0.9004** | **[0.8922, 0.9084]** | **0.7305** | **[0.7212, 0.7397]** |

| So sánh | ΔNDCG@10 | Wilcoxon p | McNemar "A đúng, B sai" : "B đúng, A sai" | McNemar p |
|---|---:|---|---|---|
| EF vs ALS | +0.2380 | < 1e-300 | **1.194 : 68** (17,6:1) | 1,3e-266 |
| EF vs ALS+BPR | +0.1806 | 1,7e-282 | 872 : 55 (15,9:1) | 4,5e-190 |
| EF vs Late Fusion | +0.1459 | 2,4e-236 | 681 : 47 (14,5:1) | 4,3e-145 |
| Late Fusion vs ALS | +0.0922 | 8,6e-145 | 726 : 234 | 3,2e-59 |

- EF thắng ALS trên **2.906 user**, thua **377**, hoà **1.717** → lợi thế đến từ **đa số user**, không phải nhóm nhỏ.
- Mở rộng **20.000 user**: EF 0.8998 (CI [0.8957, 0.9038]), thứ hạng 6 model **không đổi**, CI hẹp còn một nửa → không phải hiện tượng mẫu nhỏ.
- Kiểm định đủ nhạy: ALS vs BPR chỉ chênh 5,39 điểm NDCG@10 vẫn cho p = 4,4e-17.

## 3.4 RGH (protocol holdout N=7500) — số chốt v4

| Model | HR@5 | HR@10 | HR@20 | NDCG@10 | MRR |
|---|---:|---:|---:|---:|---:|
| ALS (chỉ user có factor CF, N=7500) | — | 0.7064 | — | 0.5338 | 0.4901 |
| ALS (toàn bộ user, coverage 62,29%) | 0.4004 | 0.4787 | 0.5665 | 0.3520 | 0.3274 |
| BPR | 0.3856 | 0.4667 | 0.5567 | 0.3194 | 0.2888 |
| LightGCN | 0.6580 | 0.7415 | 0.8145 | 0.5707 | 0.5260 |
| SBERT | 0.4697 | 0.6019 | 0.7328 | 0.3928 | 0.3440 |
| **RGH v4 (Proposed)** | **0.7227** | **0.7983** | **0.8681** | **0.6374** | **0.5945** |
| LambdaRank | 0.7244 | 0.8028 | 0.8689 | 0.6399 | 0.5961 |
| Stacked | 0.7232 | 0.8016 | 0.8675 | 0.6372 | 0.5929 |
| RGH v2 (bug mapping) | 0.6583 | 0.7427 | 0.8153 | 0.5705 | 0.5253 |

**Ba điều phải nói đúng về bảng này:**

1. **So sánh công bằng nhất: RGH 0.7983 vs LightGCN 0.7415 = +5,68 điểm** (cùng phủ 100% user). Đây là claim chính cho kiến trúc RGH.
2. **Không được nói "RGH vượt ALS +66,8%"**: dòng ALS 0.4787 là ALS đo trên toàn bộ user của benchmark, trong đó **37,71% user không có factor ALS/BPR** nên nhánh CF trả về 0. Con số 0.7064 chỉ tính trên user có factor. Phải nói rõ điều này trước khi hội đồng hỏi.
3. **RGH v2 (0.7427) ≈ LightGCN là do bug index**, không phải kết luận khoa học: nhánh ALS/BPR/SBERT bị chấm trên không gian index khác nên chỉ còn mức random (0.1176 / 0.0988 / 0.1168). Sau khi sửa bằng `build_pos_map` + `_cf_branch`, RGH lên 0.7983. Kể cả chuyện này cũng là điểm cộng về tính trung thực phương pháp.

**Trọng số RGH học được (bằng chứng gate/neo hoạt động):**

| Tham số | Khởi tạo | Sau 20 epoch |
|---|---:|---:|
| a_cf (neo CF) | 0.60 | **0.735** |
| a_gcn (graph) | 0.30 | **0.499** |
| a_txt (content) | 0.10 | **0.299** |

⇒ content được "mở" gấp 3 lần; graph tăng gần gấp đôi; CF vẫn là neo lớn nhất — đúng thiết kế.

---

# 4. MƯỜI HAI CÂU HỎI PHẢN BIỆN VÀ TRẢ LỜI

**Q1. Vì sao Early Fusion tốt hơn Late Fusion?**  
A: Late Fusion chỉ học trọng số tuyến tính trên 3 nhánh; Early Fusion học **tương tác phi tuyến** qua DCN cross và MLP, đồng thời tối ưu **listwise** đúng với ranking. Bằng chứng: +16,4% Recall@10.

**Q2. Vì sao không dùng full catalog?**  
A: Đúng, sampled 1+99 có thiên lệch (Krichene & Rendle 2020). Em báo cáo trung thực và đã có kế hoạch full-catalog. Tuy vậy, vì mọi mô hình dùng **cùng candidate**, so sánh tương đối vẫn hợp lệ; và chênh lệch lớn gấp ~28 lần sai số.

**Q3. Vì sao SBERT content-only thấp hơn ALS?**  
A: Content chỉ dùng ngữ nghĩa metadata; ALS khai thác hành vi cộng đồng. Content là **bổ trợ**, không thay thế. Sau rebuild metadata, SBERT tăng từ 0.4362 lên 0.6096 (+39,8%) — chứng minh nút thắt là dữ liệu.

**Q4. Vì sao chọn listwise softmax thay vì BCE?**  
A: BCE tối ưu xác suất pointwise, không tối ưu thứ tự; listwise softmax tối ưu trực tiếp phân phối trên candidate — khớp protocol và cải thiện NDCG.

**Q5. Gate trong RGH có thực sự học không?**  
A: Có. Gate khởi tạo bias âm (g ≈ 0.08–0.12) để mô hình bắt đầu gần CF; qua training, trọng số a_cf/a_gcn/a_txt thay đổi và gate mở dần. Ablation bỏ gate cho thấy NDCG giảm.

**Q6. Tại sao RGH gần LightGCN (bản v2)?**  
A: Bản v2 **có bug mapping index**: nhánh ALS/BPR/SBERT được chấm trên không gian compact khác (2.313.272 item / 1.594.207 user) trong khi candidate lấy từ không gian SBERT/graph (2.464.371 item / 2.559.404 user), nên các nhánh đó chỉ còn mức random — RGH thực chất chỉ dùng graph. Sau khi sửa bằng `build_pos_map` + `_cf_branch`, bản v4 đạt **0.7983 vs LightGCN 0.7415 (+5,68 điểm)**. Em giữ cả hai con số trong báo cáo vì đó là bằng chứng cho thấy kiểm tra index là bước bắt buộc trong hệ nhiều không gian.

**Q6b. Vậy RGH có vượt mọi baseline không?**  
A: Không. Trên protocol N=7500, RGH 0.7983 còn **thấp hơn nhẹ** LambdaRank 0.8028 và Stacked 0.8016 (chênh ~0,4 điểm). Em không claim "RGH > tất cả"; đóng góp của RGH là **kiến trúc có kiểm soát + bằng chứng ablation**, còn mô hình triển khai là Early Fusion V3 (Recall@10 = 0.9004).

**Q6c. Vì sao ALS trong benchmark RGH chỉ 0.4787, còn bảng khác ghi 0.7064?**  
A: Vì **coverage**. Catalogue đánh giá có 2.559.404 user nhưng chỉ 1.594.207 user (62,29%) có factor ALS/BPR. Con số 0.7064 tính riêng trên nhóm có factor; 0.4787 tính trên toàn bộ 7.500 user, trong đó 37,71% user nhận điểm CF = hằng số (xếp hạng ngẫu nhiên). Đây là giới hạn coverage của nhánh CF, không phải ALS bị "làm yếu" đi.

**Q7. Có kiểm định ý nghĩa thống kê chưa?**  
A: Có, đã chạy đầy đủ. Bootstrap 2.000 lần theo user: EF Recall@10 = 0.9004, CI95 **[0.8922, 0.9084]**; ALS 0.6752, CI95 [0.6624, 0.6878] — **không chồng lấn**. Paired Wilcoxon trên NDCG@10: p < 1e-300. McNemar exact trên hit@10: EF đúng/ALS sai = **1.194 user**, chiều ngược lại chỉ **68 user** (17,6:1), p = 1,3e-266. EF thắng trên 2.906 user, thua 377. Mở rộng 20.000 user: EF 0.8998 — kết luận không đổi.

**Q8. Chia train/test thế nào, có leakage không?**  
A: Chia theo user; tune hyperparameter trên validation; test chỉ chạy một lần. SBERT user profile chỉ dùng item trong train history — không dùng test.

**Q9. Vì sao chọn 2.464.371 item?**  
A: Đó là catalog có metadata/SBERT phủ; item ngoài catalog bị loại và báo cáo tỷ lệ (~6,13% positive out-of-catalog).

**Q10. Mô hình có chạy được thực tế không?**  
A: Có. Suy luận là dot-product/MLP nhỏ; kiến trúc two-stage retrieval–rerank giữ latency thấp trên catalog lớn.

**Q11. So với SOTA thì sao?**  
A: Em không claim SOTA tuyệt đối; em claim **vượt baseline CF mạnh nhất trên cùng dữ liệu và protocol**, với kiểm định thống kê. Đây là đóng góp có thể tái lập.

**Q12. Điểm mới của đồ án là gì?**  
A: (i) pipeline rebuild metadata + re-encode chứng minh bằng số; (ii) kiến trúc fusion có kiểm soát (neo CF + gate + listwise); (iii) protocol đánh giá nghiêm, có Recall@K và kiểm định.

**Q13. Mô hình đã bao quát được hết dữ liệu chưa?**  
A: Theo từng không gian: catalogue đánh giá chung 2.464.371 item / 2.559.404 user — content branch phủ **gần 100%** (99,97% có product_text, 99,95% có title, chỉ 1.176 item thiếu toàn bộ metadata); CF branch chỉ phủ **93,87% item và 62,29% user**. Vì vậy câu trả lời trung thực là: **content bao quát gần đủ, CF thì không** — và đây chính là lý do hệ cần hybrid chứ không thể chỉ dùng CF.

**Q14. 37,71% user không có factor CF thì mô hình xử lý sao?**  
A: Với các user đó, nhánh CF trả về 0 nên điểm CF là hằng số; Early Fusion V3 vẫn hoạt động vì học trên **nhiều nhánh** (BPR, SBERT, popularity, history) và gate học được cách giảm trọng số nhánh vô dụng. Đây cũng là lý do Early Fusion (0.9004) vượt xa ALS (0.6752) — nó không phụ thuộc vào việc user có factor CF hay không.

**Q15. Đánh giá 5.000 user có đủ tin không?**  
A: Đủ để kết luận các chênh lệch lớn: SE của Recall@10 ≈ 0,42 điểm phần trăm, CI95 ≈ ±0,83 điểm; chênh lệch EF vs ALS là 0,2252 (z ≈ 28). Ngoài ra em chạy thêm **20.000 user** (5.000 test + 15.000 user chưa từng dùng để train/tune) và bootstrap CI95 2.000 lần + paired Wilcoxon/McNemar theo từng user: xem bảng ở mục 8.6 của báo cáo. Điều em **không** claim là absolute recall ngoài production — sampled 1+99 không thay thế full-catalog.

---

# 5. CHECKLIST TRƯỚC HỘI ĐỒNG

- [ ] Slide 4 in đúng bảng 3.1, **không** trộn N=5000 với N=7500.
- [ ] Có slide riêng ghi rõ protocol: 1 positive + 99 negatives, min_interactions = 3, rating ≥ 4, tune val/test 1 lần.
- [x] Chuẩn bị sẵn câu trả lời Q2 (full catalog) và Q6/Q6b/Q6c (RGH v4, không vượt GBDT, ALS 0.4787 do coverage).
- [ ] Nếu bị hỏi sâu: nói rõ **full-catalog / 999-negative vẫn chưa chạy** — đây là việc tiếp theo, không phải đã có.
- [x] Slide có **Recall@10 = 0.9004, CI95 [0.8922, 0.9084]**, Wilcoxon p < 1e-300, McNemar 1.194:68.
- [ ] Mở sẵn `summary.json` trên máy để đối chiếu nếu bị hỏi số.
- [ ] Nêu rõ Recall@K ≡ HR@K trong protocol 1 positive **trước khi** hội đồng hỏi.
- [ ] Không dùng từ "SOTA"; dùng "vượt baseline mạnh nhất trên cùng protocol".
- [x] Hình đã vẽ sẵn trong `docs/figures/` và nhúng trong báo cáo: 6.1 pipeline · 6.2 RGH · 6.3 Late Fusion · 6.4 Early Fusion · 6.5 bản đồ chọn mô hình · 7.1 protocol.
- [ ] Biểu đồ cột Recall@10 (nếu cần slide): dựng từ bảng 3.1, không vẽ tay số khác.
- [ ] Khi bị hỏi "vì sao chọn mô hình này": mở **Hình 6.5** + mục 6.7.2 (bảng quyết định) — trả lời theo 5 tiêu chí, không trả lời "vì điểm cao nhất".

---

# 6. PHỤ LỤC — GIẢI THÍCH NHANH KHI BỊ HỎI KHÓ

## 6.1 Vì sao Recall@K = HR@K?

```
Recall@K = (1/|U|) Σ_u |R_u ∩ TopK(u)| / |R_u|
HR@K     = (1/|U|) Σ_u 1[ |R_u ∩ TopK(u)| > 0 ]
```

Protocol 1 positive ⇒ `|R_u| = 1` ⇒ hai công thức trùng nhau. Khi chuyển sang multi-positive (candidate union), hai metric tách nhau và Recall@K trở thành upper bound cho reranker.

## 6.2 Vì sao dùng z-score per-user?

Điểm ALS, BPR, SBERT có phân phối khác nhau theo từng user (độ dài history, độ phổ biến). Chuẩn hoá theo user đưa các nhánh về cùng thang đo, tránh một nhánh áp đảo chỉ vì scale lớn — đây là điều kiện để gate/trọng số học có ý nghĩa.

## 6.3 Vì sao tune trên validation?

Nếu tune trên test, kết quả bị lạc quan giả tạo. Trong đồ án, trọng số Late Fusion và checkpoint Early Fusion đều chọn trên validation (2000 user), test chỉ đánh giá một lần.

## 6.4 Một câu chốt nếu bị hỏi "kết quả có đáng tin không?"

> "Cùng dữ liệu, cùng protocol, cùng seed, tune trên validation, test một lần; mức tăng 0.2252 Recall@10 lớn hơn sai số chuẩn 28 lần. Nếu hội đồng yêu cầu, em sẵn sàng bổ sung full-catalog evaluation và paired test theo từng user — đây là phần em đã đưa vào kế hoạch phát triển."
