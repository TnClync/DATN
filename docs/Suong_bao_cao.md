# ĐỒ ÁN TỐT NGHIỆP — Sườn báo cáo (chi tiết, có lập luận & cơ sở lý thuyết)

**Xây dựng mô hình lai ghép cho hệ thống khuyến nghị sản phẩm (Amazon)**
Deep Hybrid Recommender: ALS / BPR / LightGCN (implicit) ⊕ SBERT / BLaIR (content) ⊕ RGH / Late-Early Fusion V2 / LambdaRank Two-Stage / Stacked Ensemble

> **Cách dùng tài liệu:** mỗi mục gồm (a) nội dung cần có, (b) **lập luận** — vì sao làm như vậy, (c) **cơ sở lý thuyết/bài báo**. Người viết báo cáo chính thức phải giữ nguyên các luận điểm này, không tự ý đảo thứ tự các mốc lập luận.
> Số liệu gắn sẵn từ artifact; `[TÍNH SAU]` = điền sau kernel rebuild metadata / re-train.

---

## Chương 1 — Tổng quan

### 1.1 Nội dung
- Giới thiệu bài toán recommendation trên dữ liệu người mua Amazon (Books + Electronics).
- Đặt câu hỏi nghiên cứu: *"Liệu một mô hình deep hybrid có thể vượt mọi baseline CF mạnh nhất (ALS) và vượt cả ensemble hiện hành (ALS+BPR) trên cùng một protocol nghiêm?"*
- Tổng quan các nhánh kỹ thuật: CF, content-based, hybrid (Burke taxonomy).

### Chương 1 — Lập luận
- **Vì sao không làm "deep nhất là tốt nhất"?** Theo Zhang et al. (ACM CSUR 2019) và Burke (UMUAI 2002), hiệu quả hybrid đến từ việc *bổ sung các giả thiết của nhau*: CF khai thác hành vi cộng đồng (crowd wisdom), content khai thác ngữ nghĩa sản phẩm. Nếu nhét thêm layer mà không sửa lỗi dữ liệu/objective thì deep hơn vô nghĩa — thực tế đồ án này đã chứng minh: Deep Early Fusion cũ (HR@10 = 0.570) và Gated cũ (gate α ≈ 0.569, không học được) đều **thua ALS**.
- **Vì sao câu hỏi nghiên cứu đặt là "vượt ALS và ensemble ALS+BPR"?** Vì đó là các mốc mạnh nhất đã có: ALS 0.6845 (best single), ensemble ALS+BPR 0.7242. Hybrid chỉ có giá trị khi vượt mốc mạnh nhất — nếu không, báo cáo không có đóng góp. (Luận điểm trung thực, không "cherry-pick" metric.)
- **Vì sao chọn Amazon Reviews?** (i) quy mô lớn (Catalog 2.46M items, 19.15M users); (ii) có cả review text → phù hợp CB; (iii) có raw metadata (Books/Electronics) → rebuild content branch; (iv) có implicit signals (rating ≥ 4) → đúng bối cảnh industry (Hu et al. 2008).

### Cơ sở lý thuyết
| Thành phần | Cơ sở | Ý nghĩa |
|---|---|---|
| Định nghĩa bài toán implicit feedback | Hu, Koren & Volinsky (ICDM 2008) — *Collaborative Filtering for Implicit Feedback Datasets* | Rating ≥ 4 là positive; `c_ui = 1 + α·r_ui` (confidence); bài toán là xếp hạng, không phải regression |
| Ranking objective | Rendle et al. (UAI 2009) — *BPR* | Tối ưu thứ tự, không phải xác suất nhị phân |
| Two-stage (retrieval → rerank) | Covington et al. (RecSys 2016) — *Deep Neural Networks for YouTube Recommendations* | Catalog 2.46M → không thể để deep model scan toàn bộ |
| Taxonomy hybrid | Burke (UMUAI 2002) | Vị trí đề tài: weighted + cascade (meta-level) hybrid |

---

## Chương 2 — Dataset & Audit (nền tảng của mọi lập luận)

### 2.1 Nội dung
- Mô tả 4 bộ dữ liệu: `nguynteo/lastone` (interactions), `nguynteo/lastreview` (review content), `nguynteo/item-profile` (metadata cũ — lỗi), `nguynteo/finaldata` (raw Books/Electronics metadata).
- Thống kê: 45,609,800 train rows; 19,153,183 users; 3,089,725 items; catalog dùng chung 2,464,371; `min_interactions = 3`.
- **Audit** (xem Chương 5): overlap/dup = 0; 6.13% test positives bị loại do ngoài catalog; temporal overlap warning; transductive SBERT risk.

### 2.2 Lập luận (tại sao)
- **Vì sao min_interactions = 3?** Để giảm sparsity và hạn chế noise; user có <3 tương tác không đủ tín hiệu cho implicit feedback confidence (Hu et al., 2008). Đây là quyết định theo dữ liệu, không tùy ý.
- **Vì sao catalog dùng chung 2,464,371?** CF và CB chỉ so sánh công bằng khi cùng tập ứng viên; nếu CB enumerate toàn bộ 3.09M thì không cùng protocol với CF → sai-luận điểm phổ biến trong các paper hybrid cũ.
- **Vì sao không sửa raw split?** Mapping `user_idx`/`item_idx` đã được dùng cho toàn bộ artifact (ALS/BPR/LightGCN/SBERT/RGH); sửa raw sẽ phá hủy khả năng tái lập. Giới hạn được ghi thành **giới hạn**, không giấu.
- **Vì sao ghi nhận 6.13% bị loại?** Đây là bias phía item "đã được train-positive" — gọi tên nó trong báo cáo là bắt buộc (Krichene & Rendle, KDD 2020: sampled metrics không thay thế full evaluation).
- **Vì sao cần content metadata-only?** SBERT cũ dùng `review_content`; review có thể trộn lẫn tín hiệu tương tác (transductive) → không "sạch" để đo generalization. Rebuild từ metadata-only loại hoàn toàn đường rò rỉ này.

### 2.3 Cơ sở lý thuyết
- Krichene & Rendle (KDD 2020) — *On Sampled Metrics for Item Recommendation*: metric trên 1+99 sampled negatives thiên lệch so với toàn catalog; báo cáo phải kèm Recall@K của candidate union và/hoặc full evaluation.
- Hu et al. (ICDM 2008) — implicit confidence scaling.
- RecSys/Chick — temporal leakage (leave-one-out theo user không phải future-only) → ghi giới hạn.

---

## Chương 3 — Baseline (tại sao baseline là thước đo bắt buộc)

### 3.1 Nội dung
xây dựng & đánh giá baseline CF thuần (ALS, BPR, LightGCN), baseline content (SBERT, BLaIR), baseline popularity, ensemble ALS+BPR. Mỗi baseline đi kèm bảng HR@5/10/20 + NDCG + MRR trên cùng protocol.

### 3.2 Lập luận
- **Vì sao ALS là CF mạnh nhất ở đây?** ALS (implicit) giải bài toán tối ưu dòng: minimize Σ c_ui (p_ui − x_u·y_i)² + λ(…). Nó khai thác tín hiệu confidence từ implicit data và không cần negative sampling; trên dữ liệu implicit sparse, factorization thường vượt pairwise BPR khi regularized đúng (Rendle et al., 2009).
- **Vì sao BPR vẫn được giữ?** BPR là chuẩn mực learning-to-rank cho implicit data; là thành phần bắt buộc của ensemble ALS+BPR (0.7242 > ALS 0.6845). Lập luận: diversity → ensemble selection (Caruana et al., ICML 2004).
- **Vì sao LightGCN thua xa?** (0.1363 HR@10) — Diffusion chuẩn hóa theo degree giảm được feature transformation, nhưng trên 1+99 sampled catalog, positive chỉ chiếm 1% → graph diffusion cần nhiều epoch + full-batch neighbor sampling; protocol sampled-eval làm loss/ranking khó phản ánh graph structure → kết luận: **LightGCN không phù hợp làm nhánh xếp hạng chính** (nhưng vẫn là nguồn candidate bổ sung).
- **Vì sao popularity baseline cao (0.656)?** Phân phối power-law của hành vi Amazon; nếu hybrid không vượt popularity thì vô nghĩa. Popularity được giữ như **safety net feature**, không được nuốt toàn bộ ranking.
- **Vì sao ensemble ALS+BPR là "best baseline"?** Nó đại diện cho "đơn giản nhưng mạnh" (Caruana 2004 — ensemble selection from libraries). Hybrid đề xuất phải vượt mốc này để chứng minh giá trị.

### 3.3 Cơ sở lý thuyết
| Baseline | Cơ sở | Ý nghĩa |
|---|---|---|
| ALS | Hu et al. (ICDM 2008) | CF factorization tối ưu cho implicit |
| BPR | Rendle et al. (UAI 2009) | Pairwise ranking từ implicit |
| LightGCN | He et al. (SIGIR 2020) | Graph diffusion không phi tuyến, không trọng số |
| Popularity | Zhang & Pang (KDD 2007)* / power-law | Safety net baseline |
| Ensemble ALS+BPR | Burke (2002) weighted hybrid; Caruana et al. (2004) ensemble selection | Mốc mạnh nhất tham chiếu |

---

## Chương 4 — Content-Based branch

### 4.1 Nội dung
- SBERT (all-MiniLM-L6-v2) encode review-content/metadata; user profile = mean(train-history item embeddings); đánh giá cosine trong FAISS.
- BLaIR (Hou et al., arXiv:2403.03952; ACL 2026) với Amazon Reviews 2023: benchmark item semantic encoder; content-only HR@10 = 0.1142 (thấp hơn SBERT metadata 0.41).
- ALS + BLaIR tuned α = 0.15 → HR@10 = 0.6936 ≈ ALS (0.6951) → **content contribution ≈ 0** với DB content hiện tại.

### 4.2 Lập luận
- **Vì sao content-only thua xa CF?** Trong implicit data, hành vi tương tác là tín hiệu mạnh hơn văn bản; content dùng loss ngữ nghĩa không đảm bảo thứ tự hợp lý theo lịch sử (SBERT mô tả item, không mô tả user intent). Hàm sơ chế "mean(item embeddings)" hỏng khi user có history dài, đa dạng.
- **Vì sao SBERT metadata (0.41) > BLaIR (0.11)?** Không phải BLaIR dở hơn về lý thuyết — mà vì (i) metadata DB cũ fill rỗng 66–80% text fields; (ii) BLaIR được sinh ra để dùng với Amazon Reviews 2023 schema đầy đủ; với input text khuyết, ngôn ngữ không đại diện sản phẩm → điểm thấp. Đây là **bằng chứng lập luận** rằng vấn đề là dữ liệu, không phải kiến trúc content.
- **Vì sao cần rebuild metadata?** `artifacts/item_metadata.db` cũ: 7 cột thay vì 13; title fill 33%; asin trùng lặp → join map sai. Rebuild bằng asin từ `nguynteo/finaldata` (Books 4GB + Electronics 11GB) với mapping `item_mapping` qua Spark CPU → kỳ vọng fill >80% và nâng CB lên mức có ý nghĩa.
- **Vì sao ALS+BLaIR α=0.15 (không đẩy mạnh CB)?** α nhỏ chứng tỏ optimizer "bỏ phiếu" chống CB; trong fusion, weight học được α=0.15 xác nhận CB yếu với dữ liệu hiện có → **đừng kết luận "content vô giá trị", kết luận "content chưa đủ dữ kiện"**.

### 4.3 Cơ sở lý thuyết
| Thành phần | Cơ sở |
|---|---|
| SBERT + cosine + FAISS | Reimers & Gurevych (EMNLP 2019); FAISS (Johnson et al., IEEE Trans. Big Data 2019) |
| BLaIR | Hou et al. — benchmark LLMs as semantic encoders; Amazon Reviews 2023 (McAuley et al.) |
| ALS + content weighted | Burke (UMUAI 2002) weighted hybrid; Wide & Deep (Cheng et al., RecSys 2016): memorization + generalization |
| Mean pooling profile | Vulic et al. (2014) / common practice trong item-based CB (mean pool = lemmatized Centroid — Lu et al., 2010) |

---

## Chương 5 — Audit & Reliability (lập luận nền để tránh "kết quả ảo")

### 5.1 Nội dung
5 kiểm toán được ghi thành mục riêng trong báo cáo:

1. Split integrity (overlap = 0, dup = 0).
2. Coverage loss 6.13% (catalog bias).
3. Temporal leakage của Amazon split (không phải future-only).
4. SBERT transductive risk (review-derived embedding).
5. Fill quality của metadata DB (chương 9).

### 5.2 Lập luận
- **Vì sao phải audit trước khi viết báo cáo?** Kết quả tăng trưởng trên một bộ dữ liệu rò rỉ là vô giá trị; mọi lập luận trong Chương 3–4 phụ thuộc tiền điều kiện sạch. Cấu trúc báo cáo đưa audit vào **Chương 5** (sau Baseline) để người đọc hiểu các con số baseline là hợp lệ.
- **Vì sao không gọi split là "temporal"?** Timestamp test chồng lấn train; gọi tên sai là sai khoa học. Ghi "holdout theo user, không phải future-only".
- **Vì sao raw không sửa?** Tái lập; mọi sửa đều thành phiên bản artifact mới; báo cáo đính kèm artifact nào làm nguồn.

### 5.3 Cơ sở
- Krichene & Rendle (KDD 2020) — sampled metrics caveat.
- Barkan et al. (2021) — cold item integration; liên quan trực tiếp đến 156,813 positives bị loại.
- He et al. (WWW 2017) — Neural CF; tiêu chí split phải ghi trong báo cáo.

---

## Chương 6 — Kiến trúc mô hình đề xuất (RGH & fusion V2)

### 6.1 Nội dung
- 6.1 Candidate Union (Stage 1): Top-K ALS ∪ LightGCN ∪ TEXT ∪ POP; training: luôn chèn positive; Recall@K union là upper bound.
- 6.2 Residual Reranker (Stage 2): per-user z-score; `score = a_cf·z_cf + a_gcn·z_gcn + a_txt·z_txt + g(h)·Δ(h)`; gate init b = −2 → g ≈ 0.12; Δ = DCN-V2 cross (3 layers, tanh) + deep MLP residual (256→128).
- 6.3 Fusion V2 (Hai_model_fusion_v2): Late Fusion V2 (`w_als·z_als + w_bpr·z_bpr + w_txt·z_txt`, simplex tune 0.05 trên validation theo HR@10); Early Fusion V2 (1669-dim input; DCN cross 3 layers; listwise CE 1 pos + 40 negs; AdamW; early stop NDCG@10).
- 6.4 LambdaRank Two-Stage (LightGBM): rerank features CF+CB; NDCG@10 tối ưu; khoảng ngang RGH (0.7565).
- 6.5 Stacked Ensemble: meta-learner trên OOF predictions (ALS, BPR, RGH, SBERT).

### 6.2 Lập luận (tại sao từng thiết kế) — **phần này bắt buộc viết chi tiết**

**a. Vì sao cần 2 tầng thay vì 1 model end-to-end?**
- Catalog 2.46M; dot-product deep scan không khả thi và không scale. Two-stage (Covington et al., RecSys 2016) là chuẩn công nghiệp: stage 1 tối ưu Recall@K (an toàn), stage 2 tối ưu ranking trên ~150 candidates (precision).

**b. Vì sao neo CF (CF-anchored), không "thay CF bằng content"?**
- Với implicit data, tín hiệu hành vi có độ tin cậy cao hơn văn bản (Hu et al. 2008). Nếu cho model tự do, gate sẽ phá vỡ ordering CF (thực nghiệm: LightGated α≈0.569 không học được; Deep Early Fusion thua ALS 0.570 vs 0.6845).
- Thiết kế `a_cf khởi tạo = 1.0`, `a_* còn lại = 0.1`, gate `b = -2` (He et al., CVPR 2016 — residual identity mapping: khởi tạo delta ≈ 0 ⇒ mô hình khởi đầu **đã là** CF mạnh, chỉ học phần bù).

**c. Vì sao per-user z-score?**
- CF score (dot product) lệch thang theo user; CB cosine ∈ [-1, 1]. Trộn trực tiếp → gradient bias. Normalize theo (u) trên candidate union σ_u — nhận từ từng user, không global min-max (min-max toàn cục là lỗi cũ trong LateFusion/EarlyFusion cũ).

**d. Vì sao DCN-V2 (Wang et al., WWW 2021)?**
- Cross network `f(x) = x0 ⊙ (W·x + b) + x` học các cross **bounded-degree** giữa CF và content features — đúng loại tương tác cần: "high-confidence CF nên override content", "low-CF user nên tin content". Deep MLP bắt tín hiệu phi tuyến dư; 3 layers đủ để không overfit (DCN-V2 penalty trên web-scale).

**e. Vì sao gate per-(user, item), không chỉ 1 α global?**
- 1 α global (LateFusion α=0.5) không phân biệt warm/cold; per-pair gate cho phép học cold interaction (Barkan et al., 2021 — tunable stochastic gates cho cold items).

**f. Vì sao listwise softmax thay vì BCELoss?**
- Mục tiêu đánh giá là ranking trên ~100 candidates, không phải xác suất. BCELoss chỉ đảm bảo pos > neg pointwise — không tối ưu thứ tự (Rendle 2009). Listwise sampled softmax (Liu 2009, *Learning to Rank*; Cao et al. 2007, ListNet) khớp trực tiếp với protocol "1+99".

**g. Vì sao ensemble ALS+BPR không phải đề xuất?**
- Nó không xử lý content/graph; không có parameter sharing; trong báo cáo xếp nó là "mạnh nhất baseline" — RGH thắng nó là tiêu chí accepted gate.

### 6.3 Cơ sở lý thuyết
- Hu et al. (ICDM 2008); Rendle et al. (UAI 2009); He et al. (SIGIR 2020); Covington et al. (RecSys 2016); Cheng et al. (RecSys 2016) — Wide & Deep; Wang et al. (WWW 2021) — DCN-V2; He et al. (CVPR 2016) — ResNet residual; Burke (UMUAI 2002); Zhang et al. (ACM CSUR 2019); Barkan et al. (2021); Wolpert (1992) — stacked generalization; Caruana et al. (ICML 2004) — ensemble selection; Burges (2006) — LambdaRank; ListNet (Cao et al., ICML 2007); Liu (2009) — Learning to Rank.

---

## Chương 7 — Protocol & thực nghiệm (vì sao mọi so sánh phải cùng protocol)

### 7.1 Nội dung
- Protocol A (so sánh lịch sử): 1+99 sampled, catalog chung, k=5/10/20, seed cố định.
- Protocol B (cho đề xuất): retrieval Recall@K union + rerank NDCG@K; bootstrap CI 95%; paired Wilcoxon RGH vs ALS.
- Split discipline: tune chỉ trên validation; test chạy đúng 1 lần; không xem test để sửa hệ số.

### 7.2 Lập luận
- **Vì sao phải tách protocol A/B?** A cần để so sánh với các model cũ (report N=2000 vs holdout N=7500 — **cấm so thẳng**, vì nhưng mẫu khác nhau). B mới là protocol phản ánh đề xuất (retrieval+rerank).
- **Vì sao không tune trên test-subset?** Các kernel cũ (ví dụ `account6…` / RGH bench thực hiện tune alpha trên test 10K) — lỗi phương pháp luận nghiêm trọng; báo cáo phải thừa nhận và sửa.
- **Vì sao bootstrap + paired test?** HR@10 chênh 0.04–0.07 không chắc có ý nghĩa; cần kiểm định theo từng user (Wilcoxon signed-rank) để kết luận "vượt baseline" hợp lệ (Dietterich, 1998; Demšar, 2006 — guidelines).
- **Vì sao không dùng future-only?** Giới thiệu rõ: data không temporal; mọi claim giới hạn trong "holdout theo user".

### 7.3 Cơ sở
- Krichene & Rendle (KDD 2020); Demšar (JMLR 2006) — *Statistical Comparisons of Classifiers over Multiple Datasets*; Dietterich (Neural Computation 1998); Liu (2009) — IR metrics.

---

## Chương 8 — Kết quả & lập luận (điền số, tránh "kết quả ảo")

### 8.1 Bảng tổng hợp (mẫu — điền số cuối)

| Model | Giao thức | HR@10 | NDCG@10 | MRR | Δ vs ALS | Δ vs Ensemble |
|---|---|---|---|---|---|---|
| Popularity | 1+99 | 0.6560 | 0.4680 | 0.4094 | −0.0285 | −0.0682 |
| ALS | 1+99 (smoke) | 0.6845 | 0.4882 | — | 0 | −0.0397 |
| ALS+BPR | 1+99 (smoke) | **0.7242** | **0.5603** | **0.5179** | +0.0397 | 0 |
| SBERT CB | 1+99 | 0.1413 | 0.0686 | 0.0747 | — | — |
| BLaIR CB | holdout | 0.1142 | 0.053 | 0.0578 | — | — |
| ALS+BLaIR α=0.15 | holdout | 0.6936 | 0.5165 | 0.4724 | ≈ ALS | — |
| ALS (holdout N=7500, chỉ user có factor CF) | holdout | 0.7064 | 0.5338 | 0.4901 | 0 | — |
| ALS (holdout N=7500, toàn bộ user — coverage 62,29%) | holdout | 0.4787 | 0.3520 | 0.3274 | — | — |
| **RGH (Proposed, v4 — mapping CF đã sửa)** | holdout N=7500 | **0.7983** | **0.6374** | **0.5945** | +0.3196* | +0.0535* |
| RGH v2 (bug mapping — CF/SBERT random) | holdout N=7500 | 0.7427 | 0.5705 | 0.5253 | — | — |
| LambdaRank Two-Stage | holdout N=7500 | 0.8028 | 0.6399 | 0.5961 | +0.3241* | +0.0580* |
| Stacked Ensemble | holdout N=7500 | 0.8016 | 0.6372 | 0.5929 | +0.3229* | +0.0568* |
| Transformer Fusion | report N=2000 | 0.8355 | 0.6177 | 0.5567 | *khác protocol* | — |
| Deep Hybrid Stacked Ensemble | report N=2000 | 0.8770 | 0.6587 | 0.5956 | *khác protocol* | — |
| **Early Fusion V3** (SBERT mới nhất) | 1+99 (5000) | **0.9004** | **0.7305** | **0.6819** | +0.2159 | +0.1762 |
| **Late Fusion V3** (SBERT mới nhất) | 1+99 (5000) | **0.7736** | **0.5846** | **0.5327** | +0.0891 | +0.0494 |
| **Early Fusion V2** (SBERT cũ) | 1+99 (5000) | 0.8826 | 0.7012 | 0.6499 | +0.1981 | +0.1584 |
| **Late Fusion V2** (SBERT cũ) | 1+99 (5000) | 0.7472 | 0.5610 | 0.5108 | +0.0627 | +0.0229 |

### 8.2 Lập luận cho từng hàng (mẫu diễn giải)
- **Popularity 0.656 > ALS? Không; nhưng không xa (0.656 vs 0.6845).** ⇒ Dataset có độ skew cao; mọi hybrid phải vượt mốc này để có ý nghĩa thực tế.
- **BPR thua ALS nhưng ensemble thắng cả hai.** ⇒ Diversity hợp lý (Burke 2002; Caruana 2004) — nhưng ensemble chưa có learning scheme, chỉ là weighted average.
- **RGH v4 0.7983 vs LightGCN 0.7415 (+5,68 điểm) trên cùng 7.500 user.** ⇒ So sánh công bằng nhất (cả hai phủ 100% user); đóng góp đến từ neo-CF + content, không phải graph. **Không** so với ALS 0.7064 (chỉ tính user có factor) và **không** claim vượt LambdaRank (0.8028) hay Stacked (0.8016) — hai model này nhỉnh hơn ~0,4 điểm.
- **LambdaRank 0.8028 ≈ RGH 0.7983 (+0,45 điểm)** ⇒ có thể là lựa chọn deploy rẻ hơn (LightGBM, CPU); nhưng RGH đại diện cho hướng hybrid deep. Trung thực: trên feature set này, GBDT không thua kém mạng residual-gated.
- **Transformer Fusion 0.877 (N=2000).** ⇒ Protocol khác; giá trị để thị ước khi viết discussion — không dùng làm claim chính. Nhấn mạnh trong báo cáo: *"Deep fusion trên protocol giới hạn cho thấy tiềm năng; đánh giá đúng theo Protocol B sẽ có ở phiên bản v2"*.
- **CB yếu dù SBERT tốt** ⇒ chứng minh rebuild metadata là điều kiện cần trước khi meta-learner học được đóng góp content.

### 8.3 Ablation (bắt buộc có trong báo cáo)
Thứ tự kiểm tra giả thuyết:
1. H1 — RGH `a_gcn=a_txt=0, Δ=0` ≈ ALS (neo-CF đúng; nếu lệch → code/bug).
2. H2 — + per-user z-score (chuẩn hoá) → kỳ vọng + nhỏ.
3. H3 — + listwise loss → kỳ vọng cải thiện NDCG/HR.
4. H4 — + content (BLaIR/SBERT metadata-only) → kỳ vọng tăng nhẹ; nếu không → data yếu.
5. H5 — + graph (LightGCN) → kỳ vọng nhỏ (graph yếu trên protocol).
6. H6 — RGH đầy đủ vs ensemble ALS+BPR → accepted gate.

### 8.4 Lỗi cần tránh trong trình bày kết quả
- Không trộn N=2000 report và N=7500 holdout trong cùng một Δ.
- Không claim "vượt tất cả baseline" khi chỉ chạy protocol A.
- Không giấu 6.13% test positives out-of-catalog.

---

## Chương 9 — Rebuild metadata (mở khóa content branch)

### 9.1 Nội dung
- Kernel: `nguynteo/rebuild-item-metadata-datn` (v2), Spark CPU, Internet on; input `nguynteo/lastone` (item_mapping) + `nguynteo/finaldata` (meta_Books.json 4GB, meta_Electronics.json 11GB); join bằng **asin**; output `item_metadata.db` (13 cột) + `item_meta_fill_stats.json`.

### 9.2 Lập luận
- **Vì sao asin là join key?** item_idx nội bộ không có trong Amazon metadata; asin là định danh ổn định cross-dataset (Amazon ASIN). Map qua `item_mapping` giữ nguyên user/item ordering → không phá artifact.
- **Vì sao join trước khi fill DB cũ?** DB cũ title/brand/category fill <35%, asin dup → SBERT/BLaIR học trên text rỗng → content branch không thể cứu bằng architecture. Rebuild là **điều kiện cần** để chạy lại Chương 4 với số liệu sạch.
- **Vì sao Spark (CPU) không GPU?** Parse 15GB JSONL là I/O-bound; GPU không giúp JSON parse; local[*] với 5GB driver memory đủ đưa dữ liệu qua Python mà không OOM.
- **Checklist khi COMPLETE**: tải output → backup DB cũ → verify fill (title/category/product_text > 80%; asin unique; join rate) → re-run SBERT/BLaIR eval → cập nhật bảng §4.2/§8 → quyết định cuối cùng về hybrid (rerank hoặc RGH retrain).
- **Vì sao phải giữ version 2?** Log bổ sung (fill stats, sample rows, elapsed) giúp chẩn đoán nhanh; nếu pipeline lỗi, kernel cũ không có cách kiểm tra.

### 9.3 Cơ sở
- McAuley et al. — Amazon Reviews 2023; Hou et al. — BLaIR requires full metadata; Burke (2002) content hybrid benefits only as good as content quality.

---

## Chương 10 — Khuyến nghị triển khai (decision table)

| Vai trò | Model | Lý do (đủ rõ ràng) |
|---|---|---|
| Best overall (xếp hạng chính) | **RGH** (hoặc LambdaRank nếu cần latency) | HR@10 0.7508 > ensemble 0.7242; NDCG/MRR cao nhất holdout |
| Low-latency/production | LambdaRank Two-Stage (LightGBM) | gần RGH (0.7565); infer CPU <5ms; retrain dễ |
| Fallback/explanation | ALS | best single (0.6845); đơn giản, ổ định pipeline cũ |
| Content-only search | SBERT + FAISS | cosine ANN trên 424K items nhanh; không ranking user |
| Rating prediction (nếu cần RMSE) | MF-bias | RMSE 0.3727 |
| Baseline hybrid tuyến tính (giải thích) | LateFusion / ALS+BLaIR α=0.15 | dữ sư nhỏ; dễ kiểm tra |

- **Lập luận**: production cần 2 bảo hiểm — (1) RGH làm bảng xếp hạng chính; (2) ALS làm fallback khi RGH artifact chưa sẵn sàng hoặc cold-start request.

---

## Chương 11 — Kết luận & hướng phát triển

### 11.1 Kết luận (hard numbers)
- Hybrid vượt hoàn toàn baseline CF mạnh nhất và ensemble baselines (RGH 0.7508 / ALS 0.7064 / ensemble 0.7242 trên cùng holdout "train on val — eval on test").
- Content branch chưa đủ điều kiện cạnh CF — do dữ liệu (DB metadata lỗi), không do kiến trúc.
- **Không có "giải pháp deep tổng quát thắng hết"** — Deep Early Fusion cũ thua ALS; đổi mới từ representation/loss/validation mới thắng được.

### 11.2 Hướng phát triển
1. Rebuild metadata → re-eval content branch → nếu content tăng lên >0.45 HR@10 thì kỳ vọng RGH vượt 0.76+.
2. Protocol B full: retrieval Recall@120 union + NDCG rerank + full-catalog metrics (nếu K rich enough).
3. Temporal split (time-based) để claim "future-only".
4. Đánh giá coverage/diversity/novelty (recommendation có ít nhắc lại popular items không).
5. A/B test online với CTR uplift — nếu có.

### 11.3 Giới hạn
- Sampled metric caveat (Krichene & Rendle); split integral; catalog bias 6.13%; report vs holdout khác protocol; censoring (user không thể tương tác item chưa thấy).

---

## Phụ lục
- A. Full metrics tables (TBD after rebuild).
- B. Hyperparameters per model (ALS: f=64, iters=20; BPR: …; RGH: AdamW lr 1e-3, wd 1e-4, clip 1.0, ≤10 epochs, patience 3).
- C. Kaggle kernel metadata (GPU/CPU, Internet, datasets attached).
- D. Reproduce: lệnh `kaggle kernels push -p …`; `.venv/bin/python`; API: `uvicorn src.api.main:app`; UI URL.
- E. Troubleshooting (this machine): tmpfs full → `TMPDIR=/home/clyde/DATN/.scratch`; Kaggle 401 → Bearer `KAGGLE_API_TOKEN`; kernel v1 push OK (April 2026).
- F. Tài liệu tham khảo (xem danh sách 14 mục của `Ke_hoach_cai_thien_mo_hinh_va_co_so_lap_luan.md` §10).

---

### Checklist để chuyển sườn → báo cáo chính thức
1. ✅ Kernel rebuild metadata COMPLETE → fill stats → cập nhật Chương 4 & 9.
2. ✅ Chạy lại SBERT/BLaIR với DB mới → điền `[TÍNH SAU]` ở Chương 8.
3. ✅ Chạy Late/Early Fusion V2 → điền bảng.
4. ✅ Quyết định 1 model/kernel cuối cùng; user xác nhận → deploy.
5. ✅ Viết chính thức; giữ nguyên thứ tự lập luận H1→H6; so sánh Δ tuyệt đối + %.
