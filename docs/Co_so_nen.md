# CƠ SỞ NỀN — BÀI BÁO VÀ LÝ THUYẾT LIÊN QUAN

> Tài liệu tóm tắt các bài báo và khái niệm lý thuyết làm nền cho đồ án Deep Hybrid Recommender. Nguồn chi tiết: `docs/Bao_cao_day_du.md` (Chương 2, 13) và Phụ lục F.

---

## 1. Collaborative Filtering (CF) cho Implicit Feedback

### 1.1 ALS Implicit — Hu, Koren & Volinsky (ICDM 2008)
- **Ý tưởng:** Phân rã ma trận `R = X·Yᵀ` với confidence `c_ui = 1 + α·r_ui` (α thường = 40). Mục tiêu:
  $$\min_{x_u,y_i} \sum_{u,i} c_{ui}(p_{ui} - x_u·y_i)^2 + \lambda(\|x_u\|^2+\|y_i\|^2)$$
- **Cách giải:** Lặp ALS — cố định `y_i` → giải `x_u` (ridge regression đóng kín), rồi ngược lại.
- **Vì sao ALS mạnh ở đây:** khai thác confidence từ implicit data; không cần negative sampling; ổn định trên dữ liệu sparse.
- **Tham số trong đồ án:** f=64, iters=20, λ=0,1. f=128 tăng HR@10 lên 1,0000 nhưng RMSE 4,7383 → overfitting.
- **Kết quả:** HR@10 = **0,6752** (1+99, N=5.000) — mốc CF mạnh nhất.

### 1.2 BPR — Rendle et al. (UAI 2009)
- **Ý tưởng:** Bayesian Personalized Ranking tối ưu posterior trên pairwise ordering:
  $$\text{BPR-OPT} = \sum_{(u,i,j)\in D_S} \ln \sigma(\hat{x}_{ui}-\hat{x}_{uj}) - \lambda_\Theta\|\Theta\|^2$$
- **Negative sampling:** với mỗi positive (u,i), chọn ngẫu nhiên item j mà u chưa tương tác.
- **Vì sao BPR thua ALS ở đây:** phụ thuộc vào chất lượng negative sampling; với catalog 2,46M, uniform negative quá dễ → model không học được ranking khó. ALS khai thác toàn bộ ma trận confidence → ổn định hơn.
- **Kết quả:** HR@10 = **0,6340**; nhưng ALS+BPR = **0,7370** > cả hai (diversity).

### 1.3 LightGCN — He et al. (SIGIR 2020)
- **Ý tưởng:** Graph convolution đơn giản hóa: bỏ nonlinear và weight transformation:
  $$e_u^{(k+1)} = \sum_{i\in \mathcal{N}_u} \frac{1}{\sqrt{|\mathcal{N}_u||\mathcal{N}_i|}} e_i^{(k)}$$
  Final embedding = mean của các layer.
- **Vì sao LightGCN thua xa (0,1363 HR@10 ở protocol cũ)?** Diffusion chuẩn hóa theo degree; trên protocol 1+99 sampled catalog, positive chỉ 1% → graph diffusion cần nhiều epoch + full-batch; sampled-eval khó phản ánh graph structure.
- **Vai trò trong đồ án:** không phải nhánh xếp hạng chính, nhưng là **nguồn candidate bổ sung** (Stage 1 union) và **đối chứng công bằng** cho RGH (cùng phủ 100% user).
- **Kết quả:** HR@10 = **0,7415** (holdout N=7.500).

---

## 2. Content-Based Filtering

### 2.1 SBERT — Reimers & Gurevych (EMNLP 2019)
- **Ý tưởng:** Sentence-BERT encode văn bản thành vector 384/768-dim bằng Siamese BERT-Networks; user profile = mean(train-history item embeddings); đánh giá cosine trong FAISS.
- **Vì sao SBERT hợp lệ:** cosine nhanh + FAISS ANN; nhưng cần nâng chất lượng nếu content yếu.
- **Vì sao mean pooling:** đơn giản, ổn định; weighted pooling (theo recency) là hướng cải thiện tương lai.
- **Kết quả trong đồ án:** SBERT-only HR@10 = **0,6096** sau rebuild v4 (trước rebuild: 0,4362).

### 2.2 BLaIR — Hou et al. (arXiv:2403.03952)
- **Ý tưởng:** Benchmark LLMs as semantic encoders trên Amazon Reviews 2023: item metadata + language context → semantic representation.
- **Vì sao BLaIR thấp hơn SBERT (0,1142 vs 0,41)?** Không phải BLaIR dở hơn về lý thuyết — mà vì (i) metadata DB cũ fill rỗng 66–80% text fields; (ii) BLaIR sinh ra để dùng với Amazon Reviews 2023 schema đầy đủ; với input text khuyết, ngôn ngữ không đại diện sản phẩm.
- **Kết luận:** Chọn embedding đúng domain quan trọng hơn chọn mô hình "mới hơn".

---

## 3. Hybrid và Deep Learning

### 3.1 Wide & Deep — Cheng et al. (RecSys 2016)
- **Ý tưởng:** Linear "memorization" (CF z-features) + deep "generalization" (residual).
- **Cơ sở cho residual anchor trong RGH:** neo CF, chỉ học phần bù. CF z-features là "memorization" của hành vi; deep residual là "generalization" sang content/graph. Kết hợp cho phép model vừa giữ ordering CF vừa học cross features.

### 3.2 DCN-V2 — Wang et al. (WWW 2021)
- **Ý tưởng:** Cross network `f(x) = x0 ⊙ (W·x + b) + x` học các cross **bounded-degree** giữa CF và content features.
- **Vì sao DCN-V2:** đúng loại tương tác cần — "high-confidence CF nên override content", "low-CF user nên tin content". Deep MLP bắt tín hiệu phi tuyến dư; 3 layers đủ để không overfit (DCN-V2 penalty trên web-scale).

### 3.3 Two-Stage Retrieval — Covington et al. (RecSys 2016)
- **Ý tưởng:** Retrieval union CF/content/graph rồi rerank — giữ latency, recall và quy mô catalog lớn; tránh scan 2,46M items bằng deep model.
- **Vì sao two-stage:** Catalog 2,46M; dot-product deep scan không khả thi và không scale. Stage 1 tối ưu Recall@K (an toàn), stage 2 tối ưu ranking trên ~150 candidates (precision).

### 3.4 Taxonomy Hybrid — Burke (UMUAI 2002)
- **Ý tưởng:** Phân loại hybrid: weighted, switching, mixed, feature combination, cascade, feature augmentation, meta-level.
- **Vai trò trong đồ án:** Mô hình đề xuất thuộc nhóm **weighted + cascade (meta-level) hybrid**; cung cấp khung để đặt vị trí mô hình trong literature.

### 3.5 Sampled Metrics — Krichene & Rendle (KDD 2020)
- **Cảnh báo:** Metric trên 1 positive + 99 sampled negatives **không đại diện đầy đủ full catalog**; cần thêm retrieval Recall và/hoặc full evaluation.
- **Vì sao phải báo cáo thêm Recall@K:** Vì sampled metric thiên lệch; Recall@K của union là upper bound thực sự cho reranker.

### 3.6 Residual Learning — He et al. (CVPR 2016)
- **Ý tưởng:** Identity mapping khởi tạo: delta ≈ 0 ⇒ mô hình khởi đầu **đã là** CF mạnh, chỉ học phần bù.
- **Cơ sở cho gate init** `b = -2` (g ≈ 0,12). **Vì sao residual:** Tránh vanishing gradient; cho phép model "không làm gì" khi content không giúp → an toàn.

### 3.7 Cold Item Gates — Barkan et al. (2021)
- **Ý tưởng:** Tunable stochastic gates cho cold items.
- **Vì sao quan trọng:** Cold item là điểm yếu của CF; gate per-pair cho phép học cold interaction. Liên quan trực tiếp đến ~156.813 positives bị loại khỏi eval do catalog.

---

## 4. Thống kê và Kiểm định

### 4.1 Bootstrap CI & Wilcoxon — Dietterich (1998), Demšar (2006)
- **Bootstrap CI:** Resample user có hoàn lại 2.000 lần, lấy percentile 2,5–97,5. Không giả định phân phối chuẩn.
- **Wilcoxon signed-rank:** Kiểm định ghép cặp phi tham số trên NDCG@10 per-user.
- **McNemar exact:** Kiểm định ghép cặp trên hit/miss nhị phân @10.
- **Vì sao paired design quan trọng:** Mọi model chấm trên cùng user, cùng 1 positive, cùng 99 negative, cùng seed ⇒ so sánh từng cặp trên cùng user, loại bỏ phương sai do "user dễ/khó" — power cao hơn nhiều so sánh hai trung bình độc lập.

---

## 5. Bảng tổng hợp bài báo → đóng góp trong đồ án

| Bài báo | Năm | Ứng dụng trong đồ án | Kết quả liên quan |
|---|---|---|---|
| Hu et al. (ALS) | 2008 | Baseline CF mạnh nhất + neo của RGH | HR@10 = 0,6752 |
| Rendle et al. (BPR) | 2009 | Baseline CF thứ hai + thành phần Late Fusion | HR@10 = 0,6340; ALS+BPR = 0,7370 |
| He et al. (LightGCN) | 2020 | Baseline graph + đối chứng công bằng cho RGH | HR@10 = 0,7415 (holdout) |
| Reimers & Gurevych (SBERT) | 2019 | Nhánh content duy nhất (768-d) | HR@10 = 0,6096 (sau rebuild v4) |
| Hou et al. (BLaIR) | 2024 | Ứng viên content bị loại (domain mismatch) | HR@10 = 0,1142 |
| Cheng et al. (Wide & Deep) | 2016 | Cơ sở residual anchor trong RGH | — |
| Wang et al. (DCN-V2) | 2021 | Cross network trong Early Fusion & RGH | — |
| Covington et al. (Two-stage) | 2016 | Kiến trúc retrieval → rerank | — |
| Burke (Taxonomy hybrid) | 2002 | Phân loại hybrid (weighted + cascade) | — |
| Krichene & Rendle | 2020 | Cảnh báo sampled metric; cần full-catalog | — |
| He et al. (Residual) | 2016 | Gate init b=−2, identity mapping | — |
| Barkan et al. (Cold gates) | 2021 | Gate per-pair cho cold item | — |
| Caruana et al. (Ensemble) | 2004 | Cơ sở Stacked Ensemble | HR@10 = 0,8016 |
| Wolpert (Stacked) | 1992 | Meta-learner trên OOF predictions | — |

---

## 6. Tài liệu tham khảo đầy đủ (23 bài)

1. Hu, Y., Koren, Y., & Volinsky, C. (2008). *Collaborative Filtering for Implicit Feedback Datasets*. ICDM.
2. Rendle, S., Freudenthaler, C., Gantner, Z., & Schmidt-Thieme, L. (2009). *BPR: Bayesian Personalized Ranking from Implicit Feedback*. UAI.
3. He, X., Deng, K., Wang, X., Li, Y., Zhang, Y., & Wang, M. (2020). *LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation*. SIGIR.
4. Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. EMNLP.
5. Hou, Y. et al. (2024). *Bridging Language and Items for Retrieval and Recommendation*. arXiv:2403.03952.
6. Covington, P., Adams, J., & Sargin, E. (2016). *Deep Neural Networks for YouTube Recommendations*. RecSys.
7. Cheng, H.-T. et al. (2016). *Wide & Deep Learning for Recommender Systems*. DLRS/RecSys.
8. Wang, R. et al. (2021). *DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems*. WWW.
9. Burke, R. (2002). *Hybrid Recommender Systems: Survey and Experiments*. User Modeling and User-Adapted Interaction.
10. Krichene, W., & Rendle, S. (2020). *On Sampled Metrics for Item Recommendation*. KDD.
11. Caruana, R., Niculescu-Mizil, A., Crew, G., & Ksikes, A. (2004). *Ensemble Selection from Libraries of Models*. ICML.
12. Wolpert, D. H. (1992). *Stacked Generalization*. Neural Networks.
13. He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition*. CVPR.
14. Burges, C. J. C. et al. (2006). *Learning to Rank with Nonsmooth Cost Functions*. NIPS.
15. Cao, Z. et al. (2007). *Learning to Rank: From Pairwise Approach to Listwise Approach (ListNet)*. ICML.
16. Liu, T.-Y. (2009). *Learning to Rank for Information Retrieval*. Foundations and Trends in IR.
17. Demšar, J. (2006). *Statistical Comparisons of Classifiers over Multiple Data Sets*. JMLR.
18. Dietterich, T. G. (1998). *Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms*. Neural Computation.
19. McNemar, Q. (1947). *Note on the Sampling Error of the Difference Between Correlated Proportions or Percentages*. Psychometrika.
20. Barkan, O. et al. (2021). *CB2CF: A Neural Multiview Content-to-Collaborative Filtering Model for Completely Cold Item Recommendations*. RecSys.
21. Chaudhari, A. et al. (2024). *A Hybrid Recommendation System: A Review*. IEEE Access.
22. Hou, Y. et al. (2024). *Amazon Reviews 2023 Dataset* / mô tả dữ liệu phục vụ thực nghiệm.
23. Johnson, J., Douze, M., & Jégou, H. (2019). *Billion-scale similarity search with GPUs*. IEEE Trans. Big Data.

---
*Nguồn chi tiết: `docs/Bao_cao_day_du.md` (Chương 2, 13, Phụ lục F). Bảng so sánh model: mục "Kết quả nòng cốt" và Phụ lục A. Kiến trúc chi tiết: `docs/RGH_Architecture.md`, `docs/Hai_model_fusion_v2.md`.*