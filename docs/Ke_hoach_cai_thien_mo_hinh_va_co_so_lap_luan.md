# Kế hoạch cải thiện mô hình và cơ sở lập luận

> **Vai trò tài liệu:** giải thích tại sao cần cải thiện hệ Deep Hybrid, dữ liệu hiện tại có lỗi gì, nguyên nhân mô hình hybrid chưa vượt baseline, thiết kế kiến trúc đề xuất, cách đánh giá công bằng và lộ trình triển khai để mô hình mới vượt mọi baseline.

**Ngày lập:** 11/11/2026  
**Phạm vi dữ liệu hiện có:** Amazon Reviews (Books + Electronics), implicit feedback, `rating >= 4`, ngưỡng người dùng tối thiểu `>= 3` tương tác.  
**Mục tiêu:** xây dựng một mô hình hybrid có căn cứ khoa học, tối ưu trực tiếp xếp hạng và triển khai được trên ứng dụng hiện hữu.

---

## 1. Kết luận hiện tại

Đồ án có năng lực kỹ thuật tốt nhưng chưa chứng minh được rằng **Deep Hybrid tốt hơn mô hình collaborative baseline**. Kết quả smoke (4.805 users, protocol 1 positive + 99 random negatives, catalog dùng chung 2.464.371 items):

| Model | HR@10 | NDCG@10 | MRR |
|---|---:|---:|---:|
| ALS | **0.6845** | **0.4882** | **0.4401** |
| BPR | 0.6539 | 0.4517 | 0.4028 |
| SBERT | 0.1413 | 0.0686 | 0.0747 |
| Late Fusion, α = 0.8 | 0.6133 | 0.4550 | 0.4204 |
| Light Gated Fusion | 0.5284 | 0.3037 | 0.2612 |
| Deep Early Fusion | 0.5700 | 0.3603 | 0.3129 |
| Ensemble ALS+BPR | **0.7242** | **0.5603** | **0.5179** |

Các kết quả graph/two-tower/transformer chưa đủ độ tin cậy vì trước đó 768-d SBERT bị thiếu; hiện artifact `sbert_item_embeddings_768_f16.npy` đã có (2.464.371×768, float16; đúng 3.785.273.984 bytes), nhưng cần eval lại bằng quy trình không rò rỉ.

**Kết luận cần ghi trong báo cáo:** Ensemble ALS+BPR đang là mô hình mạnh nhất hiện tại, nhưng chưa phải là mô hình đề xuất. Mục tiêu của bước cải thiện là định nghĩa một mô hình deep hybrid duy nhất có thể vượt ALS và, nếu có thể, vượt cả ensemble hiện tại trên cùng protocol.

---

## 2. Kiểm toán dữ liệu thực tế

### 2.1. Những gì đã đúng

- Train: **45.609.800** dòng, không có cặp `(user_idx, item_idx)` lặp; 19.153.183 users và 3.089.725 items.
- Validation: 4.998.646 dòng, mỗi user một dòng, toàn bộ `rating >= 4`; 2.559.404 dòng thuộc user có ít nhất 3 tương tác train.
- Test: 8.281.078 dòng, mỗi user đúng một positive, toàn bộ `rating >= 4`; 2.559.404 dòng thuộc user có ít nhất 3 tương tác train.
- Không có cặp `(user, item)` chồng lấn giữa train/test, train/validation và test/validation.
- Không có cặp `(user, item)` lặp trong train và test.
- `als_item_ids.npy` và `als_user_ids.npy` khớp 100% với các ID unique tái tính từ train sau bộ lọc `rating >= 4` và `count >= 3`.
- Không nên sửa raw train/val/test. Mapping đã được CF, DeepP và SBERT sử dụng cùng chung, nên random remap sẽ phá hỏng toàn bộ artifact.

### 2.2. Những gì phải báo cáo là giới hạn

1. **Split không phải temporal-strict tuyệt đối.** Khoảng thời gian test chồng lấn khoảng thời gian train. Đây là một dạng holdout/leave-one-out theo user trên dữ liệu review, không nên gọi là “future-only” nếu không chứng minh thêm bằng timestamp.

2. **Mất coverage khoảng 6.1% của tập test.** Trong 2.559.404 test positives của user có thể đánh giá, có **156.813 dòng (6.13%)** bị loại vì positive nằm ngoài `item_ids` của tập huấn luyện hợp lệ. Các dòng này thuộc **133.327 item unique**; chỉ **76.307 dòng** (**57.888** item unique) từng xuất hiện ở positive thô trong train. Đây là bias theo phía item “đã được một user đủ mạnh tương tác”.

3. **Validation đang bị bỏ phí.** Tập validation có 1 positive mỗi user và 2.559.404 dòng của user active, nhưng eval hiện tại chọn alpha/ensemble trên 10.000 test users rồi báo cáo trên cả tập test. Điều này làm số liệu dễ bị lạc quan.

4. **Nguy cơ transductive của content embedding.** Các SBERT embedding hiện tại được tạo từ metadata item và tối đa 3 đoạn `review_content` theo item. Nếu review lấy từ toàn bộ `lastreview`, item embedding có thể quan sát gián tiếp thông tin của tương tác test. Dù chưa chứng minh đây là leakage, không nên coi content branch hiện tại là “sạch tuyệt đối”. Content embedding cải tiến cần dùng metadata-only hoặc text phát sinh chỉ từ train.

### 2.3. Có cần “làm lại dữ liệu” hoàn toàn không?

**Không.** Không nên ghi đè `data/train`, `data/validation`, `data/test` hoặc mapping. Việc “làm lại” cần thiết nằm ở dữ liệu phái sinh:

1. Tạo lại như cũ: catalog đánh giá, history-only-from-train và exact cùng mapping;
2. Tạo/ tái kiểm tra mapping `user_idx ↔ reviewerID`, `item_idx ↔ asin`;
3. Dựng lại item text từ **metadata-only** (title, brand, category, description, features), không lấy review;
4. Tạo `user content profile = mean(item embeddings của lịch sử train)`;
5. Chuẩn hoá điểm trong từng candidate set; không đưa điểm thô khác thang đo vào một hàm fusion;
6. Tạo hard-negative từ retrieval union thay cho 1 positive + 4 negative ngẫu nhiên;
7. Tách validation thành tập tune/early stopping và test thành holdout báo cáo cuối cùng.

Nếu dữ liệu `finaldata` trên Kaggle được cung cấp đúng schema, bước đầu tiên phải là đọc schema, kiểm tra số lượng, overlap, mapping, phân phối rating và timestamp trước khi rebuild bất kỳ artifact nào.

---

## 3. Nguyên nhân Deep Hybrid hiện tại chưa vượt baseline

### 3.1. User input chưa cùng bản chất: CF và CB không được chuẩn hoá

User/candidate footprint hiện có hai nhóm tín hiệu:

- **CF:** user embedding, item embedding, dot-product/ALS-BPR-LightGCN score;
- **CB:** content profile từ history, item embedding, cosine SBERT/BLAIR score.

Hai nhóm score có thang đo khác nhau. CF dot-product có thể chênh cỡ nhiều đơn vị, trong khi embedding thường đã chuẩn hoá nên cosine sát khoảng `[-1, 1]`. Khi trộn trực tiếp, optimizer hiếm khi học được trọng số ổn định.

### 3.2. Gate hiện có gần như không mang thông tin

Trên 213 test users (protocol tương đương), LightGated score không tạo một phân biệt rõ ràng:

- ALS HR@10: **0.6479**; HR@5 **0.5305**
- LightGated HR@10: **0.4319**; HR@5 **0.1737**
- Gate α trung bình: **0.569**; p05 **0.549**; p50 **0.554**; p95 **0.756**
- Overlap top-10 với CF: **0.116**

Điều này chỉ ra mô hình đang “phá hỏng” ordering của CF hơn là học một hybrid có lợi, đồng thời α gần như không phân biệt được users có history khác nhau.

### 3.3. Mục tiêu huấn luyện không phản ánh mục tiêu đánh giá

Các head deep hiện huấn luyện theo pointwise BCE/binary cross-entropy trên 1 positive + 4 negative. Trong khi validation/test đo ranking trong tập candidate hoạt động theo thứ tự HR/NDCG/Recall/AP/MRR. Model dễ học “positive lớn hơn negative” một cách lỏng lẻo nhưng không học được xếp hạng tốt nhất candidate.

### 3.4. Content branch còn yếu và có rủi ro chất lượng

SBERT-alone đạt HR@10 chỉ **0.1413**. Điều này không nhất thiết có nghĩa content vô giá trị, nhưng với cách nhồi content hiện tại nó chưa bổ sung được đủ tín hiệu để vượt CF mạnh. Ngoài ra, review-derived embedding có nguy cơ transductive, nên cải thiện content cần rebuild từ metadata-only.

### 3.5. Không có ràng buộc an toàn cho nhánh deep

Mô hình không có cơ chế về mặt kiến trúc để đảm bảo “nếu content không giúp, quay về CF mạnh nhất”. Đây là thiếu sót của residual anchor design, không đơn thuần là lỗi training seed.

---

## 4. Cơ sở lập luận từ tài liệu và bài báo

| Thành phần cần dùng/chọn | Cơ sở từ tài liệu/bài báo | Ý nghĩa áp dụng cho đồ án |
|---|---|---|
| **ALS implicit feedback** | Hu, Koren & Volinsky (ICDM 2008), *Collaborative Filtering for Implicit Feedback Datasets* | Đặt `rating >= 4` là positive có confidence; lý giải bài toán là implicit, không phải regression rating thông thường. |
| **BPR** | Rendle et al. (UAI 2009), *BPR: Bayesian Personalized Ranking from Implicit Feedback* | Learning-to-rank bằng pairwise ranking; nền tảng cho việc chuyển head từ BCE sang objective ranking. |
| **LightGCN** | He et al. (SIGIR 2020), *LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation* | User-item graph diffusion bổ sung collaborative signal bậc cao, phù hợp làm nhánh graph thay vì chỉ dùng MF. |
| **Candidate generation / two-stage** | Covington, Adams & Sargin (RecSys 2016), *Deep Neural Networks for YouTube Recommendations* | Retrieval union CF/content/graph rồi rerank giúp giữ latency, recall và quy mô catalog lớn; tránh scan 3.7M items. |
| **LLM/text item encoder** | Hou et al., *Bridging Language and Items for Retrieval and Recommendation* (arXiv:2403.03952; ACL 2026) | Amazon Reviews 2023 chứng minh item metadata + language context có thể dùng làm semantic representation cho recommendation/retrieval; là lý do chọn BLAIR thay SBERT nếu có thể rebuild. |
| **SBERT content representation** | Reimers & Gurevych (EMNLP 2019), *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks* | Nền tảng content retrieval cosine nhanh; SBERT hợp lệ nếu đã normalization và index FAISS, nhưng cần nâng chất lượng nếu content hiện tại yếu. |
| **Linear + deep mémoire/generalization** | Cheng et al. (DLRS/RecSys 2016), *Wide & Deep Learning for Recommender Systems* | User input CF tương đương “memorization”; deep content/graph component tạo “generalization”. Wide&Deep là cơ sở cho residual anchor. |
| **Bounded-degree feature crosses** | Wang et al. (WWW 2021), *DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems* | Cross layers học tương tác tường minh giữa user/content/graph features; thích hợp cho Deep Early Fusion cải tiến. |
| **Hybrid taxonomy** | Burke. *Hybrid Recommender Systems: Survey and Experiments* (UMUAI 2002) | Mô hình đề xuất thuộc nhóm weighted/feature-based + meta-level hybrid; dùng để giải thích lựa chọn kiến trúc trong báo cáo. |
| **Deep learning for recommendation** | Zhang, Yao, Sun & Tay (ACM Computing Surveys 2019), *Deep Learning based Recommender System: A Survey and New Perspectives* | Cung cấp taxonomy cho NCF/MLP/graph/attention; giúp đặt vị trí của mô hình mới trong literature. |
| **Deep hybrid + stochastic cold gates** | Barkan et al. (2021), *Cold Item Integration in Deep Hybrid Recommenders via Tunable Stochastic Gates* | Hỗ trợ khía cạnh cold item/gated hybrid; liên quan trực tiếp đến ~156.813 positives bị loại khỏi eval do catalog. |
| **Sampled metrics** | Krichene & Rendle (KDD 2020), *On Sampled Metrics for Item Recommendation* | Cảnh báo metric trên 1 positive + 99 sampled negatives không đại diện đầy đủ full catalog; cần thêm retrieval Recall và/hoặc full evaluation. |
| **Hybrid review trong đề án** | Chaudhari et al. (IEEE Access 2024), *A Hybrid Recommendation System: A Review* | Review cho thấy hybrid có lợi vì kết hợp điểm mạnh CF/CB, nhưng vẫn bị sparsity, cold start, scale và drift; phù hợp để viết Related Work và Discussion, không phải công thức trực tiếp để tăng điểm. |

---

## 5. Mô hình đề xuất: CF-Anchored Residual Gated Hybrid (RGH)

### 5.1. Mục tiêu kiến trúc

RGH không cố “thay thế” CF bằng content. Thiết kế tốt nhất cho dữ liệu hiện tại là **neo CF, chỉ để model học phần bù từ graph/content khi phần bù có ích**.

### 5.2. Hai giai đoạn

#### Stage 1 — Candidate generation

Với mỗi user, tạo union:

`U = TopK_cf(ALS) ∪ TopK_gcn(LightGCN) ∪ TopK_text(SBERT/BLaIR) ∪ Popularity(K_pop)`

K thường 50–100 mỗi nguồn. Với training, union luôn chứa positive. Với serving, union không làm thay đổi artifact cũ; chỉ tăng chất lượng recall trước rerank.

#### Stage 2 — Residual reranker

Đặc trưng mỗi candidate:

```text
CF features      : u_als, i_als, u_bpr, i_bpr
Graph features   : u_gcn, i_gcn
Content features : u_txt = mean(embedding history_train), i_txt
Per-user scores  : z_cf, z_gcn, z_txt (z-score hoặc rank normalization trên U)
Context          : popularity, history length, domain/category if available
```

Công thức đề xuất:

```text
score(u,i) =
    a_cf  * z_cf(u,i)
  + a_gcn * z_gcn(u,i)
  + a_txt * z_txt(u,i)
  + g(h_u, h_i) * Δ(u,i)
```

Trong đó:

- `z_cf`, `z_gcn`, `z_txt` được chuẩn hoá per-user trên candidate set `U`;
- `a_*` là trọng số học được hoặc cố định ban đầu thiên CF;
- `g(h_u,h_i) = sigmoid(w·h + b)` là gated residual với `b` khởi tạo negative để Đầu ra `gΔ` nhỏ ban đầu;
- `Δ(u,i)` là cross network/MLP theo DCN-V2;
- khi `g→0`, model quy về một weighted ranker neo trên CF và graph/content; mô hình không thể “tùy tiện” phá hỏng ordering CF.

### 5.3. User input thực tế

User input **luôn có cả CF và CB**:

- **CF:** `u_als`, `u_bpr`, `u_gcn` từ đồ thị/ma trận implicit;
- **CB:** `u_txt` là trung bình embedding các item trong toàn bộ train history; nếu user không có history thực tế (cold-start), CB signal bị mask và `a_cf/a_gcn` phải chiếm ưu thế;
- **Candidate scores:** `z_cf`, `z_gcn`, `z_txt`;
- **Context:** popularity, rank of item in each branch, history length/domain.

### 5.4. Loss function

Thay binary BCE hiện tại bằng **listwise sampled softmax trên candidate union**:

```text
L_rank = - log exp(score(pos)/T) / Σ_{i∈U} exp(score(i)/T)
```

Có thể kết hợp BPR làm pairwise regularizer:

```text
L_pair = -log σ(score(pos) - score(neg_hard))
```

Negative gồm:

- hard negatives từ top CF/graph/text nhưng không phải positive;
- random negatives để giữ exploration;
- positives không bao giờ xuất hiện như negative.

Chọn checkpoint dựa trên **validation NDCG@10 và Recall@10**; test chỉ dùng một lần để báo cáo cuối cùng.

### 5.5. Tại sao đây là “một bài báo”/cụm bài báo hợp lý?

Nếu buộc chọn một bài báo xuyên suốt cho đề án, **BLaIR (Hou et al.)** là bài phù hợp nhất với Amazon Reviews 2023 vì tạo item representation từ language context và item metadata. Nhưng RGH thực tế là tích hợp: BLaIR/SBERT + ALS/BPR + LightGCN + Wide&Deep/DCN-V2 + sampled/listwise ranking + two-stage retrieval. Trong báo cáo nên trình bày đây là một **kiến trúc kết hợp có chọn lọc**, không phải mô phỏng nguyên xi một bài.

---

## 6. Protocol đánh giá bắt buộc

### 6.1. Hai loại đánh giá

**A. Protocol hiện hành để so sánh lịch sử**

- 1 positive + 99 random negatives từ catalog 2.464.371;
- k = 5/10/20;
- cùng user, cùng candidate, cùng seed; leak-free history.

**B. Protocol phục vụ mô hình đề xuất và retrieval**

- Recall@K và HR@K của `U` hợp nhất CF+graph+text;
- Recall@K và NDCG@K sau reranker;
- Báo cáo riêng: đều có và không có item text; cold item bị loại khỏi catalog;
- Nêu rõ sampled-negative metric không thay thế full catalog metric.

### 6.2. Metrics

- **HR@K / Recall@K**: hit trong top-K;
- **Precision@K**: fraction đúng trong K;
- **F1@K**: cân bằng Precision/Recall;
- **NDCG@K**: chất lượng xếp hạng với vị trí;
- **MRR**: vị trí của positive đầu tiên;
- **AP@K**: Average Precision;
- **Mean Rank**: rank trung bình của positive;
- **Retrieval Recall@K**: positive nằm trong candidate union;
- **Coverage/novelty/diversity** nếu cần phần phân tích định tính;
- **Bootstrap CI** và paired test (paired t-test hoặc Wilcoxon signed-rank) theo từng user;
- Report delta với baseline và p-value/confidence interval, không chỉ con số.

### 6.3. Quy trình tune chống leakage

1. Rebuild/run baseline ALS/BPR/LightGCN với same split;
2. Tune alpha/hyperparameters **trên validation**;
3. Early stopping/chọn model trên validation;
4. Chạy test một lần;
5. Báo cáo full N; nếu lấy mẫu thì chỉ lấy ≤20% nhưng seed cố định, không chia nhóm đặc biệt;
6. Chạy ablation;
7. Không sửa lại test sau khi đã xem kết quả; nếu sửa phải coi là thí nghiệm mới.

---

## 7. Ablation và kế hoạch thí nghiệm

Thứ tự bắt buộc:

1. **ALS | BPR | LightGCN standalone** trên validation và test.
2. **RGH Neo-CF only** (`a_gcn=a_txt=0, Δ=0`): phải gần ALS nhất.
3. **RGH + Normalized branch scores**: chuẩn hoá per-user.
4. **RGH + Listwise/hard negatives**: thay BCE.
5. **RGH + Content BLaIR/SBERT metadata-only**: content branch.
6. **RGH + Graph branch**.
7. **RGH + Full loss** và so với ensemble ALS+BPR.
8. Phân tích theo:
   - độ dài history (<5, 5–20, >20);
   - domain (Books/Electronics);
   - item warm/cold;
   - positive nằm ngoài catalog;
   - alpha/gate distribution.

**Tiêu chuẩn chấp nhận:** RGH phải vượt ALS trên NDCG@10 hoặc Recall@10/HR@10 trên test holdout, và không thua đáng kể ở metric còn lại. Lý tưởng là vượt ensemble ALS+BPR hiện tại trên NDCG@10.

---

## 8. Lộ trình triển khai

### Giai đoạn 1 — Data và evaluation repair

- [ ] Xác nhận `finaldata` từ Kaggle: schema, số lượng, overlap, mapping.
- [ ] Giữ raw split; không ghi đè.
- [ ] Dựng history leakage-free từ train.
- [ ] Chuẩn hoá catalog mapping.
- [ ] Rebuild content metadata-only nếu có metadata đã kiểm tra.
- [ ] Viết script eval tune trên validation, report test; thêm metrics và bootstrap CI.

### Giai đoạn 2 — Training và model implementation

- [ ] Tạo candidate union.
- [ ] Tạo hard negatives.
- [ ] Cài normalized branch score layer.
- [ ] Cài residual gate (DCN-V2 cross + MLP).
- [ ] Train listwise với checkpoint selection trên validation.
- [ ] Export checkpoint và metrics.

### Giai đoạn 3 — Báo cáo và deploy

- [ ] So sánh đầy đủ ALS/BPR/LightGCN/Deep/RGH/Ensemble.
- [ ] Ablation table + graph HR/NDCG.
- [ ] Error analysis, warm/cold case, qualitative examples.
- [ ] Cập nhật app chỉ sau khi model đạt acceptance gate.
- [ ] Viết chương 3 Methodology và chương 4 Results/Discussion theo cấu trúc.

---

## 9. Các điểm cần nói rõ để tránh kỹ sư không bị hỏi vặn

- **“Dữ liệu có temporal split chưa?”** → Không hoàn toàn; đây là implicit user-wise holdout với timestamp chồng lấn. Ghi là giới hạn.
- **“Có leakage không?”** → Không có overlap `(user,item)` giữa các split; chưa chứng minh review content leakage, nhưng content cần rebuilt metadata-only để loại yếu tố này.
- **“Tại sao loại 6.13%?”** → Catalog protocol chỉ xếp hạng item có đủ tín hiệu train; cần nêu giới hạn cold-item.
- **“Tuning có leak không?”** → Hiện tại eval kernel đang tune trên test-subset; đây là vấn đề cần sửa, không phải kết quả công bố cuối cùng.
- **“Vì sao hybrid thắng hoặc không thắng?”** → Phải trả lời bằng ablation, không bằng trực giác. Kiểm tra từng trong thêm, từng branch và từng objective.
- **“Tách biệt popularity và personalization?”** → Bắt buộc; ghi popularity như feature nhưng không cho nó nuốt toàn bộ ranking.

---

## 10. Tài liệu tham khảo trực tiếp

1. Hu, Y., Koren, Y., & Volinsky, C. (2008). Collaborative Filtering for Implicit Feedback Datasets. *ICDM*.
2. Rendle, S., et al. (2009). BPR: Bayesian Personalized Ranking from Implicit Feedback. *UAI*.
3. He, X., et al. (2020). LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation. *SIGIR*. arXiv:2002.02126.
4. Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *EMNLP*.
5. Hou, Y., et al. Bridging Language and Items for Retrieval and Recommendation: Benchmarking LLMs as Semantic Encoders. arXiv:2403.03952.
6. Covington, P., Adams, J., & Sargin, E. (2016). Deep Neural Networks for YouTube Recommendations. *RecSys*.
7. Cheng, H.-T., et al. (2016). Wide & Deep Learning for Recommender Systems. *DLRS/RecSys*.
8. Wang, R., et al. (2021). DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems. *WWW*. arXiv:2008.13535.
9. Burke, R. Hybrid Recommender Systems: Survey and Experiments. *UMUAI*.
10. Zhang, S., Yao, L., Sun, A., & Tay, Y. (2019). Deep Learning based Recommender System: A Survey and New Perspectives. *ACM Computing Surveys*, 52(1), 1–38. DOI:10.1145/3285029.
11. Barkan, O., et al. (2021). Cold Item Integration in Deep Hybrid Recommenders via Tunable Stochastic Gates. arXiv:2112.07615.
12. Krichene, W., & Rendle, F. (2020). On Sampled Metrics for Item Recommendation. *KDD*.
13. Caruana, R., et al. (2004). Ensemble Selection from Libraries of Models. *ICML*.
14. Chaudhari, A., Seddig, A. A. H., Sarlan, A., & Raut, R. (2024). A Hybrid Recommendation System: A Review. *IEEE Access*, 12, 157107–157129. DOI:10.1109/ACCESS.2024.3480693.

---

## 11. Kết luận lập luận

Đồ án này không cần “vờ rằng deep hybrid đã thắng”. Cần cải thiện một cách có phương pháp: giữ CF mạnh làm nền, nâng content/graph thành phần bổ sung có kiểm soát, thay objective ranking sai, và loại tuning leakage. Nếu RGH không vượt ALS sau khi sửa đúng những điểm trên, kết quả phải được báo thật và mô hình đề xuất phải được điều chỉnh; không được chọn metric hoặc split để làm số đẹp. Nếu RGH vượt validation nhưng không vượt test, nguyên nhân phải là generalization issue, không phải lỗi mapping.

**Luận điểm báo cáo:** một hybrid tốt không phải là mô hình deep hơn hoặc nhiều tham số hơn, mà là mô hình kết hợp đúng các giả thuyết của từng nhánh: CF khai thác hành vi cộng tác, graph khai thác cấu trúc bậc cao, content khai thác ngữ nghĩa sản phẩm, resolver học khi nào dùng tín hiệu nào. RGH được đề xuất chính xác để làm điều đó.
