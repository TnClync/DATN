# Deep Hybrid Recommendation Models Design

**Project Context**  
- Data: ~45.6M train interactions, 22M users × 3.7M items  
- Existing strong artifacts: **MF-bias**, **ALS / Bayesian factors**, **SBERT + FAISS**, optional FPGrowth  
- Constraints: Train on Kaggle (GPU possible), Deploy on local machine with **8GB RAM**, fast CPU inference  
- Goal: High ranking quality (HR@K, NDCG@K) while staying memory-efficient

---

## 1. Overview of Hybrid Strategies

| Strategy       | Description                                      | Pros                              | Cons                              | Best When                          |
|----------------|--------------------------------------------------|-----------------------------------|-----------------------------------|------------------------------------|
| **Early Fusion**   | Combine features / embeddings before scoring    | Joint representation learning    | Higher memory, harder to train   | Rich side information available   |
| **Late Fusion**    | Score separately then combine scores            | Simple, modular, easy to tune    | Limited interaction modeling     | Strong individual models exist    |
| **Advanced (Two-Stage + Reranker)** | Retrieve candidates → Rerank with stronger model | Best quality + controllable latency | Extra inference stage            | Production systems                |

---

## 2. Required Artifacts (Common)

| Artifact                    | Source                  | Format                  | Approx. Size (optimized) | Notes |
|----------------------------|-------------------------|-------------------------|---------------------------|-------|
| User Factors (CF)          | MF-bias / ALS / Bayesian | `.npy` (float16) + mmap | ~2.8 GB (22M × 64)       | Memory-map on deploy |
| Item Factors (CF)          | MF-bias / ALS / Bayesian | `.npy` (float16)        | ~480 MB                  | Keep in RAM |
| SBERT Item Embeddings      | all-MiniLM-L6-v2        | `.npy` or FAISS index   | ~600–800 MB              | 384-dim |
| FAISS Index (SBERT)        | HNSW or IVF+PQ          | `.index`                | 400–700 MB               | For fast candidate retrieval |
| Item Metadata / Text       | Title, category, etc.   | Parquet / JSON          | Small                    | For reranker input |
| User History (recent)      | Interaction log         | Dict / Redis / mmap     | Variable                 | Needed for personalization |
| (Optional) FPGrowth Rules  | Spark FPGrowth          | Dict[item → related]    | 100–300 MB               | Co-occurrence signal |

**Optimization for 8GB RAM**  
- Convert all factors to **float16**  
- Memory-map user factors (`mmap_mode="r"`)  
- Use two-stage retrieval (never score all 3.7M items)  
- Keep only active / popular items in RAM if needed

---

## 3. Early Fusion Models

Early Fusion combines different signals **before** the final scoring layer.

### 3.1 Simple Early Fusion (Feature Concatenation)

**Architecture**
```
User Embedding (CF) ──┐
                      ├── Concat → MLP / Dot Product → Score
Item Embedding (CF) ──┤
SBERT Item Emb ───────┘
```

**Required Artifacts**
- User CF factors
- Item CF factors
- SBERT item embeddings (aligned by item_idx)

**Training**
- Concatenate CF user + CF item + SBERT item
- Train a small MLP or use weighted sum + bias
- Loss: BPR or sampled softmax

**Pros**: Easy to implement  
**Cons**: Limited interaction between modalities  
**Memory**: Medium

---

### 3.2 Deep Early Fusion (Cross-Network style)

Inspired by DCN / AutoInt ideas but kept lightweight.

**Architecture**
```
[User CF | Item CF | SBERT Item] 
        ↓
   Cross Layers (or Self-Attention)
        ↓
   Deep MLP
        ↓
     Score
```

**Required Artifacts**
- Same as above
- Optional: categorical features (category, brand) embedded

**Training Notes**
- Use residual connections
- Keep model small (2–4 cross layers, hidden size 64–128)
- Can be trained on Kaggle GPU

**When to use**: When you want the model to learn non-linear interactions between CF and content.

---

### 3.3 Gated Early Fusion

```
CF_score_raw = User_CF · Item_CF
Content_score_raw = User_proj(SBERT) · Item_SBERT

gate = σ(MLP([User_CF, Item_CF, SBERT]))
final = gate * CF_score_raw + (1 - gate) * Content_score_raw
```

**Required Artifacts**
- CF factors
- SBERT embeddings
- Small gate network (very lightweight)

**Advantage**: Learns when to trust CF vs Content per user-item pair.

---

## 4. Late Fusion Models

Late Fusion scores each signal independently, then combines the scores.

### 4.1 Classic Late Fusion (Weighted Sum)

```python
final_score = α * normalize(CF_score) 
            + β * normalize(SBERT_score) 
            + γ * normalize(FPGrowth_score)   # optional
```

**Required Artifacts**
- CF factors (MF-bias or ALS)
- SBERT + FAISS
- (Optional) FPGrowth rules

**Tuning**
- Grid search or Optuna on validation set for α, β, γ
- Your previous experiments already showed good results with α ≈ 0.5–0.7

**Pros**: Extremely simple, fast, interpretable  
**Cons**: No deep interaction between signals

---

### 4.2 Learned Late Fusion (Light Gate)

```python
# Features for gate
gate_input = [user_activity, item_popularity, CF_score, SBERT_score, ...]

α = sigmoid(MLP(gate_input))          # or multi-weight softmax
final = α * CF + (1-α) * SBERT
```

**Required Artifacts**
- Same as Classic Late Fusion
- Small MLP (few thousand parameters) — easy to train and deploy

**Advantage**: Dynamic weighting instead of fixed α.

---

### 4.3 Multi-Signal Late Fusion

```python
final = w1*CF + w2*SBERT + w3*OtherCB + w4*FPGrowth + w5*Popularity
```

Weights can be fixed or predicted by a tiny network.

---

## 5. Advanced Models (Two-Stage + Reranker)

This is the **recommended production architecture** for your scale and 8GB constraint.

### 5.1 Two-Stage Hybrid + Cross-Encoder Reranker

```
Stage 1 – Candidate Generation (fast, recall-oriented)
├── CF (MF-bias / ALS) top-K
├── SBERT + FAISS top-K
└── FPGrowth / Popularity (optional)
        ↓  (union → 200–1000 candidates)

Stage 2 – Reranking (precision-oriented)
└── Cross-Encoder Reranker (bge-reranker-v2-m3 or ms-marco-MiniLM)
        ↓
   Final Top-N
```

**Required Artifacts**
- All CF factors + SBERT FAISS (for Stage 1)
- Item text / title / description (for reranker input)
- Pretrained cross-encoder (bge-reranker-v2-m3 recommended — free & strong)
- User recent history (to build query text for reranker)

**Why this is powerful**
- Stage 1 keeps memory and latency low
- Stage 2 brings deep semantic understanding
- Fully free (using open-source reranker)

---

### 5.2 Two-Stage + Learned Ranker (LightGBM / Small NN)

Instead of (or in addition to) cross-encoder:

**Features for ranker**:
- CF score
- SBERT score
- FPGrowth score
- Item popularity
- User-item interaction count
- Category match, price difference, etc.

Train a **LightGBM** or tiny MLP as the final ranker.

**Pros**: Extremely fast on CPU, easy to add many features  
**Cons**: Needs feature engineering

---

### 5.3 Deep Two-Tower + Reranker (Advanced)

```
User Tower: User CF + User history encoder
Item Tower: Item CF + SBERT + metadata
        ↓
Approximate retrieval (FAISS on item tower)
        ↓
Reranker (cross-encoder or gradient boosting)
```

**Required Artifacts**
- Pre-trained user & item towers
- FAISS index on item tower embeddings
- Reranker model

**Note**: Heavier to train. Only recommended if you have enough GPU time on Kaggle.

---

## 6. Recommended Model Stack (Practical Order)

| Priority | Model                              | Type          | Complexity | Expected Gain          | 8GB Friendly |
|----------|------------------------------------|---------------|------------|------------------------|--------------|
| 1        | Late Fusion (MF-bias + SBERT)     | Late          | Low        | Strong baseline        | Yes          |
| 2        | Two-Stage + bge-reranker-v2-m3    | Advanced      | Medium     | Highest quality        | Yes          |
| 3        | Learned Late Fusion (Light Gate)  | Late          | Low-Medium | Better than fixed α    | Yes          |
| 4        | Gated Early Fusion                | Early         | Medium     | Good interaction       | Yes          |
| 5        | Deep Early Fusion (Cross + MLP)   | Early         | High       | Research-oriented      | Borderline   |

---

## 7. Artifact Checklist for Implementation

### Minimum Viable Hybrid
- [ ] MF-bias user & item factors (float16)
- [ ] SBERT item embeddings + FAISS index
- [ ] Item ID mapping
- [ ] Simple Late Fusion code

### Production Two-Stage
- [ ] All of the above
- [ ] Item text data (title + description)
- [ ] bge-reranker-v2-m3 (or MiniLM)
- [ ] User recent history store
- [ ] Candidate generation + rerank pipeline

### Optional Strong Additions
- [ ] ALS / Bayesian factors (as extra CF signal)
- [ ] FPGrowth association rules
- [ ] LightGBM ranker features
- [ ] Small gate network weights

---

## 8. Training & Deployment Notes

**Training**
- Late Fusion: almost no training (just tune weights)
- Gate / Early Fusion: train on Kaggle GPU
- Reranker: use pretrained (no need to train from scratch)

**Deployment (8GB RAM)**
```python
# Example loading strategy
user_factors = np.load("user_factors_f16.npy", mmap_mode="r")
item_factors = np.load("item_factors_f16.npy")          # in RAM
faiss_index = faiss.read_index("sbert.faiss")
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", device="cpu")
```

**Inference Flow (Recommended)**
1. Get user vector (from mmap)
2. Retrieve 300–500 candidates via FAISS + CF
3. Rerank top 50 with cross-encoder
4. Return top 10–20

---

## 9. Next Steps

1. Start with **Classic Late Fusion** (reuse your existing MF-bias + SBERT)
2. Implement **Two-Stage + bge-reranker-v2-m3**
3. Measure HR@10 / NDCG@10 lift
4. Optionally add Light Gate or FPGrowth signal

---

**Document Version**: 1.0  
**Date**: 2026-11-06  
**Author**: Hybrid Recommender Design for large-scale implicit feedback + 8GB deployment constraint
