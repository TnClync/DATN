# User Flow & Client Interaction Journey

## Executive Overview

This document maps out the end-to-end user interaction flow and API request execution path for the Recommendation Inference Studio. It describes how clients (end-users, frontend applications, or API consumers) interact with the system to generate real-time personalized recommendations using hybrid collaborative and content-based algorithms.

---

## 🗺️ Visual User Flow Diagram

```
[ 1. Launch Session / API Request ]
                 │
                 ▼
[ 2. Load Checkpoints & System Diagnostic Check ]
                 │
                 ▼
[ 3. Select Target User (Warm-Start / Cold-Start) ]
                 │
                 ▼
[ 4. Tune Hybrid Fusion Parameters (α-Weight, Model Engine) ]
                 │
                 ▼
[ 5. Trigger Real-Time Hybrid Recommendation Engine ]
                 │
        ┌────────┴────────┐
        ▼                 ▼
  ┌──────────┐      ┌──────────┐
  │ CF Model │      │ CB Model │
  │ Scoring  │      │ Scoring  │
  └────┬─────┘      └────┬─────┘
       │                 │
       └────────┬────────┘
                │
                ▼
[ 6. Late Fusion Score Blending: α·CF + (1-α)·CB ]
                │
                ▼
[ 7. Mask Interaction History & Rank Top-N Candidates ]
                │
                ▼
[ 8. Render Results Dashboard / Return JSON Payload ]
```

---

## 📋 Detailed Step-by-Step User Journey

### Phase 1: Session Initialization & Dashboard Entry
1. **Application Entry**: The user accesses the web application dashboard or sends an HTTP request to the inference API endpoint.
2. **Pre-flight System Diagnostics**:
   * Verification of active pre-computed SciPy CSR matrices (`user_item_train.npz`).
   * Availability check of serialized model checkpoints (`als_model.npz`, `bm25_model.npz`, etc.).
   * PyArrow streaming memory health validation.

### Phase 2: Target User Selection & Context Setting
1. **User Profile Selection**:
   * The client specifies a target `user_id` (e.g., `User #42`).
   * The system inspects user interaction history from the sparse training matrix.
   * **Warm-Start Check**: 
     * If interactions $\ge 1$, the historical vector is retrieved.
     * If cold-start (0 interactions), the system automatically defaults to content-based or popularity-based fallbacks.
2. **Inference Constraints**:
   * Set candidate ranking cut-off ($N=5, 10, 20$).
   * Enable/disable filtering of previously interacted items.

### Phase 3: Interactive Parameter Tuning & Model Customization
1. **Model Backbone Selection**:
   * Select primary Collaborative Filtering algorithm (`ALS`, `BPR`, `LMF`, `BM25`, `Cosine`, or `TF-IDF`).
2. **Hybrid Alpha-Fusion Weighting ($\alpha$)**:
   * Adjust the fusion slider ($\alpha \in [0.0, 1.0]$):
     * $\alpha = 1.0$: Pure Collaborative Filtering (latent community behaviors).
     * $\alpha = 0.0$: Pure Content-Based Filtering (item metadata & text similarity).
     * $0.0 < \alpha < 1.0$: Weighted ensemble balancing both signals.

### Phase 4: Real-Time Score Computation & Late Fusion
1. **Inference Execution**:
   * **Collaborative Scoring**:
     $$S_{\text{CF}}(u) = U_u \cdot V_{\text{candidates}}^T$$
   * **Content-Based Scoring**:
     $$S_{\text{CB}}(u) = \text{CosineSimilarity}(P_u, F_{\text{candidates}})$$
   * **Linear Blending**:
     $$S_{\text{hybrid}}(u) = \alpha \cdot S_{\text{CF}}(u) + (1 - \alpha) \cdot S_{\text{CB}}(u)$$
2. **Candidate Filtering & Ranking**:
   * Mask out indices corresponding to items already present in the user's training history.
   * Sort the top $N$ scores descending and assign rank badges (#1, #2, ... #N).

### Phase 5: Result Inspection & Output Rendering
1. **UI Dashboard Rendering**:
   * Display candidate recommendation cards with category tags, descriptions, and rank indicators.
   * Visual score breakdown displaying CF vs. CB relative contributions.
2. **API JSON Payload Output**:
   * Returns a structured JSON payload with item identifiers, names, individual sub-scores, and final ensemble rankings.

---

## 🔄 User Feedback & Iteration Loop
* **Interactive Refinement**: If recommendations lean too heavily on general popularity, the user can lower $\alpha$ to prioritize item content similarity.
* **A/B Testing Simulation**: Test evaluation metrics against real-time user subsamples directly within the dashboard.