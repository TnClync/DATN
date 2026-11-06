# Deep Hybrid Recommendation System Architecture & Data Flow

## Executive Overview

This document presents the detailed architectural blueprint, component interactions, and multi-stage data flow of the **Deep Hybrid Recommendation Engine** (combining **Feature-Level Early Fusion** and **Score-Level Late Fusion**).

---

## 🏗️ End-to-End System Architecture Diagram

```
                              ┌────────────────────────────────────────┐
                              │    RAW DATA & SPARK STAGING PIPELINE   │
                              └───────────────────┬────────────────────┘
                                                  │
                                                  ▼
                                ┌───────────────────────────────────┐
                                │ PySpark & PyArrow Data Staging    │
                                │ • Thresholding: Rating >= 4.0     │
                                │ • CSR Matrix (user_item_train)    │
                                └─────────────────┬─────────────────┘
                                                  │
                                ┌─────────────────┴─────────────────┐
                                │                                   │
                                ▼                                   ▼
                   ┌──────────────────────────┐        ┌──────────────────────────┐
                   │ Collaborative Filtering  │        │  Content-Based Filtering │
                   │  (ALS / BPR Checkpoint)  │        │   (TF-IDF / Metadata)    │
                   └────────────┬─────────────┘        └────────────┬─────────────┘
                                │                                   │
                                │ Item Factors (Q_i)                │ TF-IDF Matrix (C_i)
                                └─────────────────┬─────────────────┘
                                                  │
                                                  ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: EARLY FUSION (Feature-Level Integration)                                               │
│                                                                                                  │
│   ┌───────────────────────────────┐                 ┌───────────────────────────────┐            │
│   │   L2-Normalized CF Factors    │                 │   L2-Normalized CB Features   │            │
│   │   Q_norm = (Q / ||Q||) * α    │                 │  C_norm = (C / ||C||) * (1-α) │            │
│   └───────────────┬───────────────┘                 └───────────────┬───────────────┘            │
│                   │                                                 │                            │
│                   └────────────────────────┬────────────────────────┘                            │
│                                            │                                                     │
│                                            ▼                                                     │
│                           ┌──────────────────────────────────┐                                   │
│                           │      Sparse Horizontal Stack     │                                   │
│                           │   E_i = [ Q_norm  ||  C_norm ]   │                                   │
│                           └────────────────┬─────────────────┘                                   │
│                                            │                                                     │
│                                            ▼                                                     │
│                           ┌──────────────────────────────────┐                                   │
│                           │     User Profile Construction    │                                   │
│                           │     U_p = History_u · E_i        │                                   │
│                           └────────────────┬─────────────────┘                                   │
│                                            │                                                     │
│                                            ▼                                                     │
│                           ┌──────────────────────────────────┐                                   │
│                           │      Cosine Similarity Layer     │                                   │
│                           │      Score_early(u, i)           │                                   │
│                           └────────────────┬─────────────────┘                                   │
└────────────────────────────────────────────┼─────────────────────────────────────────────────────┘
                                             │
                                             │ Score_early (Stage 1 Output)
                                             │
                                             ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: LATE FUSION & ENSEMBLE RANKING                                                         │
│                                                                                                  │
│       ┌────────────────────────────────────┐              ┌────────────────────────────────────┐ │
│       │    Pure CF Raw Prediction Engine   │              │   Stage 1 Early Fusion Predictor   │ │
│       │          Score_cf(u, i)            │              │          Score_early(u, i)         │ │
│       └─────────────────┬──────────────────┘              └─────────────────┬──────────────────┘ │
│                         │                                                   │                    │
│                         ▼                                                   ▼                    │
│       ┌────────────────────────────────────┐              ┌────────────────────────────────────┐ │
│       │      Min-Max Normalization         │              │      Min-Max Normalization         │ │
│       │      Score_cf_norm ∈ [0, 1]        │              │     Score_early_norm ∈ [0, 1]       │ │
│       └─────────────────┬──────────────────┘              └─────────────────┬──────────────────┘ │
│                         │                                                   │                    │
│                         └──────────────────┬────────────────────────────────┘                    │
│                                            │                                                     │
│                                            ▼                                                     │
│                           ┌──────────────────────────────────┐                                   │
│                           │     Linear Score Weighted Blend  │                                   │
│                           │   Score = β·S_cf + (1-β)·S_early │                                   │
│                           └────────────────┬─────────────────┘                                   │
│                                            │                                                     │
│                                            ▼                                                     │
│                           ┌──────────────────────────────────┐                                   │
│                           │     Mask Interacted Train Items  │                                   │
│                           │     (Liked Items -> -1e9)        │                                   │
│                           └────────────────┬─────────────────┘                                   │
│                                            │                                                     │
│                                            ▼                                                     │
│                           ┌──────────────────────────────────┐                                   │
│                           │      Top-N Candidate Sorting     │                                   │
│                           └────────────────┬─────────────────┘                                   │
└────────────────────────────────────────────┼─────────────────────────────────────────────────────┘
                                             │
                                             ▼
                              ┌──────────────────────────────┐
                              │ Final Top-N Recommendation   │
                              │       Output Payload         │
                              └──────────────────────────────┘
```

---

## 📋 Component Specification & Mathematics

### 1. Representation & Normalization Layer (Stage 1)
Given:
* $Q_i \in \mathbb{R}^d$: Collaborative Filtering item latent factor vector from Matrix Factorization.
* $C_i \in \mathbb{R}^m$: Content-Based TF-IDF feature sparse vector.
* $\alpha \in [0, 1]$: Early fusion feature weight parameter.

The normalized feature representations are computed as:

$$Q_{\text{norm}}(i) = \alpha \cdot \frac{Q_i}{\|Q_i\|_2}$$

$$C_{\text{norm}}(i) = (1 - \alpha) \cdot \frac{C_i}{\|C_i\|_2}$$

The joint representation matrix $E \in \mathbb{R}^{N \times (d + m)}$ is formed via horizontal sparse concatenation:

$$E_i = \left[ Q_{\text{norm}}(i) \;\|\; C_{\text{norm}}(i) \right]$$

### 2. Preference Aggregation Layer
For user $u$ with interaction history vector $H_u \in \mathbb{R}^{1 \times N}$, the aggregated user preference vector $U_p$ in the joint space is:

$$U_p = H_u \cdot E$$

The Stage 1 similarity score across all items is calculated via Cosine Similarity:

$$\text{Score}_{\text{early}}(u, i) = \frac{U_p \cdot E_i^T}{\|U_p\|_2 \cdot \|E_i\|_2}$$

### 3. Min-Max Score Normalization & Late Fusion Layer (Stage 2)
To align score magnitudes before final ensembling:

$$\text{Score}_{\text{norm}}(s) = \frac{s - \min(s)}{\max(s) - \min(s) + \epsilon}$$

The final hybrid rank score combines Stage 1 Early Fusion and Pure Collaborative Filtering using parameter $\beta \in [0, 1]$:

$$\text{Score}_{\text{final}}(u, i) = \beta \cdot \text{Score}_{\text{cf\_norm}}(u, i) + (1 - \beta) \cdot \text{Score}_{\text{early\_norm}}(u, i)$$