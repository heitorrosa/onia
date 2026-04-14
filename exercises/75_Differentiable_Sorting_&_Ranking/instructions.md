# Differentiable Sorting & Ranking
**Difficulty Level:** Phase 4 (Extreme) | **Time Limit:** 4 Hours

## 1. Background / Scenario
Standard sorting (like `argsort`) is not differentiable. **Neural Sorting** allows gradients to flow through a "Rank" operation. You will implement a differentiable **Sinkhorn-like Operator** to learn rank-based features on the [MSLR-WEB10K Dataset](https://www.kaggle.com/datasets/petezhishuo/mslr-web10k).

## 2. Problem Statement
1.  **Permutation Matrix**: Approximate a discrete sorting operation with a continuous, doubly-stochastic matrix.
2.  **LTR (Learning to Rank)**: Traing a model to minimize the **ListNet** loss (a distribution-based ranking loss).
3.  **End-to-End Ranking**: Optimize for **NDCG** (Normalized Discounted Cumulative Gain) through backpropagation.

## 3. Dataset Description
**Reference:** [Microsoft Learning to Rank](https://www.kaggle.com/datasets/petezhishuo/mslr-web10k)
- **Features:** 136 features per query-document pair.

## 4. Subtasks & Point Distribution (100 Points)
*   **Task 4.1: Permutation Approximation (40 pts):** Implement the Sinkhorn iteration to project an NxN matrix to a permutation-like matrix.
*   **Task 4.2: ListNet Loss (30 pts):** Calculate cross-entropy between the softmax of relevance scores and the softmax of model predictions.
*   **Task 4.3: Batch Gradient Analysis (20 pts):** Show how gradients reach the "input ranking" even through the sorting step.
*   **Task 4.4: NDCG Benchmarking (10 pts):** Compare your neural ranker against a standard XGBoost ranker.

## 5. Constraints & Technical Rules
- **Framework:** `PyTorch`.
- **No Loop Sorting:** You must find a matrix-based approximation.
