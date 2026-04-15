# XGBoost Hardware Awareness & Sparsity
**Difficulty Level:** Phase 3 (Advanced) | **Time Limit:** 2.5 Hours

## 1. Background / Scenario
Speed and efficiency are crucial in production. In this exercise, you'll use the [Safe Driver Prediction (Porto Seguro)](https://www.kaggle.com/c/porto-seguro-safe-driver-prediction) dataset, which is large, sparse, and imbalanced. You will optimize XGBoost to leverage hardware threading and internal sparsity handling.

## 2. Problem Statement
Maximize the **Gini Coefficient** for insurance claim prediction. You must:
1.  **Configure XGBoost for Sparsity**: Use the `missing` parameter to handle nulls natively without imputation.
2.  **Hardware Optimization**: Set `n_jobs` and observe the CPU utilization.
3.  **Regularized Boosting**: Apply `gamma` (minimum loss reduction) and `lambda` (L2) to keep trees shallow but effective.

## 3. Dataset Description
**Reference:** [Safe Driver Prediction](https://www.kaggle.com/c/porto-seguro-safe-driver-prediction)
- **Features:** Grouped as binary, categorical, and numeric (masked names).
- **Target:** `target=1` if a claim was filed.
- **Challenge:** Data is 96% "negative" class; requires high precision.

## 4. Subtasks & Point Distribution (100 Points)
*   **Task 4.1: Native Sparsity Handling (20 pts):** 
    -   Avoid `SimpleImputer`. Pass the raw data with NaNs to `DMatrix` and specify the missing value placeholder.
*   **Task 4.2: Hyperparameter Tuning for Imbalance (30 pts):**
    -   Tune `scale_pos_weight` to reflect the class distribution.
*   **Task 4.3: Regularization for High-Dimensional Sparse Data (40 pts):**
    -   Use `alpha` (L1) and `max_delta_step` to stabilize updates in an imbalanced setting.
*   **Task 4.4: Performance Profiling (10 pts):**
    -   Record the time taken to train with 1 thread vs. 8 threads.

## 5. Constraints & Technical Rules
- **Library:** Strictly `XGBoost`.
- **Primary Metric:** Normalized Gini Coefficient (requires a custom evaluation function in some versions).