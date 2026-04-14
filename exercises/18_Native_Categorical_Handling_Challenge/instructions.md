# Native Categorical Handling Challenge
**Difficulty Level:** Phase 3 (Advanced) | **Time Limit:** 2.5 Hours

## 1. Background / Scenario
CatBoost and LightGBM have specialized algorithms to handle categorical data natively, often outperforming manual encoding. In this exercise, you'll use the [Ames Housing Dataset](https://www.kaggle.com/datasets/shashankasubrahmanyam/ames-housing-dataset) to compare native handling against One-Hot encoding.

## 2. Problem Statement
Predict property prices by:
1.  **Benchmarking Native Handling**: Pass the list of categorical features directly to `CatBoostRegressor` or `LGBMRegressor` (with `categorical_feature` param).
2.  **Comparison to Pre-Encoded Data**: Implement a `Pipeline` with `OneHotEncoder` + `LinearRegression` and compare RMSE.
3.  **Stability Tests**: Observe how the native handling performs when categories are missing from the training data.

## 3. Dataset Description
**Reference:** [Ames Housing](https://www.kaggle.com/datasets/shashankasubrahmanyam/ames-housing-dataset)
- **Features:** 80 mixture of categorical (Foundation, Neighborhood, Electrical) and numeric.
- **Target:** SalePrice.
- **Challenge:** Features like "Neighborhood" have high cardinality; One-Hot encoding creates hundreds of columns.

## 4. Subtasks & Point Distribution (100 Points)
*   **Task 4.1: Specialized Categorical Indexing (25 pts):** Correctly identify and tell the learner which columns are categorical (e.g., indices for CatBoost).
*   **Task 4.2: Model Selection (40 pts):**
    -   Train a `CatBoostRegressor` using its internal categorical conversion (Permutation-based encoding).
*   **Task 4.3: Feature Importance Analysis (25 pts):**
    -   Which neighborhoods add the most value according to the model?
*   **Task 4.4: Sensitivity Analysis (10 pts):**
    -   Explain why native handling is often more memory-efficient than `OneHotEncoding`.

## 5. Constraints & Technical Rules
- **Boosting Library:** Use `CatBoost` or `LightGBM`.
- **Validation:** Use `mean_squared_log_error` as the housing market prices are usually log-normally distributed.
