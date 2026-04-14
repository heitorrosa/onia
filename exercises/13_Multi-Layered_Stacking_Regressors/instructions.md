# Multi-Layered Stacking Regressors
**Difficulty Level:** Phase 3 (Advanced) | **Time Limit:** 3 Hours

## 1. Background / Scenario
Stacking (Stacked Generalization) uses a "Meta-Model" to learn how to combine the predictions of multiple base regressors. In this exercise, you'll use the [Medical Cost Personal Datasets](https://www.kaggle.com/datasets/mirichoi0218/insurance) to predict individual medical costs billed by health insurance.

## 2. Problem Statement
Design a multi-layered regressor that:
1.  **First Layer (Base Learners)**: XGBoost, Random Forest, and Ridge Regression.
2.  **Second Layer (Meta-Learner)**: Use a `Lasso` regression as the "judge" to blend the base learner predictions.
3.  **Out-of-Fold (OOF) Training**: Implement the stacking logic using Scikit-Learn's `StackingRegressor` or a manual cross-val loop.

## 3. Dataset Description
**Reference:** [Insurance Cost](https://www.kaggle.com/datasets/mirichoi0218/insurance)
- **Features:** Age, sex, BMI, children, smoker, region.
- **Target:** Charges.
- **Challenge:** Categorical features (smoker/region) have massive non-linear impacts on the target.

## 4. Subtasks & Point Distribution (100 Points)
*   **Task 4.1: Mixed-Type Preprocessing (20 pts):** Use `ColumnTransformer` to handle `OrdinalEncoder` for 'sex'/'smoker' and `OneHotEncoder` for 'region'.
*   **Task 4.2: Base Learner Benchmarking (25 pts):**
    -   Train and report the MAE (Mean Absolute Error) for each individual model (Ridge, RF, XGB).
*   **Task 4.3: Stacking Architecture (40 pts):**
    -   Configure a `StackingRegressor` with a `final_estimator=LassoCV()`.
    -   Explain how the Meta-Model coefficients indicate which base learner is most reliable.
*   **Task 4.4: Residual Analysis (15 pts):**
    -   Plot the residuals of the stacked model. Identify any systemic bias (e.g., underpredicting high-cost smokers).

## 5. Constraints & Technical Rules
- **Tooling:** You must use `XGBRegressor` as one of the base models.
- **Cross-Validation:** Ensure the `StackingRegressor` uses `cv=5` to prevent the meta-model from overfitting the base-models' training predictions.
