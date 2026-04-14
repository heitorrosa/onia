# Iterative Gradient Boosting & Custom Losses
**Difficulty Level:** Phase 3 (Advanced) | **Time Limit:** 3 Hours

## 1. Background / Scenario
Standard MSE (Mean Squared Error) might not define business success. Using the [Store Item Demand Forecasting Challenge](https://www.kaggle.com/c/demand-forecasting-kernels-only/data), you will implement Gradient Boosting with a custom loss function that punishes *Under-prediction* more severely than *Over-prediction* (as running out of stock is more expensive).

## 2. Problem Statement
Build a Gradient Boosting Regressor (using LightGBM or XGBoost) that:
1.  **Implements a Custom Loss**: Define an asymmetric loss function $L(y, \hat{y})$.
2.  **Iterative Refinement**: Use `early_stopping_rounds` to prevent the boosting process from overfitting.
3.  **Feature Lagging**: Create time-series features (Rolling Mean, Lagged demand).

## 3. Dataset Description
**Reference:** [Store Item Demand](https://www.kaggle.com/c/demand-forecasting-kernels-only)
- **Features:** Date, store, item.
- **Target:** Sales.
- **Challenge:** High seasonal cycles and multi-store variance.

## 4. Subtasks & Point Distribution (100 Points)
*   **Task 4.1: Custom Objective Function (40 pts):**
    -   Write a Python function for the first and second-order derivatives (gradient and hessian) of an asymmetric loss.
*   **Task 4.2: Time-Series Feature Engineering (20 pts):**
    -   Generate 7-day and 30-day moving averages of sales per item.
*   **Task 4.3: Model Training with Early Stopping (30 pts):**
    -   Train the model using a validation set to monitor the custom loss.
*   **Task 4.4: Business Impact Report (10 pts):**
    -   Show how the model results in fewer "Out-of-Stock" scenarios compared to a standard MSE-based model.

## 5. Constraints & Technical Rules
- **Boosting Library:** Use `XGBoost` or `LightGBM`.
- **Validation:** You must use a **Time-Based Split** (e.g., train on 2013-2016, validate on 2017) instead of random shuffling.
