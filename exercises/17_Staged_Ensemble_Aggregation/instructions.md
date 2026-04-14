# Staged Ensemble Aggregation
**Difficulty Level:** Phase 2 (Intermediate) | **Time Limit:** 2 Hours

## 1. Background / Scenario
Ensemble models like `GradientBoostingClassifier` can show the evolution of progress across stages (individual trees). In this exercise, you'll use the [Heart Disease Dataset](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data) to visualize how accuracy improves as more estimators are added.

## 2. Problem Statement
Implement a classification pipeline that:
1.  **Analyzes Staged Predictions**: Loop through `staged_predict()` to plot accuracy vs. number of boosting iterations.
2.  **Regularization Intervention**: Apply `learning_rate` shrinkage (0.1, 0.05, 0.01) and observe how it slows down convergence but leads to better generalization.
3.  **Variable Importance Evolution**: Track how the most important feature changes as the ensemble deepens.

## 3. Dataset Description
**Reference:** [Heart Disease](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data)
- **Features:** Age, sex, cp (chest pain), trestbps (resting blood pressure), chol (cholesterol), etc.
- **Target:** presence (1) or absence (0) of heart disease.
- **Challenge:** Small dataset size makes it easy to overfit with deep ensembles.

## 4. Subtasks & Point Distribution (100 Points)
*   **Task 4.1: Gradient Boosting Pipeline (20 pts):** Create a standard `Pipeline` with `StandardScaler` and `GradientBoostingClassifier`.
*   **Task 4.2: Step-by-Step Accuracy Plotting (40 pts):**
    -   Use `model.staged_predict()` on the test set.
    -   Plot the test error curve across iterations (1 to 100 trees).
*   **Task 4.3: Early Stopping Intervention (30 pts):**
    -   Explain the concept of "Deviance" and use it to find the optimal number of trees before overfitting begins.
*   **Task 4.4: Model Shrinkage Analysis (10 pts):**
    -   Demonstrate the "Shrinkage" effect by comparing error curves for `learning_rate=0.1` vs `learning_rate=0.01`.

## 5. Constraints & Technical Rules
- **Library:** `scikit-learn.ensemble.GradientBoostingClassifier`.
- **Plotting:** Use `matplotlib` to render the stage-by-stage results.
