# Iterative Gradient Boosting & Custom Losses
**Difficulty Level:** Phase 4 (Intermediate) | **Time Limit:** 3 Hours

## 1. Background / Scenario
You are a lead AI architect deployed in a complex environment. For a medical diagnostics pipeline using Heart Disease data, the cost of false negatives is extreme. Construct a <code>GradientBoostingClassifier</code> utilizing <code>staged_predict()</code>. You must formulate an approach to wrap the objective function to penalize false negatives severely, combined with an epoch-based early stopping loop. To succeed, you must build algorithmic solutions that operate under strict mathematical constraints, ensuring no data leakage and optimal spatial mappings.

## 2. Problem Statement
The objective is to implement the exact structural mechanisms described:
For a medical diagnostics pipeline using Heart Disease data, the cost of false negatives is extreme. Construct a <code>GradientBoostingClassifier</code> utilizing <code>staged_predict()</code>. You must formulate an approach to wrap the objective function to penalize false negatives severely, combined with an epoch-based early stopping loop.
Generic approaches will fail. You must construct robust operational algorithms that address specific parameter expectations, exploiting structural bounds and rigorous mathematical validation.

## 3. Dataset Description
**Reference:** [Heart Disease](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)

The dataset contains explicit feature properties and schema quirks. You must accommodate high-dimensional matrices, potential zero-inflated arrays in target columns, and severely imbalanced distribution boundaries. Assume missing topologies require mathematical imputation.

## 4. Subtasks & Point Distribution
* **Task 4.1: Structural Preprocessing (30 pts):** Define precise matrix manipulations and preprocessing pipelines. Establish robust feature extraction sequences.
* **Task 4.2: Algorithmic Architecture (40 pts):** Detail the exact modeling layers and hyperparameter bounds. Construct explicit equations and network topologies/pipelines as requested.
* **Task 4.3: Metric Validation Integration (30 pts):** Explain the rigid validation and scoring constraint integration. Implement customized mathematical bounds validating inference decay and performance explicitly.

## 5. Constraints & Technical Rules
* **Libraries:** You are strictly permitted to deploy Pandas, Scikit-Learn, NumPy, and PyTorch (if specified). External high-level APIs bypassing the fundamental logic are forbidden.
* **Execution:** Must complete matrix transformations under 15 minutes utilizing strictly vectorized mathematical operators.
* **Scikit-Learn, XGBoost & Tooling Constraints (CRITICAL):** The implementation MUST systematically feature arrays scaling with `Pipeline`, `ColumnTransformer`, custom loss overrides, or advanced structural metrics like `warm_start=True` or `staged_predict()`. Epochs and partial fitting must be tracked structurally using NumPy and Pandas properties.

## 6. Evaluation Criteria
Your pipeline will be evaluated on penalized F1-weighted variance, bounded Mean Absolute Error via logarithm mapping, and execution footprint. Automated tests will analyze validation leakage implicitly via isolated holdout matrices.

## 7. Deliverables
* `pipeline.py`: Containing the explicit mathematical classes and topologies.
* `inference.ipynb`: A step-by-step analytical vector mapping breakdown demonstrating the matrix logic cleanly.
