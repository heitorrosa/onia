# Support Vector Tube Extrapolations
**Difficulty Level:** Phase 2 (Intermediate) | **Time Limit:** 2 Hours

## 1. Background / Scenario
Support Vector Regression (SVR) uses a "Tube" ($\epsilon$-insensitive zone) where errors are ignored. Using the [Diamond Price Prediction Dataset](https://www.kaggle.com/datasets/shivam2503/diamonds), you will build a regressor that values diamonds while ignoring minor price fluctuations within the margin of error.

## 2. Problem Statement
Implement an SVR model to predict diamond prices. You must:
1.  **Handle Ordinality**: Map 'cut', 'color', and 'clarity' using proper ordinal mappings.
2.  **Optimize the Tube**: Use cross-validation to find the best $\epsilon$ (tube width) and $C$ (regularization).
3.  **Kernel Selection**: Compare Linear vs. RBF (Radial Basis Function) kernels.

## 3. Dataset Description
**Reference:** [Diamonds](https://www.kaggle.com/datasets/shivam2503/diamonds)
- **Features:** Carat, cut, color, clarity, depth, table, x, y, z.
- **Target:** Price.
- **Challenge:** High variance in pricing for large carats; requires robust scaling.

## 4. Subtasks & Point Distribution (100 Points)
*   **Task 4.1: Ordinal & Numeric Pipeline (20 pts):** Use `OrdinalEncoder` for quality features and `StandardScaler` for dimensions/carat.
*   **Task 4.2: SVR Implementation (30 pts):**
    -   Train a `SVR` model with a linear kernel initially.
    -   Explain the geometric meaning of the "Support Vectors" in your price regression.
*   **Task 4.3: Support Vector Analysis (30 pts):**
    -   Identify which diamonds fall *outside* the $\epsilon$-tube (these are the vectors defining the price boundary).
*   **Task 4.4: Nonlinear Kernel Evaluation (20 pts):** Switch to an RBF kernel and analyze the improvement in RMSE.

## 5. Constraints & Technical Rules
- **Scaling:** SVR is distance-sensitive; a `StandardScaler` is **mandatory**.
- **Efficiency:** Use `TruncatedSVD` if the categorical expansions make the matrix too large for your CPU's memory.
