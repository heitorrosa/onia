# Probabilistic GMM Density Estimation
**Difficulty Level:** Phase 3 (Advanced) | **Time Limit:** 2.5 Hours

## 1. Background / Scenario
Gaussian Mixture Models (GMM) treat clusters as ellipses with different orientations and densities. Unlike K-Means, which assumes circular clusters, GMM can fit complex, overlapping shapes. Using the [Wine Quality Dataset](https://www.kaggle.com/datasets/yasserh/wine-quality-dataset) (or similar clustered data), you will determine the optimal number of data clusters using the Bayesian Information Criterion (BIC).

## 2. Problem Statement
Implement a probabilistic GMM that:
1.  **Estimates Density**: Predict the log-likelihood of each data point under the fitted GMM.
2.  **Identifies the Optimal K**: Use the `BIC` (Bayesian Information Criterion) and `AIC` (Akaike Information Criterion) to select the number of Gaussian components (1 to 10).
3.  **Covariance Structures**: Compare 'full', 'spherical', and 'diag' covariance types.

## 3. Dataset Description
**Reference:** [Wine Quality](https://www.kaggle.com/datasets/yasserh/wine-quality-dataset)
- **Features:** 11 chemical properties (fixed acidity, volatile acidity, citric acid, etc.).
- **Target:** Quality (Ignore for unsupervised clustering, use for validation).
- **Challenge:** Subtle overlap between wine qualities requires soft (probabilistic) clustering.

## 4. Subtasks & Point Distribution (100 Points)
*   **Task 4.1: Chemical Preprocessing (20 pts):** Standardize all 11 features.
*   **Task 4.2: Probabilistic Fitting (40 pts):**
    -   Train a `GaussianMixture` and use `predict_proba()` to see the probability of a wine belonging to each "style" (cluster).
*   **Task 4.3: BIC/AIC Optimization (30 pts):**
    -   Plot the BIC score vs. number of components. Identify the "minimal" BIC.
*   **Task 4.4: Outlier Detection (10 pts):**
    -   Points with very low log-likelihood are considered "anomalies" or unique wines. Identify the 5 most abnormal wines.

## 5. Constraints & Technical Rules
- **Methodology:** You must prove why you chose the final number of components using a graph.
- **Probabilities:** Explain the difference between "Hard" (Predict) and "Soft" (Predict_Proba) clustering in your report.
