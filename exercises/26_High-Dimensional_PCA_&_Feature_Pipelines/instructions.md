# High-Dimensional PCA & Feature Pipelines
**Difficulty Level:** Phase 2 (Intermediate) | **Time Limit:** 2 Hours

## 1. Background / Scenario
Principal Component Analysis (PCA) is the core tool for linear dimensionality reduction. Using the [Arcene Dataset](https://archive.ics.uci.edu/dataset/167/arcene) (which has 10,000 features per sample) or the similar [Gene Expression Cancer RNA-Seq Dataset](https://www.kaggle.com/datasets/ahmedmoorsy/gene-expression-cancer-rnaseq), you will project massive feature spaces into informative low-rank approximations.

## 2. Problem Statement
Implement a dimensionality reduction pipeline that:
1.  **Analyzes Explained Variance**: Plot the cumulative explained variance vs. the number of principal components.
2.  **Identifies the Rank**: Determine the minimum number of components needed to retain 90% and 99% of the original data's information.
3.  **Visualization**: Project the 10,000-D data into 2D and 3D space to see if cancer types are separable.

## 3. Dataset Description
**Reference:** [Gene Expression Dataset](https://archive.ics.uci.edu/dataset/401/gene+expression+cancer+rna+seq)
- **Features:** 10,000 - 20,531 gene expression levels (numeric).
- **Target:** Cancer type (BRCA, KIRC, LUAD, etc.).
- **Challenge:** Avoid "Over-fitting the PCA" on sample size smaller than feature count ($n < d$).

## 4. Subtasks & Point Distribution (100 Points)
*   **Task 4.1: High-Rank Scaling (20 pts):** Scale the genes using `StandardScaler` to ensure PCA isn't biased by high-expression genes.
*   **Task 4.2: PCA Explained Variance Analysis (40 pts):**
    -   Train `PCA(n_components=100)`.
    -   Plot the `explained_variance_ratio_` and find the "knee" in the scree plot.
*   **Task 4.3: Low-Rank Classifier Pipeline (30 pts):**
    -   Combine `PCA` (50 components) and `SVC` inside a `Pipeline`.
    -   Compare its accuracy against a raw `SVC` on the 10,000 features (observe the speed difference).
*   **Task 4.4: Result Plotting (10 pts):**
    -   Create a color-labeled 2D scatter plot using the first 2 principal components.

## 5. Constraints & Technical Rules
- **Library:** `scikit-learn.decomposition.PCA`.
- **Scaling:** PCA **requires** zero-centered and scaled features to work as intended across disparate dimensions.
