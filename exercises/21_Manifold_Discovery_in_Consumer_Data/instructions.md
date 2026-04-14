# Manifold Discovery in Consumer Data
**Difficulty Level:** Phase 3 (Advanced) | **Time Limit:** 2.5 Hours

## 1. Background / Scenario
t-SNE (t-Distributed Stochastic Neighbor Embedding) is an unsupervised technique that visualizes high-dimensional datasets by projecting them into 2D or 3D while preserving local similarities. In this exercise, you'll use the [Mall Customer Segmentation Dataset](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python) to find hidden clusters of consumer behavior.

## 2. Problem Statement
Perform unsupervised manifold discovery to segment customers. You will:
1.  **Reduce Dimensions with PCA**: First compress the data linearly to 50% variance.
2.  **Project with t-SNE**: Use t-SNE (or UMAP) to create a scatter plot of the clusters.
3.  **Perplexity Analysis**: Experiment with the 'perplexity' parameter to see how it affects local vs global structure.

## 3. Dataset Description
**Reference:** [Mall Customers](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)
- **Features:** Age, Annual Income (k$), Spending Score (1-100).
- **Target:** None (Unsupervised).
- **Challenge:** Detecting meaningful patterns without explicit labels.

## 4. Subtasks & Point Distribution (100 Points)
*   **Task 4.1: Cluster Feature Engineering (20 pts):** Normalizing the "Annual Income" and "Spending Score" to ensure they receive equal weight in the distance matrix.
*   **Task 4.2: t-SNE Projection (40 pts):**
    -   Apply `TSNE` to the data and create a 2D scatter plot.
    -   Analyze how many distinct "islands" of customers emerge.
*   **Task 4.3: Perplexity Experiments (30 pts):**
    -   Generate 3 plots with perlexity values 5, 30, and 100.
    -   Explain how higher perplexity captures more global structure.
*   **Task 4.4: Qualitative Analysis (10 pts):**
    -   Check if the clusters in t-SNE space correlate with "Annual Income" vs "Spending Score" quadrants in the original data.

## 5. Constraints & Technical Rules
- **Execution:** Since t-SNE is computationally expensive, use `PCA` beforehand to speed up the process.
- **Library:** `scikit-learn.manifold.TSNE`.
