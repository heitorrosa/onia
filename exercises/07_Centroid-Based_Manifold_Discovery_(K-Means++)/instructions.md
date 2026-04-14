# Centroid-Based Manifold Discovery (K-Means++)
**Difficulty Level:** Phase 1 (Unsupervised Learning) | **Time Limit:** 1.5 Hours

## 1. Background / Scenario
K-Means is the workhorse of unsupervised learning, but standard random initialization can lead to poor local minima. K-Means++ improves this by spreading out the initial centroids, leading to faster convergence and more stable cluster assignments.

## 2. Problem Statement
Perform an end-to-end clustering analysis using K-Means++. You will determine the optimal number of clusters ($K$) using the Elbow Method and Silhouette Analysis, then interpret the resulting centroids.

## 3. Dataset Description
The [Credit Card Dataset for Clustering](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata) from Kaggle.
* **Features:** Balance, Purchases, Credit Limit, Payments, etc.
* **Challenge:** High dimensionality requires careful feature selection or PCA before clustering.

## 4. Subtasks & Point Distribution
* **Task 4.1: Feature Prep & Imputation (20 pts):** Handle missing values (e.g., `MINIMUM_PAYMENTS`) and normalize the data.
* **Task 4.2: The Elbow Method (30 pts):** Run K-Means for $K \in [2, 12]$ and plot the **Inertia** (Within-Cluster Sum of Squares). Identify the optimal $K$.
* **Task 4.3: Silhouette Profiling (20 pts):** Calculate the Silhouette Score for your chosen $K$. Ensure the clusters are well-separated ($> 0.2$ for this complex dataset).
* **Task 4.4: Centroid Interpretation (30 pts):** Inverse-transform the centroids to original scales and describe the "persona" of each cluster (e.g., "The Big Spenders", "The Conservative Savers").

## 5. Constraints & Technical Rules
* **Initialization:** Must explicitly set `init='k-means++'` in the constructor.
* **Dimensionality:** Use only the top 5 most variance-heavy features or apply PCA (n_components=2) for visualization.

## 6. Evaluation Criteria
* Generation of a clear Elbow Plot.
* Logical interpretation of at least 3 distinct customer personas based on centroids.

## 7. Deliverables
* `kmeans_plus_plus.py`: The Python script.
* `elbow_curve.png`: Inertia plot for K selection.
* `customer_personas.txt`: A brief summary of what each cluster represents.
