# Unsupervised Density-Based Clustering (DBSCAN)
**Difficulty Level:** Phase 1 (Unsupervised Learning) | **Time Limit:** 2 Hours

## 1. Background / Scenario
Traditional clustering algorithms like K-Means struggle with clusters of arbitrary shapes and are highly sensitive to noise/outliers. DBSCAN (Density-Based Spatial Clustering of Applications with Noise) identifies clusters based on the density of data points, allowing it to discover clusters of any shape and automatically flag outliers as noise.

## 2. Problem Statement
Implement a DBSCAN clustering pipeline to segment customers or identify structural patterns in a non-linear dataset. You must optimize the `eps` (epsilon) and `min_samples` parameters using a K-Distance graph.

## 3. Dataset Description
The [Customer Segmentation Data](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python) from Kaggle.
* **Features:** Annual Income (k$), Spending Score (1-100).
* **Objective:** Group customers without prior labels and identify "noise" points (outliers).

## 4. Subtasks & Point Distribution
* **Task 4.1: Data Scaling (20 pts):** Scale the numerical features using `StandardScaler`. Density-based algorithms are highly sensitive to feature magnitude.
* **Task 4.2: Epsilon Optimization (30 pts):** Direct calculation: Use `NearestNeighbors` to compute the average distance to the *k* nearest neighbors. Plot the distances and identify the "elbow" to select the optimal `eps`.
* **Task 4.3: Model Implementation (30 pts):** Implement Scikit-Learn's `DBSCAN`. Retrieve the total number of clusters and the count of noise points (label `-1`).
* **Task 4.4: Visual Validation (20 pts):** Create a scatter plot colored by cluster ID. Clearly highlight the noise points in a distinct color (e.g., black or red).

## 5. Constraints & Technical Rules
* **No Manual Labels:** You must not use the "Gender" or "Age" labels for clustering; focus purely on the income/spending density.
* **Noise Handling:** Your script must explicitly report the percentage of data points classified as noise.

## 6. Evaluation Criteria
* Successful identification of at least 4 distinct density clusters.
* Correct use of the K-Distance elbow method for `eps` selection.
* Silhouette Score (excluding noise) > 0.45.

## 7. Deliverables
* `dbscan_segmentation.py`: The Python script.
* `k_distance_elbow.png`: The plot used to find `eps`.
* `clusters_visualization.png`: The final clustered scatter plot.
