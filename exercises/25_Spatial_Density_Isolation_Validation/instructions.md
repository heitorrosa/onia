# Spatial Density Isolation Validation
**Difficulty Level:** Phase 2 (Intermediate) | **Time Limit:** 2 Hours

## 1. Background / Scenario
Anomaly detection in spatial data often requires isolating points that exist in low-density "islands." Using the [Maritime Objects Pathing (AIS Data)](https://www.kaggle.com/datasets/maritime-objects-detection) or similar location data, you'll identify unusual vessel behaviors using **Local Outlier Factor (LOF)**.

## 2. Problem Statement
Implement a spatial anomaly detector that:
1.  **Calculates Local Density**: Use `LocalOutlierFactor` to compare the local density of a point with its neighbors.
2.  **Identifies Deviations**: Flag points that have a significantly lower density than their surrounding cluster as "anomalies" or "rogue vessels."
3.  **Impact of N-Neighbors**: Experiment with $n\_neighbors$ (10 vs 50) to see how it shifts the sensitivity to outliers.

## 3. Dataset Description
**Reference:** [Maritime Data / Ship Paths](https://www.kaggle.com/datasets/maritime-objects-detection)
- **Features:** Latitude, Longitude, Heading, Speed.
- **Target:** Potential Anomaly (None, Unsupervised).
- **Challenge:** Distinguishing between normal navigation and unusual spatial pivots.

## 4. Subtasks & Point Distribution (100 Points)
*   **Task 4.1: Coordinate Normalization (20 pts):** Use `StandardScaler` on Lat/Long as LOF is distance-based.
*   **Task 4.2: LOF Fitting (40 pts):**
    -   Compute the "negative outlier factor" for each vessel coordinate.
*   **Task 4.3: Novelty Detection Comparison (30 pts):**
    -   Compare the "Density-Based" LOF against an "Isolation-Based" Isolation Forest.
*   **Task 4.4: Result Interpretation (10 pts):**
    -   Visualize the anomalies on a scatter plot; label the "densest" areas vs "outliest" areas.

## 5. Constraints & Technical Rules
- **Metric:** You must use `minkowski` distance with $p=2$ (Euclidean).
- **Library:** `scikit-learn.neighbors.LocalOutlierFactor`.
