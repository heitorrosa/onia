# Hierarchical Feature Agglomeration
**Difficulty Level:** Phase 3 (Advanced) | **Time Limit:** 3 Hours

## 1. Background / Scenario
Feature Agglomeration uses hierarchical clustering to merge features that are similar, rather than merging samples. Using the [Human Activity Recognition (HAR)](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) dataset with 561 features, you will reduce the feature space by grouping correlated sensor signals.

## 2. Problem Statement
Implement a feature reduction pipeline that:
1.  **Agglomerates Features**: Group the 561 features into 50 clusters of similar behaviors.
2.  **Analyzes Redundancy**: Show how the 561 features are grouped into "meta-features" based on their ward-linkage.
3.  **End-to-End Performance**: Compare a classifier (e.g., Logistic Regression) trained on 561 features vs 50 agglomerated features.

## 3. Dataset Description
**Reference:** [UCI HAR Dataset](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)
- **Features:** 561 sensor measurements.
- **Target:** Action (Walking, Sitting, etc.).
- **Challenge:** High redundancy among sensor channels (e.g., X and Y axis signals are often correlated).

## 4. Subtasks & Point Distribution (100 Points)
*   **Task 4.1: Agglomeration Setup (25 pts):** Configure `FeatureAgglomeration` (n_clusters=50) using the Ward linkage criterion.
*   **Task 4.2: Training Benchmarking (30 pts):**
    -   Train a basic model on the raw 561 features.
    -   Train the same model on the 50 "Agglomerated" features.
*   **Task 4.3: Dendrogram Analysis (35 pts):**
    -   Visualize the relationship between features using a dendrogram (using `scipy.cluster.hierarchy`).
*   **Task 4.4: Feature Interpretation (10 pts):**
    -   Explain how "Agglomeration" differs from "PCA" in terms of preserving original data structure.

## 5. Constraints & Technical Rules
- **Linkage:** You must use `ward` linkage for the agglomeration.
- **Library:** `scikit-learn.cluster.FeatureAgglomeration`.
