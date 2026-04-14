# SVM Kernel Subspace Projections
**Difficulty Level:** Phase 3 (Advanced) | **Time Limit:** 3 Hours

## 1. Background / Scenario
The "Kernel Trick" allows Support Vector Machines (SVM) to find linear separation in high-dimensional projections. Using the [Smartphone-Based Recognition of Human Activities](https://academic.oup.com/bioinformatics/article/35/14/i473/5531383) dataset (or similar 561-feature sensor data), you will classify human behavior (walking, sitting, standing).

## 2. Problem Statement
Implement an SVM classifier that:
1.  **Iteratively Projects**: Compare Linear, Polynomial (degree 3), and RBF kernels.
2.  **Kernel Correlation Analysis**: Plot the SVC decision boundaries for the two most important features (e.g., body acceleration vs. gravity).
3.  **Regularization (C) and Gamma Balancing**: Find the "Goldilocks" zone where the model generalizes perfectly without overfitting specific motion transients.

## 3. Dataset Description
**Reference:** [UCI HAR Dataset](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)
- **Features:** 561 numeric attributes from accelerometer and gyroscope sensors.
- **Target:** Activity (6 classes).
- **Challenge:** Massive feature space (high d) requires effective dimensionality scaling.

## 4. Subtasks & Point Distribution (100 Points)
*   **Task 4.1: Feature Correlation Heatmap (20 pts):** Identify redundant sensor features that should be filtered before SVM training.
*   **Task 4.2: Kernel Benchmark (40 pts):**
    -   Train 3 SVC models (Linear, Poly, RBF).
    -   Report accuracy and training time for each.
*   **Task 4.3: Decoupling Hyperparameters (30 pts):**
    -   GridSearch `C` (penalty) and `gamma` (kernel width) for the RBF kernel.
*   **Task 4.4: Error Breakdown (10 pts):**
    -   Which activities (e.g., Sitting vs Standing) are confusing the RBF kernel?

## 5. Constraints & Technical Rules
- **Dimensionality:** Do not use PCA yet; let the SVM kernels handle the 561 features directly to observe performance.
- **Standardization:** Mandatory `StandardScaler` to ensure all sensor readings are on the same magnitude.
