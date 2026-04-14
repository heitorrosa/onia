# Lazy Learning Matrix Validations (KNN)
**Difficulty Level:** Phase 1 (Foundational) | **Time Limit:** 1.5 Hours

## 1. Background / Scenario
K-Nearest Neighbors (KNN) is the quintessential "lazy learner," requiring no explicit training phase but substantial memory during inference. Using the [Iris Dataset](https://www.kaggle.com/datasets/uciml/iris) (or a similar flower classification), you will build a classification model and analyze its decision boundaries.

## 2. Problem Statement
Implement a KNN classifier that:
1.  **Analyzes the "K" Influence**: Plot training error vs. validation error as $K$ increases from 1 to 50.
2.  **Analyzes Distance Metrics**: Compare Euclidean, Manhattan, and Minkowski distances.
3.  **Analyzes the Curse of Dimensionality**: Observe how KNN's accuracy declines as you add non-informative (noise) features.

## 3. Dataset Description
**Reference:** [Iris Species](https://www.kaggle.com/datasets/uciml/iris)
- **Features:** Sepal length, sepal width, petal length, petal width.
- **Target:** Species (Setosa, Versicolor, Virginica).
- **Challenge:** Two classes (Versicolor and Virginica) are subtly overlapped, requiring precise distance boundaries.

## 4. Subtasks & Point Distribution (100 Points)
*   **Task 4.1: Feature Correlation Heatmap (20 pts):** Identify redundant features (e.g., Petal Length is highly correlated with Petal Width).
*   **Task 4.2: KNN Pipeline (40 pts):**
    -   Create a `Pipeline` that includes `StandardScaler` and `KNeighborsClassifier`.
    -   Compare performance with and without scaling.
*   **Task 4.3: Elbow Method for K (30 pts):**
    -   Generate a plot of $K$ vs Error. Where does the bias-variance tradeoff peak?
*   **Task 4.4: Decision Boundary Viz (10 pts):**
    -   Render a 2D plot of the decision regions using the two most important features.

## 5. Constraints & Technical Rules
- **Scaling:** You must explain why KNN is sensitive to feature magnitude.
- **Complexity:** Use `GridSearchCV` to find the best distance metric and K simultaneously.
