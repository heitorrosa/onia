# Temporal Sequence Alignment (DTW)
**Difficulty Level:** Phase 3 (Advanced) | **Time Limit:** 2 Hours

## 1. Background / Scenario
Comparing two time-series is difficult if they have different speeds (e.g., two people saying the same word at different rates). **Dynamic Time Warping (DTW)** finds the optimal non-linear alignment. You will implement it for [Sign Language Gesture Recognition](https://www.kaggle.com/datasets/birdy654/sign-language-gesture-recognition).

## 2. Problem Statement
1.  **Cost Matrix**: Create a matrix $D(i, j)$ of distances between every point in Series A and Series B.
2.  **Cumulative Distance**: Use dynamic programming: $D(i, j) = cost(a_i, b_j) + \min(D_{i-1,j}, D_{i,j-1}, D_{i-1,j-1})$.
3.  **Warping Path**: Trace the path from $(0,0)$ to $(N,M)$ that minimizes total distance.

## 3. Dataset Description
**Reference:** [Sign Language Gestures](https://www.kaggle.com/datasets/birdy654/sign-language-gesture-recognition).

## 4. Subtasks & Point Distribution (100 Points)
*   **Task 4.1: Euclidean Distance Matrix (30 pts):** Vectorized calculation.
*   **Task 4.2: DP Table Filling (40 pts):** Loop-based alignment logic.
*   **Task 4.3: Path Visualization (20 pts):** Plot the "Warping Path" over the cost matrix.
*   **Task 4.4: Gesture Comparison (10 pts):** Classify a new gesture based on its DTW proximity to a known prototype.

## 5. Constraints & Technical Rules
- **No FastDTW:** You must implement the $O(N \times M)$ dynamic programming table yourself.
- **Language:** Python / NumPy.
