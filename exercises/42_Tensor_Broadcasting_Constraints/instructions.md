# Tensor Broadcasting Constraints
**Difficulty Level:** Phase 3 (Advanced) | **Time Limit:** 1.5 Hours

## 1. Background / Scenario
Inefficient loop-based calculations are the leading cause of slow AI pipelines. In this exercise, you will master **NumPy/PyTorch Broadcasting** using the [Global Weather Data](https://www.kaggle.com/datasets/smid80/weatherww2) to perform massive spatial normalization without `for` loops.

## 2. Problem Statement
Given a 3D tensor of weather readings (Time, Latitude, Longitude), you must:
1.  **Normalize by Region**: Subtract the mean temperature of each "Zone" using broadcasting.
2.  **Weighted Averaging**: Multiply the tensor by a 2D "Importance Weight" matrix using compatible shapes.
3.  **Vectorized Logic**: Implement a custom distance metric (e.g., Haversine) using pure tensor operations.

## 3. Dataset Description
**Reference:** [Weather in WWII](https://www.kaggle.com/datasets/smid80/weatherww2)
- **Features:** Max Temp, Min Temp, Mean Temp across thousands of stations.

## 4. Subtasks & Point Distribution (100 Points)
*   **Task 4.1: Shape Manipulation (30 pts):** Use `unsqueeze`, `view`, or `np.newaxis` to make tensors broadcast-compatible.
*   **Task 4.2: Vectorized Normalization (40 pts):** Perform `(Data - Mean) / Std` across specific axes without a single loop.
*   **Task 4.3: Computational Speed Test (20 pts):** Compare a loop-based implementation vs your vectorized one using `timeit`.
*   **Task 4.4: Logical Masking (10 pts):** Use boolean indexing (masking) to zero out invalid readings ($Temp < -50$).

## 5. Constraints & Technical Rules
- **No Loops:** Any Python `for` or `while` loop used for tensor computation results in a 50-point penalty.
- **Language:** `NumPy` or `PyTorch`.
