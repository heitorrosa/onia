# Multi-Scale Structural Proposals (mAP)
**Difficulty Level:** Phase 4 (Extreme) | **Time Limit:** 4 Hours

## 1. Background / Scenario
Object detectors like YOLO or SSD use different grid scales to detect objects of varying sizes. In this challenge, you will implement the **Mean Average Precision (mAP)** calculation metric from scratch using the [Global Wheat Head Detection Dataset](https://www.kaggle.com/c/global-wheat-detection).

## 2. Problem Statement
1.  **IoU Calculation**: Write a vectorized function to calculate the Intersection over Union between two sets of bounding boxes.
2.  **Precision-Recall Curve**: Generate the curve for a specific IoU threshold (e.g., 0.5).
3.  **mAP Logic**: Calculate the Area Under the Curve (AUC) for IoU thresholds from 0.5 to 0.95.

## 3. Dataset Description
**Reference:** [Wheat Detection](https://www.kaggle.com/c/global-wheat-detection)
- **Features:** Images of wheat heads with CSV bounding boxes.

## 4. Subtasks & Point Distribution (100 Points)
*   **Task 4.1: Vectorized IoU (30 pts):** Handle $(N, M)$ box comparisons without loops.
*   **Task 4.2: NMS (Non-Maximum Suppression) (30 pts):** Implement the algorithm to prune overlapping boxes.
*   **Task 4.3: mAP@0.5 Estimation (30 pts):** Correct logic for Average Precision per class.
*   **Task 4.4: Box Visualization (10 pts):** Render the ground truth boxes in green and predicted boxes in red.

## 5. Constraints & Technical Rules
- **Manual Metric:** You must not use `torchvision.ops.box_iou`. You must implement the coordinate math yourself.
- **Language:** `NumPy` or `PyTorch`.
