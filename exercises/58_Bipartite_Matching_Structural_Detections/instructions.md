# Bipartite Matching Structural Detections
**Difficulty Level:** Phase 4 (Extreme) | **Time Limit:** 5 Hours

## 1. Background / Scenario
Modern detectors like **DETR** removed the need for NMS by using **Bipartite Matching** (Hungarian Algorithm) to match predictions to ground truths. You will implement the assignment cost logic using the [COCO Common Objects Dataset](https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset) (subset).

## 2. Problem Statement
1.  **Hungarian Matching**: Give $N$ predictions and $M$ targets, find the optimal 1-to-1 mapping.
2.  **Cost Function**: Implement a multi-part cost: $Cost = \alpha \times ClassLoss + \beta \times L1\_BoxLoss + \gamma \times GIoU\_Loss$.
3.  **Transformer Integration**: Explain how the "Object Queries" in a transformer interact with these matches.

## 3. Dataset Description
**Reference:** [COCO 2017](https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset).

## 4. Subtasks & Point Distribution (100 Points)
*   **Task 4.1: Matching Cost Matrix (40 pts):** Generate a matrix of size $(Preds \times Targets)$ containing the cost values.
*   **Task 4.2: Scipy Linear Sum Assignment (20 pts):** Use `scipy.optimize.linear_sum_assignment` to find the indices.
*   **Task 4.3: Loss Accumulation (30 pts):** Calculate total loss only for the matched pairs.
*   **Task 4.4: Conceptual Diagram (10 pts):** Draw or describe the workflow of end-to-end detection without anchors.

## 5. Constraints & Technical Rules
- **Framework:** `PyTorch`.
- **Focus:** This is a mathematical exercise; you can use dummy model outputs to test the matching algorithm.
