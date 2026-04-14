# Real-Time Object Detection (YOLO Custom Head)
**Difficulty Level:** Phase 4 (Extreme) | **Time Limit:** 5 Hours

## 1. Background / Scenario
Object detection in industrial environments requires speed and accuracy. In this challenge, you will implement a **Custom Prediction Head** for a YOLO-like architecture using the [Pothole Detection Dataset](https://www.kaggle.com/datasets/chitholian/pothole-detection). Instead of using a pre-built library, you will architect the tensor output yourself.

## 2. Problem Statement
Design a convolutional neural network that:
1.  **Grid-Based Anchors**: Divides an image into a $7 \times 7$ grid.
2.  **Multitask Head**: Predicts a tensor of shape $(B, 7, 7, 5 + C)$ where $B$ is batch, $5$ represents $[confident, x, y, w, h]$, and $C$ is the class.
3.  **Non-Maximum Suppression (NMS)**: Implement the algorithm to prune overlapping predictions using IoU thresholds.

## 3. Dataset Description
**Reference:** [Pothole Detection](https://www.kaggle.com/datasets/chitholian/pothole-detection)
- **Features:** 640x640 RGB images.
- **Targets:** Boundary box coordinates in YOLO format.

## 4. Subtasks & Point Distribution (100 Points)
*   **Task 4.1: Feature Backbone Integration (20 pts):** Use a pre-trained `RestNet18` or `MobileNetV2` as the feature extractor (removing global pooling).
*   **Task 4.2: YOLO Loss Function (40 pts):**
    -   Implement the coordinate loss (MSE).
    -   Implement the objectness loss (Binary Cross Entropy).
    -   Address the "No-Object" imbalance by weighting the empty grid cells lower.
*   **Task 4.3: Custom Head Architecture (30 pts):** Create the final layer that maps high-level features to the grid-cells.
*   **Task 4.4: Bounding Box Post-processing (10 pts):** Convert grid-relative coordinates back to image-absolute coordinates for visualization.

## 5. Constraints & Technical Rules
- **Logic:** You must implement the IoU-based loss calculation from scratch (no `torchvision.ops` for loss).
- **Framework:** `PyTorch`.
