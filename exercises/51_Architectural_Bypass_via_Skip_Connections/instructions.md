# Architectural Bypass via Skip Connections
**Difficulty Level:** Phase 4 (Advanced) | **Time Limit:** 2.5 Hours

## 1. Background / Scenario
Vanishing gradients in very deep networks were solved by ResNet's "Skip Connections" (Identity additive paths). In this exercise, you'll implement a **Residual Block** from scratch to classify the [Breast Cancer Histopathology Images](https://www.kaggle.com/datasets/paultimothymooney/breast-cancer-wisconsin-data).

## 2. Problem Statement
1.  **Residual Block Design**: Implement a module $y = F(x) + x$ where $F(x)$ represents two convolutional layers.
2.  **Downsampling**: Handle the dimension mismatch when input/output sizes differ using a 1x1 convolution in the skip path.
3.  **Deep Stacking**: Build a 10-layer network by stacking these blocks.

## 3. Dataset Description
**Reference:** [Breast Cancer Images](https://www.kaggle.com/datasets/paultimothymooney/breast-cancer-wisconsin-data) (or the CSV variant for simple architecture testing).

## 4. Subtasks & Point Distribution (100 Points)
*   **Task 4.1: Custom ResBlock Module (40 pts):** Proper implementation of the additive skip path.
*   **Task 4.2: Handling Strides (30 pts):** Correct logic for the 1x1 "shortcut" convolution.
*   **Task 4.3: Training Benchmarking (20 pts):** Compare a 10-layer network *with* and *without* skip connections.
*   **Task 4.4: Gradients Flow Map (10 pts):** Visualize the gradient magnitude at the first layer.

## 5. Constraints & Technical Rules
- **Framework:** `PyTorch`.
- **Add Op:** You must use `x + out` (Addition), not `cat` (Concatenation).
