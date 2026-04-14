# Activation Gradients & Loss Surface Mapping
**Difficulty Level:** Phase 4 (Extreme) | **Time Limit:** 4 Hours

## 1. Background / Scenario
Why does a model stop learning? In this advanced exercise, you'll use the [Spambase Dataset](https://archive.ics.uci.edu/dataset/94/spambase) to visualize the **Vanishing Gradient** problem and map the loss surface of a Deep MLP.

## 2. Problem Statement
1.  **Gradient Tracking**: Plot the norm of the gradients per layer during training.
2.  **Activation Distribution**: Use histograms to see if activations are saturating (e.g., Sigmoid outputs staying at 0 or 1).
3.  **Loss Surface Contours**: Use a simple 2-parameter slice to visualize the "bowl" or "valley" the optimizer is navigating.

## 3. Dataset Description
**Reference:** [UCI Spambase](https://archive.ics.uci.edu/dataset/94/spambase)
- **Features:** 57 numeric properties of emails.
- **Challenge:** Deep networks on this data often suffer from saturation.

## 4. Subtasks & Point Distribution (100 Points)
*   **Task 4.1: Gradient Norm Monitoring (30 pts):** Record `weight.grad.norm()` per layer.
*   **Task 4.2: Activation Saturation Analysis (30 pts):** Identify which layers "die" (zero gradients).
*   **Task 4.3: Loss Surface Visualization (30 pts):** Perturb two weights by $\pm \epsilon$ and plot the resulting loss.
*   **Task 4.4: BatchNorm/ReLU Mitigation (10 pts):** Show how adding `BatchNorm` fixes the saturation.

## 5. Constraints & Technical Rules
- **Visualization:** Must use `Matplotlib` or `Seaborn` for the gradient histograms.
- **Framework:** `PyTorch`.
