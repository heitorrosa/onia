# Stochastic Dropout Stability MLP
**Difficulty Level:** Phase 2 (Deep Learning) | **Time Limit:** 2 Hours

## 1. Background / Scenario
Deep MLPs are prone to "Memorization" (overfitting) the specific data in the training set rather than generalizes. You must implement a custom <code>Dropout</code> layer that randomly "kills" a percentage of neuron activations to ensure the network builds redundant, robust representations.

## 2. Problem Statement
Implement a custom PyTorch <code>nn.Module</code> consisting of 2 hidden layers with a <code>nn.Dropout(p=0.5)</code> layer inserted between each linear transformation to solve the overfitting problem.

## 3. Dataset Description
The [Synthetic Moons](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html) will be used to visualize the decision boundary complexity with vs without dropout.

## 4. Subtasks & Point Distribution
* **Task 4.1: Dense Network with Dropout (30 pts):** Architect a 4-layer MLP (hidden: 128, 128, 64) with <code>p=0.5</code> dropout before each <code>nn.ReLU</code> activation.
* **Task 4.2: Inference mode (40 pts):** You must manually toggle the model between <code>model.train()</code> and <code>model.eval()</code> during the training/validation loop. You must prove that Dropout is active during training but inactive during inference.
* **Task 4.3: Decisive Boundary Plot (30 pts):** Use <code>matplotlib</code> to plot the decision boundary (contour plot) and prove that dropout results in a "smoother" boundary compared to an overfitted model (without dropout).

## 5. Constraints & Technical Rules
* Strictly forbidden: Higher-level model wrappers. All dropout layers must be explicit <code>nn.Module</code> calls.

## 6. Evaluation Criteria
* Reach 0.95 Accuracy on the moons dataset.
* Visualization of stable boundaries.

## 7. Deliverables
* <code>dropout_mlp.py</code>: The PyTorch architecture and boundary visualization.
* <code>boundary_comparison.png</code>: Chart showing the effect of dropout.
