# Hidden Layer Ablation Analysis
**Difficulty Level:** Phase 2 (Deep Learning) | **Time Limit:** 2 Hours

## 1. Background / Scenario
Adding hidden layers doesn't always improve model capacity; it can lead to "Dead Neurons" if not properly managed. You must perform an "Ablation Study" to mathematically find the optimal depth of an MLP required to solve a problem.

## 2. Problem Statement
Implement a custom PyTorch <code>nn.Module</code> with a variable number of layers. You must systematically compare a 1-layer, 3-layer, and 5-layer MLP on identical classification tasks.

## 3. Dataset Description
The [Synthetic Moons](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html) will be used to analyze how layer depth affects the complexity of binary classification.

## 4. Subtasks & Point Distribution
* **Task 4.1: Architecture with Variable Layers (30 pts):** Create a Loop that initializes 3 versions of an MLP: one with 1 hidden layer (64 units), one with 3 (64x3), and one with 5 (64x5).
* **Task 4.2: Capacity vs Overfitting (40 pts):** Train all three models for 50 epochs on the same training split and record the training vs validation error for each.
* **Task 4.3: Model Complexity Report (30 pts):** Find the "Sweet Spot" (the minimum number of layers that achieves > 95% accuracy) and justify it based on the Bias-Variance tradeoff.

## 5. Constraints & Technical Rules
* Strictly forbidden: Higher-level cross-validation tools like <code>KFold</code>. All training and validation must be implemented via PyTorch and NumPy manually.

## 6. Evaluation Criteria
* Successful ablation plot.
* Finding the optimal layer count for the moons challenge.

## 7. Deliverables
* <code>ablation_mlp.py</code>: The PyTorch variable architecture and training script.
* <code>depth_comparison.png</code>: Chart showing training/val error vs. layer count.
