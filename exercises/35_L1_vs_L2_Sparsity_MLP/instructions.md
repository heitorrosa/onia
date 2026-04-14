# L1 vs L2 Sparsity MLP
**Difficulty Level:** Phase 2 (Deep Learning) | **Time Limit:** 2 Hours

## 1. Background / Scenario
Network Weights represent "Importance." L2 Regularization (Weight Decay) penalizes large weights while L1 Regularization (Lasso) penalizes non-zero weights, effectively creating a "Sparse" network where unimportant features are zeroed. You must compare the sparsity of your network with vs without regularizers.

## 2. Problem Statement
Implement a custom PyTorch <code>nn.Module</code> consisting of a 4-layer MLP and use <code>weight_decay</code> in the optimizer to apply L2 regularization. You must manually implement the L1 penalty in the training loop.

## 3. Dataset Description
The [Heart Disease](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset) dataset will be used with its highly redundant feature set.

## 4. Subtasks & Point Distribution
* **Task 4.1: Weight Decay Configuration (30 pts):** Initialize an <code>Adam</code> optimizer and use <code>weight_decay=1e-4</code> to apply L2 regularization across all layers.
* **Task 4.2: Manual L1 Penalty (40 pts):** In every loop, manually compute the sum of absolute values of your weights: <code>l1_loss = lambda * sum(p.abs().sum() for p in model.parameters())</code> and add it to your <code>BCELoss</code>.
* **Task 4.3: Sparsity Analysis (30 pts):** Visualize the histogram of weights (x=Weight Value, y=Count) and show how L1 creates a large spike at exactly zero.

## 5. Constraints & Technical Rules
* Strictly forbidden: Use of <code>Lasso</code> from Scikit-Learn. All weight penalty logic must be inside the PyTorch backpropagation loop.

## 6. Evaluation Criteria
* Reach 0.82 Accuracy.
* Histogram of weight sparsity with and without L1 showing a higher zero count.

## 7. Deliverables
* <code>regularized_mlp.py</code>: The PyTorch script and sparsity visualization.
* <code>weight_histogram.png</code>: Chart showing weight sparsity comparison.
