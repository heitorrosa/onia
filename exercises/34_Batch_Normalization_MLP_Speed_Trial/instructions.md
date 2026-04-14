# Batch Normalization MLP Speed Trial
**Difficulty Level:** Phase 2 (Deep Learning) | **Time Limit:** 2 Hours

## 1. Background / Scenario
Adding more layers can make training slower and more numerically unstable. You must use "Batch Normalization" (BN) to normalize the activations of each layer to have zero mean and unit variance ($0 \pm 1$) during every batch.

## 2. Problem Statement
Implement a custom PyTorch <code>nn.Module</code> with 5 layers and use <code>nn.BatchNorm1d</code> to speed up convergence and stabilize the gradient.

## 3. Dataset Description
The [California Housing](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html) will be used to test convergence speed.

## 4. Subtasks & Point Distribution
* **Task 4.1: Internal Layer Normalization (30 pts):** Create a 5-layer MLP (hidden: 1024, 512, 256, 128, 64) with <code>nn.BatchNorm1d</code> after each linear layer.
* **Task 4.2: Speed Comparison (40 pts):** Train one model with BN and one without. Record the Number of Epochs required to reach a MAE of 0.60.
* **Task 4.3: Batch Stat Monitoring (30 pts):** Access the <code>running_mean</code> and <code>running_var</code> buffers of the BN layers to verify they have correctly learned the feature distribution.

## 5. Constraints & Technical Rules
* Strictly forbidden: Normalizing the entire dataset beforehand (you must normalize batch-by-batch using BN).

## 6. Evaluation Criteria
* Reach 0.60 MAE in less than 20 epochs.
* Training speed comparison report.

## 7. Deliverables
* <code>batch_norm_mlp.py</code>: The PyTorch script and BN-vs-Non-BN comparison.
* <code>speed_to_convergence.png</code>: Chart comparing number of epochs to reach a fixed accuracy.
