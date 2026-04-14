# Multi-Dimensional Housing MLP
**Difficulty Level:** Phase 2 (Deep Learning) | **Time Limit:** 2 Hours

## 1. Background / Scenario
Estimating housing costs based on dense tabular metadata requires more than linear regression. You must build a Deep Multi-Layer Perceptron (MLP) to capture non-linear interactions across geographical and industrial features.

## 2. Problem Statement
Implement a custom PyTorch <code>nn.Module</code> consisting of 3 input-to-hidden linear layers. You must successfully map 8 continuous inputs to a single price prediction output without using high-level CNN/Sequential wrappers.

## 3. Dataset Description
The [California Housing](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html) dataset will be used. Every row contains coordinates and economic stats for a housing block.

## 4. Subtasks & Point Distribution
* **Task 4.1: Linear Stack Implementation (30 pts):** Define a network with hidden dimensions (64, 32, 16). Each layer must be followed by a <code>nn.ReLU</code> activation.
* **Task 4.2: MSE Optimization Loop (40 pts):** Write a training loop using <code>torch.optim.Adam</code> to minimize Mean Squared Error. You must normalize the input using <code>StandardScaler</code> from Scikit-Learn before passing it to the PyTorch tensors.
* **Task 4.3: Absolute Error Report (30 pts):** Reach a Mean Absolute Error (MAE) of less than 0.55 on the test set.

## 5. Constraints & Technical Rules
* Strictly forbidden: Use of <code>nn.Conv2d</code> or pre-made regression models. 
* All weights must be initialized using the PyTorch default (Xavier-like).

## 6. Evaluation Criteria
* Successful training curve convergence.
* Accuracy vs Epochs plot for regression stability.

## 7. Deliverables
* <code>housing_mlp.py</code>: The PyTorch architecture and training logic.
