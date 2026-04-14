# Binary Health Risk MLP
**Difficulty Level:** Phase 2 (Deep Learning) | **Time Limit:** 2 Hours

## 1. Background / Scenario
Medical diagnostics requires a robust binary classifier that can handle categorical and numerical features alike to predict diseased vs healthy status. You must build a binary-output MLP for disease prediction.

## 2. Problem Statement
Implement a custom PyTorch <code>nn.Module</code> with a single Sigmoid-activated output. You must use <code>BCELoss</code> to train the classification boundaries from health records.

## 3. Dataset Description
The [Heart Disease](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset) dataset will be used, consisting of clinical measurements like cholesterol, age, and blood pressure.

## 4. Subtasks & Point Distribution
* **Task 4.1: Logit Projection (30 pts):** Create a 2-layer MLP (units: 32, 16) with a final <code>nn.Linear(16, 1)</code> layer.
* **Task 4.2: Binary Entropy Optimization (40 pts):** Write a training loop with <code>BCELoss</code> and <code>torch.optim.SGD</code> with momentum set to 0.9.
* **Task 4.3: Decision Threshold Analysis (30 pts):** Visualize the prediction probabilities (0.0 to 1.0) and optimize the threshold for maximum F1-score.

## 5. Constraints & Technical Rules
* Strictly forbidden: Higher-level <code>CrossEntropyLoss</code>. You must use raw <code>BCELoss</code> with a manual Sigmoid in the <code>forward</code> pass or <code>BCEWithLogitsLoss</code>.

## 6. Evaluation Criteria
* Reach at least 0.85 Accuracy on the test split.
* Final Confusion Matrix report.

## 7. Deliverables
* <code>health_mlp.py</code>: The PyTorch binary classifier.
