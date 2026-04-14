# Learning Rate Step-Down MLP
**Difficulty Level:** Phase 2 (Deep Learning) | **Time Limit:** 2 Hours

## 1. Background / Scenario
Constant Learning Rates (LR) often "Bounce" around the global minimum without ever reaching the exact bottom. You must implement a "Learning Rate Scheduler" that shrinks the LR size as the model approaches convergence.

## 2. Problem Statement
Implement a custom PyTorch <code>nn.Module</code> with a 3-layer MLP. You must use <code>torch.optim.lr_scheduler</code> to dynamically decrease the LR during training.

## 3. Dataset Description
The [Synthetic Moons](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html) will be used to analyze LR schedule stability.

## 4. Subtasks & Point Distribution
* **Task 4.1: Static vs Dynamic LR (30 pts):** Create a 3-layer MLP (hidden: 64, 32). Train one with a constant LR=0.1 and another one with an <code>exp_decay</code> LR using <code>StepLR</code>.
* **Task 4.2: Step-Down Configuration (40 pts):** Initialize a <code>torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)</code>. You must reduce the LR by 10x every 10 epochs.
* **Task 4.3: Convergence Precision (30 pts):** Verify that the Step-Down model reaches a lower loss floor than the Constant-LR model.

## 5. Constraints & Technical Rules
* Strictly forbidden: Using the <code>ReduceLROnPlateau</code> scheduler (you must use the simpler <code>StepLR</code>).

## 6. Evaluation Criteria
* Reach 0.98 Accuracy.
* Final loss comparison chart.

## 7. Deliverables
* <code>lr_scheduler_mlp.py</code>: The PyTorch script and Step-Down scheduler.
* <code>loss_floor_comparison.png</code>: Chart showing how decay improves precision.
