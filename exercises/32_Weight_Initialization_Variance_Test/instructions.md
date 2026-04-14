# Weight Initialization Variance Test
**Difficulty Level:** Phase 2 (Deep Learning) | **Time Limit:** 2 Hours

## 1. Background / Scenario
Initial weights dictate the "Starting Position" of your optimization landscape. If weights are too large, activations explode; if too small, they vanish. You must implement custom <code>weight_init</code> functions to compare zero-init, random-init, and Xavier-init.

## 2. Problem Statement
Implement a custom PyTorch <code>nn.Module</code> consisting of a 4-layer MLP. You must manually initialize the weights using <code>torch.nn.init</code> methods before every training run to find the most stable starting point.

## 3. Dataset Description
The [California Housing](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html) dataset will be used to test initialization stability.

## 4. Subtasks & Point Distribution
* **Task 4.1: Manual Weight Overrides (30 pts):** Write a <code>weights_init(m)</code> function that uses <code>nn.init.xavier_normal_</code> for <code>nn.Linear</code> layers. Then, replace it with <code>nn.init.zeros_</code> to observe the failure of the gradient.
* **Task 4.2: Variance Monitoring (40 pts):** Print the Mean and Standard Deviation of the activations (output of each layer) before training. You represent stable initial activation variance (1.0 $\pm$ 0.1).
* **Task 4.3: Initialization Stability Plot (30 pts):** Graph the training loss curves ($y=Loss, x=Epochs$) for Zero-init, Random-init, and Xavier-init on the same chart.

## 5. Constraints & Technical Rules
* Strictly forbidden: Higher-level initializers inside <code>nn.Sequential</code> (use <code>nn.Module.apply()</code>).

## 6. Evaluation Criteria
* Reach accuracy on normalized California Housing within 10 epochs.
* Proving that zero-initialization fails to learn (loss stays constant).

## 7. Deliverables
* <code>weight_init.py</code>: The PyTorch script and manual weight overrides.
* <code>init_comparison.png</code>: Chart comparing three initialization methods.
