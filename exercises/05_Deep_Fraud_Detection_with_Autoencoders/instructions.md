# Deep Fraud Detection with Autoencoders
**Difficulty Level:** Phase 4 (Advanced) | **Time Limit:** 4 Hours

## 1. Background / Scenario
You are a lead AI architect deployed in a complex digital banking environment. Fraud transaction volumes are so imbalanced that standard classification logic structurally collapses under false-positive rates. Your objective is to build a PyTorch-based Autoencoder neural network to detect anomalies in highly imbalanced credit card transaction data. By mapping transactions into a lower-dimensional latent space and reconstructing them, fraudulent anomalies will exhibit massive reconstruction loss.

## 2. Problem Statement
The objective is to implement an unsupervised deep learning pipeline. Utilizing PyTorch, architect an Autoencoder neural network to detect anomalies. Leverage high-level PyTorch modules (`nn.Linear`, `nn.Sequential`, etc.) to construct the encoder-decoder topology and compute reconstruction loss via MSE. You must efficiently separate fraudulent anomalous transactions from legitimate distributions by establishing a dynamic thresholding mechanism over the reconstruction losses calculated across a validation split. Bypassing manual from-scratch gradient building is encouraged; deploy PyTorch's native `autograd` and built-in metric mapping.

## 3. Dataset Description
**Reference:** [Credit Card Fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

The dataset contains heavily constrained dimensionality due to PCA-transformed variables (`V1` to `V28`) alongside Time and Amount parameters. It features a severe class imbalance where anomalies account for less than 0.2% of the spatial distribution. 

## 4. Subtasks & Point Distribution
* **Task 4.1: Feature Scaling & DataLoaders (25 pts):** Define precise matrix preprocessing using PyTorch `TensorDataset` and `DataLoader`. Legitimate transactions must be isolated for the training topology, utilizing robust standard scaling.
* **Task 4.2: Autoencoder Architecture (40 pts):** Detail the exact modeling layers using PyTorch `nn.Module`. Construct a multi-layered encoder to a narrow bottleneck and a mirrored decoder, utilizing non-linear activations (e.g., LeakyReLU) avoiding topology collapse.
* **Task 4.3: Reconstruction Validation & Thresholding (35 pts):** Compute reconstruction MSE on validation tensors. Programmatically extract the 95th percentile reconstruction loss of legitimate data to serve as the anomaly threshold, computing precise Recall and Precision scores.

## 5. Constraints & Technical Rules
* **Libraries:** You are strictly permitted to deploy PyTorch, Pandas, Scikit-Learn, and NumPy. 
* **Execution:** Training constraints require execution under 10 minutes utilizing parallel `DataLoader` workers and vectorized MSE functions.
* **Architecture Constraint:** You must use PyTorch's `nn.MSELoss` and an optimizer like `AdamW`. Do not manually construct forward passes or backpropagation loops using raw NumPy.

## 6. Evaluation Criteria
Your pipeline will be evaluated on Area Under the Precision-Recall Curve (AUPRC) on a hidden test dataset. Code will be assessed on its idiomatic utilization of PyTorch ecosystem features rather than convoluted standard loops.

## 7. Deliverables
* `pipeline.py`: Containing the PyTorch `nn.Module` definition and training loops.
* `inference.ipynb`: A step-by-step vector reconstruction and thresholding analysis demonstrating the loss distribution cleanly.# Statistical Metric Validation Formulation
**Difficulty Level:** Phase 4 (Advanced) | **Time Limit:** 4 Hours