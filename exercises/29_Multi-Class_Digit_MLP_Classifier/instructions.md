# Multi-Class Digit MLP Classifier
**Difficulty Level:** Phase 2 (Deep Learning) | **Time Limit:** 2 Hours

## 1. Background / Scenario
Before using CNNs for image recognition, you must understand how a flat Multi-Layer Perceptron (MLP) processes high-dimensional pixel data linearly. You must flatten a 2D image matrix into a 1D vector and classify handwritten digits.

## 2. Problem Statement
Implement a custom PyTorch <code>nn.Module</code> consisting of 2 hidden layers (units: 512, 128). You must manually flatten $(28, 28)$ tensors using <code>view(-1, 784)</code> before backpropagation begins.

## 3. Dataset Description
The [MNIST](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv) dataset will be used, consisting of 28x28 grayscale images of digits (0-9).

## 4. Subtasks & Point Distribution
* **Task 4.1: Feature Flattening (30 pts):** Create a 3-layer MLP and use <code>nn.Flatten</code> or <code>view</code> to intake $(batch, 1, 28, 28)$ and output $(batch, 10)$ logits.
* **Task 4.2: Softmax Classification Head (40 pts):** Connect the final linear layer to <code>nn.CrossEntropyLoss</code>. You must use <code>log_softmax</code> inside the model or ensure the loss function applies it automatically.
* **Task 4.3: Prediction Accuracy (30 pts):** Train for 10 epochs and reach an Accuracy of at least 95% on the test set.

## 5. Constraints & Technical Rules
* Strictly forbidden: <code>nn.Conv2d</code>, <code>nn.MaxPool2d</code>, and <code>nn.ReLU</code> for the first hidden layer (use <code>nn.Tanh</code> for Task 31).

## 6. Evaluation Criteria
* Successful classification under 10 minutes of training.
* Accuracy vs Epochs plot showing convergence.

## 7. Deliverables
* <code>digit_mlp.py</code>: The PyTorch digit classifier and training script.
