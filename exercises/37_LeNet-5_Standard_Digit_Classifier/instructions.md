# LeNet-5 Standard Digit Classifier
**Difficulty Level:** Phase 3 (Computer Vision) | **Time Limit:** 2 Hours

## 1. Background / Scenario
LeNet-5 is the "Grandfather" of Convolutional Neural Networks (CNNs). It introduced the concepts of alternating Convolutional and Subsampling (Pooling) layers. While simple, it is highly effective for grayscale image classification and serves as the baseline for all spatial reasoning.

## 2. Problem Statement
Implement the LeNet-5 architecture exactly as proposed (2x Conv + 3x FC) using PyTorch. Train it to classify handwritten digits from the MNIST dataset.

## 3. Dataset Description
The [MNIST Digit Recognizer](https://www.kaggle.com/competitions/digit-recognizer) dataset from Kaggle.
* Input: 28x28 grayscale images.
* Labels: 0-9.

## 4. Subtasks & Point Distribution
* **Task 4.1: Architecture Implementation (40 pts):**
    * Conv1: 1 -> 6 filters (5x5 kernel), Padding=2 (to reach 32x32 input).
    * Pool1: Average Pooling (2x2).
    * Conv2: 6 -> 16 filters (5x5 kernel).
    * Pool2: Average Pooling (2x2).
    * FC layers: 120 -> 84 -> 10.
* **Task 4.2: Spatial Calculation (30 pts):** Calculate and document the tensor shape transformation throughout the network in comments (e.g., `[B, 1, 28, 28] -> [B, 6, 28, 28] ...`).
* **Task 4.3: Training & Inference (30 pts):** Achieve >98% test accuracy using <code>CrossEntropyLoss</code>.

## 5. Constraints & Technical Rules
* Strictly forbidden: Using <code>nn.MaxPool2d</code>. Use <code>nn.AvgPool2d</code> as per the original 1998 paper.
* Strictly forbidden: Pre-trained weights.

## 6. Evaluation Criteria
* Reach 0.98 Accuracy.
* Correct implementation of the "Flatten" operation using <code>view()</code> or <code>nn.Flatten</code>.

## 7. Deliverables
* <code>lenet5_mnist.py</code>: The PyTorch implementation.
* <code>shapes.txt</code>: Log of tensor shapes at each layer.
