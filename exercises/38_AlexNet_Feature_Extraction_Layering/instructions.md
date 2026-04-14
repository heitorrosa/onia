# AlexNet Feature Extraction Layering
**Difficulty Level:** Phase 3 (Computer Vision) | **Time Limit:** 3 Hours

## 1. Background / Scenario
AlexNet was the first GPU-accelerated CNN to win the ImageNet competition, proving that Deep Learning is superior to manual feature engineering. It introduces Max Pooling, ReLU activation, and Dropout for regularization.

## 2. Problem Statement
Implement a simplified AlexNet-style architecture for 3-channel RGB image classification. You will use a Kaggle-sourced object detection dataset.

## 3. Dataset Description
The [Stanford Dogs](https://www.kaggle.com/datasets/miljan/stanford-dogs-dataset-tensors) or any standard Kaggle RGB dataset.
* Input: 64x64 or 224x224 RGB images.
* Labels: 120 categories of dogs.

## 4. Subtasks & Point Distribution
* **Task 4.1: Architecture Implementation (40 pts):**
    * Conv1: 3 -> 64 filters (11x11 kernel, Stride=4).
    * Conv2: 64 -> 192 filters (5x5 kernel, Padding=2).
    * Conv3/4/5: Consecutive 3x3 layers (384 -> 256).
    * MaxPool2d: Use 3x3 kernels with Stride=2.
* **Task 4.2: ReLU Activation (30 pts):** Replace LeNet-5's Sigmoid/Tanh activations with <code>nn.ReLU</code> across all layers.
* **Task 4.3: Dropout Regularization (30 pts):** Insert <code>nn.Dropout(0.5)</code> before the final Linear layers to prevent overfitting on the specialized dog breeds.

## 5. Constraints & Technical Rules
* Strictly forbidden: Use of <code>transforms.Normalize</code> from ImageNet coefficients. Use locally computed Mean/Std of your dataset subset.

## 6. Evaluation Criteria
* Reach 0.35 Accuracy on the complex "Stanford Dogs" dataset (Note: This is a difficult task; even base convergence shows success).
* Correct placement of <code>ReLU</code> vs <code>Dropout</code>.

## 7. Deliverables
* <code>alexnet_dogs.py</code>: The PyTorch script.
* <code>overfitting_plot.png</code>: Chart showing training vs test accuracy over 50 epochs.
