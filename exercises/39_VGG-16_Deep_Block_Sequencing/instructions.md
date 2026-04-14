# VGG-16 Deep Block Sequencing
**Difficulty Level:** Phase 3 (Computer Vision) | **Time Limit:** 3 Hours

## 1. Background / Scenario
If LeNet is the grandfather, VGG-16 is the "Architectural Standard." Simonyan and Zisserman showed that depth (16+ layers) and small (3x3) filters are more effective than AlexNet's large (11x11) filters. Each "VGG Block" consists of 2-3 Convolutions followed by a MaxPool.

## 2. Problem Statement
Implement a VGG-16 clone using a modular approach (e.g., using <code>nn.Sequential</code> or a <code>make_layers</code> helper function). Use this to classify 10 types of objects in 32x32 color images.

## 3. Dataset Description
The [CIFAR-10](https://www.kaggle.com/c/cifar-10) dataset from Kaggle.
* Input: 32x32 RGB images.
* Labels: 10 (Airplane, Automobile, Bird, Cat, etc.).

## 4. Subtasks & Point Distribution
* **Task 4.1: VGG Block Factory (40 pts):** Create a <code>vgg_block(in_dims, out_dims, num_convs)</code> function to instantiate 2x or 3x Conv blocks automatically.
* **Task 4.2: Architectural Depth (40 pts):** Stack 13 Convolutional layers and 3 Fully-Connected layers.
* **Task 4.3: Parameter Analysis (20 pts):** Count the total trainable parameters and report them in the training log (Expected: ~138 Million).

## 5. Constraints & Technical Rules
* Strictly forbidden: Use of <code>models.vgg16(pretrained=True)</code> or <code>pretrained=False</code> from Torchvision. All blocks must be manually coded from <code>nn.Conv2d</code>.

## 6. Evaluation Criteria
* Reach 0.70 Accuracy on CIFAR-10 without advanced augmentation.
* Correct use of <code>Padding=1</code> and <code>Stride=1</code> to maintain spatial dimensions within blocks.

## 7. Deliverables
* <code>vgg16_cifar.py</code>: The PyTorch script.
* <code>block_visualization.txt</code>: Printout of <code>print(model)</code> showing the sequential blocks.
