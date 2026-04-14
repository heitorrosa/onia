# Data Augmentation & Generalization
**Difficulty Level:** Phase 3 (Computer Vision) | **Time Limit:** 2 Hours

## 1. Background / Scenario
The "Small Dataset Problem" occurs when your model learns a fixed representation of your images (e.g., only identifying birds facing the left). Data Augmentation creates synthetic variations via transformations (rotation, scaling, noise) to force the CNN to learn invariant features.

## 2. Problem Statement
Implement a custom <code>torchvision.transforms.Compose</code> pipeline that effectively doubles or triples your training set's variance and use it with any base CNN (e.g., a simple 4-layer ConvNet).

## 3. Dataset Description
The [Cat and Dog](https://www.kaggle.com/datasets/tongpython/cat-and-dog) or [Fruit Recognition](https://www.kaggle.com/datasets/chrisfilo/fruit-recognition) dataset from Kaggle.
* Input: RGB imagery.
* Scenario: High intra-class variance (different poses, lighting).

## 4. Subtasks & Point Distribution
* **Task 4.1: Geometrical Transformations (30 pts):** Integrate <code>RandomHorizontalFlip</code>, <code>RandomRotation(20)</code>, and <code>RandomResizedCrop</code> into your <code>TrainLoader</code>.
* **Task 4.2: Photometric Jitter (30 pts):** Use <code>ColorJitter</code> to vary brightness and saturation of your incoming images.
* **Task 4.3: Comparative Validation (40 pts):** Train a baseline CNN on raw images and then the exact same CNN architecture with the data augmentation pipeline and visualize the reduction in training vs test accuracy gap.

## 5. Constraints & Technical Rules
* Strictly forbidden: Off-line augmentation (saving new images to disk). All transformations must be on-the-fly inside the PyTorch <code>Dataset</code> or <code>DataLoader</code> pipeline.

## 6. Evaluation Criteria
* Performance Gap: The gap between <code>TrainAcc</code> and <code>TestAcc</code> should decrease by at least 3% compared to the baseline.
* Reach 0.72 Accuracy on the CAT/DOG test set.

## 7. Deliverables
* <code>augmentation_trial.py</code>: The PyTorch script with the <code>transforms</code> definition.
* <code>train_test_gap.png</code>: Chart showing how augmentation reduced overfitting.
