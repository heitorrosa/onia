# Self-Supervised Contrastive Formulations
**Difficulty Level:** Phase 4 (Extreme) | **Time Limit:** 5 Hours

## 1. Background / Scenario
What if you have images but no labels? **SimCLR** or **MoCo** use contrastive learning to learn representations by comparing different "views" of the same image. Using the [CIFAR-100](https://www.kaggle.com/datasets/imsparsh/cifar100-dataset), you'll train a self-supervised model.

## 2. Problem Statement
1.  **Data Augmentation (Dual-View)**: For every image, generate two different random augmentations (Crop, Jitter, Blur).
2.  **NT-Xent Loss**: Implement the "Normalized Temperature-scaled Cross Entropy" loss to pull the views of the same image together and push others away.
3.  **Linear Batch Evaluation**: Train a linear head on top of the frozen encoder and see if it can classify images without the encoder seeing labels during pre-training.

## 3. Dataset Description
**Reference:** [CIFAR-100](https://www.kaggle.com/datasets/imsparsh/cifar100-dataset).

## 4. Subtasks & Point Distribution (100 Points)
*   **Task 4.1: Contrastive Augmentation Pipeline (30 pts):** Custom Dataset class that returns two tensors for one image index.
*   **Task 4.2: Projection Head (20 pts):** Implement the MLP head that projects the feature space to a unit sphere.
*   **Task 4.3: NT-Xent Loss Implementation (40 pts):** Use cosine similarity and temperature scaling.
*   **Task 4.4: Efficiency Analysis (10 pts):** Explain how "Batch Size" is the most critical hyperparameter for SimCLR.

## 5. Constraints & Technical Rules
- **Framework:** `PyTorch`.
- **No Labels:** The encoder **must not** see the target labels during the first 10 epochs.
