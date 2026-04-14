# Transfer_Learning_Freezing_Limits
**Difficulty Level:** Phase 3 (Advanced) | **Time Limit:** 2 Hours

## 1. Background / Scenario
Using a pre-trained model (like ResNet-18) is efficient, but choosing "where to freeze" is an art. Using the [Intel Image Classification (Natural Scenes)](https://www.kaggle.com/datasets/puneet6060/intel-image-classification) dataset, you'll optimize transfer learning.

## 2. Problem Statement
1.  **Feature Extraction**: Freeze all layers of a ResNet-18 and only train the final FC layer.
2.  **Fine-Tuning**: Unfreeze the last "Stage" (e.g. `layer4`) and compare accuracy.
3.  **Convergence Analysis**: Compare how fast the loss drops when weights are already close to optimal.

## 3. Dataset Description
**Reference:** [Intel Natural Scenes](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)
- **Categories:** Forest, Glacier, Mountain, Sea, etc.

## 4. Subtasks & Point Distribution (100 Points)
*   **Task 4.1: Pre-trained Loading (20 pts):** Correctly load `weights=ResNet18_Weights.DEFAULT`.
*   **Task 4.2: Model "Surgery" (40 pts):** Replace the final `out_features=1000` with `out_features=6`.
*   **Task 4.3: Freezing Strategy Experiment (30 pts):** Plot Accuracy for "Frozen" vs "Partially Unfrozen".
*   **Task 4.4: Image Augmentation (10 pts):** Implement `RandomResizedCrop` and `Normalize` match for ImageNet.

## 5. Constraints & Technical Rules
- **Framework:** `PyTorch / Torchvision`.
- **Normalization:** Must use `mean=[0.485, 0.456, 0.406]` matching the original ImageNet training.
