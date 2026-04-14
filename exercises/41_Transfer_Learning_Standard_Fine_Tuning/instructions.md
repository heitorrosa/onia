# Transfer Learning Standard Fine-Tuning
**Difficulty Level:** Phase 3 (Computer Vision) | **Time Limit:** 3 Hours

## 1. Background / Scenario
Training large CNNs (ResNet, Inception) from scratch requires trillions of floating-point operations. Transfer Learning leverages pre-trained "feature extractors" to solve specialized tasks on small datasets with an order of magnitude less time and data.

## 2. Problem Statement
Implement a Transfer Learning pipeline where you freeze the backbone of a pre-trained ResNet-18 or EfficientNet-B0 and replace the final fully-connected "Head" to solve a medical imaging classification task.

## 3. Dataset Description
The [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset from Kaggle.
* Input: Grayscale X-rays (Normal vs. Pneumonia).
* Labels: 2 (Binary Classification).

## 4. Subtasks & Point Distribution
* **Task 4.1: Backbone Freezing (30 pts):** Initialize a pre-trained ResNet-18 model via <code>models.resnet18(weights='DEFAULT')</code> and freeze all gradients: <code>for param in model.parameters(): param.requires_grad = False</code>.
* **Task 4.2: FC Head Replacement (30 pts):** Calculate the number of input features for the original <code>model.fc</code> and replace it with a new <code>nn.Linear</code> layer with 2 output units.
* **Task 4.3: Stage 2 - Fine Tuning (40 pts):** After training the Head for 5 epochs, unfreeze only the last "Residual Block" (Layer 4) and train for another 5 epochs using a smaller learning rate (e.g., <code>1e-5</code>) to specialize the deep feature maps.

## 5. Constraints & Technical Rules
* Strictly forbidden: Retraining the entire backbone from epoch 1. You must demonstrate the 2-stage "Warm Up Head then Fine-Tune" strategy.

## 6. Evaluation Criteria
* Reach 0.90 Accuracy on the test set.
* Reach 0.88 Recall (Sensitivity is critical in pneumonia detection; false negatives are dangerous).

## 7. Deliverables
* <code>transfer_learning_pneumonia.py</code>: The PyTorch script with the unfreeze logic.
* <code>recall_matrix.png</code>: Confusion matrix showing true positives for pneumonia.
