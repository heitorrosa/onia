# Transfer Learning Standard Fine-Tuning
**Difficulty Level:** Phase 3 (Computer Vision) | **Time Limit:** 3 Hours

## 1. Background / Scenario
Training large CNNs like ResNet-18 from scratch requires significant computational resources and massive datasets. Transfer Learning allows us to take a model trained on a general dataset (ImageNet) and adapt it to a specialized domain (Medical Imaging) by repurposing its learned feature extractors. This is often the only way to achieve high accuracy on small, specialized datasets.

## 2. Problem Statement
Implement a 2-stage Transfer Learning pipeline. First, you will perform "Feature Extraction" by freezing the backbone and training a new classification head. Second, you will perform "Fine-Tuning" by unfreezing the final layers of the backbone and training with a high-precision learning rate.

## 3. Dataset Description
The [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset from Kaggle.
* **Input:** RGB Chest X-rays (converted/resized to 224x224).
* **Classes:** 2 (Normal vs. Pneumonia).
* **Challenge:** High class imbalance and critical need for high recall (minimizing missed pneumonia cases).

## 4. Subtasks & Point Distribution
* **Task 4.1: Backbone Immobilization (30 pts):** Initialize <code>torchvision.models.resnet18(weights='DEFAULT')</code>. Freeze all parameters so they do not update during backpropagation: <code>param.requires_grad = False</code>.
* **Task 4.2: Head Surgery (30 pts):** Replace the existing <code>model.fc</code> with a new <code>nn.Sequential</code> block containing:
    * <code>nn.Linear(512, 256)</code>
    * <code>nn.ReLU()</code>
    * <code>nn.Dropout(0.2)</code>
    * <code>nn.Linear(256, 2)</code>
* **Task 4.3: Stage 2 Fine-Tuning (40 pts):** After 5 epochs of training only the head, unfreeze the <code>layer4</code> parameters of ResNet-18. Update your optimizer to use a very small learning rate (e.g., <code>1e-5</code>) and train for 5 more epochs.

## 5. Constraints & Technical Rules
* **No Scratch Training:** You must not train from random weights.
* **Normalization:** Use the specific ImageNet Mean <code>[0.485, 0.456, 0.406]</code> and Std <code>[0.229, 0.224, 0.225]</code> in your <code>transforms.Normalize</code>.

## 6. Evaluation Criteria
* **Recall (Sensitivity):** Must reach > 0.90 on the Pneumonia class.
* **Accuracy:** Must reach > 0.88 overall.
* **Visuals:** Provide a Confusion Matrix to analyze False Negatives.

## 7. Deliverables
* <code>pneumonia_transfer.py</code>: The PyTorch implementation with 2-stage training logic.
* <code>confusion_matrix.png</code>: Visualization of model performance.
