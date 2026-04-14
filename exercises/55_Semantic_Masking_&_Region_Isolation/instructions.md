# Semantic Masking & Region Isolation
**Difficulty Level:** Phase 4 (Extreme) | **Time Limit:** 4 Hours

## 1. Background / Scenario
Image Segmentation involves labeling every pixel. In this exercise, you'll implement a **U-Net** architecture to isolate tumors or structures in the [Lung Mask Image Dataset](https://www.kaggle.com/datasets/kmader/finding-lungs-in-ct-data).

## 2. Problem Statement
1.  **Encoder-Decoder Flow**: Implement the contracting path (pooling) and the expansive path (up-sampling).
2.  **Skip Connections (Concat)**: Pass feature maps from the encoder to the decoder via concatenation.
3.  **Dice Loss**: Implement the Dice Coefficient loss function (intersection over union) which is better for imbalanced segmentation masks than Cross Entropy.

## 3. Dataset Description
**Reference:** [Lung Masks](https://www.kaggle.com/datasets/kmader/finding-lungs-in-ct-data) or similar CT scan data.

## 4. Subtasks & Point Distribution (100 Points)
*   **Task 4.1: U-Net Architecture (40 pts):** 4 levels of depth with concatenation skip paths.
*   **Task 4.2: Up-Convolution Logic (20 pts):** Use `nn.ConvTranspose2d` or `Upsample` layers.
*   **Task 4.3: Dice Loss Implementation (30 pts):** Correct formula for $2 \times |P \cap G| / (|P| + |G|)$.
*   **Task 4.4: Inference Visualization (10 pts):** Display the original image, ground truth mask, and your predicted mask side-by-side.

## 5. Constraints & Technical Rules
- **Loss:** Standard `BCEWithLogitsLoss` is acceptable only if combined with Dice.
- **Framework:** `PyTorch`.
