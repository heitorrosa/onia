# ViT - Vision Transformer Architectures
**Difficulty Level:** Phase 4 (Extreme) | **Time Limit:** 5 Hours

## 1. Background / Scenario
"An Image is Worth 16x16 Words." Vision Transformers (ViT) replaced convolutions with global self-attention. Using the [MedNIST Dataset](https://www.kaggle.com/datasets/andrewmvd/medical-mnist), you will implement the ViT architecture from the ground up.

## 2. Problem Statement
1.  **Patch Embedding**: Convert an image into a sequence of $16 \times 16$ flattened patches.
2.  **Learnable [CLS] Token**: Inject a class token into the sequence to aggregate global information.
3.  **Transformer Encoder**: Stack Multi-Head Attention blocks with Layer Normalization and MLP heads.

## 3. Dataset Description
**Reference:** [Medical MNIST](https://www.kaggle.com/datasets/andrewmvd/medical-mnist)
- **Features:** 64x64 Grayscale images of medical scans (CXR, MRI, etc.).

## 4. Subtasks & Point Distribution (100 Points)
*   **Task 4.1: Image Patching Module (30 pts):** Implement the `einops` or `unfold` logic to split images into patches.
*   **Task 4.2: Positional Encoding (20 pts):** Implement learnable 1D positional embeddings.
*   **Task 4.3: Multi-Head Self-Attention (40 pts):** Build the core ViT block that allows patches to "attend" to each other.
*   **Task 4.4: Attention Map Visualization (10 pts):** Extract and plot the attention weights of the [CLS] token to see which part of the scan the model is "looking at."

## 5. Constraints & Technical Rules
- **Architecture:** You must not use `torchvision.models.vit`. You must implement the `TransformerEncoderLayer` manually.
- **Framework:** `PyTorch`.
