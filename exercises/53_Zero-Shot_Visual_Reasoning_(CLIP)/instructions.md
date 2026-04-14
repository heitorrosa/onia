# Zero-Shot Visual Reasoning (CLIP)
**Difficulty Level:** Phase 4 (Extreme) | **Time Limit:** 4 Hours

## 1. Background / Scenario
Contrastive Language-Image Pre-training (CLIP) allows models to "understand" images through natural language. In this challenge, you will implement a **Dual-Encoder Contrastive System** using the [Tiny-ImageNet](https://www.kaggle.com/datasets/akash2sharma/tiny-imagenet) dataset and text captions.

## 2. Problem Statement
1.  **Dual Arch**: Implement an Image Encoder (Vision Transformer) and a Text Encoder (Small Transformer/BERT).
2.  **Projection Space**: Project both embeddings into a shared $d=512$ latent space.
3.  **Contrastive Objective**: Implement the symmetric cross-entropy loss that maximizes the cosine similarity between matched (image, text) pairs in a batch.

## 3. Dataset Description
**Reference:** [Tiny-ImageNet](https://www.kaggle.com/datasets/akash2sharma/tiny-imagenet)
- **Challenge:** Creating text prompts (e.g., "A photo of a [class]") for zero-shot inference.

## 4. Subtasks & Point Distribution (100 Points)
*   **Task 4.1: Multi-Modal Dataloader (30 pts):** Return a batch of images and their corresponding text descriptions.
*   **Task 4.2: Shared Latent Projection (30 pts):** Implement the linear layers and L2-normalization for the embeddings.
*   **Task 4.3: CLIP Contrastive Loss (30 pts):** Build the similarity matrix ($N \times N$) and apply CrossEntropy along both axes.
*   **Task 4.4: Zero-Shot Classification (10 pts):** Predict the class of an image by comparing it to 10 "Text Queries" and picking the highest similarity.

## 5. Constraints & Technical Rules
- **Framework:** `PyTorch`.
- **Pre-training:** You may use pre-trained backbones, but the **Fusion Logic** and **Contrastive Loss** must be implemented from scratch.
