# Contrastive Speech Quantization Masks
**Difficulty Level:** Phase 4 (Extreme) | **Time Limit:** 5 Hours

## 1. Background / Scenario
Models like **wav2vec 2.0** learn to "hear" by masking parts of raw audio and trying to predict the masked segment. This is **Self-Supervised Speech Pre-training**. You will implement the quantization and masking logic on the [LibriSpeech Dataset](https://www.kaggle.com/datasets/rashikrahmanpritom/librispeech-clean-100-subset).

## 2. Problem Statement
1.  **Feature Encoder**: Build a CNN that compresses raw audio samples into latent vectors.
2.  **Vector Quantization (VQ)**: Map each latent vector to the nearest entry in a fixed "Codebook."
3.  **Masked Prediction**: Mask 50% of the latent vectors and use a Transformer to predict the *codebook index* of the masked area.

## 3. Dataset Description
**Reference:** [LibriSpeech Clean 100](https://www.kaggle.com/datasets/rashikrahmanpritom/librispeech-clean-100-subset).

## 4. Subtasks & Point Distribution (100 Points)
*   **Task 4.1: CNN Audio Encoder (30 pts):** Strided convolutions to reduce resolution.
*   **Task 4.2: Gumbel-Softmax Quantizer (40 pts):** Implement the differentiable codebook selection.
*   **Task 4.3: Masking logic (20 pts):** Continuous spans of masked frames.
*   **Task 4.4: Contrastive Objective (10 pts):** Pull the target codebook vector closer to the prediction.

## 5. Constraints & Technical Rules
- **Complexity:** This is a structural exercise—focus on the quantization math.
- **Framework:** `PyTorch`.
