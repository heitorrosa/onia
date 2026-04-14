# GAN-Powered Anomaly Synthesis
**Difficulty Level:** Phase 4 (Extreme) | **Time Limit:** 5 Hours

## 1. Background / Scenario
Training anomaly detectors requires "Anomalies," which are rare in real production. You will implement a **Conditional GAN (cGAN)** to generate *synthetic* steel defects for data augmentation.

## 2. Problem Statement
1.  **Imbalance Solution**: Use a GAN to generate "Class 3" (rare) defects on top of clean steel sheet textures.
2.  **Texture Blending**: Implement a loss that ensures the generated defect blends seamlessly with the background texture.
3.  **Augmentation Test**: Train a simple U-Net with and without your synthetic GAN-generated defects.

## 3. Dataset Description
**Reference:** [Steel Defect Detection](https://www.kaggle.com/c/severstal-steel-defect-detection).

## 4. Subtasks & Point Distribution (100 Points)
*   **Task 4.1: Texture Discriminator (30 pts):** A network that tells if a texture is "Natural Steel" or "Neural Noise".
*   **Task 4.2: Pix2Pix-style Generator (40 pts):** Map a grayscale mask of a defect to a realistic RGB steel patch.
*   **Task 4.3: Perceptual Loss (20 pts):** Use a pre-trained VGG-16 to compare high-level features of real vs fake defects.
*   **Task 4.4: FID Score Analysis (10 pts):** Calculate the Fréchet Inception Distance between the real and synthetic domains.

## 5. Constraints & Technical Rules
- **Framework:** `PyTorch`.
- **Generation:** Defects must be of size 256x256.
