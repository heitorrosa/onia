# Generative Adversarial Regularizations
**Difficulty Level:** Phase 4 (Advanced) | **Time Limit:** 4 Hours

## 1. Background / Scenario
GANs (Generative Adversarial Networks) involve a "Cat-and-Mouse" game between a **Generator** and a **Discriminator**. Using the [Anime Faces Dataset](https://www.kaggle.com/datasets/splcher/anime-faces), you will train a DCGAN to generate high-quality cartoon faces.

## 2. Problem Statement
1.  **Architecture**: Implement a Generator that uses `ConvTranspose2d` to upsample seeds.
2.  **Minimax Loss**: Implement the standard GAN loss function.
3.  **Stabilization**: Use `Spectral Normalization` or `InstanceNorm` to prevent "Mode Collapse" (where the generator produces only one image).

## 3. Dataset Description
**Reference:** [Anime Faces](https://www.kaggle.com/datasets/splcher/anime-faces)
- **Features:** 64x64 RGB images.

## 4. Subtasks & Point Distribution (100 Points)
*   **Task 4.1: DCGAN Generator Design (30 pts):** Project 100-D noise into a 64x64x3 image.
*   **Task 4.2: Discriminator Design (20 pts):** Standard CNN classifier (real vs. fake).
*   **Task 4.3: Training Loop (40 pts):** Alternating updates for D and G.
*   **Task 4.4: Generation GIF (10 pts):** Save snapshots every 5 epochs to show the "learning" process.

## 5. Constraints & Technical Rules
- **Loss:** Both models must be optimized using `Adam(lr=0.0002, betas=(0.5, 0.999))`.
- **Framework:** `PyTorch`.
