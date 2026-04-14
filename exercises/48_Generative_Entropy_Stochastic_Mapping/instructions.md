# Generative Entropy Stochastic Mapping
**Difficulty Level:** Phase 4 (Extreme) | **Time Limit:** 4 Hours

## 1. Background / Scenario
Diffusion models rely on the concept of **Stochastic Mapping**—reversing entropy to generate images from noise. In this exercise, you'll implement the core logic of a **DDPM (Denoising Diffusion Probabilistic Model)** on a small scale.

## 2. Problem Statement
1.  **Forward Diffusion**: Implement the "Noise Schedule" that gradually destroys information in an image.
2.  **Reverse Diffusion**: Create a U-Net (or simple CNN) that predicts the *noise* added to an image.
3.  **Sampling**: Starting from pure Gaussian noise, iteratively "clean" the tensor using your model to generate a digit.

## 3. Dataset Description
**Reference:** MNIST or FashionMNIST.

## 4. Subtasks & Point Distribution (100 Points)
*   **Task 4.1: Noise Schedule ($\beta_t$) (20 pts):** Linear or Cosine schedule implementation.
*   **Task 4.2: Denoising Model Pipeline (40 pts):** Train a network to minimize the L2 distance between predicted noise and actual noise.
*   **Task 4.3: DDPM Sampling Loop (30 pts):** Implement the Langevin dynamics sampling.
*   **Task 4.4: Generation Viz (10 pts):** Show the "un-noising" process step-by-step.

## 5. Constraints & Technical Rules
- **Time Embeddings:** You must use positional embeddings (like Sinusoidal) to tell the model which "Time-step" it is in.
- **Framework:** `PyTorch`.
