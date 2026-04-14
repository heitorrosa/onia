# ELBO Matrix Optimization (VAE)
**Difficulty Level:** Phase 4 (Extreme) | **Time Limit:** 4 Hours

## 1. Background / Scenario
Variational Autoencoders (VAEs) don't just compress data; they learn a continuous latent space. Using the [Handwritten Digits (MNIST)](https://www.kaggle.com/datasets/hojjatk/mnist-dataset), you will maximize the **Evidence Lower Bound (ELBO)**.

## 2. Problem Statement
Implement a VAE that:
1.  **Reparameterization Trick**: Encode images into a $(\mu, \sigma)$ pair and sample $z = \mu + \epsilon \times \sigma$.
2.  **KL-Divergence Loss**: Implement the regularization term that forces the latent space to follow a Unit Gaussian distribution.
3.  **Generation**: Sample points from the latent space to generate *new*, never-before-seen digits.

## 3. Dataset Description
**Reference:** [MNIST Digits](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)

## 4. Subtasks & Point Distribution (100 Points)
*   **Task 4.1: Encoder/Decoder Design (30 pts):** Symmetrical CNN or MLP architecture.
*   **Task 4.2: ELBO Loss Function (40 pts):** Combine `BCE_Loss` (Reconstruction) + `0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)` (KLD).
*   **Task 4.3: Latent Manifold Walk (20 pts):** Linearly interpolate between two digits in the latent space and visualize the transition images.
*   **Task 4.4: VAE vs. AE Comparison (10 pts):** Explain why VAE produces better image samples than a standard Autoencoder.

## 5. Constraints & Technical Rules
- **Framework:** `PyTorch`.
- **Sampling:** You must implement the reparameterization from scratch (no high-level VAE wrappers).
