# Attention Mechanism Decoding Frameworks
**Difficulty Level:** Phase 4 (Extreme) | **Time Limit:** 3 Hours

## 1. Background / Scenario
"Attention is All You Need." In this exercise, you'll implement the mathematical core of **Scaled Dot-Product Attention** ($Attention(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$).

## 2. Problem Statement
1.  **Attention Core**: Implement the score calculation without using `nn.MultiheadAttention`.
2.  **Masking**: Implement a "Look-ahead Mask" (triangular matrix) to prevent the model from seeing future tokens during training (Causal Attention).
3.  **Softmax Visualization**: Plot the attention weights (Heatmap) between words in a sentence like "The cat sat on the mat."

## 3. Dataset Description
**Reference:** Any short text snippet or [IMDB Review Subset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).

## 4. Subtasks & Point Distribution (100 Points)
*   **Task 4.1: Score Matrix Calculation (40 pts):** Vectorized $(Q \times K^T)$ with scaling factor.
*   **Task 4.2: Causal Masking (30 pts):** Zero-out upper triangle before softmax.
*   **Task 4.3: Weighted Sum (20 pts):** Combine scores with $V$.
*   **Task 4.4: Analysis of $d_k$ (10 pts):** Explain why dividing by $\sqrt{d_k}$ is necessary for gradient stability.

## 5. Constraints & Technical Rules
- **No Built-ins:** You must use `torch.matmul` and `F.softmax`.
- **Dimensionality:** Assume tensors are of shape $(Batch, Heads, Seq, Dim)$.
