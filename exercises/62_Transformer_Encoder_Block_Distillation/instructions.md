# Transformer Encoder Block Distillation
**Difficulty Level:** Phase 4 (Extreme) | **Time Limit:** 5 Hours

## 1. Background / Scenario
A Transformer Encoder consists of Multi-head Attention + Feed Forward + LayerNorm. In this challenge, you will build one full "Layer" and use it for **Text Classification** on the [AG News Classification Dataset](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset).

## 2. Problem Statement
1.  **Multi-Head Attention (MHA)**: Split the hidden dimension into $H$ heads and process in parallel.
2.  **Position-wise Feed Forward**: Two linear layers with a non-linearity between them.
3.  **Positional Encoding**: Implement `sin`/`cos` embeddings to give the model a sense of order.

## 3. Dataset Description
**Reference:** [AG News](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset)
- **Features:** Headlines and descriptions of news.
- **Target:** 4 categories (World, Sports, Business, Sci/Tech).

## 4. Subtasks & Point Distribution (100 Points)
*   **Task 4.1: MHA Implementation (40 pts):** Head splitting and concatenation logic.
*   **Task 4.2: Positional Encoding (20 pts):** Static matrix logic for $Seq \times Dim$.
*   **Task 4.3: Full Layer Integration (30 pts):** Multi-head Attention + Add & Norm + Feed Forward.
*   **Task 4.4: Performance Benchmarking (10 pts):** Does a 1-layer Transformer beat a standard RNN on AG News?

## 5. Constraints & Technical Rules
- **Focus:** This is about the *block* structural knowledge.
- **Framework:** `PyTorch`.
