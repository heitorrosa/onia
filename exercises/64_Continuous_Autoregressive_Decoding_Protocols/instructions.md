# Continuous Autoregressive Decoding Protocols
**Difficulty Level:** Phase 4 (Extreme) | **Time Limit:** 4 Hours

## 1. Background / Scenario
Generating text involves predicting the next token, appending it to the input, and repeating. In this exercise, you'll implement two sampling strategies: **Beam Search** and **Top-k/Top-p Sampling** on a small [Language Model](https://www.kaggle.com/datasets/shrutimechlearn/large-text-corpus-for-language-modeling).

## 2. Problem Statement
1.  **Greedy vs. Stochastic**: Compare the "safest" next token vs. sampling from the distribution.
2.  **Beam Search**: Keep track of the top-3 most likely paths simultaneously.
3.  **Nucleus Sampling (Top-p)**: Only sample from tokens that make up the top 95% of the cumulative probability.

## 3. Dataset Description
**Reference:** Use a pre-trained character-level or word-level model.

## 4. Subtasks & Point Distribution (100 Points)
*   **Task 4.1: Beam Search Implementation (40 pts):** Recursive tree-search for sequence probability.
*   **Task 4.2: Temperature Scaling (20 pts):** Implement $P_i = \exp(Logit_i / T) / \sum \exp(Logit_j / T)$.
*   **Task 4.3: Top-p Filtering Logic (30 pts):** Correct sorting and cumulative sum logic.
*   **Task 4.4: Qualitative Comparison (10 pts):** Which strategy produces the most "human-like" text?

## 5. Constraints & Technical Rules
- **Loop:** Decoding is inherently sequential. You must manage the context window (KV-cache concept).
- **Language:** Python/PyTorch.
