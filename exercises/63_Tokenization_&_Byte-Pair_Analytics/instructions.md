# Tokenization & Byte-Pair Analytics
**Difficulty Level:** Phase 2 (Intermediate) | **Time Limit:** 2 Hours

## 1. Background / Scenario
LLMs don't read words; they read "Tokens." **Byte-Pair Encoding (BPE)** is how GPT-style models break down text. Using the [English Quotes Dataset](https://www.kaggle.com/datasets/akashaditya17/quotes-dataset), you will implement a BPE tokenizer.

## 2. Problem Statement
1.  **Merge Search**: Find the most frequent pair of adjacent bytes/characters in a corpus.
2.  **Vocabulary Expansion**: Add the merged pair to the vocabulary.
3.  **Compression Ratio**: Calculate how much the total sequence length is reduced by using tokens instead of characters.

## 3. Dataset Description
**Reference:** [Quotes Dataset](https://www.kaggle.com/datasets/akashaditya17/quotes-dataset).

## 4. Subtasks & Point Distribution (100 Points)
*   **Task 4.1: Character Frequency Map (20 pts):** Initial byte mapping.
*   **Task 4.2: Pair-Freq Algorithm (40 pts):** Iterative loop finding $(e, t)$ or $(i, n)$ pairs.
*   **Task 4.3: Token Encoder (30 pts):** Convert a sentence into a list of token IDs based on your dictionary.
*   **Task 4.4: Subword Analysis (10 pts):** Explain why "unhappy" is split into "un" + "happy" instead of being one word.

## 5. Constraints & Technical Rules
- **Code:** No `HuggingFace Tokenizers` permitted. Implement the `while` loop logic yourself.
- **Language:** Python.
