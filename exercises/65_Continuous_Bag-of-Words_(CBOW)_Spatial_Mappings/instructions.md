# Continuous Bag-of-Words (CBOW) Spatial Mappings
**Difficulty Level:** Phase 2 (Intermediate) | **Time Limit:** 2 Hours

## 1. Background / Scenario
Word2Vec revolutionized NLP by mapping words into a vector space where "King - Man + Woman = Queen." In this exercise, you will implement **CBOW (Continuous Bag of Words)** to predict a middle word given its context using the [Text8 Corpus](https://www.kaggle.com/datasets/amanajay/text8-corpus).

## 2. Problem Statement
1.  **Sliding Context Window**: Create $(Context, Target)$ pairs (e.g., "[The, sat, on]" -> "cat").
2.  **Embedding Layer**: Learn a 100-D vector for every word in a 10,000-word vocabulary.
3.  **Spatial Analysis**: Using Cosine Similarity, find the 5 words closest to "science."

## 3. Dataset Description
**Reference:** [Text8](https://www.kaggle.com/datasets/amanajay/text8-corpus) (Pre-cleaned Wikipedia text).

## 4. Subtasks & Point Distribution (100 Points)
*   **Task 4.1: N-Gram Generator (20 pts):** Extract windows of size 2+1+2.
*   **Task 4.2: Projection Model (40 pts):** Average the context embeddings and pass them to a softmax classifier.
*   **Task 4.3: Negative Sampling Logic (30 pts):** Explain how to speed up training by only sampling a few "non-target" words.
*   **Task 4.4: Vector Visualization (10 pts):** Use t-SNE to plot "apple," "orange," and "laptop."

## 5. Constraints & Technical Rules
- **Logic:** You must explain the difference between CBOW and Skip-Gram.
- **Framework:** `PyTorch` or `Gensim` (if focus is on analysis).
