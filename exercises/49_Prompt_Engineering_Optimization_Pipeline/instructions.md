# Prompt Engineering Optimization Pipeline
**Difficulty Level:** Phase 1 (Foundational) | **Time Limit:** 1 Hour

## 1. Background / Scenario
Quality of LLM outputs depends on the "System Prompt." In this challenge, you will build an automated tester to find the best prompt for the [Twitter Sentiment Dataset](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis).

## 2. Problem Statement
Build a Python script that:
1.  **Tests 3 Prompt Strategies**: Zero-Shot, Few-Shot, and Chain-of-Thought (CoT).
2.  **Automated Scoring**: Use an LLM API (or a local Llama model) to classify tweets.
3.  **Precision Benchmarking**: Compare the LLM's classification against the dataset's ground truth.

## 3. Dataset Description
**Reference:** [Twitter Sentiment Analysis](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis)
- **Target:** Positive, Negative, Neutral.

## 4. Subtasks & Point Distribution (100 Points)
*   **Task 4.1: Prompt Template Design (30 pts):** Create well-structured JSON templates for the 3 strategies.
*   **Task 4.2: API Orchestration (30 pts):** Implement batch processing to send 50 tweets to the model.
*   **Task 4.3: Accuracy Matrix (30 pts):** Report the F1-score for each prompt strategy.
*   **Task 4.4: Hallucination Analysis (10 pts):** Flag responses that don't match the required categorical format.

## 5. Constraints & Technical Rules
- **Framework:** Use `LangChain`, `OpenAI API`, or `Groq`.
- **Validation:** Prompt must include a "negative constraint" (e.g., "Do not use markdown formatting").
