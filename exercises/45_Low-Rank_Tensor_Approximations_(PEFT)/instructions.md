# Low-Rank Tensor Approximations (PEFT)
**Difficulty Level:** Phase 4 (Extreme) | **Time Limit:** 5 Hours

## 1. Background / Scenario
Fine-tuning 7B+ parameter models on a single GPU requires efficiency. **LoRA (Low-Rank Adaptation)** is the industry standard for Parameter-Efficient Fine-Tuning (PEFT). In this exercise, you'll implement the mathematical core of LoRA on a small MLP.

## 2. Problem Statement
1.  **Weight Decomposition**: For a weight matrix $W$, freeze $W$ and introduce two trainable low-rank matrices $A$ and $B$, such that $W_{new} = W + (A \times B)$.
2.  **Rank Control (r)**: Experiment with rank $r=4$ vs $r=16$ and observe how many parameters are trainable.
3.  **Benchmark**: Fine-tune a pre-trained model on [Medical Question Pairs](https://www.kaggle.com/datasets/iammustafatz/medical-question-pairs) using your LoRA implementation.

## 3. Dataset Description
**Reference:** [Medical Question Pairs](https://www.kaggle.com/datasets/iammustafatz/medical-question-pairs)
- **Task:** Duplicate detection.

## 4. Subtasks & Point Distribution (100 Points)
*   **Task 4.1: LoRA Layer Implementation (40 pts):** Create a custom PyTorch layer that injects $A \times B$ into a linear projection.
*   **Task 4.2: Trainable Param Count (20 pts):** Calculate the % of total weights that are being updated.
*   **Task 4.3: Model Training (30 pts):** Train on the question pairs.
*   **Task 4.4: Inference Speed (10 pts):** Show how A and B can be "merged" back into W for zero-latency inference.

## 5. Constraints & Technical Rules
- **Framework:** `PyTorch`.
- **Rank:** $r$ must be much smaller than the hidden dimension $d$.
- **No PEFT Library:** You must implement the decomposition logic yourself.
