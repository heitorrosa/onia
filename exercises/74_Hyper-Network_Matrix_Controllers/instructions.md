# Hyper-Network Matrix Controllers
**Difficulty Level:** Phase 4 (Extreme) | **Time Limit:** 5 Hours

## 1. Background / Scenario
Few-Shot learning requires adapting to new tasks with only 5 images. **Hyper-networks** are neural networks that generate the *weights* for another neural network. You will implement a weight-generating controller for the [Omniglot Dataset](https://www.kaggle.com/datasets/sainikhileswar/omniglot-dataset).

## 2. Problem Statement
1.  **Primary Network**: A simple CNN tasked with classifying characters.
2.  **Hyper-Network**: An MLP that takes a character's "embedding" and generates the kernels for the Primary Network's final layer.
3.  **Meta-Learning**: Train the system so that after seeing 5 images of a new alphabet, the Hyper-network generalizes the weights correctly.

## 3. Dataset Description
**Reference:** [Omniglot Characters](https://www.kaggle.com/datasets/sainikhileswar/omniglot-dataset).
- **Challenge:** 1,623 different characters with only 20 samples each.

## 4. Subtasks & Point Distribution (100 Points)
*   **Task 4.1: Weight Generator Architecture (40 pts):** Correctly map an input vector to a $(K \times K \times C_{in} \times C_{out})$ weight tensor.
*   **Task 4.2: Functional Convolutions (30 pts):** Use `torch.nn.functional.conv2d` to apply the generated weights dynamically.
*   **Task 4.3: Episodic Training (20 pts):** Implement a "Support Set" and "Query Set" training loop.
*   **Task 4.4: Weight Space Viz (10 pts):** Use t-SNE to show that for similar characters, the Hyper-network generates similar weight kernels.

## 5. Constraints & Technical Rules
- **Framework:** `PyTorch`.
- **No Weight Freezing:** The Hyper-network must be fully trainable.
