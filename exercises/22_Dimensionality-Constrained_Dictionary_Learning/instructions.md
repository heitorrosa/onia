# Dimensionality-Constrained Dictionary Learning
**Difficulty Level:** Phase 4 (Advanced) | **Time Limit:** 3 Hours

## 1. Background / Scenario
Dictionary Learning is a matrix factorization technique that seeks a sparse representation of data. Using the [Olivetti Faces Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_olivetti_faces.html) (or a similar image dataset), you will decompose facial images into a "dictionary" of basis features (e.g., noses, eyes, lips).

## 2. Problem Statement
Implement a sparse coding model that:
1.  **Learns a Basis**: Use `DictionaryLearning` to find 100 "atoms" that can reconstruct the faces.
2.  **Sparse Reconstruction**: Show how a specific face is reconstructed using only 5-10 atoms from the dictionary.
3.  **Regularization (Alpha)**: Control the sparsity of the encoding (fewer atoms used = higher alpha).

## 3. Dataset Description
**Reference:** [Olivetti Faces](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_olivetti_faces.html)
- **Features:** 400 images of 64x64 pixels (4096 features after flattening).
- **Target:** Person ID (Not used in the learning phase).
- **Challenge:** Massive feature space (4096) with low sample size (400).

## 4. Subtasks & Point Distribution (100 Points)
*   **Task 4.1: Image Preprocessing (20 pts):** Standardize the pixels to [0, 1] and center the images around the mean face.
*   **Task 4.2: Dictionary Extraction (40 pts):**
    -   Initialize `DictionaryLearning` with 100 components.
    -   Visualize the top 10 dictionary atoms as images.
*   **Task 4.3: Sparse Encoding Experiment (30 pts):**
    -   Use `SparseCoder` to reconstruct a face using the learned dictionary.
    -   Analyze the reconstruction error as you vary the number of allowed atoms (e.g., `n_nonzero_coefs`).
*   **Task 4.4: Alpha Stability (10 pts):**
    -   Demonstrate how increasing the $\alpha$ parameter simplifies the reconstruction at the cost of detail.

## 5. Constraints & Technical Rules
- **Memory Management:** Pixels are 4096 features; ensure you do not run out of RAM during the factorization.
- **Visualization:** You must provide images of both the "Dictionary Atoms" and the "Reconstructed Face".
