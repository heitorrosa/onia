# Implicit Neural SDF Reconstruction
**Difficulty Level:** Phase 4 (Extreme) | **Time Limit:** 5 Hours

## 1. Background / Scenario
Implicit Neural Representations (INRs) store 3D shapes within the weights of an MLP without using voxels or meshes. You will implement a **Signed Distance Function (SDF)** network to reconstruct 3D objects from the [ModelNet40 Dataset](https://www.kaggle.com/datasets/balraj98/modelnet40-princeton-3d-object-dataset).

## 2. Problem Statement
1.  **Coordinate-Based MLP**: Write a network that takes $(x, y, z)$ coordinates and predicts the distance to the nearest surface.
2.  **Distance Loss**: Calculate the L1 loss between the predicted distance and the ground truth distance from a point cloud.
3.  **Marching Cubes**: At inference time, sample a $V \times V \times V$ grid of coordinates to extract a 3D mesh from the zero-level set of your network.

## 3. Dataset Description
**Reference:** [ModelNet40 (3D shapes)](https://www.kaggle.com/datasets/balraj98/modelnet40-princeton-3d-object-dataset).

## 4. Subtasks & Point Distribution (100 Points)
*   **Task 4.1: Point Cloud Sampling (20 pts):** Sample interior and exterior points from a 3D mesh to create a distance training set.
*   **Task 4.2: Siren/ReLU MLP (40 pts):** Implement a network using SIREN (Sinusoidal activations) to capture high-frequency 3D details.
*   **Task 4.3: Normal Gradient Loss (30 pts):** Use `autograd` to calculate $\|\nabla f(x)\| = 1$ (the Eikonal constraint).
*   **Task 4.4: 3D Visualization (10 pts):** Render the reconstructed 3D object using `Open3D` or `Matplotlib` voxels.

## 5. Constraints & Technical Rules
- **Framework:** `PyTorch`.
- **Optimization:** You must use the Eikonal Regularizer to ensure the network truly learns a Distance Function.
