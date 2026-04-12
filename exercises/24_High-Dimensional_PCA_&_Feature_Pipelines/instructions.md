# High-Dimensional PCA & Feature Pipelines
**Difficulty Level:** Phase 3 (Intermediate) | **Time Limit:** 3 Hours

## 1. Background / Scenario
You are a data architect running compression metrics over satellite image transmissions. Due to transmission layer lag, projecting high-dimensional visual spaces into optimized lower-dimensional sub-structures is required before deep analysis. You will work directly within the MNIST visual baseline to synthesize optimal compression, bypassing purely manual Eigenvalue formulations for native Scikit-Learn processing.

## 2. Problem Statement
Working specifically on high-dimensional visual matrices, natively scale a raw pixel feature array. Construct entirely customized pipeline systems deploying Scikit-Learn's `PCA` instead of executing manual Eigendecomposition loops iteratively. You must extract dimensions retaining strictly 98% variances directly into an optimal downstream Support Vector Machine classifier, tuning computational efficiency and recall accuracy simultaneously.

## 3. Dataset Description
**Reference:** [MNIST in CSV](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)

The dataset contains high-dimensional arrays of hand-written digits (0-9) represented by up to 784 spatial pixel limits. The schema contains heavy sparsities, redundant spatial background limits, and massive multi-collinearity across specific digit loops.

## 4. Subtasks & Point Distribution
* **Task 4.1: Feature Subspace Rescaling (30 pts):** Define precise preprocessing pipelines leveraging `StandardScaler` to map properties consistently across identical statistical distributions.
* **Task 4.2: PCA Pipeline Orchestration (40 pts):** Implement a `PCA` transformation strictly configuring `n_components` parametrically to retain 98% explained variance rather than fixed dimensional structures. Print the final component scalar counts iteratively.
* **Task 4.3: Downstream SVM Inference (30 pts):** Append the dimensional sub-structure sequentially into an `SVC` implementation configured mapping complex interactions internally via an RBF or Polynomial kernel dynamically mapped via `cross_val_score`. 

## 5. Constraints & Technical Rules
* **Libraries:** You are strictly permitted to deploy Pandas, Scikit-Learn, and NumPy arrays. Math derivations must exclusively utilize Scikit-Learn's modules.
* **Execution:** Do not calculate raw Covariance Matrices nor orthogonal Eigenvectors manually. Deploy high-level `PCA` algorithms ensuring stable SVD calculations underlying.
* **Scikit-Learn:** The structural layout must systematically deploy a Scikit-Learn `Pipeline` bounding the scaler, PCA decomposition, and SVM classification to entirely prevent data leakage in holdout testing constraints.

## 6. Evaluation Criteria
Must complete matrix transformations mathematically under reasonable algorithmic duration limits. Evaluated heavily on validation execution time footprints via testing classification reports outputting Precision, Recall, and explicit F1 metrics. 

## 7. Deliverables
* `pipeline.py`: Code implementing the exact Scikit-Learn pipeline and parameters.
* `inference_script.ipynb`: Analytical mappings logging elapsed durations, testing matrix reports, and reconstructed variance curves systematically.