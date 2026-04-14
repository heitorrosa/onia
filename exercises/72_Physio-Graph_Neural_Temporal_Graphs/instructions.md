# Physio-Graph: Neural Temporal Graphs
**Difficulty Level:** Phase 4 (Extreme) | **Time Limit:** 5 Hours

## 1. Background / Scenario
Hospital data is often irregularly sampled. In this challenge, you will implement a **Graph Neural Network (GNN)** using `PyTorch Geometric` to model patient physiological sensors as nodes in a dynamic graph.

## 2. Problem Statement
1.  **Graph Construction**: Define nodes (sensors) and edges (correlations between sensors).
2.  **Message Passing**: Implement a custom `GNNLayer` that aggregates information from neighbors to predict the future state of a patient.
3.  **Relational Induction**: Learn the "Edge Weights" dynamically using an attention mechanism (GAT).

## 3. Dataset Description
**Reference:** [PhysioNet 2012 Patient Data](https://www.kaggle.com/datasets/koki25/physionet-2012-challenge-dataset)
- **Challenge:** Sparse, irregularly timed observations for each patient.

## 4. Subtasks & Point Distribution (100 Points)
*   **Task 4.1: Adjacency Matrix Engineering (30 pts):** Calculate $A = \exp(-\|x_i - x_j\|^2 / \sigma)$ to create the initial graph structure.
*   **Task 4.2: GCN/GAT Layer (40 pts):** Implement $h_i^{(l+1)} = \sigma(\sum_{j \in \mathcal{N}_i} \alpha_{ij} W h_j^{(l)})$.
*   **Task 4.3: Temporal Aggregation (20 pts):** Pipe the GNN outputs into a GRU cell to handle sequences of graphs.
*   **Task 4.4: Graph Connectivity Viz (10 pts):** Use `NetworkX` to plot the learned graph of patient features.

## 5. Constraints & Technical Rules
- **Framework:** `PyTorch` and `PyTorch Geometric`.
- **Inference:** The model must predict the 24-hour mortality risk.
