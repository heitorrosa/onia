# Advanced Decision Tree Pruning & Ensembling
**Difficulty Level:** Phase 3 (Intermediate) | **Time Limit:** 2 Hours

## 1. Background / Scenario
You are a senior viticulturist analyzing high-dimensional chemical assays to predict wine varietals. Instead of writing pure mathematical entropy splits by hand, your objective is to fully utilize modern Scikit-Learn tools. You will structurally prune overgrown decision trees to prevent over-fitting on noisy parameters and integrate randomized Forest architectures to analyze feature importances efficiently. 

## 2. Problem Statement
Leveraging the Wine dataset, implement a rigorous Scikit-Learn `DecisionTreeClassifier`. Systematically optimize tree complexity by validating `ccp_alpha` (Cost Complexity Pruning) parameters over a cross-validated split. Extend this architectural approach by building a `RandomForestClassifier` baseline, using its internal algorithms to extract and plot structural feature importance arrays safely. Bypassing from-scratch root split mathematical formulas is required; rely on Scikit-Learn`s internal metrics natively.

## 3. Dataset Description
**Reference:** [Wine Dataset (Scikit-Learn)](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html)

The dataset contains chemical analyses of wines grown in the same region in Italy but derived from three different cultivars. It exhibits numeric properties across 13 continuous features representing chemical constituents (e.g., Alcohol, Malic acid, Ash).

## 4. Subtasks & Point Distribution
* **Task 4.1: Feature Prep & Base Tree Formulation (25 pts):** Initiate exactly standard scaling distributions and fit a baseline, unpruned `DecisionTreeClassifier`. 
* **Task 4.2: Cost Complexity Pruning Matrix (40 pts):** Calculate the effective alphas for the pruning path via `cost_complexity_pruning_path`. Build an iterative loop extracting cross-validated accuracy arrays to isolate the optimal `ccp_alpha` threshold structurally.
* **Task 4.3: Ensembled Feature Interpretability (35 pts):** Instantiate a `RandomForestClassifier`. Aggregate and map the explicit feature importances derived directly by the Scikit-Learn API using PyPlot mappings cleanly. 

## 5. Constraints & Technical Rules
* **Libraries:** You are strictly permitted to deploy Pandas, Scikit-Learn, and NumPy arrays. Matplotlib/Seaborn allowed for visualizations. 
* **Execution Constraint:** Do not write raw matrix equations for Shannon Entropy or Gini Impurity. Use Scikit-Learn's native implementations explicitly. 
* **Scikit-Learn Constraints:** Must leverage `cross_val_score` or `GridSearchCV` dynamically to ascertain the explicit tuning parameters perfectly.

## 6. Evaluation Criteria
Penalized accuracy matrices tracking test-set stabilization post-pruning. Over-fitted unpruned models will fail the structural holdout thresholds iteratively. Execution efficiency deploying ensemble metrics implicitly validates accuracy.

## 7. Deliverables
* `pipeline.py`: Code building the structural Decision Tree and Forest logic.
* `inference_script.ipynb`: Presenting the Cost-Complexity Pruning path graph and the Forest Feature Importances.