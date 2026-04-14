import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def load_data():
    dataset = load_wine(as_frame=True)
    dataset = dataset.frame

    return dataset.drop(columns={'target'}), dataset['target']

if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    dt = DecisionTreeClassifier(random_state=42)
    path = dt.cost_complexity_pruning_path(X_train_scaled, y_train)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities

    dt = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        param_grid={'ccp_alpha': ccp_alphas[:-1]},
        cv=5
    )
    dt.fit(X_train_scaled, y_train)

    dt_best_alpha = dt.best_params_['ccp_alpha']
    dt_best = dt.best_estimator_
    dt_importances = pd.Series(dt_best.feature_importances_, index=X.columns)

    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train_scaled, y_train)
    
    rf_importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

    print(classification_report(y_test, rf.predict(X_test_scaled)))
    print(classification_report(y_test, dt.predict(X_test_scaled)))
    print(f'best dt alpha: {dt_best_alpha}')

    plt.figure(figsize=(10, 6))
    plt.plot(ccp_alphas[:-1], impurities[:-1], marker='o', drawstyle="steps-post", color='darkgreen')
    plt.xlabel("Effective Alpha (ccp_alpha)")
    plt.ylabel("Total Impurity of Leaves")
    plt.title("Cost-Complexity Pruning Path: Impurity vs Alpha")
    plt.axvline(x=dt_best_alpha, color='red', linestyle='--', label=f'Best Alpha: {dt_best_alpha:.4f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("exercises/08_Advanced_Decision_Tree_Pruning_&_Ensembling/pruning_path.png")

    importance_comparison = pd.DataFrame({
    'Decision Tree (Pruned)': dt_importances,
    'Random Forest': rf_importances
    }).sort_values(by='Random Forest', ascending=False)

    plt.figure(figsize=(12, 6))
    importance_comparison.plot(kind='bar', figsize=(12, 6), width=0.8)
    plt.title("Comparação de Importância de Características: DTC vs RF")
    plt.ylabel("Gini Importance Score")
    plt.xlabel("Chemical Assays")
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("exercises/08_Advanced_Decision_Tree_Pruning_&_Ensembling/importance_comparison.png")