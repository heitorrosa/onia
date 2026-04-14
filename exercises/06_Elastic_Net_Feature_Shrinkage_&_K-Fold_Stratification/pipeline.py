import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler

def load_data():
    dataset = load_diabetes(as_frame=True)
    dataset = dataset.frame

    return dataset.drop(columns={'target'}), dataset['target']

if __name__ == "__main__":
    X, y = load_data()
    y_binned = pd.qcut(y, q=5, labels=False)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    coef_paths = []
    mean_scores = []
    best_alphas = []

    l1_ratios = np.linspace(0.01, 1.0, 75)
    for l1 in l1_ratios:
        fold_coefs = []
        fold_scores = []
        fold_alphas = []
        for train_index, test_index in skf.split(X, y_binned):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            model = ElasticNetCV(l1_ratio=l1, alphas=np.logspace(-3, 2, 75), cv=3, max_iter=100000)
            model.fit(X_train_scaled, y_train)

            fold_coefs.append(model.coef_)
            fold_scores.append(model.score(X_test_scaled, y_test))
            fold_alphas.append(model.alpha_)

        coef_paths.append(np.mean(fold_coefs, axis=0))
        mean_scores.append(np.mean(fold_scores))
        best_alphas.append(np.mean(fold_alphas))

    coef_paths = np.array(coef_paths)

    plt.figure(figsize=(10, 6))
    for i in range(coef_paths.shape[1]):
        plt.plot(l1_ratios, coef_paths[:, i], label=X.columns[i])

    plt.axvline(x=0.5, linestyle='--', color='grey', alpha=0.5)
    plt.xlabel('L1 Ratio (0=Ridge, 1=Lasso)')
    plt.ylabel('Coefficient Value')
    plt.title('Elastic Net Coefficient Decay Paths')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.savefig("exercises/06_Elastic_Net_Feature_Shrinkage_&_K-Fold_Stratification/elasticnet.png",  bbox_inches='tight')

    plt.figure(figsize=(10, 4))
    plt.plot(l1_ratios, mean_scores, marker='o', color='red', label='Mean R^2 Score')

    best_idx = np.argmax(mean_scores)
    best_l1 = l1_ratios[best_idx]
    best_score = mean_scores[best_idx]

    plt.axvline(x=best_l1, linestyle='--', color='green', label=f'Best L1: {best_l1:.2f}')
    plt.title(f'Performance Mapping (Max R^2: {best_score:.4f})')
    plt.xlabel('L1 Ratio')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.savefig("exercises/06_Elastic_Net_Feature_Shrinkage_&_K-Fold_Stratification/performance.png", bbox_inches='tight')
    
print(f"mean optimal alpha: {np.mean(best_alphas):.6f}")
print(f"best alpha at best l1 ({best_l1:.2f}): {best_alphas[best_idx]:.6f}")