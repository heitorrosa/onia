import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix

def load_data():
    X, y = make_classification(
        n_samples=2000,
        n_features=25,
        n_informative=15,
        n_redundant=5,
        n_classes=3,
        weights=[0.6, 0.3, 0.1],
        flip_y=0.05,
        random_state=42
    )

    dataset = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(25)])
    dataset['target'] = y

    #print(dataset['target'].value_counts())

    return dataset.drop(columns={'target'}), dataset['target']

if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    svm_model = SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42)
    knn_model = KNeighborsClassifier(weights='distance', metric='euclidean')
    lr_model = LogisticRegression(class_weight='balanced', max_iter=100000, random_state=42)

    param_grid = {
        'weights': [
            [1, 1, 1], # equal
            [2, 1, 1], # svm
            [1, 1, 2], # logistic
            [2, 1, 2], # svm and logistic
        ],
        'voting': ['soft', 'hard'],
        'knn__n_neighbors': np.unique(np.linspace(1, 50, 10, dtype=int)) 
    }

    vc_model = VotingClassifier(estimators=[
        ('svm', svm_model),
        ('knn', knn_model),
        ('lr', lr_model)
    ])

    model = GridSearchCV(vc_model, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
    model.fit(X_train_scaled, y_train)

    best_model = model.best_estimator_

    print(f"params: {model.best_params_}")
    print(f"f1: {model.best_score_:.4f}")

    y_pred = best_model.predict(X_test_scaled)
            
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    """
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_train_scaled)

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train, cmap='viridis', alpha=0.6, edgecolors='w')
    plt.title("Synthetic Multi-Class Cluster Visualization (2D PCA)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()
    """