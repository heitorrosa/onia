import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import confusion_matrix, log_loss, f1_score

def loadDataset():
    df = pd.read_csv("2.A.1/Iris.csv")
    df = df.dropna()

    X = df.drop(columns=['Id', 'Species']) if 'Id' in df.columns else df.drop(columns=['Species'])
    y = df['Species']

    return X, y

if __name__ == '__main__':
    X, y = loadDataset()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', SGDClassifier(loss='log_loss', penalty='elasticnet', max_iter=100000, random_state=42))
    ])

    param_grid = {
        'classifier__alpha': [0.0001, 0.001, 0.01, 0.1],
        'classifier__l1_ratio': [0.0, 0.15, 0.5, 0.85, 1.0]
    }

    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)

    ce = log_loss(y_test, y_prob)
    f_score = f1_score(y_test, y_pred, average='weighted')

    print(f"best alpha: {grid_search.best_params_['classifier__alpha']}")
    print(f"best l1 ratio: {grid_search.best_params_['classifier__l1_ratio']}")

    print(f"ce: {ce:.4f}")
    print(f"f-score: {f_score:.4f}")

    w = best_model.named_steps['classifier'].coef_
    features = X.columns
    classes = best_model.named_steps['classifier'].classes_

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    #plt.show()

# ce: 0.0152 | f-score: 1.0000