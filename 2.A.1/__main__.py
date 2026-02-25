import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
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
        ('poly', PolynomialFeatures(degree=6, include_bias=False)),
        ('scaler', StandardScaler()),
        ('classifier', SGDClassifier(
            loss='log_loss',            # cross-entropy
            penalty='elasticnet',       # regulator
            l1_ratio=0.15,              # regulator
            alpha=0.0001,               # regulator
            learning_rate='adaptive',
            eta0=0.01,                  # initial learning rate
            max_iter=100000,
            tol=0.000004,
            random_state=42
        ))

    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    ce = log_loss(y_test, y_prob)
    f_score = f1_score(y_test, y_pred, average='weighted')

    print(f"ce: {ce:.4f}")
    print(f"f-score: {f_score:.4f}")

    w = model.named_steps['classifier'].coef_
    features = X.columns
    classes = model.named_steps['classifier'].classes_

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    #plt.show()

# ce: 0.0263 | f-score: 1.0000