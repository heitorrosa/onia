import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.experimental import enable_iterative_imputer

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report

from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

def load_data():
    dataset = pd.read_csv("exercises/04_Dimensionality_Synthesis_&_Constraint_Modeling/breast_cancer.csv")
    dataset = dataset.drop(columns={'id'})
    dataset = dataset.dropna(axis=1, how='all')
    
    dataset['diagnosis'] = dataset['diagnosis'].map({'M': 1, 'B': 0})

    return dataset.drop(columns={'diagnosis'}), dataset['diagnosis']

if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = {
        'svc__gamma': np.logspace(-4, 2, 20),
        'svc__C': [0.1, 1, 10, 100]
    }

    model = Pipeline([
        ('imputer', IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=42)),
        ('scaler', StandardScaler()),
        #('poly', PolynomialFeatures(degree=3, interaction_only=True)),
        ('pca', PCA(n_components=0.95)),
        ('svc', SVC(kernel='rbf', class_weight='balanced'))
    ])

    grid = GridSearchCV(model, param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid.fit(X_train, y_train)

    y_pred = grid.predict(X_test)

    print(f'params: {grid.best_params_}')
    print(f"{classification_report(y_test, y_pred)}")