import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, OneHotEncoder, TargetEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss

"""
StandardScalar: 'fnlwgt', 'age', 'capital.gain', 'capital.loss', 'hours.per.week', 'education.num'
TargetEncoding: 'native.country', 'occupation', 'education', 'workclass'
OneHotEncoding: 'marital.status', 'marital.status', 'race', 'occupation', 'education', 'workclass', 'relationship', 'sex', 'income'
"""

def load_data():
    dataset = pd.read_csv("exercises/02_High-Cardinality_Target_Encoding/adult.csv")
    dataset = dataset.replace('?', 'Unknown')
    dataset['income'] = dataset['income'].map({'<=50K': 0, '>50K': 1})

    return dataset.drop(columns=['income']), dataset['income']

if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    standradScaler_columns = ['fnlwgt', 'age', 'capital.gain', 'capital.loss', 'hours.per.week', 'education.num']
    oneHotEncoding_columns = ['marital.status', 'race', 'relationship', 'sex', 'workclass', 'education']
    targetEncoding_columns = ['native.country', 'occupation']

    transformer = ColumnTransformer(transformers=[
        ('num', StandardScaler(), standradScaler_columns),
        ('low_complexity', OneHotEncoder(handle_unknown='ignore', sparse_output=False), oneHotEncoding_columns),
        ('high_complexity', TargetEncoder(smooth='auto'), targetEncoding_columns)
    ])

    X_train_scaled = transformer.fit_transform(X_train, y_train)
    X_test_scaled = transformer.transform(X_test)

    param_grid = {
        'C': [0.1, 1.0, 10.0],
        'l1_ratio': np.linspace(0, 1, 5),
    }

    lr_model = LogisticRegression(solver='saga', max_iter=100000)
    model = GridSearchCV(lr_model, param_grid, cv=3, n_jobs=-1)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)

    ce = log_loss(y_test, y_prob)
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"params: {model.best_params_}")
    print(f"f1: {f1:.4f}")
    print(f"ce: {ce:.4f}")