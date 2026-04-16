import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import f1_score, classification_report

def load_data():
    df = pd.read_csv('exercises/17_Staged_Ensemble_Aggregation/heart.csv')
    df = df.drop(columns={'id'})

    #print(df['num'].value_counts())

    return df.drop(columns={'num'}), df['num']

if __name__ == "__main__":
    X, y = load_data()
    #y = y.apply(lambda x: 1 if x > 0 else 0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    cat_pipeline = Pipeline([
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    ord_pipeline = Pipeline([
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('encode', OrdinalEncoder())
    ])

    ct = ColumnTransformer([
        ('ordinal', ord_pipeline, ['sex', 'fbs', 'exang']),
        ('onehot', cat_pipeline, ['dataset', 'cp', 'restecg', 'slope', 'thal'])
    ], remainder=SimpleImputer(strategy='median'))

    X_train_scaled = ct.fit_transform(X_train)
    X_test_scaled = ct.transform(X_test)

    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 4, 5],
        'subsample': [0.8, 1.0],
        'min_samples_leaf': [1, 3, 5],
    }
    model = GridSearchCV(GradientBoostingClassifier(random_state=42), param_grid, cv=5, n_jobs=-1, scoring='f1_macro', verbose=1)
    model.fit(X_train_scaled, y_train, sample_weight=sample_weights)

    y_pred = model.predict(X_test_scaled)

    print(f'f1: {f1_score(y_test, y_pred, average="macro")}')
    print(f'{classification_report(y_test, y_pred)}')
    print(f'{model.best_params_}')