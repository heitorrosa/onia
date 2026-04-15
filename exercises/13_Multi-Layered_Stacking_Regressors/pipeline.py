import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV, LassoCV
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_percentage_error, r2_score

def load_data():
    df = pd.read_csv("exercises/13_Multi-Layered_Stacking_Regressors/insurance.csv")

    return df.drop(columns={'charges'}), df['charges']

if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    ct = ColumnTransformer([
        ('ordinal', OrdinalEncoder(), ['sex', 'smoker']),
        ('onehot', OneHotEncoder(), ['region']),
        ('scaler', StandardScaler(), ['age', 'bmi', 'children'])
    ])

    X_train_scaled = ct.fit_transform(X_train)
    X_test_scaled = ct.transform(X_test)

    xgb_params = {
        'n_estimators': [100, 250],
        'max_depth': [3, 5],
        'learning_rate': [0.05, 0.1]
    }

    rf_params = {
        'n_estimators': [50, 100],
        'max_features': ['sqrt', None],
        'min_samples_leaf': [1, 5]
    }

    xgb = GridSearchCV(XGBRegressor(random_state=42), xgb_params, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
    rf = GridSearchCV(RandomForestRegressor(random_state=42), rf_params, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)

    xgb.fit(X_train_scaled, y_train)
    rf.fit(X_train_scaled, y_train)

    xgb_best = xgb.best_estimator_
    rf_best = rf.best_estimator_

    model = StackingRegressor(
        estimators=[
            ('xgb', xgb_best),
            ('rf', rf_best),
            ('ridge', RidgeCV(cv=5))
        ],
        final_estimator=LassoCV(),
        cv=5 
    )
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    mape = mean_absolute_percentage_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'mape: {mape:.4f} | r2: {r2:.4f}')
    print(model.final_estimator_.coef_)
    print(xgb.best_params_)
    print(rf.best_params_)

    residuals = y_test - y_pred

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(y_pred, residuals, alpha=0.5, color='teal')
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Preds ($)')
    plt.ylabel('Error (real - predict)')
    plt.subplot(1, 2, 2)
    plt.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('Error ($)')
    plt.tight_layout()
    plt.savefig('exercises/13_Multi-Layered_Stacking_Regressors/model_evaluation.png')

    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Real Values ($)')
    plt.ylabel('Preds ($)')
    plt.title('Real vs. Previsto')
    plt.savefig('exercises/13_Multi-Layered_Stacking_Regressors/comparison.png')