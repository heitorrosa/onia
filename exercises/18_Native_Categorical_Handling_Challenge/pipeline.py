import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor, cv, Pool

def load_data():
    df = pd.read_csv('exercises/18_Native_Categorical_Handling_Challenge/AmesHousing.csv')
    df = df.drop(columns={'Order', 'PID'})

    cat_features = df.select_dtypes(include=['object']).columns.tolist()
    df[cat_features] = df[cat_features].astype(str)

    return df.drop(columns={'SalePrice'}), df['SalePrice']

if __name__ == "__main__":
    X, y = load_data()
    y = np.log1p(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    cat_features = X.select_dtypes(include=['object']).columns.tolist()

    model = CatBoostRegressor(
        iterations=2000,
        learning_rate=0.05,
        loss_function='RMSE',
        random_seed=42,
        early_stopping_rounds=25,
        verbose=100, task_type="GPU"
    )
    model.fit(X_train, y_train, cat_features=cat_features, eval_set=(X_test, y_test))

    y_test_original = np.expm1(y_test)
    y_pred = np.expm1(model.predict(X_test))

    rmse = np.sqrt(mean_squared_error(y_test_original, y_pred))
    r2 = r2_score(y_test_original, y_pred)

    print(f'r2: {r2:.4f} | rmse: {rmse:.4f}')

    importances = model.get_feature_importance(type="PredictionValuesChange")
    feature_names = X.columns
    feat_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)

    print(feat_importances.head(10))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.scatter(y_test_original, y_pred, alpha=0.5, color='teal')
    line_vals = [y_test_original.min(), y_test_original.max()]
    ax1.plot(line_vals, line_vals, '--', color='red', linewidth=2)
    ax1.set_xlabel('Actual Price ($)')
    ax1.set_ylabel('Predicted Price ($)')
    ax1.grid(True, linestyle='--', alpha=0.7)

    top_15_features = feat_importances.head(15)
    top_15_features.plot(kind='barh', ax=ax2, color='skyblue')
    ax2.invert_yaxis()
    ax2.set_xlabel('Importance Score')
    ax2.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('exercises/18_Native_Categorical_Handling_Challenge/catboost.png')

    evals_result = model.get_evals_result()
    train_loss = evals_result['learn']['RMSE']
    test_loss = evals_result['validation']['RMSE']

    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label='Training Loss (RMSE)', color='blue')
    plt.plot(test_loss, label='Validation Loss (RMSE)', color='red')
    plt.ylabel('RMSE (Log Scale)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('exercises/18_Native_Categorical_Handling_Challenge/loss_curve.png')