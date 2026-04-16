import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

def gini(y_true, y_pred):
    return 2 * roc_auc_score(y_true, y_pred) - 1

def load_data():
    df = pd.read_csv('exercises/16_XGBoost_Hardware_Awareness_&_Sparsity/driver.csv')
    df = df.drop(columns={'id'})
    
    #print(df['target'].value_counts())

    return df.drop(columns={'target'}), df['target']

if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    counts = y_train.value_counts()
    scale_weight = counts[0] / counts[1]

    xgb = XGBClassifier(
        scale_pos_weight=scale_weight,
        alpha=15,
        reg_lambda=5,
        max_delta_step=1,
        max_depth=4,
        learning_rate=0.066,
        n_estimators=10000,
        early_stopping_rounds=100,
        eval_metric='auc',
        missing=-1, device='cuda'
    )
    xgb.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=False)

    y_pred = xgb.predict_proba(X_test)[:, 1]

    print(f'gini: {gini(y_test, y_pred):.4f}')

    results = xgb.evals_result()
    epochs = len(results['validation_0']['auc'])
    x_axis = range(0, epochs)

    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, results['validation_0']['auc'], label='Train')
    plt.plot(x_axis, results['validation_1']['auc'], label='Test')
    plt.axvline(xgb.best_iteration, color='r', linestyle='--', label=f'Best ({xgb.best_iteration})')
    plt.xlabel('Epochs')
    plt.ylabel('AUC')
    plt.title('XGBoost AUC over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('exercises/16_XGBoost_Hardware_Awareness_&_Sparsity/loss_plot.png')