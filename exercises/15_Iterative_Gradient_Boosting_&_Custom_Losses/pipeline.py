import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score, make_scorer
from xgboost import XGBRegressor

def asymmetric_mse(y_true, y_pred):
    residual = y_true - y_pred
    grad = np.where(residual > 0, -2 * 5.0 * residual, -2 * 1.0 * residual)
    hess = np.where(residual > 0, 2 * 5.0, 2 * 1.0)
    return grad, hess

def asymmetric_eval(y_true, y_pred):
    residual = y_true - y_pred
    loss = np.where(residual > 0, 5.0 * (residual**2), 1.0 * (residual**2))
    return np.mean(loss)

asymmetric_scorer = make_scorer(asymmetric_eval, greater_is_better=False)

def load_data():
    df = pd.read_csv("exercises/15_Iterative_Gradient_Boosting_&_Custom_Losses/store-item.csv")
    df = df.sort_values(['store', 'item', 'date'])
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')

    group = df.groupby(['store', 'item'])['sales']

    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)

    df['lag_1'] = group.shift(1)
    df['lag_7'] = group.shift(7)

    df['MA7'] = group.transform(lambda x: x.rolling(7).mean())
    df['MA50'] = group.transform(lambda x: x.rolling(50).mean())

    df['sales'] = np.log1p(df['sales'])

    df = df.dropna()
    
    return df.drop(columns={'sales'}), df['sales']

if __name__ == "__main__":
    X, y = load_data()

    X_train = X[X.index < '2017-01-01']
    X_test = X[X.index >= '2017-01-01']
    y_train = y[y.index < '2017-01-01']
    y_test = y[y.index >= '2017-01-01']

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = XGBRegressor(
        learning_rate=0.05,
        max_depth=3,
        early_stopping_rounds=50,
        objective=asymmetric_mse, eval_metric=asymmetric_eval, device='cuda'
    )
    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_train_scaled, y_train), (X_test_scaled, y_test)],
        verbose=False,
    )

    preds = np.expm1(model.predict(X_test_scaled))
    y_test_original = np.expm1(y_test)

    rmse = np.sqrt(mean_squared_error(y_test_original, preds))
    mae = mean_absolute_error(y_test_original, preds)
    mape = mean_absolute_percentage_error(y_test_original, preds)
    r2 = r2_score(y_test_original, preds)

    print(f"rmse: {rmse:.4f} | mae: {mae:.4f} | mape: {mape:.4f} | r2: {r2:.4f}")

    results = model.evals_result()

    train_loss = results['validation_0']['asymmetric_eval']
    test_loss = results['validation_1']['asymmetric_eval']
    best_iteration = model.best_iteration
    epochs = range(len(train_loss))

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label='Train (Asymmetric Loss)')
    plt.plot(epochs, test_loss, label='Test (Asymmetric Loss)')
    plt.axvline(best_iteration, color='r', linestyle='--', label=f'Early Stop ({best_iteration})')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.grid(True)
    plt.savefig('exercises/15_Iterative_Gradient_Boosting_&_Custom_Losses/loss.png')

    plt.figure(figsize=(15, 6))
    plt.plot(y_test_original.values[:100], label='Actual Sales (2017)', color='blue', alpha=0.6, linewidth=2)
    plt.plot(preds[:100], label='Predicted Sales (Asymmetric)', color='orange', linestyle='--', linewidth=2)
    plt.xlabel('Days')
    plt.ylabel('Sales Volume')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('exercises/15_Iterative_Gradient_Boosting_&_Custom_Losses/comparison_line.png')