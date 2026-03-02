import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import TimeSeriesSplit

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.metrics import mean_absolute_error, mean_squared_error

def loadDataset():
    df = pd.read_csv("2.A/financial_forecasting_dataset.csv")
    df = df[df['Ticker'] == 'AAPL'].sort_values('Date')
    for i in range(1, 6):
        df[f'Close Lag {i}'] = df['Close'].shift(i)
    df = df.dropna()
    return df

if __name__ == '__main__':
    df = loadDataset()
    X = df[[f'Close Lag {i}' for i in range(1,6)]]
    y = df['Close']

    tscv = TimeSeriesSplit(n_splits=5)
    for trainIndex, testIndex in tscv.split(X):
        X_train, X_test = X.iloc[trainIndex], X.iloc[testIndex]
        y_train, y_test = y.iloc[trainIndex], y.iloc[testIndex]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    ridge_alphas = np.logspace(-15, -2, 10)
    lasso_alphas = np.logspace(-9, -3, 10)

    l2_model = RidgeCV(
            alphas=ridge_alphas,
            cv=tscv
        )

    l1_model = LassoCV(
            alphas=lasso_alphas,
            cv=tscv
        )

    plt.figure(figsize=(14, 7))
    plt.plot(y_test.index, y_test.values, color='black', label='Actual Price', linewidth=1.5, alpha=0.7)

    models = {'L1 (Lasso)': l1_model, 'L2 (Ridge)': l2_model}
    colors = {'L1 (Lasso)': 'red', 'L2 (Ridge)': 'blue'}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        print(f"{name} - mae: {mae:.4f} | rmse: {rmse:.4f}")
        plt.plot(y_test.index, y_pred, color=colors[name], label=f'{name} Prediction', linewidth=2)

    plt.title('L1 vs L2 (Scikit-learn)')
    plt.xlabel('Sample Index')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('2.A/2.A_scikitlearn.png')