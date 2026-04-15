import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR

# cut - 5
# color - 7
# clarity - 8

def load_data():
    df = pd.read_csv("exercises/14_Support_Vector_Tube_Extrapolations/diamonds.csv")
    df = df.drop(columns=['Unnamed: 0'])

    return df.drop(columns={'carat'}), df['carat']

if __name__ == "__main__":
    start_time = time.time()

    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    ct = ColumnTransformer([
        ('ordinal', OrdinalEncoder(), ['cut', 'color', 'clarity']),
        ('scaler', StandardScaler(), ['depth', 'table', 'price', 'x', 'y', 'z'])
    ])

    X_train_scaled = ct.fit_transform(X_train)
    X_test_scaled = ct.transform(X_test)

    X_train_small, _, y_train_small, _ = train_test_split(
        X_train_scaled, y_train, train_size=5000, random_state=42
    )

    c_range = np.linspace(0.1, 10, 5)
    epsilon_range = np.linspace(0.01, 1, 5)
    gamma_range = np.linspace(0.001, 0.1, 3)

    param_grid = [
        {
            'kernel': ['rbf'],
            'C': c_range,
            'epsilon': epsilon_range,
            'gamma': gamma_range
        },
        {
            'kernel': ['linear'],
            'C': c_range,
            'epsilon': epsilon_range
        }
    ]

    svr = GridSearchCV(SVR(), param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
    svr.fit(X_train_scaled, y_train)

    print(f"mae: {-svr.best_score_:.4f}")
    print(f"params: {svr.best_params_}")

    y_pred = svr.predict(X_test_scaled)
    
    sort_idx = np.argsort(X_test_scaled[:500, 6])
    x_line = X_test_scaled[:500, 6][sort_idx]
    y_line = y_pred[:500][sort_idx]

    eps = svr.best_params_['epsilon']
    
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test_scaled[:500, 6], y_test[:500], alpha=0.3, label='Real Data', color='gray')
    plt.plot(x_line, y_line, color='red', label='SVR Prediction')
    plt.fill_between(x_line, y_line - eps, y_line + eps, color='red', alpha=0.2, label=f'SVR Tube (eps={eps})')
    plt.title('SVR Tube Extrapolation (Carat Prediction)')
    plt.xlabel('Diamond Length (X) - Scaled')
    plt.ylabel('Carat Weight')
    plt.legend()
    plt.savefig('exercises/14_Support_Vector_Tube_Extrapolations/svr.png')

    print(f'{time.time() - start_time}s')