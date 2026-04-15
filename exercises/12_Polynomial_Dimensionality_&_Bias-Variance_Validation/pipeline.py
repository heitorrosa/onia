import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, TargetEncoder, OneHotEncoder, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error

def load_data():
    df = pd.read_csv('exercises/12_Polynomial_Dimensionality_&_Bias-Variance_Validation/auto-mpg.csv')
    
    df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')
    df['horsepower'] = df['horsepower'].fillna(df['horsepower'].median())

    return df.drop(columns=['mpg']), df['mpg']

if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    ct = ColumnTransformer([
        ('scaler', StandardScaler(), ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year']),
        ('one_hot_encoding', OneHotEncoder(), ['origin']),
        ('target_encoder', TargetEncoder(target_type='continuous'), ['car name'])
    ])

    X_train_scaled = ct.fit_transform(X_train, y_train)
    X_test_scaled = ct.transform(X_test)

    train_maes = []
    test_maes = []
    for d in range(1, 7):
        ridge = Pipeline([
            ('poly', PolynomialFeatures(degree=d, include_bias=False)),
            ('scaler', StandardScaler()),
            ('ridge', RidgeCV(alphas=np.logspace(-3, 3, 50), cv=5, scoring='neg_mean_absolute_error'))
        ])
        ridge.fit(X_train_scaled, y_train)

        train_err = mean_absolute_error(y_train, ridge.predict(X_train_scaled))
        test_err = mean_absolute_error(y_test, ridge.predict(X_test_scaled))

        train_maes.append(train_err)
        test_maes.append(test_err)

        print(f"d: {d} | train mae: {train_err:.4f} |test mae: {test_err:.4f} | alpha: {ridge.named_steps['ridge'].alpha_:.3f}")

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 7), train_maes, label='Train MAE', marker='o', linestyle='--')
    plt.plot(range(1, 7), test_maes, label='Test MAE', marker='s', linewidth=2)
    plt.xlabel('Polynomial Degree')
    plt.ylabel('MAE')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('exercises/12_Polynomial_Dimensionality_&_Bias-Variance_Validation/validation_curve.png')