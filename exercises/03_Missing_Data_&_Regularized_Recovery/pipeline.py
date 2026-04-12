import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.experimental import enable_iterative_imputer

from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import r2_score

def load_data(split_size):
    dataset = pd.read_csv("exercises/03_Missing_Data_&_Regularized_Recovery/house.csv")
    dataset = dataset.drop(columns={'Id'})

    X = dataset.drop(columns={'SalePrice'})
    y = dataset['SalePrice']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_size, random_state=42)

    column_audit = {
        col: {
            'dtype': str(X[col].dtype),
            'unique_count': X[col].nunique(),
            'missing_count': X[col].isnull().sum(),
            'values': X[col].unique().tolist()[:5]
        } 
        for col in X.columns
    }

    standardScalar_columns = [col for col, data in column_audit.items() if data['dtype'] != 'object' and col != 'SalePrice']
    targetEncoding_columns = [col for col, data in column_audit.items() 
                          if data['dtype'] == 'object' and data['unique_count'] > 10]
    oneHotEncoding_columns = [col for col, data in column_audit.items() 
                         if data['dtype'] == 'object' and data['unique_count'] <= 10]

    num_pipe = Pipeline([
        ('imputer', IterativeImputer(estimator=Ridge(), random_state=42)),
        ('scaler', StandardScaler())
    ])

    high_card_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', TargetEncoder())
    ])

    low_card_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    transformer = ColumnTransformer([
        ('numerical', num_pipe, standardScalar_columns),
        ('high_complexity', high_card_pipe, targetEncoding_columns),
        ('low_complexity', low_card_pipe, oneHotEncoding_columns),
    ])

    X_train_scaled = transformer.fit_transform(X_train, y_train)
    X_test_scaled = transformer.transform(X_test)

    return X_train_scaled, X_train, X_test_scaled, X_test, y_train, y_test

if __name__ == "__main__":
    X_train_scaled, X_train, X_test_scaled, X_test, y_train, y_test = load_data(0.2)
    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test)

    param_grid = {
        'alpha': np.logspace(-4, -1, 50)
    }

    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    lasso_model = Lasso(max_iter=100000)
    model = GridSearchCV(lasso_model, param_grid, cv=kf, n_jobs=-1, scoring='neg_mean_squared_error')
    model.fit(X_train_scaled, y_train_log)

    y_pred = model.predict(X_test_scaled)

    r2 = r2_score(y_test_log, y_pred)

    print(f'params: {model.best_params_}')
    print(f'mse: {model.best_score_:.4f}')
    print(f'r2: {r2:.4f}')