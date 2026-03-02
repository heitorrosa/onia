import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearnex import patch_sklearn, config_context
patch_sklearn(verbose=False)

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, train_test_split
from sklearn.metrics import classification_report, RocCurveDisplay
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC

def loadDataset():
    df = pd.read_csv('2.B/stock_market_data_large.csv')
    df = df.drop(columns={'Sentiment', 'Volume', 'Target', 'Low', 'High', 'Open'})
    df = df.set_index('Date')

    df['Returns'] = df['Close'].pct_change()
    df['Volatility'] = df['Returns'].rolling(10).std()

    df['Next Day Return'] = df['Returns'].shift(-1) 
    df['Target'] = (df['Next Day Return'] > 0.02).astype(int)

    df = df.drop(columns={'Next Day Return'})
    df = df.dropna()

    X = df.drop(columns={'Target', 'Close'})
    y = df['Target']

    return X, y

if __name__ == "__main__":
    X, y = loadDataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    dt_params = {
        'max_depth': [3, 4, 5],
        'min_samples_leaf': [25, 50],
        'criterion': ['gini', 'entropy']
    }

    svm_params = {
        'C': [0.01, 0.1, 1],
        'gamma': ['scale'],
        'kernel': ['rbf']
    }

    cv = 5
    grid_dt = GridSearchCV(DecisionTreeClassifier(random_state=42, class_weight='balanced'), dt_params, cv=cv, scoring='f1', n_jobs=-1)
    grid_dt.fit(X_train, y_train)
    
    grid_svm = GridSearchCV(SVC(kernel='rbf', degree=4, probability=True, class_weight='balanced'), svm_params, cv=cv, scoring='f1', n_jobs=-1)
    grid_svm.fit(X_train_s, y_train)

    dt = grid_dt.best_estimator_
    svm = grid_svm.best_estimator_

    dt.fit(X_train, y_train)
    svm.fit(X_train_s, y_train)

    y_pred_tree = dt.predict(X_test)
    y_pred_svm = svm.predict(X_test_s)

    print("Decision Tree Report:")
    print(classification_report(y_test, y_pred_tree))

    print("\nSVM Report:")
    print(classification_report(y_test, y_pred_svm))

    # ROC Curve
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    RocCurveDisplay.from_estimator(dt, X_test, y_test, ax=ax)
    RocCurveDisplay.from_estimator(svm, X_test_s, y_test, ax=ax)
    plt.savefig('2.B/roc_curve.png')
    plt.close()

    # Decision Tree
    plt.figure(figsize=(30, 20))
    plot_tree(dt, 
              feature_names=X.columns.tolist(), 
              class_names=['Bear', 'Bull'], 
              filled=True, 
              rounded=True,
              max_depth=4,
              fontsize=10)
    plt.savefig('2.B/decision_tree.png')
    plt.close()

"""
    Decision Tree Report:
              precision    recall  f1-score   support

           0       0.75      0.04      0.08       1337
           1       0.42      0.98      0.59       935

    accuracy                           0.43       2272
    macro avg       0.58      0.51     0.33       2272
    weighted avg    0.61      0.43     0.29       2272

    SVM Report:
              precision    recall  f1-score   support

           0       0.62      0.39      0.48      1337
           1       0.43      0.66      0.52      935

    accuracy                           0.50      2272
    macro avg       0.53      0.53     0.50      2272
    weighted avg    0.54      0.50     0.50      2272
"""
