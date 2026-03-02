import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix
from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

def loadDataset():
    df = pd.read_csv("2.B.1/WA_Fn-UseC_-Telco-Customer-Churn.csv")

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)

    categorical_cols = [
        'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
        'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 
        'PaperlessBilling', 'PaymentMethod'
    ]

    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    df = df.drop(columns=['customerID'])
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    df = df.astype({col: 'int' for col in df.select_dtypes('bool').columns})
    df = df.dropna()

    X = df.drop(columns={'Churn'})
    y = df['Churn']

    return X, y
if __name__ == "__main__":
    X, y = loadDataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    dt_param_grid = {
        'max_depth': [4, 6, 8, 10],
        'criterion': ['gini', 'entropy'],
        'min_samples_split': [2, 5, 10],
        'class_weight': ['balanced']
    }
        
    svm_param_grid = {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto', 0.1, 0.01],
        'class_weight': ['balanced']
    }

    cv = 5
    dt_grid = GridSearchCV(DecisionTreeClassifier(random_state=42), dt_param_grid, cv=cv, scoring='f1', n_jobs=-1)
    dt_grid.fit(X_train, y_train)
    print(f"dt: {dt_grid.best_params_}")

    svm_grid = GridSearchCV(SVC(probability=True, random_state=42), svm_param_grid, cv=cv, scoring='f1', n_jobs=-1)
    svm_grid.fit(X_train_s, y_train)
    print(f"svm: {svm_grid.best_params_}")

    cv_dt = dt_grid.best_params_
    cv_svm = svm_grid.best_params_

    dt = DecisionTreeClassifier(
        max_depth=cv_dt['max_depth'],
        criterion=cv_dt['criterion'],
        min_samples_split=cv_dt['min_samples_split'],
        class_weight='balanced',
        random_state=42
    )

    svm = SVC(
        kernel='rbf',
        C=cv_svm['C'],
        gamma=cv_svm['gamma'],
        class_weight='balanced',
        probability=True,
        random_state=42
    )

    dt.fit(X_train, y_train)
    svm.fit(X_train_s, y_train)

    y_pred_tree = dt.predict(X_test)
    y_pred_svm = svm.predict(X_test_s)
    
    print("Decision Tree Report:")
    print(classification_report(y_test, y_pred_tree))

    print("\nSVM Report:")
    print(classification_report(y_test, y_pred_svm))

    # 1. Plot ROC Curves
    plt.figure(figsize=(10, 6))
    for model, name, X_data in [(dt, 'Decision Tree', X_test), (svm, 'SVM', X_test_s)]:
        probs = model.predict_proba(X_data)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC Curve Comparison')
    plt.legend()
    plt.savefig('2.B.1/roc_comparison_churn.png')
    plt.close()

    # 2. Plot Confusion Matrix for SVM
    plt.figure(figsize=(6, 4))
    cm = confusion_matrix(y_test, y_pred_svm)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('SVM Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig('2.B.1/svm_confusion_matrix.png')
    plt.close()

    # 3. Decision Tree Feature Importance
    plt.figure(figsize=(8, 6))
    importances = pd.Series(dt.feature_importances_, index=X.columns)
    importances.nlargest(5).sort_values().plot(kind='barh')
    plt.title('Top 5 Predictors of Churn (Decision Tree)')
    plt.savefig('2.B.1/decision_tree_feature_importance.png')
    plt.close()