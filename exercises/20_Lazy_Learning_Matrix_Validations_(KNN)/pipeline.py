import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, log_loss

def load_data():
    df = pd.read_csv('exercises/20_Lazy_Learning_Matrix_Validations_(KNN)/iris_extended.csv')

    return df.drop(columns={'species'}), df['species']

if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    ct = ColumnTransformer([
        ('num', StandardScaler(), X.drop(columns=['soil_type']).columns.tolist()),
        ('cat', OneHotEncoder(), ['soil_type'])
    ])
    X_train_scaled = ct.fit_transform(X_train)
    X_test_scaled = ct.transform(X_test)

    param_grid = {
        'n_neighbors': np.arange(1, 51)
    }

    model = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, n_jobs=-1)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_probs = model.predict_proba(X_test_scaled)

    print(model.best_params_)
    print(f'ce: {log_loss(y_test, y_probs):.4f}')
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    """
    error_rate = []
    for i in range(1, 51):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train_scaled, y_train)
        pred_i = knn.predict(X_test_scaled)
        error_rate.append(np.mean(pred_i != y_test))

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 51), error_rate, color='blue', linestyle='dashed', 
            marker='o', markerfacecolor='red', markersize=8)
    plt.xlabel('K')
    plt.ylabel('Error Rate')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('exercises/20_Lazy_Learning_Matrix_Validations_(KNN)/loss.png')
    """