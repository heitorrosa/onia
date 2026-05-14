import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC

def load_data():
    df = pd.read_csv('exercises/19_SVM_Kernel_Subspace_Projections/activity.csv')
    
    #print(df['Activity'].value_counts())

    return df.drop(columns={'Activity'}), df['Activity']

if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    param_grid = {
        'C': np.logspace(-3, 2, 25),
        'kernel': ['linear'],
    }
    svm = GridSearchCV(SVC(random_state=42), param_grid, cv=5, n_jobs=-1)
    svm.fit(X_train_scaled, y_train)

    y_pred = svm.predict(X_test_scaled)

    print(svm.best_params_)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    labels = ['LAYING', 'SITTING', 'STANDING', 'WALKING', 'WALKING_DOWN', 'WALKING_UP']
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.ylabel('True')
    plt.xlabel('Pred')
    plt.savefig('exercises/19_SVM_Kernel_Subspace_Projections/confusion_matrix.png')