import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearnex import patch_sklearn, config_context
patch_sklearn(verbose=False)

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier

from train import trainModel

def loadDataset():
    df = pd.read_csv("2.A.2/diabetes.csv")
    df = df.dropna()

    X = df.drop(columns=['Outcome'])
    y = df['Outcome']

    return X, y

if __name__ == "__main__":
    X, y = loadDataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # gpu
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    param_grid = {
        'alpha': np.logspace(-6, -1, 10).tolist(),
        'eta0': [0.001, 0.002, 0.005, 0.008, 0.01, 0.012, 0.015, 0.02, 0.05, 0.1],
        'l1_ratio': [0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.5, 0.8, 1.0],

        'tol': [1e-5, 1e-6]
    }

    with config_context(target_offload="gpu"):
        gs_model = SGDClassifier(loss='log_loss', penalty='elasticnet', learning_rate='adaptive', random_state=42, max_iter=1000)
        grid_search = GridSearchCV(gs_model, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
        grid_search.fit(X_train, y_train)
    
    best_params = grid_search.best_params_

    model = SGDClassifier(
            loss='log_loss',

            penalty='elasticnet',
            l1_ratio=best_params['l1_ratio'],
            alpha=best_params['alpha'],

            max_iter=1,
            learning_rate='adaptive',
            eta0=best_params['eta0'],
            tol=None,

            n_jobs=-1,
            random_state=42,
            warm_start=True,
        )
    
    results = trainModel(
        model,
        X_train, y_train,
        X_test, y_test,
        epochs=50,
        patience=2,
        tol=best_params['tol'],
        monitor='val_loss',
        enable_early_stopping=True
    )

    epochs_range = results.index + 1
    best_epoch_idx = results['val_loss'].idxmin()
    best_loss = results.loc[best_epoch_idx, 'val_loss']
    best_acc = results.loc[best_epoch_idx, 'val_acc']
    best_mae = results.loc[best_epoch_idx, 'val_mae']

    fig, (ax1,ax2,ax3) = plt.subplots(1,3, figsize=(18, 6))

    # plot 1: loss
    ax1.plot(epochs_range, results['train_loss'], label='Training Loss')
    ax1.plot(epochs_range, results['val_loss'], label='Validation Loss')
    ax1.set_title('Loss over Epochs')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # plot 2: acc
    ax2.plot(epochs_range, results['train_acc'], label='Training Accuracy')
    ax2.plot(epochs_range, results['val_acc'], label='Validation Accuracy')
    ax2.set_title('Accuracy over Epochs')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    # plot 3: mae
    ax3.plot(epochs_range, results['train_mae'], label='Training MAE')
    ax3.plot(epochs_range, results['val_mae'], label='Validation MAE')
    ax3.set_title('MAE over Epochs')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('MAE')
    ax3.legend()

    plt.tight_layout()
    plt.savefig("2.A.2/2.A.2.png")

    print(f"best params: {best_params}")
    print(f"ce: {best_loss:.4f} | acc: {best_acc:.4f} | mae: {best_mae:.4f}")

    # best params: {'alpha': 0.007742636826811277, 'eta0': 0.005, 'l1_ratio': 1.0, 'tol': 1e-05}
    # ce: 0.5029 | acc: 0.7662 | mae: 0.3369