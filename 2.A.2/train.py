import pandas as pd
import numpy as np

from sklearnex import patch_sklearn, config_context
patch_sklearn(verbose=False)

from sklearn.metrics import log_loss, mean_absolute_error, accuracy_score

def trainModel(model, X_train, y_train, X_val, y_val, epochs=100, patience=10, tol=1e-5, monitor='val_loss', enable_early_stopping=True):
    history_list = []

    best_val_loss = float('inf')
    best_val_acc = -float('inf')
    best_weights = None
    best_intercept = None
    counter = 0

    for epoch in range(epochs):
        with config_context(target_offload="gpu"):
            model.fit(X_train, y_train)

            y_train_prob = model.predict_proba(X_train)
            y_val_prob = model.predict_proba(X_val)
            
            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)

        metrics = {
            'train_loss': log_loss(y_train, y_train_prob),
            'val_loss': log_loss(y_val, y_val_prob),
            'train_acc': accuracy_score(y_train, y_train_pred),
            'val_acc': accuracy_score(y_val, y_val_pred),
            'train_mae': mean_absolute_error(y_train, y_train_prob[:, 1]),
            'val_mae': mean_absolute_error(y_val, y_val_prob[:, 1])
        }
        history_list.append(metrics)

        v_loss, v_acc = metrics['val_loss'], metrics['val_acc']
        improved = False
        if monitor == 'val_acc':
            if v_acc > (best_val_acc + tol):
                best_val_acc = v_acc
                improved = True
        else:
            if v_loss < (best_val_loss - tol):
                best_val_loss = v_loss
                improved = True

        if improved:
            best_weights, best_intercept = model.coef_.copy(), model.intercept_.copy()
            counter = 0
        else:
            counter += 1

        if enable_early_stopping and counter >= patience:
            print(f"early stopping at epoch {epoch+1}")
            break

    if best_weights is not None:
        model.coef_, model.intercept_ = best_weights, best_intercept

    return pd.DataFrame(history_list)