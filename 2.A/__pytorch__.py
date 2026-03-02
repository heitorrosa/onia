import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def loadDataset():
    df = pd.read_csv("2.A/financial_forecasting_dataset.csv")
    df = df[df['Ticker'] == 'AAPL'].sort_values('Date')
    for i in range(1, 6):
        df[f'Close Lag {i}'] = df['Close'].shift(i)
    df = df.dropna()

    X = df[[f'Close Lag {i}' for i in range(1, 6)]].values
    y = df['Close'].values.reshape(-1, 1)

    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class Model(nn.Module):
    def __init__(self, input_dim, l1_lambda=0.0, l2_lambda=0.0):
        super(Model, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda

    def forward(self, x):
        return self.linear(x)

class ModelPersistence:
    @staticmethod
    def saveBestModels(best_models_state, metrics, path_prefix="2.A/"):
        for name, state in best_models_state.items():
            clean_name = name.replace(" ", "_").replace("(", "").replace(")", "").lower()
            avg_mae = np.mean(metrics[name]['mae'])
            path = f"{path_prefix}{clean_name} ({avg_mae:.4f}).pth"
            torch.save(state, path)

    @staticmethod
    def loadModel(input_dim, path, l1=0.0, l2=0.0):
        model = Model(input_dim, l1_lambda=l1, l2_lambda=l2)
        model.load_state_dict(torch.load(path))
        model.to(device)
        model.eval()
        return model

class CrossValidationTrainer:
    def __init__(self, ridge_alphas, lasso_alphas, epochs=100):
        self.ridge_alphas = ridge_alphas
        self.lasso_alphas = lasso_alphas
        self.epochs = epochs
        self.metrics = {'L1 (Lasso)': {'mae': [], 'rmse': []}, 'L2 (Ridge)': {'mae': [], 'rmse': []}}
        self.last_preds = {}
        self.best_models_state = {}

    def trainAnalyticalRidge(self, model, X, y):
        I = torch.eye(X.size(1)).to(device)
        X_t = X.t()
        w = torch.linalg.solve(X_t @ X + model.l2_lambda * I, X_t @ y)
        model.linear.weight.data, model.linear.bias.data = w.t(), torch.tensor([0.0]).to(device)

    def trainGradientDescent(self, model, X, y, X_val=None, y_val=None):
        optimizer = optim.Adam(model.parameters(), lr=0.2)
        criterion = nn.MSELoss()
        history = {'loss': [], 'val_loss': []}
        
        for _ in range(self.epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            if model.l1_lambda > 0:
                loss += model.l1_lambda * sum(p.abs().sum() for p in model.parameters())
            loss.backward()
            optimizer.step()
            
            history['loss'].append(loss.item())
            
            if X_val is not None:
                model.eval()
                with torch.no_grad():
                    v_loss = criterion(model(X_val), y_val)
                    history['val_loss'].append(v_loss.item())
        
        return history

    def runCrossValidation(self, X, y, n_splits=5):
        tscv = TimeSeriesSplit(n_splits=n_splits)
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train_raw, X_test_raw = X[train_idx], X[test_idx]
            y_train_raw, y_te_raw = y[train_idx], y[test_idx]
            
            scaler_x, scaler_y = StandardScaler(), StandardScaler()
            X_train = torch.tensor(scaler_x.fit_transform(X_train_raw), dtype=torch.float32).to(device)
            y_train = torch.tensor(scaler_y.fit_transform(y_train_raw), dtype=torch.float32).to(device)
            X_test = torch.tensor(scaler_x.transform(X_test_raw), dtype=torch.float32).to(device)

            print(f"--- Fold {fold+1} ---")
            for name in self.metrics.keys():
                best_mae, alphas = float('inf'), (self.lasso_alphas if 'L1' in name else self.ridge_alphas)
                for alpha in alphas:
                    model = Model(5, l1_lambda=(alpha if 'L1' in name else 0.0), l2_lambda=(alpha if 'L2' in name else 0.0)).to(device)
                    self.trainAnalyticalRidge(model, X_train, y_train) if 'L2' in name else self.trainGradientDescent(model, X_train, y_train)
                    
                    with torch.no_grad():
                        preds = scaler_y.inverse_transform(model(X_test).cpu().numpy())
                    mae = mean_absolute_error(y_te_raw.numpy(), preds)
                    if mae < best_mae:
                        best_mae, best_rmse, best_pred = mae, np.sqrt(mean_squared_error(y_te_raw.numpy(), preds)), preds
                        self.best_models_state[name] = model.state_dict()

                self.metrics[name]['mae'].append(best_mae)
                self.metrics[name]['rmse'].append(best_rmse)
                self.last_preds[name] = best_pred
                print(f"  {name} - mae: {best_mae:.4f} | rmse: {best_rmse:.4f}")

if __name__ == '__main__':
    X, y = loadDataset()
    trainer = CrossValidationTrainer(np.logspace(-10, 1, 5), np.logspace(-5, -2, 5))
    trainer.runCrossValidation(X, y)

    ModelPersistence.saveBestModels(trainer.best_models_state, trainer.metrics)

    for name, m in trainer.metrics.items():
        print(f"{name} - Mean MAE: {np.mean(m['mae']):.4f} | Mean RMSE: {np.mean(m['rmse']):.4f}")

    plt.figure(figsize=(14, 7))
    plt.plot(y[-len(trainer.last_preds['L1 (Lasso)']):], color='black', label='Actual', alpha=0.7)
    for name, pred in trainer.last_preds.items():
        plt.plot(pred, label=f"{name} Prediction")
    plt.legend()
    plt.savefig('2.A/2.A_pytorch.png')