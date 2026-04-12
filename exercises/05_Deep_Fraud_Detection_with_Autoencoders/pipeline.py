import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, average_precision_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{device}")

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_data():
    dataset = pd.read_csv("exercises/05_Deep_Fraud_Detection_with_Autoencoders/creditcard.csv")

    return dataset.drop(columns={'Class'}), dataset['Class']

class CCFraudDetectorAE(nn.Module):
    def __init__(self, input_dim=30):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 32)
        )

        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, input_dim)
        )

    def forward(self, X):
        return self.decoder(self.encoder(X))
    
if __name__ == "__main__":
    set_seed()
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    X_train_normal = X_train[y_train == 0]

    scalar = StandardScaler()
    X_train_scaled = scalar.fit_transform(X_train_normal)
    X_test_scaled = scalar.transform(X_test) 

    X_train_scaled = torch.FloatTensor(X_train_scaled).to(device)
    X_test_scaled = torch.FloatTensor(X_test_scaled).to(device)

    model = CCFraudDetectorAE(input_dim=30).to(device)
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    history = {'train_loss': [], 'val_auprc': []}
    model.train()
    for epoch in range(400):
        optimizer.zero_grad()
        output = model(X_train_scaled)

        loss = loss_function(output, X_train_scaled)
        loss.backward()

        optimizer.step()

        model.eval()
        with torch.no_grad():
            history['train_loss'].append(loss.item())

            test_recon = model(X_test_scaled)
            test_loss_all = torch.mean((X_test_scaled - test_recon)**2, dim=1).cpu().numpy()
            current_auprc = average_precision_score(y_test, test_loss_all)
            history['val_auprc'].append(current_auprc)

        if epoch % 10 == 0:
            print(f'epoch {epoch:03d} | loss: {loss.item():.6f} | auprc: {current_auprc:.4f}')

    model.eval()
    with torch.no_grad():
        latent_features = model.encoder(X_train_scaled).cpu().numpy()

        train_recon = model(X_train_scaled)
        train_loss = torch.mean((X_train_scaled - train_recon)**2, dim=1).cpu().numpy()

        threshold = np.percentile(train_loss, 95)

        test_recon = model(X_test_scaled)
        test_loss = torch.mean((X_test_scaled - test_recon)**2, dim=1).cpu().numpy()

        y_pred = (test_loss > threshold).astype(int)
            
        auprc = average_precision_score(y_test, test_loss)

        print(f"auprc: {auprc:.4f}")
        print(classification_report(y_test, y_pred))
        print(confusion_matrix(y_test, y_pred))

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train MSE Loss', color='blue')
    plt.title('Training Loss Progress')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(history['val_auprc'], label='Validation AUPRC', color='green')
    plt.title('AUPRC Progress')
    plt.xlabel('Epoch')
    plt.ylabel('AUPRC Score')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('exercises/05_Deep_Fraud_Detection_with_Autoencoders/metrics.png')