import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_data():
    dataset = pd.read_csv("exercises/01_Temporal_Anomaly_Detection/AEP_hourly.csv")
    dataset['Datetime'] = pd.to_datetime(dataset['Datetime'])

    dataset['Datetime'] = dataset['Datetime'].dt.hour

     # 2pi = 24 horas
     # para descobrir o sin e cos de x horas, basta calcular o rad dessa hora
     # rad∝ = 2 * np.pi * x / 24
     # x = hora desejada
    dataset['hourly_sin'] = np.sin( 2 * np.pi * dataset['Datetime'] / 24)
    dataset['hourly_cos'] = np.cos( 2 * np.pi * dataset['Datetime'] / 24)
    dataset = dataset.drop(columns={'Datetime'})

    return dataset

class AnomalyDetectionAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super(AnomalyDetectionAutoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

        self.decoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )

    def forward(self, X):
        return self.decoder(self.encoder(X))
    
if __name__ == "__main__":
    set_seed(42)
    X = load_data()

    scalar = StandardScaler()
    X_scaled = scalar.fit_transform(X)
    X_scaled = torch.FloatTensor(X_scaled)

    model = AnomalyDetectionAutoencoder(input_dim=3)
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        output = model(X_scaled)

        loss = loss_function(output, X_scaled)
        loss.backward()

        optimizer.step()

        if epoch % 10 == 0:
            print(f'epoch {epoch}, loss: {loss.item():.6f}')

    model.eval()
    with torch.no_grad():
        latent_features = model.encoder(X_scaled).numpy()

        reconstructed = model(X_scaled)
        sample_losses = torch.mean((X_scaled - reconstructed)**2, dim=1).numpy()

    threashold = np.percentile(sample_losses, 99)
    anomalies = sample_losses > threashold

    normal_latent = latent_features[~anomalies]
    anomaly_latent = latent_features[anomalies]

    print(f"anomalies: {anomalies.sum()}")

    plt.figure(figsize=(10, 8))
    
    plt.scatter(normal_latent[:, 0], normal_latent[:, 1], 
                c='blue', alpha=0.1, s=2, label='Normal')

    plt.scatter(anomaly_latent[:, 0], anomaly_latent[:, 1], 
                c='red', alpha=0.5, s=10, label='Anomaly')
    
    plt.title("Latent Space")
    plt.xlabel("LF1")
    plt.ylabel("LF2")
    plt.legend()
    plt.savefig('exercises/01_Temporal_Anomaly_Detection/laetent_space.png')