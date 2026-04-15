import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score

def load_data():
    df = pd.read_csv("exercises/06_Unsupervised_Density-Based_Clustering_(DBSCAN)/Mall_Customers.csv")
    df = df.drop(columns={'CustomerID'})
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})

    return df.drop(columns={'Gender', 'Age'})

if __name__ == "__main__":
    X = load_data()
    X_scaled = StandardScaler().fit_transform(X)

    neighbors = NearestNeighbors(n_neighbors=5)
    neighbors_fit = neighbors.fit(X_scaled)
    distances, indices = neighbors_fit.kneighbors(X_scaled)

    distances = np.sort(distances[:, 4], axis=0)
    plt.plot(distances)
    plt.title('K-Distance Graph')
    plt.xlabel('Points sorted by distance')
    plt.ylabel('5th Nearest Neighbor Distance (eps)')
    plt.savefig('exercises/06_Unsupervised_Density-Based_Clustering_(DBSCAN)/nearestneighbors_eps.png')

    dbscan = DBSCAN(eps=0.5, min_samples=5)
    clusters = dbscan.fit_predict(X_scaled)

    score = silhouette_score(X_scaled, clusters)
    print(score)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    variance_ratio = np.sum(pca.explained_variance_ratio_) * 100
    plt.figure(figsize=(10, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', edgecolors='k', alpha=0.7)
    plt.title(f'Clusters DBSCAN (Score: {score:.3f})')
    plt.savefig("exercises/06_Unsupervised_Density-Based_Clustering_(DBSCAN)/pca_clusters.png")