import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def load_data():
    df = pd.read_csv("exercises/07_Centroid-Based_Manifold_Discovery_(K-Means++)/cc.csv")
    df = df.drop(columns={'CUST_ID'})
    df = df.fillna(df.median())

    return df

if __name__ == "__main__":
    X = load_data()
    X_scaled = StandardScaler().fit_transform(X)

    kmeans = KMeans(init='k-means++', max_iter=100000, n_clusters=12, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, clusters)

    print(score)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    var_pca = pca.explained_variance_ratio_
    
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='tab10', edgecolors='k', alpha=0.6)
    centroids_pca = pca.transform(kmeans.cluster_centers_)
    plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], c='red', s=200, marker='X', label='Centroides')
    plt.title('K-Means (K=12)')
    plt.xlabel(f'PC1 ({var_pca[0]:.2%})')
    plt.ylabel(f'PC2 ({var_pca[1]:.2%})')
    plt.legend()
    plt.savefig("exercises/07_Centroid-Based_Manifold_Discovery_(K-Means++)/clusters_2d.png")

    X['Cluster'] = clusters
    cluster_means = X.groupby('Cluster').mean()
    global_mean = X.drop(columns='Cluster').mean()
    global_std = X.drop(columns='Cluster').std()
    
    relevance = (cluster_means - global_mean) / global_std

    persona_map = {
        0: "The Debt Bound",                # High MINIMUM_PAYMENTS
        1: "The Diligent Payers",           # High PRC_FULL_PAYMENT
        2: "The Credit Conscious",          # Also high PRC_FULL_PAYMENT
        3: "The Quick Cash Reliant",        # High CASH_ADVANCE_FREQUENCY
        4: "The Daily Shoppers",            # High PURCHASES_TRX
        5: "The Cash Advance Users",        # High CASH_ADVANCE
        6: "The Luxury Buyers",             # High ONEOFF_PURCHASES (+13.11!)
        7: "The Long-Term Holders",         # TENURE
        8: "The Strategic Planners",        # ONEOFF_PURCHASES_FREQUENCY
        9: "The Occasional Borrowers",      # Moderate CASH_ADVANCE_FREQUENCY
        10: "The Buy-Now-Pay-Later Fans",   # PURCHASES_INSTALLMENTS_FREQUENCY
        11: "The Constant Transactors"      # BALANCE_FREQUENCY
    
    }
    for cid in persona_map:
        top_feat = relevance.loc[cid].idxmax()
        top_val = relevance.loc[cid].max()
        print(f"Cluster {cid:02d} | {persona_map[cid]:<25} | Key: {top_feat} (+{top_val:.2f} std)")

"""
    inertia = []
    k_range = range(1, 32 + 1)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, max_iter=100000, random_state=42)
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertia)
    plt.xlabel('Clusters (K)')
    plt.ylabel('WCSS')
    plt.grid(True)
    plt.savefig("exercises/07_Centroid-Based_Manifold_Discovery_(K-Means++)/elbow.png")
"""