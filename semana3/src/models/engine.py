import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.stats import pearsonr


class ClusteringModelEngine:
    """
    Manages clustering algorithms (KMeans, DBSCAN) and
    dimensionality reduction techniques (PCA, t-SNE).
    """

    @staticmethod
    def compute_correlation(series1, series2):
        """Computes Pearson correlation between two series."""
        corr, p_value = pearsonr(series1, series2)
        print(f"Correlación de Pearson: {corr:.4f} (p-value: {p_value:.4e})")
        return corr, p_value

    @staticmethod
    def find_optimal_k(X_scaled, k_range=range(1, 10)):
        """Runs KMeans for each k and returns inertia values for elbow method."""
        print("\nCalculando inercia para método del codo...")
        inertia = []
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=42)
            km.fit(X_scaled)
            inertia.append(km.inertia_)
        return inertia

    @staticmethod
    def run_kmeans(X_scaled, n_clusters=4):
        """Fits KMeans and returns labels and model."""
        print(f"\nEntrenando KMeans con k={n_clusters}...")
        model = KMeans(n_clusters=n_clusters, random_state=42)
        labels = model.fit_predict(X_scaled)
        print(f"Clusters encontrados: {np.unique(labels)}")
        return labels, model

    @staticmethod
    def get_cluster_centers_real(model, scaler, columns):
        """Inverse-transforms centroids to original scale and returns DataFrame."""
        centros_reales = scaler.inverse_transform(model.cluster_centers_)
        df_centros = pd.DataFrame(centros_reales, columns=columns)
        df_centros['Cluster'] = range(len(model.cluster_centers_))
        return df_centros

    @staticmethod
    def run_dbscan(X_scaled, eps=0.6, min_samples=5):
        """Fits DBSCAN and returns labels and model."""
        print(f"\nEntrenando DBSCAN (eps={eps}, min_samples={min_samples})...")
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(X_scaled)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        print(f"Clusters encontrados: {n_clusters}, Puntos de ruido: {n_noise}")
        return labels, model

    @staticmethod
    def run_pca(X_scaled, n_components=2):
        """Fits PCA and returns transformed data and model."""
        print(f"\nAplicando PCA (n_components={n_components})...")
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        print(f"Varianza explicada: {pca.explained_variance_ratio_}")
        return X_pca, pca

    @staticmethod
    def run_tsne(X_scaled, n_components=2, perplexity=30, learning_rate=200):
        """Fits t-SNE and returns transformed data."""
        print(f"\nAplicando t-SNE (perplexity={perplexity}, lr={learning_rate})...")
        tsne = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            learning_rate=learning_rate,
            random_state=42
        )
        X_tsne = tsne.fit_transform(X_scaled)
        return X_tsne
