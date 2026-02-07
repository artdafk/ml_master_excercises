# -*- coding: utf-8 -*-
"""
Main entry point for Unsupervised Learning Models Analysis.
Refactored into Modular + OOP Architecture.
"""

import os
import sys

import pandas as pd

# Ensure src is in the python path to find modules if run from root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from semana3.src.data.processor import UserBehaviorDataProcessor
from semana3.src.models.engine import ClusteringModelEngine
from semana3.src.utils.visualizer import ClusteringVisualizer


def main():
    print("=" * 60)
    print("  UNSUPERVISED LEARNING MODEL ANALYSIS (MODULAR ARCHITECTURE)")
    print("=" * 60)

    # --- Configuration ---
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    assets_dir = os.path.join(project_root, "assets")
    dataset_path = os.path.join(project_root, "src", "data", "kaggle", "user_behavior_dataset.csv")

    print(f"Project Root: {project_root}")
    print(f"Assets Directory: {assets_dir}")
    print(f"Dataset Path: {dataset_path}")

    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset not found at {dataset_path}")
        return

    # --- Phase 1: Data Processing ---
    print("\n[PHASE 1] Data Processing")
    processor = UserBehaviorDataProcessor(dataset_path)
    df = processor.load_data()
    processor.preprocess()
    X_scaled = processor.get_scaled_data()

    # --- Phase 2: Exploratory Visualization ---
    print("\n[PHASE 2] Exploratory Visualization")
    visualizer = ClusteringVisualizer(output_dir=assets_dir)
    engine = ClusteringModelEngine()

    # Pairplot of selected variables
    pairplot_cols = ['Age', 'Screen On Time (hours/day)', 'Data Usage (MB/day)', 'Number of Apps Installed']
    visualizer.plot_pairplot(df, pairplot_cols)

    # Correlation: App Usage Time vs Screen On Time
    corr, _ = engine.compute_correlation(
        df['App Usage Time (hours/day)'],
        df['Screen On Time (hours/day)']
    )
    visualizer.plot_correlation(
        df,
        x_col='App Usage Time (hours/day)',
        y_col='Screen On Time (hours/day)',
        corr_value=corr
    )

    # --- Phase 3: KMeans Clustering ---
    print("\n[PHASE 3] KMeans Clustering")
    k_range = range(1, 10)
    inertia = engine.find_optimal_k(X_scaled, k_range)
    visualizer.plot_elbow_method(k_range, inertia)

    kmeans_labels, kmeans_model = engine.run_kmeans(X_scaled, n_clusters=4)
    df['KMeans_Cluster'] = kmeans_labels

    visualizer.plot_cluster_scatter(
        X_scaled, kmeans_labels,
        title='Segmentación por K-Means (Screen on Time vs Data Usage)',
        x_label='Screen On Time (escalado)',
        y_label='Data Usage (escalado)',
        save_path='segmentacion_kmeans.png'
    )

    # Centroids in original scale
    df_centros = engine.get_cluster_centers_real(
        kmeans_model, processor.scaler, processor.selected_variables
    )
    print("\nCentroides de KMeans (escala original):")
    print(df_centros)

    # Group statistics
    print("\nEstadísticas por cluster KMeans:")
    print(df.groupby('KMeans_Cluster')[processor.selected_variables].mean())

    # --- Phase 4: DBSCAN Clustering ---
    print("\n[PHASE 4] DBSCAN Clustering")
    dbscan_labels, dbscan_model = engine.run_dbscan(X_scaled, eps=0.6, min_samples=5)
    df['DBSCAN_Cluster'] = dbscan_labels

    visualizer.plot_cluster_scatter(
        X_scaled, dbscan_labels,
        title='Segmentación por DBSCAN (Screen on Time vs Data Usage)',
        x_label='Screen On Time (escalado)',
        y_label='Data Usage (escalado)',
        save_path='segmentacion_dbscan.png'
    )

    # Group statistics
    print("\nDistribución de clusters DBSCAN:")
    print(df['DBSCAN_Cluster'].value_counts())
    print("\nEstadísticas por cluster DBSCAN:")
    print(df.groupby('DBSCAN_Cluster')[processor.selected_variables].mean())

    # --- Phase 5: Dimensionality Reduction ---
    print("\n[PHASE 5] Dimensionality Reduction")

    # PCA
    X_pca, pca_model = engine.run_pca(X_scaled, n_components=2)
    df_pca = pd.DataFrame(X_pca, columns=['Componente 1', 'Componente 2'])
    df_pca['Cluster'] = df['KMeans_Cluster']
    visualizer.plot_pca_scatter(df_pca)

    componentes = pd.DataFrame(
        pca_model.components_,
        columns=processor.selected_variables,
        index=['Componente 1', 'Componente 2']
    )
    visualizer.plot_pca_heatmap(componentes)

    # t-SNE
    X_tsne = engine.run_tsne(X_scaled, n_components=2, perplexity=30, learning_rate=200)
    visualizer.plot_tsne(X_tsne, df['KMeans_Cluster'].values)

    # --- Phase 6: Analysis Summary ---
    print("\n[PHASE 6] Analysis Summary")
    print("=" * 60)
    print(f"Clusters únicos KMeans: {df['KMeans_Cluster'].unique()}")
    print(f"Clusters únicos DBSCAN: {df['DBSCAN_Cluster'].unique()}")
    print("=" * 60)
    print("  ANALYSIS COMPLETE")
    print("  Check generated .png files in semana3/assets/ for results.")
    print("=" * 60)


if __name__ == "__main__":
    main()
