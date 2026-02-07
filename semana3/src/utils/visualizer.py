import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class ClusteringVisualizer:
    """
    Handles all visualization tasks for the Clustering analysis.
    """
    def __init__(self, output_dir='.'):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Created output directory: {self.output_dir}")

    def _get_save_path(self, filename):
        return os.path.join(self.output_dir, filename)

    def plot_pairplot(self, df, columns, save_path='pairplot_distribucion.png'):
        """Plots pairwise relationships for selected columns."""
        full_path = self._get_save_path(save_path)
        sns.pairplot(df[columns])
        plt.savefig(full_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved pairplot to {full_path}")

    def plot_correlation(self, df, x_col, y_col, corr_value, save_path='correlacion_app_vs_screen_time.png'):
        """Plots regression plot with correlation annotation."""
        full_path = self._get_save_path(save_path)
        plt.figure(figsize=(8, 6))
        sns.regplot(x=x_col, y=y_col, data=df,
                    scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'})
        plt.title(f'Redundancia: App vs Screen Time (Correlación: {corr_value:.2f})')
        plt.xlabel('Tiempo en Apps (Horas)')
        plt.ylabel('Tiempo de Pantalla (Horas)')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(full_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved correlation plot to {full_path}")

    def plot_elbow_method(self, k_range, inertia, save_path='metodo_del_codo_kmeans.png'):
        """Plots elbow method line chart."""
        full_path = self._get_save_path(save_path)
        plt.figure(figsize=(8, 5))
        plt.plot(list(k_range), inertia, marker='o')
        plt.xlabel('Número de Clusters')
        plt.ylabel('Inercia')
        plt.title('Método del Codo para K-Means')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(full_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved elbow method plot to {full_path}")

    def plot_cluster_scatter(self, X, labels, title, x_label, y_label, save_path):
        """Plots scatter plot colored by cluster labels."""
        full_path = self._get_save_path(save_path)
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=X[:, 1], y=X[:, 2], hue=labels, palette='tab10')
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.savefig(full_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved cluster scatter to {full_path}")

    def plot_pca_scatter(self, df_pca, save_path='visualizacion_pca_kmeans.png'):
        """Plots PCA 2D scatter colored by cluster."""
        full_path = self._get_save_path(save_path)
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='Componente 1', y='Componente 2', hue='Cluster',
                        data=df_pca, palette='tab10')
        plt.title('Visualización PCA (Dimensiones comprimidas en 2)')
        plt.savefig(full_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved PCA scatter to {full_path}")

    def plot_pca_heatmap(self, components_df, save_path='heatmap_componentes_pca.png'):
        """Plots heatmap of PCA component weights."""
        full_path = self._get_save_path(save_path)
        plt.figure(figsize=(10, 2))
        sns.heatmap(components_df, annot=True, cmap='coolwarm')
        plt.title('¿Qué significan los ejes del PCA?')
        plt.savefig(full_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved PCA heatmap to {full_path}")

    def plot_tsne(self, X_tsne, labels, save_path='visualizacion_tsne_kmeans.png'):
        """Plots t-SNE scatter colored by cluster labels."""
        full_path = self._get_save_path(save_path)
        plt.figure(figsize=(10, 6))
        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='tab10')
        plt.title('Visualización t-SNE de Clusters K-Means')
        plt.colorbar(label='Cluster')
        plt.savefig(full_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved t-SNE plot to {full_path}")
