import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix
import pandas as pd

class ResultsVisualizer:
    """
    Handles all visualization tasks for the Social Network Ads analysis.
    """
    def __init__(self, output_dir='.'):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Created output directory: {self.output_dir}")

    def _get_save_path(self, filename):
        return os.path.join(self.output_dir, filename)
    
    def plot_pairplot(self, df, save_path='pairplot.png'):
        """Plots pairwise relationships in the dataset."""
        full_path = self._get_save_path(save_path)
        plt.figure(figsize=(12, 8))
        sns.pairplot(df, hue='Purchased')
        plt.suptitle('Relaciones entre variables y clase objetivo', y=1.02)
        plt.savefig(full_path)
        plt.close()
        print(f"Saved pairplot to {full_path}")

    def plot_correlation_matrix(self, df, save_path='correlation_matrix.png'):
        """Plots correlation matrix of numeric features."""
        full_path = self._get_save_path(save_path)
        plt.figure(figsize=(10, 8))
        numeric_df = df.select_dtypes(include=['int64', 'float64'])
        correlation = numeric_df.corr()
        sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Matriz de Correlación')
        plt.savefig(full_path)
        plt.close()
        print(f"Saved correlation matrix to {full_path}")

    def plot_confusion_matrix(self, y_true, y_pred, model_name, save_path=None):
        """Plots confusion matrix for a specific model."""
        if save_path is None:
            save_path = f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png'
        full_path = self._get_save_path(save_path)
            
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['No Compra', 'Compra'],
                    yticklabels=['No Compra', 'Compra'])
        plt.xlabel('Predicción')
        plt.ylabel('Valor Real')
        plt.title(f'Matriz de Confusión - {model_name}')
        plt.savefig(full_path)
        plt.close()
        print(f"Saved confusion matrix for {model_name} to {full_path}")

    def plot_model_comparison(self, cv_results, save_path='cross_validation_comparison.png'):
        """Plots bar chart comparing model performance."""
        full_path = self._get_save_path(save_path)
        cv_df = pd.DataFrame(cv_results)
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Modelo', y='Accuracy Promedio', data=cv_df, palette='viridis')
        plt.errorbar(x=range(len(cv_df)), y=cv_df['Accuracy Promedio'],
                     yerr=cv_df['Desviación Estándar'], fmt='none', color='black', capsize=5)
        plt.title('Comparación de Modelos - Validación Cruzada')
        plt.ylim(0.7, 1.0)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(full_path)
        plt.close()
        print(f"Saved comparison plot to {full_path}")
        
    def plot_decision_boundary(self, model, X, y, title, save_path=None):
        """Plots decision boundary for 2D data (Age vs EstimatedSalary)."""
        if save_path is None:
            save_path = f'decision_boundary_{title.lower().replace(" ", "_")}.png'
        full_path = self._get_save_path(save_path)
        
        # Ensure we only use the last 2 columns (Age, Salary) for visualization context
        X_vis = X[:, -2:] if X.shape[1] > 2 else X
        
        h = 0.02  # step size in the mesh
        x_min, x_max = X_vis[:, 0].min() - 1, X_vis[:, 0].max() + 1
        y_min, y_max = X_vis[:, 1].min() - 1, X_vis[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        
        try:
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            
            plt.figure(figsize=(10, 8))
            plt.contourf(xx, yy, Z, alpha=0.8, cmap=ListedColormap(['#FFAAAA', '#AAFFAA']))
            
            scatter = plt.scatter(X_vis[:, 0], X_vis[:, 1], c=y,
                                  edgecolors='k', cmap=ListedColormap(['#FF0000', '#00FF00']))
                                  
            plt.xlim(xx.min(), xx.max())
            plt.ylim(yy.min(), yy.max())
            plt.title(title)
            plt.xlabel('Edad (estandarizada)')
            plt.ylabel('Salario estimado (estandarizado)')
            plt.legend(*scatter.legend_elements(), title="Compra")
            
            plt.savefig(full_path)
            plt.close()
            print(f"Saved decision boundary to {full_path}")
            
        except Exception as e:
            print(f"Could not plot decision boundary for {title}: {e}")
