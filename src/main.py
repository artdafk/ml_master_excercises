# -*- coding: utf-8 -*-
"""
Main entry point for Supervised Learning Models Analysis.
Refactored into Modular + OOP Architecture.
"""

import os
import sys

# Ensure src is in the python path to find modules if run from root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.data.processor import SocialAdDataProcessor
from src.models.engine import SupervisedModelEngine
from src.utils.visualizer import ResultsVisualizer

def main():
    print("="*60)
    print("  SUPERVISED LEARNING MODEL ANALYSIS (MODULAR ARCHITECTURE)")
    print("="*60)
    
    # --- Configuration ---
    # Determine project root (assuming src/main.py structure)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Define critical paths
    assets_dir = os.path.join(project_root, "assets")
    dataset_path = os.path.join(project_root, "src", "data", "kaggle", "Social_Network_Ads.csv")
    
    print(f"Project Root: {project_root}")
    print(f"Assets Directory: {assets_dir}")
    print(f"Dataset Path: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset not found at {dataset_path}")
        return
    
    # --- 1. Data Processing ---
    print("\n[PHASE 1] Data Processing")
    processor = SocialAdDataProcessor(dataset_path)
    df = processor.load_data()
    processor.preprocess()
    X_train, X_test, y_train, y_test = processor.get_data_splits()

    # --- 2. Initial Visualization ---
    print("\n[PHASE 2] Exploratory Visualization")
    visualizer = ResultsVisualizer(output_dir=assets_dir)
    visualizer.plot_pairplot(df)
    visualizer.plot_correlation_matrix(df)
    
    # --- 3. Model Training & Evaluation ---
    print("\n[PHASE 3] Model Training & Evaluation")
    engine = SupervisedModelEngine()
    results = engine.train_evaluate_all(X_train, y_train, X_test, y_test)
    
    # Visualize Confusion Matrices
    for model_name, res in results.items():
        visualizer.plot_confusion_matrix(y_test, res['predictions'], model_name)
        
    # --- 4. Cross-Validation Comparison ---
    print("\n[PHASE 4] Cross-Validation Comparison")
    cv_results = engine.run_cross_validation(processor.X_scaled, processor.y)
    visualizer.plot_model_comparison(cv_results)
    
    # --- 5. Decision Boundaries ---
    print("\n[PHASE 5] Decision Boundary Visualization (2D Projection)")
    # Train simplified 2D models for visualization
    models_2d = engine.train_2d_models_for_viz(X_train, y_train)
    
    # Get 2D data for plotting (Visualizer handles slicing)
    X_vis = processor.X_scaled
    
    for name, model in models_2d.items():
        visualizer.plot_decision_boundary(model, X_vis, processor.y, name)
        
    print("\n" + "="*60)
    print("  ANALYSIS COMPLETE")
    print("  Check generated .png files for results.")
    print("="*60)

if __name__ == "__main__":
    main()
