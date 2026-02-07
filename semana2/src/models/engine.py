import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score

class SupervisedModelEngine:
    """
    Manages training, evaluation, and comparison of multiple supervised models.
    """
    def __init__(self):
        self.models = {
            'Regresión Logística': LogisticRegression(random_state=42),
            'SVM': SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42),
            'Árbol de Decisión': DecisionTreeClassifier(max_depth=4, random_state=42)
        }
        self.results = {}
        
    def train_evaluate_all(self, X_train, y_train, X_test, y_test):
        """
        Trains all defined models and evaluates them on the test set.
        Returns a dictionary of results.
        """
        print("\nStarting model training and evaluation...")
        
        for name, model in self.models.items():
            print(f"\n{'='*30}\nTraining {name}...\n{'='*30}")
            
            # Train
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Evaluate
            report = classification_report(y_test, y_pred, output_dict=True)
            cm = confusion_matrix(y_test, y_pred)
            
            print(f"Confusion Matrix for {name}:\n{cm}")
            
            # Store results
            self.results[name] = {
                'model': model,
                'predictions': y_pred,
                'confusion_matrix': cm,
                'report': report,
                'accuracy': report['accuracy']
            }
            
        return self.results
            
    def run_cross_validation(self, X_scaled, y, cv=5):
        """
        Runs cross-validation for all models and returns summary statistics.
        """
        print("\nRunning Cross-Validation...")
        cv_summary = []
        
        for name, model in self.models.items():
            scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy')
            cv_summary.append({
                'Modelo': name,
                'Accuracy Promedio': scores.mean(),
                'Desviación Estándar': scores.std(),
                'Scores': scores
            })
            print(f"{name}: Mean Accuracy = {scores.mean():.3f} (+/- {scores.std():.3f})")
            
        return cv_summary

    def train_2d_models_for_viz(self, X_train, y_train):
        """
        Trains simplified classifiers on only the last 2 features (Age, Salary)
        specifically for 2D decision boundary visualization.
        """
        # Assumes X_train has [Gender, Age, Salary] or [Age, Salary]
        # We take the last 2 columns
        X_2d = X_train[:, -2:] if X_train.shape[1] > 2 else X_train
        
        models_2d = {
            'Regresión Logística': LogisticRegression(random_state=42),
            'SVM': SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42),
            'Árbol de Decisión': DecisionTreeClassifier(max_depth=4, random_state=42)
        }
        
        trained_models = {}
        for name, model in models_2d.items():
            model.fit(X_2d, y_train)
            trained_models[name] = model
            
        return trained_models
