import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.multiclass import type_of_target

# Classification Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier

# Regression Models
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor

# Metrics
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, 
    mean_absolute_error, mean_squared_error, r2_score, 
    roc_auc_score, roc_curve, precision_score, recall_score, f1_score
)

# Set professional plotting style
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 12})

class MultiModelClassifier:

    def __init__(self, X, y, test_size=0.3, scaled_data=False):
        self.X = X
        self.y = y
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=42)
        
        # Initialize Scaler
        self.scaler = StandardScaler()
        
        # Logic updated to use 'scaled_data' flag
        if scaled_data:
            self.X_train_scaled = self.scaler.fit_transform(X_train)
            self.X_test_scaled = self.scaler.transform(X_test)
        else:
            self.X_train_scaled = X_train
            self.X_test_scaled = X_test
            
        self.y_train = y_train
        self.y_test = y_test
    
    @staticmethod
    def evaluate_model(model, X_test, y_true):
        predicted = model.predict(X_test)
        
        report = classification_report(y_true, predicted)
        matrix = confusion_matrix(y_true, predicted)
        accuracy = accuracy_score(y_true, predicted)
        
        precision = precision_score(y_true, predicted, average='weighted')
        recall = recall_score(y_true, predicted, average='weighted')
        f1 = f1_score(y_true, predicted, average='weighted')
        
        if hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_pred_proba = model.decision_function(X_test)
        else:
            y_pred_proba = predicted
        
        try:
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            roc_auc = roc_auc_score(y_true, y_pred_proba)
        except ValueError:
            fpr, tpr, roc_auc = [0], [0], 0.5

        return report, matrix, accuracy, precision, recall, f1, fpr, tpr, roc_auc
        
    def Logistic_model(self):
        model = LogisticRegression()
        model.fit(self.X_train_scaled, self.y_train)
        return self.evaluate_model(model, self.X_test_scaled, self.y_test)

    def Support_vector_model(self):
        svc = SVC(kernel='linear', probability=True)
        svc.fit(self.X_train_scaled, self.y_train)
        return self.evaluate_model(svc, self.X_test_scaled, self.y_test)

    def DecisionTree_model(self):
        model = DecisionTreeClassifier()
        model.fit(self.X_train_scaled, self.y_train)
        return self.evaluate_model(model, self.X_test_scaled, self.y_test)

    def KNN_model(self):
        model = KNeighborsClassifier(n_neighbors=10)
        model.fit(self.X_train_scaled, self.y_train)
        return self.evaluate_model(model, self.X_test_scaled, self.y_test)

    def Naive_Bayes_model(self):
        model = GaussianNB()
        model.fit(self.X_train_scaled, self.y_train)
        return self.evaluate_model(model, self.X_test_scaled, self.y_test)

    def RandomForest_model(self):
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(self.X_train_scaled, self.y_train)
        return self.evaluate_model(model, self.X_test_scaled, self.y_test)

    def GradientBoosting_model(self):
        model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        model.fit(self.X_train_scaled, self.y_train)
        return self.evaluate_model(model, self.X_test_scaled, self.y_test)

    def AdaBoost_model(self):
        model = AdaBoostClassifier(n_estimators=50, random_state=42)
        model.fit(self.X_train_scaled, self.y_train)
        return self.evaluate_model(model, self.X_test_scaled, self.y_test)

    def run_all_models(self):
        models = [
            ('Logistic Regression', *self.Logistic_model()),
            ('SVM', *self.Support_vector_model()),
            ('Decision Tree', *self.DecisionTree_model()),
            ('KNN', *self.KNN_model()),
            ('Naive Bayes', *self.Naive_Bayes_model()),
            ('Random Forest', *self.RandomForest_model()),
            ('Gradient Boosting', *self.GradientBoosting_model()),
            ('AdaBoost', *self.AdaBoost_model())
        ]
        return models
    
    def show_tabular_report(self, models):
        """Displays all model metrics in a clean tabular format and recommends the best model."""
        data = []
        for name, report, matrix, accuracy, precision, recall, f1, fpr, tpr, roc_auc in models:
            data.append({
                'Model': name,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1,
                'ROC AUC': roc_auc
            })
        
        df = pd.DataFrame(data)
        # Sort by Accuracy
        df = df.sort_values(by='Accuracy', ascending=False).reset_index(drop=True)
        
        print(f"\n{'='*60}\n 📊 MODEL COMPARISON TABLE \n{'='*60}")
        print(df)
        print(f"{'='*60}\n")
        
        # Recommend Best Model
        best_model = df.iloc[0]
        print(f"⭐ BEST MODEL RECOMMENDATION: {best_model['Model']}")
        print(f"   Accuracy: {best_model['Accuracy']:.4f} | F1 Score: {best_model['F1 Score']:.4f}")
        print(f"{'='*60}\n")
        
        return None

    def plot_confusion_matrices(self, models):
        """Plots confusion matrices for all models with different colors."""
        num_models = len(models)
        cols = 2
        rows = (num_models + 1) // cols
        
        # List of distinct colormaps to cycle through
        colormaps = ['Blues', 'Greens', 'Oranges', 'Purples', 'Reds', 'YlOrBr', 'GnBu', 'PuBu']
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        fig.suptitle('Confusion Matrices', fontsize=20, fontweight='bold')
        axes = axes.flatten()
        
        for idx, (name, report, matrix, accuracy, precision, recall, f1, fpr, tpr, roc_auc) in enumerate(models):
            # Select color based on index
            cmap_choice = colormaps[idx % len(colormaps)]
            
            sns.heatmap(matrix, annot=True, fmt='d', cmap=cmap_choice, ax=axes[idx], cbar=False)
            axes[idx].set_title(f'{name}\nAcc: {accuracy:.2f}', fontsize=18, fontweight='bold')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('True')
        
        # Hide unused subplots
        for i in range(idx + 1, len(axes)):
            axes[i].axis('off')
            
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def plot_roc_curves(self, models):
        """Plots ROC curves for all models in a single combined plot."""
        plt.figure(figsize=(12, 8))
        
        for name, report, matrix, accuracy, precision, recall, f1, fpr, tpr, roc_auc in models:
            plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve Comparison', fontsize=16, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()
 
    def get_summary(self, models):
        """Legacy function to run full reporting pipeline."""
        self.show_tabular_report(models)
        self.plot_confusion_matrices(models)
        self.plot_roc_curves(models)
        self.plot_comparison(models)

    def plot_comparison(self, models):
        model_names = [m[0] for m in models]
        accuracy = [m[3] for m in models]
        precision = [m[4] for m in models]
        recall = [m[5] for m in models]
        f1 = [m[6] for m in models]
        
        data = {
            'Model': model_names * 4,
            'Score': accuracy + precision + recall + f1,
            'Metric': ['Accuracy']*len(models) + ['Precision']*len(models) + ['Recall']*len(models) + ['F1 Score']*len(models)
        }
        df_plot = pd.DataFrame(data)

        plt.figure(figsize=(14, 8))
        ax = sns.barplot(x='Model', y='Score', hue='Metric', data=df_plot, palette="viridis")
        plt.title("Comprehensive Model Comparison", fontsize=18, pad=20, fontweight='bold')
        plt.ylim(0, 1.1)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45, ha='right')
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', padding=3, fontsize=9)
        plt.tight_layout()
        plt.show()

class MultiModelRegressior:
    def __init__(self, X, y, test_size=0.3, scaled_data=False):
        self.X = X
        self.y = y
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=42)
        
        self.scaler = StandardScaler()
        if scaled_data:
            self.X_train_scaled = self.scaler.fit_transform(X_train)
            self.X_test_scaled = self.scaler.transform(X_test)
        else:
            self.X_train_scaled = X_train
            self.X_test_scaled = X_test
            
        self.y_train = y_train
        self.y_test = y_test

    @staticmethod
    def evaluate_model(model, X_test, y_true):
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        return mae, mse, rmse, r2, y_pred

    def LinearRegression_model(self): return self.evaluate_model(LinearRegression().fit(self.X_train_scaled, self.y_train), self.X_test_scaled, self.y_test)
    def Lasso_model(self): return self.evaluate_model(Lasso(alpha=0.1).fit(self.X_train_scaled, self.y_train), self.X_test_scaled, self.y_test)
    def Ridge_model(self): return self.evaluate_model(Ridge(alpha=1.0).fit(self.X_train_scaled, self.y_train), self.X_test_scaled, self.y_test)
    def SVR_model(self): return self.evaluate_model(SVR(kernel='rbf').fit(self.X_train_scaled, self.y_train), self.X_test_scaled, self.y_test)
    def DecisionTree_model(self): return self.evaluate_model(DecisionTreeRegressor(random_state=42).fit(self.X_train_scaled, self.y_train), self.X_test_scaled, self.y_test)
    def RandomForest_model(self): return self.evaluate_model(RandomForestRegressor(n_estimators=100, random_state=42).fit(self.X_train_scaled, self.y_train), self.X_test_scaled, self.y_test)
    def GradientBoosting_model(self): return self.evaluate_model(GradientBoostingRegressor(n_estimators=100, random_state=42).fit(self.X_train_scaled, self.y_train), self.X_test_scaled, self.y_test)

    def run_all_models(self):
        return [
            ('Linear Regression', *self.LinearRegression_model()),
            ('Lasso Regression', *self.Lasso_model()),
            ('Ridge Regression', *self.Ridge_model()),
            ('SVR', *self.SVR_model()),
            ('Decision Tree Regressor', *self.DecisionTree_model()),
            ('Random Forest Regressor', *self.RandomForest_model()),
            ('Gradient Boosting Regressor', *self.GradientBoosting_model())
        ]

    def show_tabular_report(self, models):
        """Displays all regression metrics in a clean tabular format and recommends the best model."""
        data = []
        for name, mae, mse, rmse, r2, y_pred in models:
            data.append({
                'Model': name,
                'MAE': mae,
                'MSE': mse,
                'RMSE': rmse,
                'R2 Score': r2
            })
        
        df = pd.DataFrame(data)
        df = df.sort_values(by='R2 Score', ascending=False).reset_index(drop=True)
        
        print(f"\n{'='*60}\n 📊 REGRESSION MODEL COMPARISON TABLE \n{'='*60}")
        print(df)
        print(f"{'='*60}\n")
        
        # Recommend Best Model
        best_model = df.iloc[0]
        print(f"⭐ BEST MODEL RECOMMENDATION: {best_model['Model']}")
        print(f"   R2 Score: {best_model['R2 Score']:.4f} | RMSE: {best_model['RMSE']:.4f}")
        print(f"{'='*60}\n")
        
        return None

    def plot_true_vs_predicted(self, models):
        """Plots True vs Predicted values for all models (Regression equivalent of Confusion Matrix)."""
        num_models = len(models)
        cols = 2
        rows = (num_models + 1) // cols
        
        # Colors to cycle through
        colors = ['blue', 'green', 'orange', 'purple', 'red', 'cyan', 'magenta']
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        fig.suptitle('True vs Predicted Values', fontsize=20, fontweight='bold')
        axes = axes.flatten()
        
        for idx, (name, mae, mse, rmse, r2, y_pred) in enumerate(models):
            color = colors[idx % len(colors)]
            
            axes[idx].scatter(self.y_test, y_pred, alpha=0.6, color=color, label=name)
            
            # Diagonal line for perfect prediction
            p1 = max(max(y_pred), max(self.y_test))
            p2 = min(min(y_pred), min(self.y_test))
            axes[idx].plot([p1, p2], [p1, p2], 'k--', lw=2)
            
            axes[idx].set_title(f'{name}\nR2: {r2:.2f} | RMSE: {rmse:.2f}', fontsize=14, fontweight='bold')
            axes[idx].set_xlabel('True Values')
            axes[idx].set_ylabel('Predicted Values')
            axes[idx].legend()
        
        # Hide unused subplots
        for i in range(idx + 1, len(axes)):
            axes[i].axis('off')
            
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def get_summary(self, models):
        self.show_tabular_report(models)
        self.plot_true_vs_predicted(models)
        self.plot_comparison(models)

    def plot_comparison(self, models):
            """Plots comparison bar charts with values on top for Regression."""
            model_names = [m[0] for m in models]
            r2_scores = [m[4] for m in models]
            
            plt.figure(figsize=(10, 6))
            ax = sns.barplot(x=model_names, y=r2_scores, palette='viridis')
            plt.title("Regressor R2 Score Comparison", pad=20, fontweight='bold')
            plt.ylabel("R2 Score")
            plt.xticks(rotation=45)
            
            # Add values on top of bars
            for container in ax.containers:
                ax.bar_label(container, fmt='%.2f', padding=3, fontsize=10)
                
            plt.tight_layout()
            plt.show()

# =============================================================================
#  SMART MAIN BLOCK: AUTOMATIC DETECTION
# =============================================================================
if __name__ == '__main__':
    print("Initializing Multi-Model Analysis...")

    # NOTE: This block expects 'X' and 'y' to be defined in the namespace
    if 'X' in locals() and 'y' in locals():
        print(f"Data Loaded. X Shape: {X.shape}, y Shape: {y.shape}")
        
        target_type = type_of_target(y)
        unique_values = len(np.unique(y))
        
        if 'continuous' in target_type or (unique_values > 20 and target_type != 'multiclass'):
            print(f"\n[INFO] Detected REGRESSION problem (Target type: {target_type})")
            print("Running MultiModelRegressior...")
            
            regressor = MultiModelRegressior(X, y, scaled_data=True)
            results = regressor.run_all_models()
            
            # New modular functions for Regression
            regressor.show_tabular_report(results)
            regressor.plot_true_vs_predicted(results)
            regressor.plot_comparison(results)
            
        else:
            print(f"\n[INFO] Detected CLASSIFICATION problem (Target type: {target_type})")
            print("Running MultiModelClassifier...")
            
            classifier = MultiModelClassifier(X, y, scaled_data=True)
            results = classifier.run_all_models()
            
            # Run all visualizations and reports
            classifier.show_tabular_report(results)
            classifier.plot_confusion_matrices(results)
            classifier.plot_roc_curves(results)
            classifier.plot_comparison(results)
            
    else:
        print("Error: X and y are not defined. Please define them.")