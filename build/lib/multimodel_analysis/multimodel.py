import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
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


def _safe_print(*args, **kwargs):
    """Safely prints strings without throwing UnicodeEncodeError on terminals with non-UTF-8 encodings."""
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        cleaned_args = []
        for arg in args:
            if isinstance(arg, str):
                cleaned_args.append(arg.encode('ascii', errors='replace').decode('ascii'))
            else:
                cleaned_args.append(arg)
        print(*cleaned_args, **kwargs)


class MultiModelClassifier:

    def __init__(self, X, y, test_size=0.3, scaled_data=False, random_state=42, stratify=True):
        self.X = X
        self.y = y
        self.random_state = random_state
        
        # Label Encoding for y (handles strings, categorical types, and numbers)
        self.label_encoder = LabelEncoder()
        if isinstance(self.y, (pd.Series, pd.DataFrame)):
            y_arr = self.y.values.ravel()
        else:
            y_arr = np.asarray(self.y).ravel()
            
        self.y_encoded = self.label_encoder.fit_transform(y_arr)
        self.classes_ = self.label_encoder.classes_
        self.n_classes_ = len(self.classes_)
        
        # Stratification setup for classification
        stratify_y = None
        if stratify and self.n_classes_ > 1:
            counts = np.bincount(self.y_encoded)
            if np.min(counts) >= 2:
                stratify_y = self.y_encoded
                
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y_encoded, test_size=test_size, random_state=self.random_state, stratify=stratify_y
        )
        
        # Initialize Scaler
        self.scaler = StandardScaler()
        
        # Feature scaling with DataFrame column preservation
        if scaled_data:
            if isinstance(X_train, pd.DataFrame):
                scaled_train = self.scaler.fit_transform(X_train)
                scaled_test = self.scaler.transform(X_test)
                self.X_train_scaled = pd.DataFrame(scaled_train, columns=X_train.columns, index=X_train.index)
                self.X_test_scaled = pd.DataFrame(scaled_test, columns=X_test.columns, index=X_test.index)
            else:
                self.X_train_scaled = self.scaler.fit_transform(X_train)
                self.X_test_scaled = self.scaler.transform(X_test)
        else:
            self.X_train_scaled = X_train
            self.X_test_scaled = X_test
            
        self.y_train = y_train
        self.y_test = y_test

    def evaluate_model(self, model, X_test, y_true):
        predicted = model.predict(X_test)
        
        # Invert target encoding for reports
        y_true_orig = self.label_encoder.inverse_transform(y_true)
        pred_orig = self.label_encoder.inverse_transform(predicted)
        
        report = classification_report(y_true_orig, pred_orig)
        matrix = confusion_matrix(y_true, predicted)
        accuracy = accuracy_score(y_true, predicted)
        
        precision = precision_score(y_true, predicted, average='weighted', zero_division=0)
        recall = recall_score(y_true, predicted, average='weighted', zero_division=0)
        f1 = f1_score(y_true, predicted, average='weighted', zero_division=0)
        
        # Predict probabilities or decision scores for ROC-AUC
        y_pred_proba = None
        if hasattr(model, "predict_proba"):
            try:
                y_pred_proba = model.predict_proba(X_test)
            except Exception:
                pass
        if y_pred_proba is None and hasattr(model, "decision_function"):
            try:
                df_val = model.decision_function(X_test)
                if df_val.ndim == 1:
                    prob1 = 1 / (1 + np.exp(-np.clip(df_val, -500, 500)))
                    y_pred_proba = np.vstack([1 - prob1, prob1]).T
                else:
                    exp_df = np.exp(df_val - np.max(df_val, axis=1, keepdims=True))
                    y_pred_proba = exp_df / np.sum(exp_df, axis=1, keepdims=True)
            except Exception:
                pass
                
        fpr_dict, tpr_dict, roc_auc = {}, {}, 0.5
        
        if y_pred_proba is not None:
            try:
                if self.n_classes_ == 2:
                    if y_pred_proba.ndim > 1 and y_pred_proba.shape[1] >= 2:
                        pos_proba = y_pred_proba[:, 1]
                    else:
                        pos_proba = y_pred_proba.ravel()
                    fpr, tpr, _ = roc_curve(y_true, pos_proba)
                    roc_auc = roc_auc_score(y_true, pos_proba)
                    fpr_dict['binary'] = fpr
                    tpr_dict['binary'] = tpr
                else:
                    roc_auc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
                    y_true_bin = label_binarize(y_true, classes=np.arange(self.n_classes_))
                    for i in range(self.n_classes_):
                        if i < y_pred_proba.shape[1]:
                            f_i, t_i, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
                            fpr_dict[i] = f_i
                            tpr_dict[i] = t_i
            except Exception:
                fpr_dict, tpr_dict, roc_auc = {}, {}, 0.5

        return report, matrix, accuracy, precision, recall, f1, fpr_dict, tpr_dict, roc_auc

    def Logistic_model(self):
        model = LogisticRegression(random_state=self.random_state, max_iter=1000)
        model.fit(self.X_train_scaled, self.y_train)
        return self.evaluate_model(model, self.X_test_scaled, self.y_test)

    def Support_vector_model(self):
        svc = SVC(kernel='linear', probability=True, random_state=self.random_state)
        svc.fit(self.X_train_scaled, self.y_train)
        return self.evaluate_model(svc, self.X_test_scaled, self.y_test)

    def DecisionTree_model(self):
        model = DecisionTreeClassifier(random_state=self.random_state)
        model.fit(self.X_train_scaled, self.y_train)
        return self.evaluate_model(model, self.X_test_scaled, self.y_test)

    def KNN_model(self):
        model = KNeighborsClassifier(n_neighbors=min(10, max(1, len(self.y_train) - 1)))
        model.fit(self.X_train_scaled, self.y_train)
        return self.evaluate_model(model, self.X_test_scaled, self.y_test)

    def Naive_Bayes_model(self):
        model = GaussianNB()
        model.fit(self.X_train_scaled, self.y_train)
        return self.evaluate_model(model, self.X_test_scaled, self.y_test)

    def RandomForest_model(self):
        model = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        model.fit(self.X_train_scaled, self.y_train)
        return self.evaluate_model(model, self.X_test_scaled, self.y_test)

    def GradientBoosting_model(self):
        model = GradientBoostingClassifier(n_estimators=100, random_state=self.random_state)
        model.fit(self.X_train_scaled, self.y_train)
        return self.evaluate_model(model, self.X_test_scaled, self.y_test)

    def AdaBoost_model(self):
        model = AdaBoostClassifier(n_estimators=50, random_state=self.random_state)
        model.fit(self.X_train_scaled, self.y_train)
        return self.evaluate_model(model, self.X_test_scaled, self.y_test)

    def run_all_models(self):
        model_methods = [
            ('Logistic Regression', self.Logistic_model),
            ('SVM', self.Support_vector_model),
            ('Decision Tree', self.DecisionTree_model),
            ('KNN', self.KNN_model),
            ('Naive Bayes', self.Naive_Bayes_model),
            ('Random Forest', self.RandomForest_model),
            ('Gradient Boosting', self.GradientBoosting_model),
            ('AdaBoost', self.AdaBoost_model)
        ]
        
        models = []
        for name, method in model_methods:
            try:
                eval_res = method()
                models.append((name, *eval_res))
            except Exception as e:
                _safe_print(f"[WARNING] Model '{name}' failed to train or evaluate: {e}")
                
        return models

    def show_tabular_report(self, models, return_df=False):
        """Displays all model metrics in a clean tabular format and recommends the best model."""
        if not models:
            _safe_print("No models were successfully evaluated.")
            return None

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
        df = df.sort_values(by='Accuracy', ascending=False).reset_index(drop=True)
        
        _safe_print(f"\n{'='*60}\n MODEL COMPARISON TABLE \n{'='*60}")
        _safe_print(df.to_string(index=False))
        _safe_print(f"{'='*60}\n")
        
        best_model = df.iloc[0]
        _safe_print(f"* BEST MODEL RECOMMENDATION: {best_model['Model']}")
        _safe_print(f"   Accuracy: {best_model['Accuracy']:.4f} | F1 Score: {best_model['F1 Score']:.4f}")
        _safe_print(f"{'='*60}\n")
        
        if return_df:
            return df
        return None

    def plot_confusion_matrices(self, models):
        """Plots confusion matrices for all models with custom colormaps and class labels."""
        if not models:
            return
            
        num_models = len(models)
        cols = min(2, num_models)
        rows = (num_models + cols - 1) // cols
        
        colormaps = ['Blues', 'Greens', 'Oranges', 'Purples', 'Reds', 'YlOrBr', 'GnBu', 'PuBu']
        
        fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5 * rows))
        fig.suptitle('Confusion Matrices', fontsize=18, fontweight='bold')
        
        if num_models == 1:
            axes_list = [axes]
        else:
            axes_list = axes.flatten() if hasattr(axes, 'flatten') else [axes]
            
        class_labels = [str(c) for c in self.classes_]
        
        for idx, (name, report, matrix, accuracy, precision, recall, f1, fpr, tpr, roc_auc) in enumerate(models):
            cmap_choice = colormaps[idx % len(colormaps)]
            ax = axes_list[idx]
            
            sns.heatmap(matrix, annot=True, fmt='d', cmap=cmap_choice, ax=ax, cbar=False,
                        xticklabels=class_labels, yticklabels=class_labels)
            ax.set_title(f'{name}\nAcc: {accuracy:.2f}', fontsize=14, fontweight='bold')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
        
        for i in range(num_models, len(axes_list)):
            axes_list[i].axis('off')
            
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def plot_roc_curves(self, models):
        """Plots ROC curves for all models in a single combined plot."""
        if not models:
            return
            
        plt.figure(figsize=(10, 7))
        
        for name, report, matrix, accuracy, precision, recall, f1, fpr_dict, tpr_dict, roc_auc in models:
            if isinstance(fpr_dict, dict) and 'binary' in fpr_dict:
                plt.plot(fpr_dict['binary'], tpr_dict['binary'], lw=2, label=f'{name} (AUC = {roc_auc:.2f})')
            elif isinstance(fpr_dict, dict) and len(fpr_dict) > 0:
                plt.plot([0, 1], [0, 1], lw=1, linestyle=':', label=f'{name} (Multiclass AUC = {roc_auc:.2f})')
            else:
                plt.plot([0, 1], [0, 1], lw=1, linestyle=':', label=f'{name} (AUC = N/A)')
        
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
        """Runs full reporting and visualization pipeline."""
        self.show_tabular_report(models)
        self.plot_confusion_matrices(models)
        self.plot_roc_curves(models)
        self.plot_comparison(models)

    def plot_comparison(self, models):
        if not models:
            return
            
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

        plt.figure(figsize=(12, 7))
        ax = sns.barplot(x='Model', y='Score', hue='Metric', data=df_plot, palette="viridis")
        plt.title("Comprehensive Model Comparison", fontsize=16, pad=20, fontweight='bold')
        plt.ylim(0, 1.1)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45, ha='right')
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', padding=3, fontsize=9)
        plt.tight_layout()
        plt.show()


class MultiModelRegressor:

    def __init__(self, X, y, test_size=0.3, scaled_data=False, random_state=42):
        self.X = X
        self.y = y
        self.random_state = random_state
        
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=self.random_state
        )
        
        self.scaler = StandardScaler()
        if scaled_data:
            if isinstance(X_train, pd.DataFrame):
                scaled_train = self.scaler.fit_transform(X_train)
                scaled_test = self.scaler.transform(X_test)
                self.X_train_scaled = pd.DataFrame(scaled_train, columns=X_train.columns, index=X_train.index)
                self.X_test_scaled = pd.DataFrame(scaled_test, columns=X_test.columns, index=X_test.index)
            else:
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

    def LinearRegression_model(self):
        return self.evaluate_model(LinearRegression().fit(self.X_train_scaled, self.y_train), self.X_test_scaled, self.y_test)

    def Lasso_model(self):
        return self.evaluate_model(Lasso(alpha=0.1, random_state=self.random_state).fit(self.X_train_scaled, self.y_train), self.X_test_scaled, self.y_test)

    def Ridge_model(self):
        return self.evaluate_model(Ridge(alpha=1.0, random_state=self.random_state).fit(self.X_train_scaled, self.y_train), self.X_test_scaled, self.y_test)

    def SVR_model(self):
        return self.evaluate_model(SVR(kernel='rbf').fit(self.X_train_scaled, self.y_train), self.X_test_scaled, self.y_test)

    def DecisionTree_model(self):
        return self.evaluate_model(DecisionTreeRegressor(random_state=self.random_state).fit(self.X_train_scaled, self.y_train), self.X_test_scaled, self.y_test)

    def RandomForest_model(self):
        return self.evaluate_model(RandomForestRegressor(n_estimators=100, random_state=self.random_state).fit(self.X_train_scaled, self.y_train), self.X_test_scaled, self.y_test)

    def GradientBoosting_model(self):
        return self.evaluate_model(GradientBoostingRegressor(n_estimators=100, random_state=self.random_state).fit(self.X_train_scaled, self.y_train), self.X_test_scaled, self.y_test)

    def run_all_models(self):
        model_methods = [
            ('Linear Regression', self.LinearRegression_model),
            ('Lasso Regression', self.Lasso_model),
            ('Ridge Regression', self.Ridge_model),
            ('SVR', self.SVR_model),
            ('Decision Tree Regressor', self.DecisionTree_model),
            ('Random Forest Regressor', self.RandomForest_model),
            ('Gradient Boosting Regressor', self.GradientBoosting_model)
        ]
        
        models = []
        for name, method in model_methods:
            try:
                eval_res = method()
                models.append((name, *eval_res))
            except Exception as e:
                _safe_print(f"[WARNING] Model '{name}' failed to train or evaluate: {e}")
                
        return models

    def show_tabular_report(self, models, return_df=False):
        """Displays all regression metrics in a clean tabular format and recommends the best model."""
        if not models:
            _safe_print("No models were successfully evaluated.")
            return None

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
        
        _safe_print(f"\n{'='*60}\n REGRESSION MODEL COMPARISON TABLE \n{'='*60}")
        _safe_print(df.to_string(index=False))
        _safe_print(f"{'='*60}\n")
        
        best_model = df.iloc[0]
        _safe_print(f"* BEST MODEL RECOMMENDATION: {best_model['Model']}")
        _safe_print(f"   R2 Score: {best_model['R2 Score']:.4f} | RMSE: {best_model['RMSE']:.4f}")
        _safe_print(f"{'='*60}\n")
        
        if return_df:
            return df
        return None

    def plot_true_vs_predicted(self, models):
        """Plots True vs Predicted values for all models."""
        if not models:
            return
            
        num_models = len(models)
        cols = min(2, num_models)
        rows = (num_models + cols - 1) // cols
        
        colors = ['blue', 'green', 'orange', 'purple', 'red', 'cyan', 'magenta']
        
        fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5 * rows))
        fig.suptitle('True vs Predicted Values', fontsize=18, fontweight='bold')
        
        if num_models == 1:
            axes_list = [axes]
        else:
            axes_list = axes.flatten() if hasattr(axes, 'flatten') else [axes]
            
        y_test_arr = np.asarray(self.y_test).ravel()
        
        for idx, (name, mae, mse, rmse, r2, y_pred) in enumerate(models):
            color = colors[idx % len(colors)]
            ax = axes_list[idx]
            
            y_pred_arr = np.asarray(y_pred).ravel()
            ax.scatter(y_test_arr, y_pred_arr, alpha=0.6, color=color, label=name)
            
            p1 = max(np.max(y_pred_arr), np.max(y_test_arr))
            p2 = min(np.min(y_pred_arr), np.min(y_test_arr))
            ax.plot([p1, p2], [p1, p2], 'k--', lw=2)
            
            ax.set_title(f'{name}\nR2: {r2:.2f} | RMSE: {rmse:.2f}', fontsize=14, fontweight='bold')
            ax.set_xlabel('True Values')
            ax.set_ylabel('Predicted Values')
            ax.legend()
        
        for i in range(num_models, len(axes_list)):
            axes_list[i].axis('off')
            
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def get_summary(self, models):
        self.show_tabular_report(models)
        self.plot_true_vs_predicted(models)
        self.plot_comparison(models)

    def plot_comparison(self, models):
        """Plots comparison bar charts for R2 score."""
        if not models:
            return
            
        model_names = [m[0] for m in models]
        r2_scores = [m[4] for m in models]
        
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x=model_names, y=r2_scores, palette='viridis')
        plt.title("Regressor R2 Score Comparison", pad=20, fontweight='bold')
        plt.ylabel("R2 Score")
        plt.xticks(rotation=45, ha='right')
        
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', padding=3, fontsize=10)
            
        plt.tight_layout()
        plt.show()


# Backwards compatibility alias for typo
MultiModelRegressior = MultiModelRegressor


# =============================================================================
#  SMART MAIN BLOCK: AUTOMATIC DETECTION & SYNTHETIC DEMONSTRATION
# =============================================================================
if __name__ == '__main__':
    _safe_print("Initializing Multi-Model Analysis...")

    if 'X' in locals() and 'y' in locals():
        _safe_print(f"Data Loaded. X Shape: {X.shape}, y Shape: {y.shape}")
        
        target_type = type_of_target(y)
        unique_values = len(np.unique(y))
        
        if 'continuous' in target_type or (unique_values > 20 and target_type != 'multiclass'):
            _safe_print(f"\n[INFO] Detected REGRESSION problem (Target type: {target_type})")
            _safe_print("Running MultiModelRegressor...")
            
            regressor = MultiModelRegressor(X, y, scaled_data=True)
            results = regressor.run_all_models()
            regressor.get_summary(results)
        else:
            _safe_print(f"\n[INFO] Detected CLASSIFICATION problem (Target type: {target_type})")
            _safe_print("Running MultiModelClassifier...")
            
            classifier = MultiModelClassifier(X, y, scaled_data=True)
            results = classifier.run_all_models()
            classifier.get_summary(results)
            
    else:
        _safe_print("\n[DEMO MODE] Running automated benchmark test on synthetic datasets...")
        from sklearn.datasets import make_classification, make_regression
        
        _safe_print("\n--- 1. Testing MultiModelClassifier (Binary Classification) ---")
        X_clf, y_clf = make_classification(n_samples=200, n_features=8, random_state=42)
        clf = MultiModelClassifier(X_clf, y_clf, scaled_data=True)
        clf_results = clf.run_all_models()
        clf.show_tabular_report(clf_results)

        _safe_print("\n--- 2. Testing MultiModelRegressor (Regression) ---")
        X_reg, y_reg = make_regression(n_samples=200, n_features=8, noise=0.1, random_state=42)
        reg = MultiModelRegressor(X_reg, y_reg, scaled_data=True)
        reg_results = reg.run_all_models()
        reg.show_tabular_report(reg_results)