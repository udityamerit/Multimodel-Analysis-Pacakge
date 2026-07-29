import os
import sys
import warnings
import contextlib
from typing import Union, Optional, List, Dict, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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


@contextlib.contextmanager
def _apply_plot_style():
    """Context manager to temporarily set plotting style without mutating global state."""
    style = 'seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'whitegrid'
    if style in plt.style.available:
        with plt.style.context(style):
            yield
    else:
        with plt.style.context('default'):
            yield


def save_report(df: Optional[pd.DataFrame] = None, filepath: str = "report.csv") -> None:
    """Saves a comparison tabular report DataFrame to CSV, Excel, HTML, or JSON based on extension."""
    if df is None:
        _safe_print("[WARNING] No DataFrame provided to save_report.")
        return
    if df.empty:
        _safe_print("[WARNING] Empty DataFrame provided to save_report.")
        return

    filepath = filepath if filepath else "report.csv"
    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.csv':
        df.to_csv(filepath, index=False)
    elif ext in ('.xlsx', '.xls'):
        df.to_excel(filepath, index=False)
    elif ext in ('.html', '.htm'):
        df.to_html(filepath, index=False)
    elif ext == '.json':
        df.to_json(filepath, orient='records', indent=4)
    else:
        df.to_csv(filepath, index=False)
    _safe_print(f"[INFO] Tabular report saved successfully to '{filepath}'.")


class _BaseMultiModel:
    """Base class providing shared initialization, preprocessing, scaling, and utility methods."""

    def __init__(
        self,
        X: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y: Optional[Union[pd.Series, pd.DataFrame, np.ndarray]] = None,
        test_size: float = 0.3,
        scaled_data: bool = False,
        random_state: Optional[int] = 42,
        n_jobs: Optional[int] = -1
    ):
        if X is None or y is None:
            raise ValueError("Both features 'X' and target 'y' must be provided to initialize MultiModel.")
        
        self.X = X
        self.y_raw = y
        self.test_size = test_size
        self.scaled_data = scaled_data
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.models_ = []
        self._last_df_report = None

        # Target flattening (raveling)
        if isinstance(y, (pd.Series, pd.DataFrame)):
            self.y_flat = y.values.ravel()
        else:
            self.y_flat = np.asarray(y).ravel()

        self.scaler = StandardScaler()

    def _split_and_scale(self, y_split: np.ndarray, stratify_y: Optional[np.ndarray] = None):
        """Splits features and targets and applies optional scaling preserving DataFrame structures."""
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, y_split, test_size=self.test_size, random_state=self.random_state, stratify=stratify_y
        )

        if self.scaled_data:
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
    def _save_and_show(fig: plt.Figure, save_path: Optional[str] = None, show_plot: bool = True):
        """Helper method to handle saving and displaying matplotlib figures."""
        if save_path:
            fig.savefig(save_path, bbox_inches='tight')
            _safe_print(f"[INFO] Plot saved to '{save_path}'.")
        if show_plot:
            plt.show()
        else:
            plt.close(fig)

    def save_report(self, df_or_filepath: Optional[Union[pd.DataFrame, str]] = None, filepath: Optional[str] = None) -> None:
        """
        Saves tabular report to CSV/Excel/HTML/JSON.
        Can be called as:
          - save_report(df, "metrics.csv")           [standalone function]
          - model.save_report(df, "metrics.csv")     [method on classifier/regressor instance]
          - model.save_report("metrics.csv")         [method using auto-stored report DataFrame]
          - model.save_report()                      [method using auto-stored report DataFrame and default path 'report.csv']
        """
        if df_or_filepath is None:
            target_df = getattr(self, '_last_df_report', None)
            target_path = filepath if filepath else "report.csv"
            if target_df is None:
                _safe_print("[WARNING] No previous tabular report DataFrame found. Run show_tabular_report() first or pass a DataFrame.")
                return
        elif isinstance(df_or_filepath, str):
            target_path = df_or_filepath
            target_df = getattr(self, '_last_df_report', None)
            if target_df is None:
                _safe_print("[WARNING] No previous tabular report DataFrame found. Run show_tabular_report() first or pass a DataFrame.")
                return
        else:
            target_df = df_or_filepath
            target_path = filepath if filepath else "report.csv"

        save_report(target_df, target_path)

    def _resolve_models(self, models: Optional[List[Tuple]] = None) -> List[Tuple]:
        """Resolves models list; if omitted, uses self.models_ or auto-runs run_all_models()."""
        if models is not None and len(models) > 0:
            return models
        if hasattr(self, 'models_') and self.models_:
            return self.models_
        _safe_print("[INFO] No models provided. Automatically running all models...")
        return self.run_all_models()


class MultiModelClassifier(_BaseMultiModel):
    """
    Automated multi-model analysis tool for classification datasets.
    """

    def __init__(
        self,
        X: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y: Optional[Union[pd.Series, pd.DataFrame, np.ndarray]] = None,
        test_size: float = 0.3,
        scaled_data: bool = False,
        random_state: Optional[int] = 42,
        stratify: bool = True,
        n_jobs: Optional[int] = -1
    ):
        super().__init__(
            X=X, y=y, test_size=test_size, scaled_data=scaled_data,
            random_state=random_state, n_jobs=n_jobs
        )

        # Label Encoding for y (handles strings, categorical types, and numbers)
        self.label_encoder = LabelEncoder()
        self.y_encoded = self.label_encoder.fit_transform(self.y_flat)
        self.classes_ = self.label_encoder.classes_
        self.n_classes_ = len(self.classes_)

        # Stratification setup for classification
        stratify_y = None
        if stratify and self.n_classes_ > 1:
            counts = np.bincount(self.y_encoded)
            if np.min(counts) >= 2:
                stratify_y = self.y_encoded

        self._split_and_scale(self.y_encoded, stratify_y=stratify_y)

    def evaluate_model(
        self,
        model: Any,
        X_test: Optional[Any] = None,
        y_true: Optional[np.ndarray] = None
    ):
        if X_test is None:
            X_test = self.X_test_scaled
        if y_true is None:
            y_true = self.y_test

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
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
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    y_pred_proba = model.predict_proba(X_test)
            except Exception:
                pass
        if y_pred_proba is None and hasattr(model, "decision_function"):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    df_val = model.decision_function(X_test)
                if df_val.ndim == 1:
                    prob1 = 1 / (1 + np.exp(-np.clip(df_val, -500, 500)))
                    y_pred_proba = np.vstack([1 - prob1, prob1]).T
                else:
                    exp_df = np.exp(df_val - np.max(df_val, axis=1, keepdims=True))
                    y_pred_proba = exp_df / np.sum(exp_df, axis=1, keepdims=True)
            except Exception:
                pass

        fpr_dict, tpr_dict = {}, {}
        roc_auc = np.nan

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

                    all_fpr = []
                    all_tpr = []
                    for i in range(self.n_classes_):
                        if i < y_pred_proba.shape[1]:
                            f_i, t_i, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
                            fpr_dict[i] = f_i
                            tpr_dict[i] = t_i
                            all_fpr.append(f_i)
                            all_tpr.append(t_i)

                    if all_fpr and all_tpr:
                        mean_fpr = np.unique(np.concatenate(all_fpr))
                        mean_tpr = np.zeros_like(mean_fpr)
                        for i in range(len(all_fpr)):
                            mean_tpr += np.interp(mean_fpr, all_fpr[i], all_tpr[i])
                        mean_tpr /= len(all_fpr)
                        fpr_dict['macro'] = mean_fpr
                        tpr_dict['macro'] = mean_tpr
            except Exception as e:
                _safe_print(f"[WARNING] ROC AUC computation failed for {model.__class__.__name__}: {e}")
                fpr_dict, tpr_dict, roc_auc = {}, {}, np.nan

        return report, matrix, accuracy, precision, recall, f1, fpr_dict, tpr_dict, roc_auc

    def Logistic_model(
        self,
        random_state: Optional[int] = None,
        max_iter: int = 1000,
        **kwargs
    ):
        rs = random_state if random_state is not None else self.random_state
        model = LogisticRegression(random_state=rs, max_iter=max_iter, n_jobs=self.n_jobs, **kwargs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(self.X_train_scaled, self.y_train)
        return self.evaluate_model(model, self.X_test_scaled, self.y_test)

    def Support_vector_model(
        self,
        random_state: Optional[int] = None,
        kernel: str = 'linear',
        probability: bool = True,
        **kwargs
    ):
        rs = random_state if random_state is not None else self.random_state
        svc = SVC(kernel=kernel, probability=probability, random_state=rs, **kwargs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            svc.fit(self.X_train_scaled, self.y_train)
        return self.evaluate_model(svc, self.X_test_scaled, self.y_test)

    def DecisionTree_model(
        self,
        random_state: Optional[int] = None,
        **kwargs
    ):
        rs = random_state if random_state is not None else self.random_state
        model = DecisionTreeClassifier(random_state=rs, **kwargs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(self.X_train_scaled, self.y_train)
        return self.evaluate_model(model, self.X_test_scaled, self.y_test)

    def KNN_model(
        self,
        n_neighbors: Optional[int] = None,
        **kwargs
    ):
        kwargs.pop('random_state', None)
        if n_neighbors is None:
            n_neighbors = min(10, max(1, len(self.y_train) - 1))
        model = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=self.n_jobs, **kwargs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(self.X_train_scaled, self.y_train)
        return self.evaluate_model(model, self.X_test_scaled, self.y_test)

    def Naive_Bayes_model(
        self,
        **kwargs
    ):
        kwargs.pop('random_state', None)
        model = GaussianNB(**kwargs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(self.X_train_scaled, self.y_train)
        return self.evaluate_model(model, self.X_test_scaled, self.y_test)

    def RandomForest_model(
        self,
        n_estimators: int = 100,
        random_state: Optional[int] = None,
        **kwargs
    ):
        rs = random_state if random_state is not None else self.random_state
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=rs, n_jobs=self.n_jobs, **kwargs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(self.X_train_scaled, self.y_train)
        return self.evaluate_model(model, self.X_test_scaled, self.y_test)

    def GradientBoosting_model(
        self,
        n_estimators: int = 100,
        random_state: Optional[int] = None,
        **kwargs
    ):
        rs = random_state if random_state is not None else self.random_state
        model = GradientBoostingClassifier(n_estimators=n_estimators, random_state=rs, **kwargs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(self.X_train_scaled, self.y_train)
        return self.evaluate_model(model, self.X_test_scaled, self.y_test)

    def AdaBoost_model(
        self,
        n_estimators: int = 50,
        random_state: Optional[int] = None,
        **kwargs
    ):
        rs = random_state if random_state is not None else self.random_state
        model = AdaBoostClassifier(n_estimators=n_estimators, random_state=rs, **kwargs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(self.X_train_scaled, self.y_train)
        return self.evaluate_model(model, self.X_test_scaled, self.y_test)

    def run_all_models(
        self,
        custom_models: Optional[Dict[str, Any]] = None,
        random_state: Optional[int] = None
    ) -> List[Tuple]:
        model_methods = [
            ('Logistic Regression', lambda: self.Logistic_model(random_state=random_state)),
            ('SVM', lambda: self.Support_vector_model(random_state=random_state)),
            ('Decision Tree', lambda: self.DecisionTree_model(random_state=random_state)),
            ('KNN', lambda: self.KNN_model(random_state=random_state)),
            ('Naive Bayes', lambda: self.Naive_Bayes_model(random_state=random_state)),
            ('Random Forest', lambda: self.RandomForest_model(random_state=random_state)),
            ('Gradient Boosting', lambda: self.GradientBoosting_model(random_state=random_state)),
            ('AdaBoost', lambda: self.AdaBoost_model(random_state=random_state))
        ]

        models = []
        for name, method in model_methods:
            try:
                eval_res = method()
                models.append((name, *eval_res))
            except Exception as e:
                _safe_print(f"[WARNING] Model '{name}' failed to train or evaluate: {e}")

        if custom_models and isinstance(custom_models, dict):
            for name, model_inst in custom_models.items():
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model_inst.fit(self.X_train_scaled, self.y_train)
                    eval_res = self.evaluate_model(model_inst, self.X_test_scaled, self.y_test)
                    models.append((name, *eval_res))
                except Exception as e:
                    _safe_print(f"[WARNING] Custom model '{name}' failed: {e}")

        self.models_ = models
        return models

    def show_tabular_report(self, models: Optional[List[Tuple]] = None, return_df: bool = False) -> Optional[pd.DataFrame]:
        """Displays all model metrics in a clean tabular format and recommends the best model."""
        models = self._resolve_models(models)
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
                'ROC AUC': roc_auc if not np.isnan(roc_auc) else "N/A"
            })

        df = pd.DataFrame(data)
        df = df.sort_values(by='Accuracy', ascending=False).reset_index(drop=True)
        self._last_df_report = df

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

    def plot_confusion_matrices(
        self,
        models: Optional[List[Tuple]] = None,
        save_path: Optional[str] = None,
        show_plot: bool = True
    ):
        """Plots confusion matrices for all models with custom colormaps and class labels."""
        models = self._resolve_models(models)
        if not models:
            return

        num_models = len(models)
        cols = min(2, num_models)
        rows = (num_models + cols - 1) // cols

        colormaps = ['Blues', 'Greens', 'Oranges', 'Purples', 'Reds', 'YlOrBr', 'GnBu', 'PuBu']

        with _apply_plot_style():
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
            self._save_and_show(fig, save_path=save_path, show_plot=show_plot)

    def plot_roc_curves(
        self,
        models: Optional[List[Tuple]] = None,
        save_path: Optional[str] = None,
        show_plot: bool = True
    ):
        """Plots ROC curves for all models in a single combined plot."""
        models = self._resolve_models(models)
        if not models:
            return

        with _apply_plot_style():
            fig, ax = plt.subplots(figsize=(10, 7))

            for name, report, matrix, accuracy, precision, recall, f1, fpr_dict, tpr_dict, roc_auc in models:
                auc_str = f"AUC = {roc_auc:.2f}" if not np.isnan(roc_auc) else "AUC = N/A"

                if isinstance(fpr_dict, dict) and 'binary' in fpr_dict:
                    ax.plot(fpr_dict['binary'], tpr_dict['binary'], lw=2, label=f'{name} ({auc_str})')
                elif isinstance(fpr_dict, dict) and 'macro' in fpr_dict:
                    ax.plot(fpr_dict['macro'], tpr_dict['macro'], lw=2, label=f'{name} (Macro {auc_str})')
                elif isinstance(fpr_dict, dict) and len(fpr_dict) > 0:
                    first_k = next(iter(fpr_dict))
                    ax.plot(fpr_dict[first_k], tpr_dict[first_k], lw=2, label=f'{name} ({auc_str})')
                else:
                    _safe_print(f"[INFO] Skipping ROC curve plot line for '{name}' (No probability data).")

            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Chance')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curve Comparison', fontsize=16, fontweight='bold')
            ax.legend(loc="lower right")
            ax.grid(True)
            self._save_and_show(fig, save_path=save_path, show_plot=show_plot)

    def plot_comparison(
        self,
        models: Optional[List[Tuple]] = None,
        save_path: Optional[str] = None,
        show_plot: bool = True
    ):
        """Plots comparison bar chart for Accuracy, Precision, Recall, and F1 Score."""
        models = self._resolve_models(models)
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

        with _apply_plot_style():
            fig, ax = plt.subplots(figsize=(12, 7))
            sns.barplot(x='Model', y='Score', hue='Metric', data=df_plot, palette="viridis", ax=ax)
            ax.set_title("Comprehensive Model Comparison", fontsize=16, pad=20, fontweight='bold')
            ax.set_ylim(0, 1.1)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.tick_params(axis='x', rotation=45)
            for container in ax.containers:
                ax.bar_label(container, fmt='%.2f', padding=3, fontsize=9)
            plt.tight_layout()
            self._save_and_show(fig, save_path=save_path, show_plot=show_plot)

    def get_summary(
        self,
        models: Optional[List[Tuple]] = None,
        save_prefix: Optional[str] = None,
        show_plot: bool = True
    ):
        """Runs full reporting and visualization pipeline."""
        models = self._resolve_models(models)
        df_report = self.show_tabular_report(models, return_df=True)

        if save_prefix:
            if df_report is not None:
                save_report(df_report, f"{save_prefix}_report.csv")
            self.plot_confusion_matrices(models, save_path=f"{save_prefix}_confusion_matrices.png", show_plot=show_plot)
            self.plot_roc_curves(models, save_path=f"{save_prefix}_roc_curves.png", show_plot=show_plot)
            self.plot_comparison(models, save_path=f"{save_prefix}_comparison.png", show_plot=show_plot)
        else:
            self.plot_confusion_matrices(models, show_plot=show_plot)
            self.plot_roc_curves(models, show_plot=show_plot)
            self.plot_comparison(models, show_plot=show_plot)


class MultiModelRegressor(_BaseMultiModel):
    """
    Automated multi-model analysis tool for regression datasets.
    """

    def __init__(
        self,
        X: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y: Optional[Union[pd.Series, pd.DataFrame, np.ndarray]] = None,
        test_size: float = 0.3,
        scaled_data: bool = False,
        random_state: Optional[int] = 42,
        n_jobs: Optional[int] = -1
    ):
        super().__init__(
            X=X, y=y, test_size=test_size, scaled_data=scaled_data,
            random_state=random_state, n_jobs=n_jobs
        )
        self._split_and_scale(self.y_flat)

    def evaluate_model(
        self,
        model: Any,
        X_test: Optional[Any] = None,
        y_true: Optional[np.ndarray] = None
    ):
        if X_test is None:
            X_test = self.X_test_scaled
        if y_true is None:
            y_true = self.y_test

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        return mae, mse, rmse, r2, y_pred

    def LinearRegression_model(
        self,
        **kwargs
    ):
        kwargs.pop('random_state', None)
        model = LinearRegression(n_jobs=self.n_jobs, **kwargs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(self.X_train_scaled, self.y_train)
        return self.evaluate_model(model, self.X_test_scaled, self.y_test)

    def Lasso_model(
        self,
        alpha: float = 0.1,
        random_state: Optional[int] = None,
        **kwargs
    ):
        rs = random_state if random_state is not None else self.random_state
        model = Lasso(alpha=alpha, random_state=rs, **kwargs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(self.X_train_scaled, self.y_train)
        return self.evaluate_model(model, self.X_test_scaled, self.y_test)

    def Ridge_model(
        self,
        alpha: float = 1.0,
        random_state: Optional[int] = None,
        **kwargs
    ):
        rs = random_state if random_state is not None else self.random_state
        model = Ridge(alpha=alpha, random_state=rs, **kwargs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(self.X_train_scaled, self.y_train)
        return self.evaluate_model(model, self.X_test_scaled, self.y_test)

    def SVR_model(
        self,
        kernel: str = 'rbf',
        **kwargs
    ):
        kwargs.pop('random_state', None)
        model = SVR(kernel=kernel, **kwargs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(self.X_train_scaled, self.y_train)
        return self.evaluate_model(model, self.X_test_scaled, self.y_test)

    def DecisionTree_model(
        self,
        random_state: Optional[int] = None,
        **kwargs
    ):
        rs = random_state if random_state is not None else self.random_state
        model = DecisionTreeRegressor(random_state=rs, **kwargs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(self.X_train_scaled, self.y_train)
        return self.evaluate_model(model, self.X_test_scaled, self.y_test)

    def RandomForest_model(
        self,
        n_estimators: int = 100,
        random_state: Optional[int] = None,
        **kwargs
    ):
        rs = random_state if random_state is not None else self.random_state
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=rs, n_jobs=self.n_jobs, **kwargs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(self.X_train_scaled, self.y_train)
        return self.evaluate_model(model, self.X_test_scaled, self.y_test)

    def GradientBoosting_model(
        self,
        n_estimators: int = 100,
        random_state: Optional[int] = None,
        **kwargs
    ):
        rs = random_state if random_state is not None else self.random_state
        model = GradientBoostingRegressor(n_estimators=n_estimators, random_state=rs, **kwargs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(self.X_train_scaled, self.y_train)
        return self.evaluate_model(model, self.X_test_scaled, self.y_test)

    def AdaBoost_model(
        self,
        n_estimators: int = 50,
        random_state: Optional[int] = None,
        **kwargs
    ):
        rs = random_state if random_state is not None else self.random_state
        model = AdaBoostRegressor(n_estimators=n_estimators, random_state=rs, **kwargs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(self.X_train_scaled, self.y_train)
        return self.evaluate_model(model, self.X_test_scaled, self.y_test)

    def run_all_models(
        self,
        custom_models: Optional[Dict[str, Any]] = None,
        random_state: Optional[int] = None
    ) -> List[Tuple]:
        model_methods = [
            ('Linear Regression', lambda: self.LinearRegression_model(random_state=random_state)),
            ('Lasso Regression', lambda: self.Lasso_model(random_state=random_state)),
            ('Ridge Regression', lambda: self.Ridge_model(random_state=random_state)),
            ('SVR', lambda: self.SVR_model(random_state=random_state)),
            ('Decision Tree Regressor', lambda: self.DecisionTree_model(random_state=random_state)),
            ('Random Forest Regressor', lambda: self.RandomForest_model(random_state=random_state)),
            ('Gradient Boosting Regressor', lambda: self.GradientBoosting_model(random_state=random_state)),
            ('AdaBoost Regressor', lambda: self.AdaBoost_model(random_state=random_state))
        ]

        models = []
        for name, method in model_methods:
            try:
                eval_res = method()
                models.append((name, *eval_res))
            except Exception as e:
                _safe_print(f"[WARNING] Model '{name}' failed to train or evaluate: {e}")

        if custom_models and isinstance(custom_models, dict):
            for name, model_inst in custom_models.items():
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model_inst.fit(self.X_train_scaled, self.y_train)
                    eval_res = self.evaluate_model(model_inst, self.X_test_scaled, self.y_test)
                    models.append((name, *eval_res))
                except Exception as e:
                    _safe_print(f"[WARNING] Custom regressor model '{name}' failed: {e}")

        self.models_ = models
        return models

    def show_tabular_report(self, models: Optional[List[Tuple]] = None, return_df: bool = False) -> Optional[pd.DataFrame]:
        """Displays all regression metrics in a clean tabular format and recommends the best model."""
        models = self._resolve_models(models)
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
        self._last_df_report = df

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

    def plot_true_vs_predicted(
        self,
        models: Optional[List[Tuple]] = None,
        save_path: Optional[str] = None,
        show_plot: bool = True
    ):
        """Plots True vs Predicted values for all models."""
        models = self._resolve_models(models)
        if not models:
            return

        num_models = len(models)
        cols = min(2, num_models)
        rows = (num_models + cols - 1) // cols

        colors = ['blue', 'green', 'orange', 'purple', 'red', 'cyan', 'magenta']

        with _apply_plot_style():
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

                line_max = max(np.max(y_pred_arr), np.max(y_test_arr))
                line_min = min(np.min(y_pred_arr), np.min(y_test_arr))
                ax.plot([line_min, line_max], [line_min, line_max], 'k--', lw=2)

                ax.set_title(f'{name}\nR2: {r2:.2f} | RMSE: {rmse:.2f}', fontsize=14, fontweight='bold')
                ax.set_xlabel('True Values')
                ax.set_ylabel('Predicted Values')
                ax.legend()

            for i in range(num_models, len(axes_list)):
                axes_list[i].axis('off')

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            self._save_and_show(fig, save_path=save_path, show_plot=show_plot)

    def plot_comparison(
        self,
        models: Optional[List[Tuple]] = None,
        save_path: Optional[str] = None,
        show_plot: bool = True
    ):
        """Plots comparison bar charts for R2 score."""
        models = self._resolve_models(models)
        if not models:
            return

        model_names = [m[0] for m in models]
        r2_scores = [m[4] for m in models]

        with _apply_plot_style():
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x=model_names, y=r2_scores, palette='viridis', ax=ax)
            ax.set_title("Regressor R2 Score Comparison", pad=20, fontweight='bold')
            ax.set_ylabel("R2 Score")
            ax.tick_params(axis='x', rotation=45)

            for container in ax.containers:
                ax.bar_label(container, fmt='%.2f', padding=3, fontsize=10)

            plt.tight_layout()
            self._save_and_show(fig, save_path=save_path, show_plot=show_plot)

    def get_summary(
        self,
        models: Optional[List[Tuple]] = None,
        save_prefix: Optional[str] = None,
        show_plot: bool = True
    ):
        """Runs full reporting and visualization pipeline."""
        models = self._resolve_models(models)
        df_report = self.show_tabular_report(models, return_df=True)

        if save_prefix:
            if df_report is not None:
                save_report(df_report, f"{save_prefix}_report.csv")
            self.plot_true_vs_predicted(models, save_path=f"{save_prefix}_true_vs_pred.png", show_plot=show_plot)
            self.plot_comparison(models, save_path=f"{save_prefix}_comparison.png", show_plot=show_plot)
        else:
            self.plot_true_vs_predicted(models, show_plot=show_plot)
            self.plot_comparison(models, show_plot=show_plot)


# Backwards compatibility alias for typo in earlier version
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