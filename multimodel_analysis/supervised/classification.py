"""Supervised classification: ``MultiModelClassifier``."""
import warnings
from typing import Any, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.multiclass import type_of_target

from ..core.base import _BaseMultiModel
from ..core.plotting import comparison_figure
from ..core.registry import ModelRegistry, ModelSpec
from ..core.utils import _apply_plot_style, _safe_print, save_report


def _cw(ctx, params):
    """Adds ``class_weight`` from the build context unless the caller set it explicitly."""
    if ctx.class_weight and "class_weight" not in params:
        return {**params, "class_weight": ctx.class_weight}
    return params


def _knn_clf(ctx, n_neighbors=None, **kw):
    n = n_neighbors if n_neighbors is not None else min(10, max(1, ctx.n_train - 1))
    return KNeighborsClassifier(n_neighbors=n, n_jobs=ctx.n_jobs, **kw)


def _xgb_clf(ctx, **p):
    from xgboost import XGBClassifier
    return XGBClassifier(random_state=ctx.random_state, n_jobs=ctx.n_jobs, verbosity=0, **p)


def _lgbm_clf(ctx, **p):
    from lightgbm import LGBMClassifier
    return LGBMClassifier(random_state=ctx.random_state, n_jobs=ctx.n_jobs, verbose=-1, **_cw(ctx, p))


def _cat_clf(ctx, **p):
    from catboost import CatBoostClassifier
    return CatBoostClassifier(random_seed=ctx.random_state, thread_count=ctx.n_jobs, verbose=0,
                              allow_writing_files=False, **p)


CLASSIFIER_REGISTRY = ModelRegistry([
    ModelSpec("logistic_regression", "Logistic Regression",
              lambda ctx, **p: LogisticRegression(random_state=ctx.random_state, **_cw(ctx, p)),
              defaults={"max_iter": 1000},
              search_space={"C": [0.01, 0.1, 1, 10, 100]}),
    ModelSpec("svm", "SVM",
              lambda ctx, **p: SVC(random_state=ctx.random_state, **_cw(ctx, p)),
              defaults={"kernel": "rbf", "probability": True},
              search_space={"C": [0.1, 1, 10, 100], "gamma": ["scale", "auto", 0.01, 0.1],
                            "kernel": ["rbf", "linear"]}),
    ModelSpec("decision_tree", "Decision Tree",
              lambda ctx, **p: DecisionTreeClassifier(random_state=ctx.random_state, **_cw(ctx, p)),
              search_space={"max_depth": [None, 3, 5, 8, 12], "min_samples_split": [2, 5, 10],
                            "min_samples_leaf": [1, 2, 4], "criterion": ["gini", "entropy"]}),
    ModelSpec("knn", "KNN", _knn_clf, defaults={"n_neighbors": None},
              search_space={"n_neighbors": [3, 5, 7, 9, 11, 15], "weights": ["uniform", "distance"], "p": [1, 2]}),
    ModelSpec("naive_bayes", "Naive Bayes", lambda ctx, **p: GaussianNB(**p),
              search_space={"var_smoothing": [1e-11, 1e-10, 1e-9, 1e-8, 1e-7]}),
    ModelSpec("random_forest", "Random Forest",
              lambda ctx, **p: RandomForestClassifier(random_state=ctx.random_state, n_jobs=ctx.n_jobs, **_cw(ctx, p)),
              defaults={"n_estimators": 100},
              search_space={"n_estimators": [100, 200, 300], "max_depth": [None, 5, 10, 20],
                            "min_samples_split": [2, 5, 10], "min_samples_leaf": [1, 2, 4],
                            "max_features": ["sqrt", "log2", None]}),
    ModelSpec("gradient_boosting", "Gradient Boosting",
              lambda ctx, **p: GradientBoostingClassifier(random_state=ctx.random_state, **p),
              defaults={"n_estimators": 100},
              search_space={"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.05, 0.1, 0.2],
                            "max_depth": [2, 3, 4, 5], "subsample": [0.7, 0.85, 1.0]}),
    ModelSpec("adaboost", "AdaBoost",
              lambda ctx, **p: AdaBoostClassifier(random_state=ctx.random_state, **p),
              defaults={"n_estimators": 50},
              search_space={"n_estimators": [25, 50, 100, 200], "learning_rate": [0.01, 0.1, 0.5, 1.0]}),

    # ---- extra models (Phase 5): run with include=[...] or include='all' ------------------------
    ModelSpec("extra_trees", "Extra Trees",
              lambda ctx, **p: ExtraTreesClassifier(random_state=ctx.random_state, n_jobs=ctx.n_jobs, **_cw(ctx, p)),
              defaults={"n_estimators": 200}, default=False,
              search_space={"n_estimators": [100, 200, 300], "max_depth": [None, 5, 10, 20],
                            "min_samples_split": [2, 5, 10], "min_samples_leaf": [1, 2, 4]}),
    ModelSpec("hist_gradient_boosting", "Hist Gradient Boosting",
              lambda ctx, **p: HistGradientBoostingClassifier(random_state=ctx.random_state, **p), default=False,
              search_space={"learning_rate": [0.03, 0.1, 0.2], "max_iter": [100, 200, 300],
                            "max_depth": [None, 3, 6], "l2_regularization": [0.0, 0.1, 1.0]}),
    ModelSpec("mlp", "Neural Network (MLP)",
              lambda ctx, **p: MLPClassifier(random_state=ctx.random_state, **p),
              defaults={"hidden_layer_sizes": (64,), "max_iter": 500}, default=False,
              search_space={"hidden_layer_sizes": [(32,), (64,), (64, 32)], "alpha": [1e-5, 1e-4, 1e-3, 1e-2],
                            "learning_rate_init": [1e-3, 1e-2]}),
    ModelSpec("lda", "Linear Discriminant Analysis", lambda ctx, **p: LinearDiscriminantAnalysis(**p), default=False,
              search_space={"solver": ["svd", "lsqr", "eigen"]}),
    ModelSpec("ridge_classifier", "Ridge Classifier",
              lambda ctx, **p: RidgeClassifier(random_state=ctx.random_state, **_cw(ctx, p)), default=False,
              search_space={"alpha": [0.01, 0.1, 1, 10, 100]}),
    ModelSpec("linear_svm", "Linear SVM",
              lambda ctx, **p: LinearSVC(random_state=ctx.random_state, **_cw(ctx, p)),
              defaults={"max_iter": 5000, "dual": "auto"}, default=False,
              search_space={"C": [0.01, 0.1, 1, 10, 100]}),

    # ---- optional boosters: pip install xgboost / lightgbm / catboost -----------------------------
    ModelSpec("xgboost", "XGBoost", _xgb_clf, requires="xgboost", default=False,
              defaults={"n_estimators": 200, "learning_rate": 0.1},
              search_space={"n_estimators": [100, 200, 400], "learning_rate": [0.03, 0.1, 0.2],
                            "max_depth": [3, 4, 6, 8], "subsample": [0.7, 0.85, 1.0],
                            "colsample_bytree": [0.7, 0.85, 1.0]}),
    ModelSpec("lightgbm", "LightGBM", _lgbm_clf, requires="lightgbm", default=False,
              defaults={"n_estimators": 200, "learning_rate": 0.1},
              search_space={"n_estimators": [100, 200, 400], "learning_rate": [0.03, 0.1, 0.2],
                            "num_leaves": [15, 31, 63], "subsample": [0.7, 0.85, 1.0],
                            "colsample_bytree": [0.7, 0.85, 1.0]}),
    ModelSpec("catboost", "CatBoost", _cat_clf, requires="catboost", default=False,
              defaults={"iterations": 300, "learning_rate": 0.1},
              search_space={"iterations": [200, 300, 500], "learning_rate": [0.03, 0.1, 0.2],
                            "depth": [4, 6, 8]}),
])


class MultiModelClassifier(_BaseMultiModel):
    registry = CLASSIFIER_REGISTRY
    task = "classification"
    default_scoring = "accuracy"
    """
    Automated multi-model analysis tool for classification datasets.

    Parameters
    ----------
    X, y : features and target. ``y`` may hold strings, categories or numbers; continuous
        targets are rejected (use ``MultiModelRegressor`` instead).
    test_size : fraction (float) or number (int) of samples held out for testing.
    scaled_data : standardise features (fit on the training split only). Default ``True``.
    random_state : seed for the split and for every stochastic model.
    stratify : keep class proportions in the split when the class counts allow it.
    n_jobs : parallel workers for estimators that support it.
    verbose : print warnings captured while models train (they are always stored in ``warnings_``).
    preprocess : automatically impute missing values and one-hot encode text columns (fit on the
        training split only). Default ``False`` keeps the strict "numeric, no NaN" input check.
    class_weight : ``None`` or ``'balanced'`` to counter imbalanced classes. Applied by Logistic
        Regression, SVM, Decision Tree and Random Forest (and any registry model that supports it).
    """

    def __init__(
        self,
        X: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y: Optional[Union[pd.Series, pd.DataFrame, np.ndarray]] = None,
        test_size: float = 0.3,
        scaled_data: bool = True,
        random_state: Optional[int] = 42,
        stratify: bool = True,
        n_jobs: Optional[int] = -1,
        verbose: bool = True,
        preprocess: bool = False,
        class_weight: Optional[str] = None
    ):
        super().__init__(
            X=X, y=y, test_size=test_size, scaled_data=scaled_data,
            random_state=random_state, n_jobs=n_jobs, verbose=verbose, preprocess=preprocess
        )
        if class_weight not in (None, "balanced"):
            raise ValueError("class_weight must be None or 'balanced'.")
        self.class_weight = class_weight

        target_kind = type_of_target(self.y_flat)
        if target_kind in ('continuous', 'continuous-multioutput'):
            raise ValueError(
                "The target 'y' looks continuous (real-valued), which is a regression problem. "
                "Use MultiModelRegressor, or convert y to class labels before using MultiModelClassifier."
            )

        # Label Encoding for y (handles strings, categorical types, and numbers)
        self.label_encoder = LabelEncoder()
        self.y_encoded = self.label_encoder.fit_transform(self.y_flat)
        self.classes_ = self.label_encoder.classes_
        self.n_classes_ = len(self.classes_)

        if self.n_classes_ < 2:
            raise ValueError("y contains only one class; classification needs at least two classes.")
        if self.n_classes_ > 50 and self.n_classes_ > 0.5 * len(self.y_flat):
            warnings.warn(
                f"y has {self.n_classes_} distinct values for {len(self.y_flat)} samples - "
                "this may be a regression target. Consider MultiModelRegressor.",
                UserWarning, stacklevel=2
            )

        # Stratification setup for classification
        stratify_y = None
        if stratify:
            counts = np.bincount(self.y_encoded)
            if np.min(counts) >= 2:
                stratify_y = self.y_encoded

        self._split_and_scale(self.y_encoded, stratify_y=stratify_y)

    # ------------------------------------------------------------------ evaluation
    def _get_scores(self, model: Any, X_test: Any) -> Optional[np.ndarray]:
        """Returns class probabilities, or softmax/sigmoid-mapped decision scores, or ``None``."""
        y_pred_proba = None
        if hasattr(model, "predict_proba"):
            try:
                y_pred_proba = np.asarray(model.predict_proba(X_test))
            except Exception:
                y_pred_proba = None
        if y_pred_proba is None and hasattr(model, "decision_function"):
            try:
                df_val = np.asarray(model.decision_function(X_test))
                if df_val.ndim == 1:
                    prob1 = 1 / (1 + np.exp(-np.clip(df_val, -500, 500)))
                    y_pred_proba = np.vstack([1 - prob1, prob1]).T
                else:
                    exp_df = np.exp(df_val - np.max(df_val, axis=1, keepdims=True))
                    y_pred_proba = exp_df / np.sum(exp_df, axis=1, keepdims=True)
            except Exception:
                y_pred_proba = None
        if y_pred_proba is None:
            return None

        # If the model never saw some class during training its probability matrix has fewer columns.
        model_classes = getattr(model, "classes_", None)
        if (model_classes is not None and y_pred_proba.ndim == 2
                and y_pred_proba.shape[1] == len(model_classes) != self.n_classes_):
            try:
                full = np.zeros((y_pred_proba.shape[0], self.n_classes_))
                full[:, np.asarray(model_classes, dtype=int)] = y_pred_proba
                y_pred_proba = full
            except Exception:
                pass
        return y_pred_proba

    def _compute_roc(self, model: Any, X_test: Any, y_true: np.ndarray):
        """Computes ``(fpr_dict, tpr_dict, roc_auc)`` using only the classes present in ``y_true``."""
        label = model.__class__.__name__
        y_pred_proba = self._get_scores(model, X_test)
        if y_pred_proba is None:
            return {}, {}, np.nan

        present = np.unique(y_true)
        if len(present) < 2:
            self._warn(label, "ROC AUC is undefined because the test split contains only one class.")
            return {}, {}, np.nan

        fpr_dict, tpr_dict = {}, {}
        try:
            if self.n_classes_ == 2:
                if y_pred_proba.ndim > 1 and y_pred_proba.shape[1] >= 2:
                    pos_proba = y_pred_proba[:, 1]
                else:
                    pos_proba = y_pred_proba.ravel()
                fpr, tpr, _ = roc_curve(y_true, pos_proba)
                fpr_dict['binary'], tpr_dict['binary'] = fpr, tpr
                return fpr_dict, tpr_dict, roc_auc_score(y_true, pos_proba)

            # Multiclass: restrict to the classes that actually occur in y_true and renormalise.
            if len(present) < self.n_classes_:
                self._warn(label, f"ROC AUC computed on the {len(present)} of {self.n_classes_} classes "
                                  "present in the test split.")
            sub = y_pred_proba[:, present]
            row_sum = sub.sum(axis=1, keepdims=True)
            sub = np.where(row_sum > 0, sub / np.where(row_sum > 0, row_sum, 1), 1.0 / len(present))

            if len(present) == 2:
                y_bin = (y_true == present[1]).astype(int)
                fpr, tpr, _ = roc_curve(y_bin, sub[:, 1])
                fpr_dict['binary'], tpr_dict['binary'] = fpr, tpr
                return fpr_dict, tpr_dict, roc_auc_score(y_bin, sub[:, 1])

            roc_auc = roc_auc_score(y_true, sub, multi_class='ovr', average='weighted', labels=present)
            all_fpr, all_tpr = [], []
            for j, cls in enumerate(present):
                f_i, t_i, _ = roc_curve((y_true == cls).astype(int), sub[:, j])
                fpr_dict[int(cls)], tpr_dict[int(cls)] = f_i, t_i
                all_fpr.append(f_i)
                all_tpr.append(t_i)
            mean_fpr = np.unique(np.concatenate(all_fpr))
            mean_tpr = np.zeros_like(mean_fpr)
            for f_i, t_i in zip(all_fpr, all_tpr):
                mean_tpr += np.interp(mean_fpr, f_i, t_i)
            fpr_dict['macro'], tpr_dict['macro'] = mean_fpr, mean_tpr / len(all_fpr)
            return fpr_dict, tpr_dict, roc_auc
        except Exception as e:
            self._warn(label, f"ROC AUC computation failed: {e}")
            return {}, {}, np.nan

    def evaluate_model(
        self,
        model: Any,
        X_test: Optional[Any] = None,
        y_true: Optional[np.ndarray] = None
    ):
        """Evaluates a fitted classifier.

        Returns ``(report, confusion_matrix, accuracy, precision, recall, f1, fpr_dict, tpr_dict, roc_auc)``.
        The confusion matrix always has shape ``(n_classes, n_classes)`` aligned with ``self.classes_``.
        """
        if X_test is None:
            X_test = self.X_test_scaled
        if y_true is None:
            y_true = self.y_test
        y_true = np.asarray(y_true)

        predicted = np.asarray(model.predict(X_test))

        # Invert target encoding for reports
        y_true_orig = self.label_encoder.inverse_transform(y_true)
        pred_orig = self.label_encoder.inverse_transform(predicted)

        report = classification_report(y_true_orig, pred_orig, zero_division=0)
        matrix = confusion_matrix(y_true, predicted, labels=np.arange(self.n_classes_))
        accuracy = accuracy_score(y_true, predicted)

        precision = precision_score(y_true, predicted, average='weighted', zero_division=0)
        recall = recall_score(y_true, predicted, average='weighted', zero_division=0)
        f1 = f1_score(y_true, predicted, average='weighted', zero_division=0)

        fpr_dict, tpr_dict, roc_auc = self._compute_roc(model, X_test, y_true)
        return report, matrix, accuracy, precision, recall, f1, fpr_dict, tpr_dict, roc_auc

    # ------------------------------------------------------------------ models
    # Each wrapper keeps its historical signature; they all delegate to the registry (see run_model()).
    def Logistic_model(self, random_state: Optional[int] = None, max_iter: int = 1000, **kwargs):
        return self._run_spec("logistic_regression", random_state=random_state, max_iter=max_iter, **kwargs)

    def Support_vector_model(self, random_state: Optional[int] = None, kernel: str = "rbf",
                             probability: bool = True, **kwargs):
        return self._run_spec("svm", random_state=random_state, kernel=kernel, probability=probability, **kwargs)

    def DecisionTree_model(self, random_state: Optional[int] = None, **kwargs):
        return self._run_spec("decision_tree", random_state=random_state, **kwargs)

    def KNN_model(self, n_neighbors: Optional[int] = None, **kwargs):
        return self._run_spec("knn", n_neighbors=n_neighbors, **kwargs)

    def Naive_Bayes_model(self, **kwargs):
        return self._run_spec("naive_bayes", **kwargs)

    def RandomForest_model(self, n_estimators: int = 100, random_state: Optional[int] = None, **kwargs):
        return self._run_spec("random_forest", random_state=random_state, n_estimators=n_estimators, **kwargs)

    def GradientBoosting_model(self, n_estimators: int = 100, random_state: Optional[int] = None, **kwargs):
        return self._run_spec("gradient_boosting", random_state=random_state, n_estimators=n_estimators, **kwargs)

    def AdaBoost_model(self, n_estimators: int = 50, random_state: Optional[int] = None, **kwargs):
        return self._run_spec("adaboost", random_state=random_state, n_estimators=n_estimators, **kwargs)

    # ------------------------------------------------------------------ reporting
    def show_tabular_report(
        self,
        models: Optional[List[Tuple]] = None,
        return_df: bool = False,
        sort_by: str = 'Accuracy'
    ) -> Optional[pd.DataFrame]:
        """Displays all model metrics in a clean tabular format and recommends the best model.

        ``sort_by`` can be 'Accuracy', 'Precision', 'Recall', 'F1 Score' or 'ROC AUC'.
        """
        models = self._resolve_models(models)
        if not models:
            _safe_print("No models were successfully evaluated.")
            self._print_failures()
            return None

        data = []
        for name, report, matrix, accuracy, precision, recall, f1, fpr, tpr, roc_auc in models:
            data.append({
                'Model': name,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1,
                'ROC AUC': roc_auc,
                'Fit Time (s)': self.fit_times_.get(name, np.nan)
            })

        df = pd.DataFrame(data)
        if sort_by not in df.columns or sort_by == 'Model':
            raise ValueError(f"sort_by must be one of {list(df.columns[1:])}, got '{sort_by}'.")
        df = df.sort_values(by=sort_by, ascending=False, na_position='last').reset_index(drop=True)
        self._last_df_report = df

        _safe_print(f"\n{'='*60}\n MODEL COMPARISON TABLE \n{'='*60}")
        _safe_print(df.to_string(index=False, na_rep="N/A"))
        _safe_print(f"{'='*60}\n")
        self._print_failures()

        best_model = df.iloc[0]
        _safe_print(f"* BEST MODEL RECOMMENDATION (by {sort_by}): {best_model['Model']}")
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
            fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5 * rows), squeeze=False)
            fig.suptitle('Confusion Matrices', fontsize=18, fontweight='bold')
            axes_list = axes.flatten()

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
        show_plot: bool = True,
        metric: str = 'F1 Score'
    ):
        """Plots all models ranked on one metric (default 'F1 Score'; or 'Accuracy', 'Precision', 'Recall', 'ROC AUC')."""
        models = self._resolve_models(models)
        if not models:
            return

        columns = {'Accuracy': 3, 'Precision': 4, 'Recall': 5, 'F1 Score': 6, 'ROC AUC': 9}
        if metric not in columns:
            raise ValueError(f"metric must be one of {list(columns)}, got '{metric}'.")

        with _apply_plot_style():
            fig = comparison_figure(
                [m[0] for m in models], [m[columns[metric]] for m in models], metric=metric,
                title=f"Model comparison: {metric}",
                footnote="Measured on the held-out test split.",
            )
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
