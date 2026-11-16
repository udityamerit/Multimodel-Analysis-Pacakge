import contextlib
import os
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)

# Classification Models
# Regression Models
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge

# Metrics
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils.multiclass import type_of_target

# Warning categories worth showing to the user while models train.
_VISIBLE_WARNINGS = ("ConvergenceWarning", "UndefinedMetricWarning", "RuntimeWarning", "UserWarning")

# Report columns where a LOWER value is better (used when sorting the comparison table).
_LOWER_IS_BETTER = ("MAE", "MSE", "RMSE")


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


def _validate_inputs(X: Any, y: Any, test_size: Union[int, float]) -> np.ndarray:
    """Validates features / target / test_size and returns the flattened target array.

    Raises clear ``ValueError`` / ``TypeError`` messages instead of cryptic scikit-learn errors.
    """
    if X is None or y is None:
        raise ValueError("Both features 'X' and target 'y' must be provided to initialize MultiModel.")

    # --- target: must be 1-D (or a single column) ---
    if isinstance(y, (pd.Series, pd.DataFrame)):
        y_arr = y.values
    else:
        y_arr = np.asarray(y)
    if y_arr.ndim == 2 and y_arr.shape[1] > 1:
        raise ValueError(
            f"y must be one-dimensional or a single column, but it has {y_arr.shape[1]} columns. "
            "Multi-output targets are not supported."
        )
    y_flat = y_arr.ravel()

    n_samples = len(X)
    if len(y_flat) != n_samples:
        raise ValueError(f"X and y have different lengths: X has {n_samples} rows but y has {len(y_flat)} values.")
    if pd.isnull(y_flat).any():
        raise ValueError(f"y contains {int(pd.isnull(y_flat).sum())} missing value(s). Drop or fill them first.")

    # --- test_size ---
    if isinstance(test_size, (float, np.floating)):
        if not 0.0 < float(test_size) < 1.0:
            raise ValueError(f"test_size must be between 0 and 1 when given as a float, got {test_size}.")
    elif isinstance(test_size, (int, np.integer)) and not isinstance(test_size, bool):
        if not 0 < int(test_size) < n_samples:
            raise ValueError(f"test_size must be between 1 and {n_samples - 1} when given as an int, got {test_size}.")
    else:
        raise TypeError(f"test_size must be a float or an int, got {type(test_size).__name__}.")

    # --- features: numeric, finite ---
    if isinstance(X, pd.DataFrame):
        non_numeric = [str(c) for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
        if non_numeric:
            raise ValueError(
                f"X contains non-numeric column(s): {non_numeric[:10]}. Encode them first "
                "(e.g. pd.get_dummies(X)) - text/categorical columns cannot be fed to the models directly."
            )
        values = X.to_numpy(dtype=float)
    else:
        try:
            values = np.asarray(X, dtype=float)
        except (ValueError, TypeError):
            raise ValueError("X contains non-numeric values. Encode categorical/text features first.") from None
    if not np.isfinite(values).all():
        n_bad = int((~np.isfinite(values)).sum())
        raise ValueError(
            f"X contains {n_bad} missing (NaN) or infinite value(s). Impute or drop them first "
            "(e.g. X.fillna(X.median()))."
        )
    return y_flat


def save_report(df: Optional[pd.DataFrame] = None, filepath: str = "report.csv") -> None:
    """Saves a comparison tabular report DataFrame to CSV, Excel (.xlsx), HTML, or JSON based on extension."""
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
    elif ext == '.xlsx':
        try:
            df.to_excel(filepath, index=False)
        except ImportError as exc:
            raise ImportError(
                "Saving to Excel needs the 'openpyxl' package. Install it with: "
                "pip install openpyxl   (or: pip install multimodel-analysis[excel])"
            ) from exc
    elif ext == '.xls':
        raise ValueError("The legacy '.xls' format is no longer supported. Please use '.xlsx' instead.")
    elif ext in ('.html', '.htm'):
        df.to_html(filepath, index=False)
    elif ext == '.json':
        df.to_json(filepath, orient='records', indent=4)
    else:
        df.to_csv(filepath, index=False)
    _safe_print(f"[INFO] Tabular report saved successfully to '{filepath}'.")


class _BaseMultiModel:
    """Base class providing shared initialization, validation, scaling, and utility methods.

    Attributes populated while working:
        models_          list of evaluation tuples from the last ``run_all_models()`` call.
        failed_models_   ``{model name: error message}`` for models that failed to train/evaluate.
        warnings_        ``{model name: [warning strings]}`` captured while models trained.
    """

    def __init__(
        self,
        X: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y: Optional[Union[pd.Series, pd.DataFrame, np.ndarray]] = None,
        test_size: float = 0.3,
        scaled_data: bool = True,
        random_state: Optional[int] = 42,
        n_jobs: Optional[int] = -1,
        verbose: bool = True
    ):
        self.y_flat = _validate_inputs(X, y, test_size)

        self.X = X
        self.y_raw = y
        self.test_size = test_size
        self.scaled_data = scaled_data
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.models_ = []
        self.failed_models_ = {}
        self.warnings_ = {}
        self._last_df_report = None

        self.scaler = StandardScaler()

    # ------------------------------------------------------------------ helpers
    def _warn(self, label: str, message: str, show: bool = True) -> None:
        """Records a warning for ``label`` (deduplicated) and optionally prints it."""
        bucket = self.warnings_.setdefault(label, [])
        if message not in bucket:
            bucket.append(message)
            if show and self.verbose:
                _safe_print(f"[WARNING] {label}: {message}")

    @contextlib.contextmanager
    def _track_warnings(self, label: str):
        """Captures warnings raised inside the block (instead of silencing them) and records them."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            yield
        for w in caught:
            text = str(w.message).strip()
            first_line = text.splitlines()[0] if text else ""
            self._warn(label, f"{w.category.__name__}: {first_line}",
                       show=w.category.__name__ in _VISIBLE_WARNINGS)

    def _fit_and_evaluate(self, model: Any, name: Optional[str] = None):
        """Fits ``model`` on the training split and evaluates it on the test split."""
        label = name or model.__class__.__name__
        with self._track_warnings(label):
            model.fit(self.X_train_scaled, self.y_train)
            return self.evaluate_model(model, self.X_test_scaled, self.y_test)

    def _split_and_scale(self, y_split: np.ndarray, stratify_y: Optional[np.ndarray] = None):
        """Splits features and targets and applies optional scaling preserving DataFrame structures."""
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, y_split, test_size=self.test_size, random_state=self.random_state, stratify=stratify_y
            )
        except ValueError as exc:
            if stratify_y is None:
                raise
            _safe_print(f"[WARNING] Stratified split not possible ({exc}). Falling back to a random split.")
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, y_split, test_size=self.test_size, random_state=self.random_state
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

    def _run_model_list(self, model_methods: List[Tuple[str, Any]], custom_models: Optional[Dict[str, Any]]) -> List[Tuple]:
        """Runs the built-in model callables plus any custom estimators; records failures instead of hiding them."""
        self.failed_models_ = {}
        self.warnings_ = {}
        models = []
        for name, method in model_methods:
            try:
                models.append((name, *method()))
            except Exception as e:
                self.failed_models_[name] = f"{type(e).__name__}: {e}"
                _safe_print(f"[WARNING] Model '{name}' failed to train or evaluate: {e}")

        if custom_models and isinstance(custom_models, dict):
            for name, model_inst in custom_models.items():
                try:
                    models.append((name, *self._fit_and_evaluate(model_inst, name=name)))
                except Exception as e:
                    self.failed_models_[name] = f"{type(e).__name__}: {e}"
                    _safe_print(f"[WARNING] Custom model '{name}' failed: {e}")

        self.models_ = models
        return models

    def _print_failures(self) -> None:
        if self.failed_models_:
            _safe_print("Models that FAILED (not in the table above):")
            for name, reason in self.failed_models_.items():
                _safe_print(f"   - {name}: {reason}")
            _safe_print(f"{'='*60}\n")


class MultiModelClassifier(_BaseMultiModel):
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
        verbose: bool = True
    ):
        super().__init__(
            X=X, y=y, test_size=test_size, scaled_data=scaled_data,
            random_state=random_state, n_jobs=n_jobs, verbose=verbose
        )

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
    def Logistic_model(
        self,
        random_state: Optional[int] = None,
        max_iter: int = 1000,
        **kwargs
    ):
        rs = random_state if random_state is not None else self.random_state
        model = LogisticRegression(random_state=rs, max_iter=max_iter, **kwargs)
        return self._fit_and_evaluate(model)

    def Support_vector_model(
        self,
        random_state: Optional[int] = None,
        kernel: str = 'rbf',
        probability: bool = True,
        **kwargs
    ):
        rs = random_state if random_state is not None else self.random_state
        svc = SVC(kernel=kernel, probability=probability, random_state=rs, **kwargs)
        return self._fit_and_evaluate(svc)

    def DecisionTree_model(
        self,
        random_state: Optional[int] = None,
        **kwargs
    ):
        rs = random_state if random_state is not None else self.random_state
        model = DecisionTreeClassifier(random_state=rs, **kwargs)
        return self._fit_and_evaluate(model)

    def KNN_model(
        self,
        n_neighbors: Optional[int] = None,
        **kwargs
    ):
        kwargs.pop('random_state', None)
        if n_neighbors is None:
            n_neighbors = min(10, max(1, len(self.y_train) - 1))
        model = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=self.n_jobs, **kwargs)
        return self._fit_and_evaluate(model)

    def Naive_Bayes_model(
        self,
        **kwargs
    ):
        kwargs.pop('random_state', None)
        model = GaussianNB(**kwargs)
        return self._fit_and_evaluate(model)

    def RandomForest_model(
        self,
        n_estimators: int = 100,
        random_state: Optional[int] = None,
        **kwargs
    ):
        rs = random_state if random_state is not None else self.random_state
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=rs, n_jobs=self.n_jobs, **kwargs)
        return self._fit_and_evaluate(model)

    def GradientBoosting_model(
        self,
        n_estimators: int = 100,
        random_state: Optional[int] = None,
        **kwargs
    ):
        rs = random_state if random_state is not None else self.random_state
        model = GradientBoostingClassifier(n_estimators=n_estimators, random_state=rs, **kwargs)
        return self._fit_and_evaluate(model)

    def AdaBoost_model(
        self,
        n_estimators: int = 50,
        random_state: Optional[int] = None,
        **kwargs
    ):
        rs = random_state if random_state is not None else self.random_state
        model = AdaBoostClassifier(n_estimators=n_estimators, random_state=rs, **kwargs)
        return self._fit_and_evaluate(model)

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
        return self._run_model_list(model_methods, custom_models)

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
                'ROC AUC': roc_auc
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

    Parameters
    ----------
    X, y : features and a continuous target (a single-column DataFrame is flattened automatically).
    test_size : fraction (float) or number (int) of samples held out for testing.
    scaled_data : standardise features (fit on the training split only). Default ``True``.
    random_state : seed for the split and for every stochastic model.
    n_jobs : parallel workers for estimators that support it.
    verbose : print warnings captured while models train (they are always stored in ``warnings_``).
    """

    def __init__(
        self,
        X: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y: Optional[Union[pd.Series, pd.DataFrame, np.ndarray]] = None,
        test_size: float = 0.3,
        scaled_data: bool = True,
        random_state: Optional[int] = 42,
        n_jobs: Optional[int] = -1,
        verbose: bool = True
    ):
        super().__init__(
            X=X, y=y, test_size=test_size, scaled_data=scaled_data,
            random_state=random_state, n_jobs=n_jobs, verbose=verbose
        )
        try:
            self.y_flat = self.y_flat.astype(float)
        except (ValueError, TypeError):
            raise ValueError("The regression target 'y' must be numeric.") from None
        self._split_and_scale(self.y_flat)

    def evaluate_model(
        self,
        model: Any,
        X_test: Optional[Any] = None,
        y_true: Optional[np.ndarray] = None
    ):
        """Evaluates a fitted regressor. Returns ``(mae, mse, rmse, r2, y_pred)``."""
        if X_test is None:
            X_test = self.X_test_scaled
        if y_true is None:
            y_true = self.y_test

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
        return self._fit_and_evaluate(model)

    def Lasso_model(
        self,
        alpha: float = 0.1,
        random_state: Optional[int] = None,
        **kwargs
    ):
        rs = random_state if random_state is not None else self.random_state
        model = Lasso(alpha=alpha, random_state=rs, **kwargs)
        return self._fit_and_evaluate(model)

    def Ridge_model(
        self,
        alpha: float = 1.0,
        random_state: Optional[int] = None,
        **kwargs
    ):
        rs = random_state if random_state is not None else self.random_state
        model = Ridge(alpha=alpha, random_state=rs, **kwargs)
        return self._fit_and_evaluate(model)

    def SVR_model(
        self,
        kernel: str = 'rbf',
        **kwargs
    ):
        """SVR is sensitive to the scale of the target, so the target is standardised internally
        (predictions are mapped back to the original units)."""
        kwargs.pop('random_state', None)
        model = TransformedTargetRegressor(regressor=SVR(kernel=kernel, **kwargs), transformer=StandardScaler())
        return self._fit_and_evaluate(model, name="SVR")

    def KNN_model(
        self,
        n_neighbors: Optional[int] = None,
        **kwargs
    ):
        kwargs.pop('random_state', None)
        if n_neighbors is None:
            n_neighbors = min(5, max(1, len(self.y_train) - 1))
        model = KNeighborsRegressor(n_neighbors=n_neighbors, n_jobs=self.n_jobs, **kwargs)
        return self._fit_and_evaluate(model)

    def DecisionTree_model(
        self,
        random_state: Optional[int] = None,
        **kwargs
    ):
        rs = random_state if random_state is not None else self.random_state
        model = DecisionTreeRegressor(random_state=rs, **kwargs)
        return self._fit_and_evaluate(model)

    def RandomForest_model(
        self,
        n_estimators: int = 100,
        random_state: Optional[int] = None,
        **kwargs
    ):
        rs = random_state if random_state is not None else self.random_state
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=rs, n_jobs=self.n_jobs, **kwargs)
        return self._fit_and_evaluate(model)

    def GradientBoosting_model(
        self,
        n_estimators: int = 100,
        random_state: Optional[int] = None,
        **kwargs
    ):
        rs = random_state if random_state is not None else self.random_state
        model = GradientBoostingRegressor(n_estimators=n_estimators, random_state=rs, **kwargs)
        return self._fit_and_evaluate(model)

    def AdaBoost_model(
        self,
        n_estimators: int = 50,
        random_state: Optional[int] = None,
        **kwargs
    ):
        rs = random_state if random_state is not None else self.random_state
        model = AdaBoostRegressor(n_estimators=n_estimators, random_state=rs, **kwargs)
        return self._fit_and_evaluate(model)

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
            ('KNN Regressor', lambda: self.KNN_model(random_state=random_state)),
            ('Decision Tree Regressor', lambda: self.DecisionTree_model(random_state=random_state)),
            ('Random Forest Regressor', lambda: self.RandomForest_model(random_state=random_state)),
            ('Gradient Boosting Regressor', lambda: self.GradientBoosting_model(random_state=random_state)),
            ('AdaBoost Regressor', lambda: self.AdaBoost_model(random_state=random_state))
        ]
        return self._run_model_list(model_methods, custom_models)

    def show_tabular_report(
        self,
        models: Optional[List[Tuple]] = None,
        return_df: bool = False,
        sort_by: str = 'R2 Score'
    ) -> Optional[pd.DataFrame]:
        """Displays all regression metrics in a clean tabular format and recommends the best model.

        ``sort_by`` can be 'R2 Score', 'MAE', 'MSE' or 'RMSE' (error metrics sort ascending).
        """
        models = self._resolve_models(models)
        if not models:
            _safe_print("No models were successfully evaluated.")
            self._print_failures()
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
        if sort_by not in df.columns or sort_by == 'Model':
            raise ValueError(f"sort_by must be one of {list(df.columns[1:])}, got '{sort_by}'.")
        df = df.sort_values(by=sort_by, ascending=sort_by in _LOWER_IS_BETTER).reset_index(drop=True)
        self._last_df_report = df

        _safe_print(f"\n{'='*60}\n REGRESSION MODEL COMPARISON TABLE \n{'='*60}")
        _safe_print(df.to_string(index=False))
        _safe_print(f"{'='*60}\n")
        self._print_failures()

        best_model = df.iloc[0]
        _safe_print(f"* BEST MODEL RECOMMENDATION (by {sort_by}): {best_model['Model']}")
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
            fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5 * rows), squeeze=False)
            fig.suptitle('True vs Predicted Values', fontsize=18, fontweight='bold')
            axes_list = axes.flatten()

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
            colors = sns.color_palette('viridis', len(model_names))
            ax.bar(model_names, r2_scores, color=colors)
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


# Backwards compatibility alias for a historical misspelling of the class name
MultiModelRegressior = MultiModelRegressor


# =============================================================================
#  DEMO: run `python -m multimodel_analysis.multimodel` for a synthetic benchmark
# =============================================================================
if __name__ == '__main__':
    from sklearn.datasets import make_classification, make_regression

    _safe_print("[DEMO MODE] Running automated benchmark on synthetic datasets...")

    _safe_print("\n--- 1. MultiModelClassifier (binary classification) ---")
    X_clf, y_clf = make_classification(n_samples=200, n_features=8, random_state=42)
    clf = MultiModelClassifier(X_clf, y_clf)
    clf.show_tabular_report(clf.run_all_models())

    _safe_print("\n--- 2. MultiModelRegressor (regression) ---")
    X_reg, y_reg = make_regression(n_samples=200, n_features=8, noise=0.1, random_state=42)
    reg = MultiModelRegressor(X_reg, y_reg)
    reg.show_tabular_report(reg.run_all_models())
