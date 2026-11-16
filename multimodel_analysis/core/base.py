"""Base class shared by every analyser: validation, scaling, warning capture, reporting helpers."""
import contextlib
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .registry import BuildContext, ModelRegistry
from .utils import _VISIBLE_WARNINGS, _safe_print, _validate_inputs, save_report


class _BaseMultiModel:
    """Base class providing shared initialization, validation, scaling, and utility methods.

    Attributes populated while working:
        models_          list of evaluation tuples from the last ``run_all_models()`` call.
        failed_models_   ``{model name: error message}`` for models that failed to train/evaluate.
        warnings_        ``{model name: [warning strings]}`` captured while models trained.
        estimators_      ``{model name: fitted estimator}`` - use ``get_model(name)`` to fetch one.
        fit_times_       ``{model name: seconds spent in fit()}``.
    """

    registry: ModelRegistry = ModelRegistry()  # overridden by each subclass

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
        self.estimators_ = {}
        self.fit_times_ = {}
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
            start = time.perf_counter()
            model.fit(self.X_train_scaled, self.y_train)
            self.fit_times_[label] = time.perf_counter() - start
            self.estimators_[label] = model
            return self.evaluate_model(model, self.X_test_scaled, self.y_test)

    # ------------------------------------------------------------------ registry-driven model running
    def _run_spec(self, ref: Any, random_state: Optional[int] = None, **params):
        """Builds the registered model ``ref`` (key, display name or ModelSpec), fits and evaluates it."""
        spec = self.registry.resolve(ref)
        ctx = BuildContext(
            random_state=random_state if random_state is not None else self.random_state,
            n_jobs=self.n_jobs,
            n_train=len(self.y_train),
        )
        return self._fit_and_evaluate(spec.build(ctx, **params), name=spec.name)

    def run_model(self, name: str, random_state: Optional[int] = None, **params) -> Tuple:
        """Runs ONE built-in model by key or display name, e.g. ``run_model('random_forest', n_estimators=300)``.

        Returns the same evaluation tuple as the old ``*_model()`` methods.
        """
        return self._run_spec(name, random_state=random_state, **params)

    @classmethod
    def available_models(cls) -> List[str]:
        """Keys of the built-in models, usable in ``run_model`` / ``include`` / ``exclude``."""
        return cls.registry.keys()

    def get_model(self, name: str) -> Any:
        """Returns the FITTED estimator for ``name`` (display name, as shown in the report)."""
        if name in self.estimators_:
            return self.estimators_[name]
        try:
            name = self.registry.resolve(name).name
            if name in self.estimators_:
                return self.estimators_[name]
        except KeyError:
            pass
        raise KeyError(f"No fitted model called '{name}'. Fitted so far: {list(self.estimators_)}")

    def run_all_models(
        self,
        custom_models: Optional[Dict[str, Any]] = None,
        random_state: Optional[int] = None,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
    ) -> List[Tuple]:
        """Runs every built-in model (optionally filtered with ``include`` / ``exclude``) plus ``custom_models``.

        ``custom_models`` is ``{"display name": unfitted_sklearn_estimator}``.
        """
        specs = self.registry.select(include=include, exclude=exclude)
        model_methods = [
            (spec.name, lambda spec=spec: self._run_spec(spec, random_state=random_state)) for spec in specs
        ]
        return self._run_model_list(model_methods, custom_models)

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
        self.estimators_ = {}
        self.fit_times_ = {}
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
