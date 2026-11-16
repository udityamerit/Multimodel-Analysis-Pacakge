"""Base class shared by every analyser: validation, scaling, warning capture, reporting helpers."""
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .mixins import _ReportingMixin
from .persistence import TrainedModel
from .preprocessing import build_preprocessor, to_dataframe, transform_to_frame
from .registry import BuildContext, ModelRegistry
from .utils import _safe_print, _validate_inputs


class _BaseMultiModel(_ReportingMixin):
    """Base class providing shared initialization, validation, scaling, and utility methods.

    Attributes populated while working:
        models_          list of evaluation tuples from the last ``run_all_models()`` call.
        failed_models_   ``{model name: error message}`` for models that failed to train/evaluate.
        warnings_        ``{model name: [warning strings]}`` captured while models trained.
        estimators_      ``{model name: fitted estimator}`` - use ``get_model(name)`` to fetch one.
        fit_times_       ``{model name: seconds spent in fit()}``.
    """

    registry: ModelRegistry = ModelRegistry()  # overridden by each subclass
    task: str = "supervised"  # 'classification' or 'regression' in the subclasses
    default_scoring: str = "r2"

    def __init__(
        self,
        X: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y: Optional[Union[pd.Series, pd.DataFrame, np.ndarray]] = None,
        test_size: float = 0.3,
        scaled_data: bool = True,
        random_state: Optional[int] = 42,
        n_jobs: Optional[int] = -1,
        verbose: bool = True,
        preprocess: bool = False
    ):
        self.y_flat = _validate_inputs(X, y, test_size, preprocess=preprocess)

        self.preprocess = preprocess
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
        self.cv_results_ = None
        self.best_params_ = {}
        self.tuning_results_ = None

        self.scaler = StandardScaler()
        self.transformer_ = None  # fitted scaler / preprocessor (None when scaled_data=False and preprocess=False)
        self.feature_names_ = None  # columns of the matrix the models are trained on

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
        ctx = self._build_context(random_state)
        return self._fit_and_evaluate(spec.build(ctx, **params), name=spec.name)

    def _build_context(self, random_state: Optional[int] = None) -> BuildContext:
        return BuildContext(
            random_state=random_state if random_state is not None else self.random_state,
            n_jobs=self.n_jobs,
            n_train=len(self.y_train),
            class_weight=getattr(self, "class_weight", None),
        )

    def run_model(self, name: str, random_state: Optional[int] = None, **params) -> Tuple:
        """Runs ONE built-in model by key or display name, e.g. ``run_model('random_forest', n_estimators=300)``.

        Returns the same evaluation tuple as the old ``*_model()`` methods.
        """
        return self._run_spec(name, random_state=random_state, **params)

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
        X_all = to_dataframe(self.X) if self.preprocess else self.X
        self.input_columns_ = [str(c) for c in X_all.columns] if isinstance(X_all, pd.DataFrame) else None
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_all, y_split, test_size=self.test_size, random_state=self.random_state, stratify=stratify_y
            )
        except ValueError as exc:
            if stratify_y is None:
                raise
            _safe_print(f"[WARNING] Stratified split not possible ({exc}). Falling back to a random split.")
            X_train, X_test, y_train, y_test = train_test_split(
                X_all, y_split, test_size=self.test_size, random_state=self.random_state
            )
        self.X_train_raw, self.X_test_raw = X_train, X_test  # untransformed splits (used by cross-validation)

        if self.preprocess:
            self.transformer_ = build_preprocessor(X_train, scale=self.scaled_data)
            self.X_train_scaled = transform_to_frame(self.transformer_, X_train, fit=True)
            self.X_test_scaled = transform_to_frame(self.transformer_, X_test)
        elif self.scaled_data:
            self.transformer_ = self.scaler
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

        if isinstance(self.X_train_scaled, pd.DataFrame):
            self.feature_names_ = [str(c) for c in self.X_train_scaled.columns]

        self.y_train = y_train
        self.y_test = y_test

    # ------------------------------------------------------------------ prediction, saving, cross-validation
    def _make_transformer(self) -> Any:
        """Returns a NEW unfitted copy of the preprocessing (or ``None``), for use inside CV pipelines."""
        if self.preprocess:
            return build_preprocessor(self.X_train_raw, scale=self.scaled_data)
        if self.scaled_data:
            return StandardScaler()
        return None

    def _best_model_name(self) -> str:
        if self._last_df_report is not None and len(self._last_df_report):
            return str(self._last_df_report.iloc[0]["Model"])
        if len(self.estimators_) == 1:
            return next(iter(self.estimators_))
        raise ValueError(
            "Say which model to use, e.g. model='Random Forest', or call show_tabular_report() first "
            f"so the best model can be picked. Fitted so far: {list(self.estimators_)}"
        )

    def export_model(self, model: Optional[str] = None) -> TrainedModel:
        """Bundles a fitted model with its preprocessing into a :class:`TrainedModel` (default: the best one)."""
        name = model if model is not None else self._best_model_name()
        estimator = self.get_model(name)
        if model is not None and name not in self.estimators_:
            name = self.registry.resolve(name).name
        from .. import __version__

        return TrainedModel(
            estimator=estimator,
            name=name,
            task=self.task,
            transformer=self.transformer_,
            label_encoder=getattr(self, "label_encoder", None),
            columns=self.input_columns_,
            feature_names=self.feature_names_,
            metadata={"package_version": __version__, "random_state": self.random_state},
        )

    def predict(self, X_new: Any, model: Optional[str] = None) -> np.ndarray:
        """Predicts for NEW raw data with a fitted model (default: the best model of the last report).

        The same scaling / imputation / encoding used in training is applied automatically, and
        classification returns the original labels.
        """
        return self.export_model(model).predict(X_new)

    def save_model(self, path: str, model: Optional[str] = None) -> str:
        """Saves a fitted model + its preprocessing to ``path`` (joblib). Reload with ``load_model(path)``."""
        return self.export_model(model).save(path)

    def tune_models(
        self,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        search: str = "random",
        n_iter: int = 20,
        cv: Union[int, Any] = 5,
        scoring: Optional[str] = None,
        param_grids: Optional[Dict[str, Dict[str, List[Any]]]] = None,
        random_state: Optional[int] = None,
    ) -> List[Tuple]:
        """Tunes hyper-parameters of the built-in models, then refits and evaluates each on the test split.

        search       'random' (default, ``n_iter`` candidates), 'grid' (every combination) or 'halving'
                     (successive halving: fast on big data).
        cv, scoring  cross-validation folds and a scikit-learn scoring string (default: accuracy for
                     classification, r2 for regression). The search runs on the TRAINING split only and
                     preprocessing is re-fitted in every fold.
        param_grids  optional ``{'random_forest': {'max_depth': [3, 5]}}`` to replace a model's default
                     search space (see ``registry`` entries for the defaults).

        Returns the usual list of evaluation tuples, so ``show_tabular_report()`` and the plots work on it.
        ``best_params_`` holds the winning parameters per model and ``tuning_results_`` a summary DataFrame.
        """
        from sklearn.model_selection import GridSearchCV, ParameterGrid, RandomizedSearchCV
        from sklearn.pipeline import Pipeline

        if search not in ("random", "grid", "halving"):
            raise ValueError("search must be 'random', 'grid' or 'halving'.")
        if not isinstance(n_iter, (int, np.integer)) or n_iter < 1:
            raise ValueError("n_iter must be a positive integer.")
        scoring = scoring or self.default_scoring
        overrides = {self.registry.resolve(k).key: v for k, v in (param_grids or {}).items()}
        specs = self.registry.select(include, exclude)
        ctx = self._build_context(random_state)
        seed = ctx.random_state

        self.best_params_ = {}
        rows: List[Dict[str, Any]] = []

        def make_runner(spec):
            def run():
                space = overrides.get(spec.key, spec.search_space)
                if not space:
                    raise ValueError("no search space is defined for this model; pass param_grids={...}.")
                prefix = "model__" + spec.param_prefix
                grid = {prefix + k: list(v) for k, v in space.items()}
                transformer = self._make_transformer()
                pipe = Pipeline([("prep", transformer if transformer is not None else "passthrough"),
                                 ("model", spec.build(ctx))])
                n_candidates = len(ParameterGrid(grid))
                if search == "grid":
                    searcher = GridSearchCV(pipe, grid, cv=cv, scoring=scoring, refit=False)
                elif search == "halving":
                    from sklearn.experimental import enable_halving_search_cv  # noqa: F401
                    from sklearn.model_selection import HalvingRandomSearchCV
                    searcher = HalvingRandomSearchCV(pipe, grid, n_candidates=min(n_iter, n_candidates), cv=cv,
                                                     scoring=scoring, refit=False, random_state=seed)
                else:
                    searcher = RandomizedSearchCV(pipe, grid, n_iter=min(n_iter, n_candidates), cv=cv,
                                                  scoring=scoring, refit=False, random_state=seed)
                start = time.perf_counter()
                with self._track_warnings(f"{spec.name} (tuning)"):
                    searcher.fit(self.X_train_raw, self.y_train)
                elapsed = time.perf_counter() - start
                cut = len(prefix)
                best = {k[cut:]: v for k, v in searcher.best_params_.items()}
                self.best_params_[spec.name] = best
                rows.append({"Model": spec.name, f"Best CV {scoring}": float(searcher.best_score_),
                             "Best Params": best, "Candidates": len(searcher.cv_results_["params"]),
                             "Search Time (s)": elapsed})
                return self._fit_and_evaluate(spec.build(ctx, **best), name=spec.name)
            return run

        models = self._run_model_list([(spec.name, make_runner(spec)) for spec in specs], None)
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values(by=f"Best CV {scoring}", ascending=False).reset_index(drop=True)
        self.tuning_results_ = df
        if self.verbose and not df.empty:
            _safe_print(f"\n{'='*60}\n HYPER-PARAMETER TUNING ({search} search, {cv}-fold CV on the training "
                        f"split) \n{'='*60}")
            _safe_print(df.drop(columns=["Best Params"]).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
            for _, r in df.iterrows():
                _safe_print(f"   {r['Model']}: {r['Best Params']}")
            _safe_print(f"{'='*60}\n")
        return models

    def cross_validate_models(
        self,
        cv: Union[int, Any] = 5,
        scoring: Optional[Union[str, List[str]]] = None,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        custom_models: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """Cross-validates every model on the TRAINING split (the test split stays untouched).

        Preprocessing is re-fitted inside every fold, so there is no data leakage.
        ``scoring`` is any scikit-learn scoring string or a list of them (default: accuracy for
        classification, r2 for regression). ``neg_*`` scorers are negative: higher is still better.
        Returns a DataFrame sorted by the first scorer and stores it in ``cv_results_``.
        """
        from sklearn.base import clone
        from sklearn.model_selection import cross_validate
        from sklearn.pipeline import make_pipeline

        scorers = [scoring] if isinstance(scoring, str) else list(scoring or [self.default_scoring])
        ctx = self._build_context()
        candidates = [(spec.name, lambda spec=spec: spec.build(ctx)) for spec in self.registry.select(include, exclude)]
        for cname, inst in (custom_models or {}).items():
            candidates.append((cname, lambda inst=inst: clone(inst)))

        rows, failed = [], {}
        for name, build in candidates:
            try:
                estimator = build()
                transformer = self._make_transformer()
                pipe = estimator if transformer is None else make_pipeline(transformer, estimator)
                with self._track_warnings(f"{name} (cv)"):
                    res = cross_validate(pipe, self.X_train_raw, self.y_train, cv=cv, scoring=scorers)
                row = {"Model": name}
                for s in scorers:
                    row[f"{s} mean"] = float(np.mean(res[f"test_{s}"]))
                    row[f"{s} std"] = float(np.std(res[f"test_{s}"]))
                row["Fit Time (s)"] = float(np.mean(res["fit_time"]))
                rows.append(row)
            except Exception as e:
                failed[name] = f"{type(e).__name__}: {e}"
                _safe_print(f"[WARNING] Cross-validation failed for '{name}': {e}")

        self.cv_failed_ = failed
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values(by=f"{scorers[0]} mean", ascending=False).reset_index(drop=True)
        self.cv_results_ = df
        if self.verbose and not df.empty:
            _safe_print(f"\n{'='*60}\n CROSS-VALIDATION ({cv if isinstance(cv, int) else 'custom'}-fold, "
                        f"on the training split) \n{'='*60}")
            _safe_print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
            _safe_print(f"{'='*60}\n")
        return df

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
