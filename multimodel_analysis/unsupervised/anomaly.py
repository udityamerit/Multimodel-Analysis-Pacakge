"""Anomaly (outlier) detection: run many detectors on the same data and compare them.

Every detector gives each row an *anomaly score* (higher = stranger). The analyser flags the
``contamination`` share of rows with the highest score, so all models flag the same number of rows and can be
compared fairly. Rows are labelled ``1`` = anomaly, ``0`` = normal.

Without labels the report shows how much each model *agrees* with the others. If you know which rows really are
anomalies, pass them as ``y`` (1 = anomaly) to get Precision, Recall, F1, ROC AUC and Average Precision.
"""
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.svm import OneClassSVM

from ..core.persistence import TrainedModel
from ..core.plotting import comparison_figure
from ..core.preprocessing import to_dataframe
from ..core.registry import BuildContext, ModelRegistry, ModelSpec
from ..core.utils import _apply_plot_style, _safe_print
from ._base import _UnsupervisedBase

_NORMAL = "#8fb3c9"
_ANOMALY = "#d62728"
_LOWER = ("Fit Time (s)",)
_SUPERVISED = ("Precision", "Recall", "F1 Score", "ROC AUC", "Avg Precision")

# O(n^2) models are skipped on big data unless ``allow_large=True``.
_MAX_SAMPLES = {"one_class_svm": 20_000, "elliptic_envelope": 100_000}


# --------------------------------------------------------------------------- two simple detectors
class RobustZScore(BaseEstimator):
    """Anomaly score = the largest robust z-score of a row: ``|x - median| / (1.4826 * MAD)`` per feature.

    Simple, fast and easy to explain ("this value is 6 robust standard deviations from the typical one"), so it
    makes a good baseline. It looks at every feature on its own and cannot see odd *combinations* of values.
    """

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.median_ = np.median(X, axis=0)
        mad = np.median(np.abs(X - self.median_), axis=0) * 1.4826
        self.scale_ = np.where(mad > 1e-12, mad, np.where(X.std(axis=0) > 1e-12, X.std(axis=0), 1.0))
        return self

    def score_samples(self, X):
        z = np.abs((np.asarray(X, dtype=float) - self.median_) / self.scale_)
        return -z.max(axis=1)  # scikit-learn convention: LOWER = stranger

    def predict(self, X):
        return np.where(self.score_samples(X) < -3.5, -1, 1)  # the usual "modified z-score > 3.5" rule


class KNNDistance(BaseEstimator):
    """Anomaly score = distance to the k-th nearest neighbour: rows far from everyone else are strange."""

    def __init__(self, n_neighbors: int = 10):
        self.n_neighbors = n_neighbors

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.k_ = max(1, min(self.n_neighbors, len(X) - 1))
        self.nn_ = NearestNeighbors(n_neighbors=self.k_ + 1).fit(X)
        self.train_scores_ = self.nn_.kneighbors(X)[0][:, -1]  # first column is the row itself
        return self

    def score_samples(self, X):
        dist = self.nn_.kneighbors(np.asarray(X, dtype=float), n_neighbors=self.k_)[0]
        return -dist[:, -1]


# --------------------------------------------------------------------------- model factories
def _isolation_forest(ctx: BuildContext, **p):
    return IsolationForest(random_state=ctx.random_state, n_jobs=ctx.n_jobs, **p)


def _lof(ctx: BuildContext, n_neighbors: int = 20, **p):
    return LocalOutlierFactor(n_neighbors=max(1, min(n_neighbors, ctx.n_train - 1)), novelty=True,
                              n_jobs=ctx.n_jobs, **p)


def _one_class_svm(ctx: BuildContext, **p):
    return OneClassSVM(**p)


def _sgd_one_class_svm(ctx: BuildContext, **p):
    from sklearn.linear_model import SGDOneClassSVM
    return SGDOneClassSVM(random_state=ctx.random_state, **p)


def _elliptic(ctx: BuildContext, **p):
    return EllipticEnvelope(random_state=ctx.random_state, **p)


def _zscore(ctx: BuildContext, **p):
    return RobustZScore()


def _knn_distance(ctx: BuildContext, **p):
    return KNNDistance(**p)


ANOMALY_REGISTRY = ModelRegistry([
    # ---- standard set -----------------------------------------------------------------------------
    ModelSpec("isolation_forest", "Isolation Forest", _isolation_forest, {"n_estimators": 200}),
    ModelSpec("lof", "Local Outlier Factor", _lof, {"n_neighbors": 20}),
    ModelSpec("one_class_svm", "One-Class SVM", _one_class_svm, {"kernel": "rbf", "gamma": "scale"}),
    ModelSpec("elliptic_envelope", "Elliptic Envelope", _elliptic, {}),
    ModelSpec("knn_distance", "KNN Distance", _knn_distance, {"n_neighbors": 10}),
    ModelSpec("robust_zscore", "Robust Z-Score", _zscore, {}),
    # ---- extra ------------------------------------------------------------------------------------
    ModelSpec("sgd_one_class_svm", "SGD One-Class SVM", _sgd_one_class_svm, {}, default=False),
])

# parameters that are set from the ``contamination`` share
_FROM_CONTAMINATION = {"one_class_svm": "nu", "sgd_one_class_svm": "nu", "elliptic_envelope": "contamination"}


class _Detector:
    """A fitted detector + the score threshold learned on the training data.

    ``score_samples`` returns the anomaly score (higher = stranger), ``predict`` returns 1 = anomaly, 0 = normal.
    This is the object stored inside an exported / saved ``TrainedModel``.
    """

    def __init__(self, estimator: Any, threshold: float):
        self.estimator = estimator
        self.threshold = threshold

    def score_samples(self, X) -> np.ndarray:
        return _raw_scores(self.estimator, X)

    def predict(self, X) -> np.ndarray:
        return (self.score_samples(X) > self.threshold).astype(int)


def _raw_scores(est: Any, X) -> np.ndarray:
    """Anomaly score (higher = stranger) of an unlabelled-data estimator, for rows it may not have seen."""
    if hasattr(est, "score_samples"):
        return -np.asarray(est.score_samples(X), dtype=float)
    if hasattr(est, "decision_function"):
        return -np.asarray(est.decision_function(X), dtype=float)
    raise TypeError(f"{type(est).__name__} has neither score_samples nor decision_function, "
                    "so it cannot give anomaly scores.")


def _train_scores(est: Any, values: Any) -> np.ndarray:
    """Scores of the rows the model was fitted on (some models must exclude the row itself)."""
    if hasattr(est, "negative_outlier_factor_"):  # Local Outlier Factor
        return -np.asarray(est.negative_outlier_factor_, dtype=float)
    if hasattr(est, "train_scores_"):  # KNN Distance
        return np.asarray(est.train_scores_, dtype=float)
    return _raw_scores(est, values)


@dataclass
class AnomalyResult:
    """What one detector produced. ``labels``: 1 = anomaly, 0 = normal. ``scores``: higher = stranger."""
    name: str
    scores: np.ndarray = field(repr=False)
    labels: np.ndarray = field(repr=False)
    threshold: float
    n_anomalies: int
    metrics: Dict[str, float] = field(default_factory=dict)
    fit_time: float = float("nan")

    @property
    def rate(self) -> float:
        return float(self.n_anomalies / len(self.labels) * 100)


class MultiModelAnomalyDetector(_UnsupervisedBase):
    """Benchmarks anomaly detectors on one dataset.

    Parameters
    ----------
    X              features (DataFrame or array). Numeric and finite unless ``preprocess=True``.
    y              optional TRUE anomaly labels (1 = anomaly, 0 = normal). Never used for fitting - only to
                   score the detectors (Precision, Recall, F1, ROC AUC, Average Precision).
    contamination  the share of rows to flag as anomalies, between 0 and 0.5. Default: the share of 1s in ``y``
                   when ``y`` is given, otherwise 0.05 (5%).
    scaled_data    standardise the features first (recommended).
    preprocess     impute missing values, one-hot encode text columns and scale.

    Typical use::

        det = MultiModelAnomalyDetector(X, contamination=0.02)
        models = det.run_all_models()
        det.show_tabular_report(models)
        det.top_anomalies(10)
        det.plot_anomalies(models)
    """

    registry: ModelRegistry = ANOMALY_REGISTRY
    task = "anomaly detection"

    def __init__(
        self,
        X: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        contamination: Optional[float] = None,
        scaled_data: bool = True,
        random_state: Optional[int] = 42,
        n_jobs: Optional[int] = -1,
        verbose: bool = True,
        preprocess: bool = False,
    ):
        n_samples = self._check_features(X, preprocess, 20, "detect anomalies")
        self.y_true_ = None
        if y is not None:
            y_arr = np.asarray(y).ravel()
            if len(y_arr) != n_samples:
                raise ValueError(f"X and y have different lengths: X has {n_samples} rows but y has {len(y_arr)}.")
            if pd.isnull(y_arr).any():
                raise ValueError("y contains missing values. Drop or fill them first.")
            if not set(np.unique(y_arr.astype(float))) <= {0.0, 1.0}:
                raise ValueError("y must contain only 0 (normal) and 1 (anomaly).")
            y_arr = y_arr.astype(int)
            if y_arr.min() == y_arr.max():
                raise ValueError("y must contain both normal (0) and anomalous (1) rows to score the detectors.")
            self.y_true_ = y_arr
        if contamination is None:
            contamination = float(self.y_true_.mean()) if self.y_true_ is not None else 0.05
        if not isinstance(contamination, (int, float, np.number)) or not 0 < contamination < 0.5:
            raise ValueError("contamination must be a number between 0 and 0.5 (the share of rows to flag).")

        self.X = X
        self.n_samples_ = n_samples
        self.contamination = float(contamination)
        self.scaled_data = scaled_data
        self.preprocess = preprocess
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose

        self.models_: List[AnomalyResult] = []
        self.failed_models_: Dict[str, str] = {}
        self.warnings_: Dict[str, List[str]] = {}
        self.estimators_: Dict[str, Any] = {}
        self.fit_times_: Dict[str, float] = {}
        self._results: Dict[str, AnomalyResult] = {}
        self._detectors: Dict[str, _Detector] = {}
        self._last_df_report: Optional[pd.DataFrame] = None

        self._prepare_features()

    # ------------------------------------------------------------------ fitting
    def _params_for(self, spec: ModelSpec, user: Dict[str, Any]) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        if spec.key in _FROM_CONTAMINATION:
            params[_FROM_CONTAMINATION[spec.key]] = self.contamination
        params.update(user)
        return params

    def _metrics(self, scores: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        if self.y_true_ is None:
            return {}
        t = self.y_true_
        return {"Precision": float(precision_score(t, labels, zero_division=0)),
                "Recall": float(recall_score(t, labels, zero_division=0)),
                "F1 Score": float(f1_score(t, labels, zero_division=0)),
                "ROC AUC": float(roc_auc_score(t, scores)),
                "Avg Precision": float(average_precision_score(t, scores))}

    def _fit_and_evaluate(self, model: Any, name: str) -> AnomalyResult:
        with self._track_warnings(name):
            start = time.perf_counter()
            model.fit(self.X_scaled)
            seconds = time.perf_counter() - start
            scores = _train_scores(model, self.X_scaled)
        if not np.all(np.isfinite(scores)):
            raise ValueError("the model produced non-finite anomaly scores.")
        threshold = float(np.quantile(scores, 1 - self.contamination))
        labels = (scores > threshold).astype(int)
        if labels.sum() == 0:  # a flat score (e.g. many ties): flag at least the top row
            labels[int(np.argmax(scores))] = 1
        result = AnomalyResult(name, scores, labels, threshold, int(labels.sum()), self._metrics(scores, labels), seconds)
        self.estimators_[name] = model
        self._detectors[name] = _Detector(model, threshold)
        self.fit_times_[name] = seconds
        self._results[name] = result
        return result

    def _run_spec(self, ref: Any, random_state: Optional[int] = None, **params) -> AnomalyResult:
        spec = self.registry.resolve(ref)
        model = spec.build(self._build_context(random_state), **self._params_for(spec, params))
        return self._fit_and_evaluate(model, spec.name)

    def run_model(self, name: str, random_state: Optional[int] = None, **params) -> AnomalyResult:
        """Runs ONE built-in detector by key or display name, e.g. ``run_model('lof', n_neighbors=35)``."""
        return self._run_spec(name, random_state=random_state, **params)

    def run_all_models(
        self,
        custom_models: Optional[Dict[str, Any]] = None,
        random_state: Optional[int] = None,
        include: Optional[Any] = None,
        exclude: Optional[List[str]] = None,
        allow_large: bool = False,
    ) -> List[AnomalyResult]:
        """Runs every standard detector (``include='all'`` for all installed ones) plus ``custom_models``.

        ``custom_models`` is ``{"display name": unfitted_estimator}``; the estimator needs ``fit`` and
        ``score_samples`` (or ``decision_function``), like scikit-learn's outlier detectors.
        """
        self.failed_models_, self.warnings_, self.estimators_, self.fit_times_ = {}, {}, {}, {}
        self._results, self._detectors = {}, {}
        self._last_df_report = None
        runs: List[Tuple[str, Any]] = []
        for spec in self.registry.select(include=include, exclude=exclude):
            limit = _MAX_SAMPLES.get(spec.key)
            if limit is not None and self.n_samples_ > limit and not allow_large:
                self.failed_models_[spec.name] = (f"skipped: {self.n_samples_} rows is above {limit}; this model is "
                                                  "slow on big data. Use allow_large=True to run it anyway.")
                continue
            runs.append((spec.name, lambda spec=spec: self._run_spec(spec, random_state=random_state)))
        for cname, inst in (custom_models or {}).items():
            runs.append((cname, lambda cname=cname, inst=inst: self._fit_and_evaluate(clone(inst), cname)))

        models = []
        for name, run in runs:
            try:
                models.append(run())
            except Exception as e:  # keep going: one broken model must not stop the benchmark
                self.failed_models_[name] = f"{type(e).__name__}: {e}"
                _safe_print(f"[WARNING] Model '{name}' failed: {e}")
        self.models_ = models
        return models

    # ------------------------------------------------------------------ reporting
    def _resolve_models(self, models: Optional[List[AnomalyResult]] = None) -> List[AnomalyResult]:
        if models:
            return models
        if self.models_:
            return self.models_
        _safe_print("[INFO] No models provided. Automatically running all models...")
        return self.run_all_models()

    @staticmethod
    def _agreement(models: List[AnomalyResult]) -> Dict[str, float]:
        """Spearman correlation between each model's scores and the average rank given by the OTHER models."""
        if len(models) < 2:
            return {m.name: float("nan") for m in models}
        ranks = np.column_stack([pd.Series(m.scores).rank().to_numpy() for m in models])
        out = {}
        for i, m in enumerate(models):
            others = np.delete(ranks, i, axis=1).mean(axis=1)
            out[m.name] = float(np.corrcoef(ranks[:, i], others)[0, 1]) if ranks[:, i].std() > 0 and others.std() > 0 else float("nan")
        return out

    def _default_sort(self) -> str:
        return "F1 Score" if self.y_true_ is not None else "Agreement"

    def _report_df(self, models: List[AnomalyResult], sort_by: Optional[str] = None) -> pd.DataFrame:
        sort_by = sort_by or self._default_sort()
        agree = self._agreement(models)
        rows = []
        for m in models:
            row = {"Model": m.name, "Anomalies": m.n_anomalies, "Anomaly Rate (%)": m.rate, "Agreement": agree[m.name]}
            row.update(m.metrics)
            row["Fit Time (s)"] = m.fit_time
            rows.append(row)
        df = pd.DataFrame(rows)
        valid = [c for c in df.columns if c not in ("Model", "Anomalies", "Anomaly Rate (%)")]
        if sort_by not in valid:
            raise ValueError(f"sort_by must be one of {valid}, got '{sort_by}'.")
        return df.sort_values(by=sort_by, ascending=sort_by in _LOWER, na_position="last",
                              kind="stable").reset_index(drop=True)

    def show_tabular_report(
        self,
        models: Optional[List[AnomalyResult]] = None,
        return_df: bool = False,
        sort_by: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        """Prints every detector's results and the recommended one.

        Default order: 'F1 Score' when you passed ``y``, otherwise 'Agreement' (how closely a model's ranking of the
        rows matches the other models; 1 = identical). Agreement is a consensus hint, NOT accuracy.
        """
        models = self._resolve_models(models)
        if not models:
            _safe_print("No models were successfully evaluated.")
            self._print_failures()
            return None
        sort_by = sort_by or self._default_sort()
        df = self._report_df(models, sort_by)
        self._last_df_report = df

        _safe_print(f"\n{'='*70}\n ANOMALY DETECTOR COMPARISON \n{'='*70}")
        _safe_print(df.to_string(index=False, na_rep="N/A", float_format=lambda v: f"{v:.4f}"))
        _safe_print(f"{'='*70}\n")
        self._print_failures()
        best = df.iloc[0]
        _safe_print(f"* BEST MODEL (by {sort_by}): {best['Model']}")
        _safe_print(f"  Every model flags the top {self.contamination * 100:.1f}% of rows by anomaly score, "
                    "so the counts are about equal.")
        if self.y_true_ is None:
            _safe_print("  No true labels were given: 'Agreement' only shows which model sees the data like the others.")
            _safe_print("  Look at top_anomalies() and the plots, and check a few flagged rows yourself.")
        else:
            _safe_print("  Precision = flagged rows that are real anomalies; Recall = real anomalies found.")
            if abs(self.contamination - float(self.y_true_.mean())) < 1e-9:
                _safe_print("  Each model flags exactly as many rows as there are real anomalies, so Precision, Recall and")
                _safe_print("  F1 are equal. ROC AUC and Avg Precision do not depend on the cut-off.")
        _safe_print(f"{'='*70}\n")
        return df if return_df else None

    def _best_model_name(self) -> str:
        if self._last_df_report is not None and len(self._last_df_report):
            return str(self._last_df_report.iloc[0]["Model"])
        if self.models_:
            return str(self._report_df(self.models_).iloc[0]["Model"])
        raise ValueError("No model has been run yet. Call run_all_models() or run_model(...) first.")

    def _name(self, model: Optional[str]) -> str:
        if model is None:
            return self._best_model_name()
        if model in self._results:
            return model
        try:
            name = self.registry.resolve(model).name
            if name in self._results:
                return name
        except KeyError:
            pass
        raise KeyError(f"No fitted model called '{model}'. Fitted so far: {list(self._results)}")

    # ------------------------------------------------------------------ looking at the anomalies
    def get_scores(self, model: Optional[str] = None) -> np.ndarray:
        """Anomaly score of every row for ``model`` (default: the best one). Higher = stranger."""
        return self._results[self._name(model)].scores

    def get_labels(self, model: Optional[str] = None) -> np.ndarray:
        """1 = anomaly, 0 = normal, for every row (default model: the best one)."""
        return self._results[self._name(model)].labels

    def assign_anomalies(self, model: Optional[str] = None) -> pd.DataFrame:
        """Your data with ``anomaly`` (0/1) and ``anomaly_score`` columns added."""
        df = to_dataframe(self.X, self.input_columns_).copy()
        res = self._results[self._name(model)]
        df["anomaly"] = res.labels
        df["anomaly_score"] = res.scores
        return df

    def top_anomalies(self, n: int = 10, model: Optional[str] = None) -> pd.DataFrame:
        """The ``n`` strangest rows (original values + score), strangest first."""
        df = self.assign_anomalies(model)
        return df.sort_values("anomaly_score", ascending=False).head(max(1, int(n)))

    def consensus(self, min_votes: Optional[int] = None, models: Optional[List[AnomalyResult]] = None) -> pd.DataFrame:
        """Your data with a ``votes`` column (how many models flagged the row) and a ``consensus`` flag.

        A row is a consensus anomaly when at least ``min_votes`` models flagged it (default: more than half).
        Rows flagged by many different methods are the ones to look at first.
        """
        models = self._resolve_models(models)
        if not models:
            raise ValueError("No fitted models. Call run_all_models() first.")
        votes = np.sum([m.labels for m in models], axis=0)
        need = int(min_votes) if min_votes is not None else len(models) // 2 + 1
        if not 1 <= need <= len(models):
            raise ValueError(f"min_votes must be between 1 and {len(models)}.")
        df = to_dataframe(self.X, self.input_columns_).copy()
        df["votes"] = votes
        df["consensus"] = (votes >= need).astype(int)
        return df

    # ------------------------------------------------------------------ predicting new data
    def export_model(self, model: Optional[str] = None) -> TrainedModel:
        """Bundles a fitted detector, its score threshold and its preprocessing so it can judge NEW rows."""
        name = self._name(model)
        det = self._detectors[name]
        est = det.estimator
        if not hasattr(est, "score_samples") and not hasattr(est, "decision_function"):
            raise TypeError(f"'{name}' cannot score new data.")
        from .. import __version__
        return TrainedModel(estimator=det, name=name, task=self.task, transformer=self.transformer_,
                            columns=self.input_columns_, feature_names=self.feature_names_,
                            metadata={"package_version": __version__, "contamination": self.contamination,
                                      "threshold": det.threshold})

    def predict(self, X_new: Any, model: Optional[str] = None) -> np.ndarray:
        """1 = anomaly, 0 = normal for NEW rows, using the threshold learned on the training data."""
        return self.export_model(model).predict(X_new)

    def score_new(self, X_new: Any, model: Optional[str] = None) -> np.ndarray:
        """Anomaly score of NEW rows (higher = stranger); compare with ``threshold`` of the model."""
        tm = self.export_model(model)
        return tm.estimator.score_samples(tm._prepare(X_new))

    def save_model(self, path: str, model: Optional[str] = None) -> str:
        """Saves a fitted detector + its preprocessing (joblib). Reload with ``load_model(path)``."""
        return self.export_model(model).save(path)

    # ------------------------------------------------------------------ plots
    def plot_comparison(
        self,
        models: Optional[List[AnomalyResult]] = None,
        save_path: Optional[str] = None,
        show_plot: bool = True,
        metric: Optional[str] = None,
    ):
        """Ranks all detectors on one metric (default 'F1 Score' with labels, else 'Agreement')."""
        models = self._resolve_models(models)
        if not models:
            return
        metric = metric or self._default_sort()
        df = self._report_df(models, metric)
        note = ("Agreement is a consensus hint, not accuracy." if metric == "Agreement"
                else f"Every model flags the top {self.contamination * 100:.1f}% of rows.")
        with _apply_plot_style():
            fig = comparison_figure(list(df["Model"]), list(df[metric]), metric=metric,
                                    title=f"Detector comparison: {metric}", lower_is_better=metric in _LOWER,
                                    footnote=note)
            self._save_and_show(fig, save_path=save_path, show_plot=show_plot)

    def plot_anomalies(
        self,
        models: Optional[List[AnomalyResult]] = None,
        top: int = 6,
        save_path: Optional[str] = None,
        show_plot: bool = True,
    ):
        """Scatter plots of the best ``top`` detectors; flagged rows are red crosses. With more than 2 features
        the data is shown on its first two principal components."""
        import math

        models = self._resolve_models(models)
        if not models:
            return
        order = {n: i for i, n in enumerate(self._report_df(models)["Model"])}
        models = sorted(models, key=lambda m: order[m.name])[:max(1, top)]
        xy, axis_names = self._project_2d()
        ncols = min(3, len(models))
        nrows = math.ceil(len(models) / ncols)
        with _apply_plot_style():
            fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 4.0 * nrows), squeeze=False)
            for ax, m in zip(axes.ravel(), models):
                flag = m.labels == 1
                ax.scatter(xy[~flag, 0], xy[~flag, 1], s=16, color=_NORMAL, alpha=0.8, edgecolor="white", linewidth=0.3)
                ax.scatter(xy[flag, 0], xy[flag, 1], s=30, color=_ANOMALY, marker="x", linewidth=1.3)
                ax.set_title(f"{m.name}\n{m.n_anomalies} flagged ({m.rate:.1f}%)", fontsize=11, loc="left",
                             fontweight="bold")
                ax.set_xticks([])
                ax.set_yticks([])
                for side in ("top", "right"):
                    ax.spines[side].set_visible(False)
            for ax in axes.ravel()[len(models):]:
                ax.axis("off")
            for ax in axes[-1]:
                ax.set_xlabel(axis_names[0], fontsize=9)
            for ax in axes[:, 0]:
                ax.set_ylabel(axis_names[1], fontsize=9)
            fig.tight_layout()
            self._save_and_show(fig, save_path=save_path, show_plot=show_plot)

    def plot_scores(self, model: Optional[str] = None, save_path: Optional[str] = None, show_plot: bool = True):
        """Histogram of the anomaly scores with the cut-off line: rows to the right of it are flagged."""
        name = self._name(model)
        res = self._results[name]
        with _apply_plot_style():
            fig, ax = plt.subplots(figsize=(8, 4.5))
            bins = np.histogram_bin_edges(res.scores, bins=50)
            ax.hist(res.scores[res.labels == 0], bins=bins, color=_NORMAL, label="normal")
            ax.hist(res.scores[res.labels == 1], bins=bins, color=_ANOMALY, label="flagged as anomaly")
            ax.axvline(res.threshold, color="#2b2b2b", ls="--", lw=1.2)
            ax.set_title(f"Anomaly scores - {name}", loc="left", fontweight="bold")
            ax.set_xlabel("Anomaly score (higher = stranger)")
            ax.set_ylabel("Number of rows")
            ax.legend(frameon=False)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            fig.tight_layout()
            self._save_and_show(fig, save_path=save_path, show_plot=show_plot)

    def _report_subtitle(self) -> str:
        known = "with true labels" if self.y_true_ is not None else "no labels"
        return (f"Anomaly detection - {self.n_samples_} rows, {self._values.shape[1]} features, "
                f"{len(self._results)} models, contamination {self.contamination:.1%} ({known})")

    def _report_plots(self):
        return [("Model comparison", lambda p: self.plot_comparison(save_path=p, show_plot=False)),
                ("Where each model sees anomalies", lambda p: self.plot_anomalies(save_path=p, show_plot=False)),
                ("Score distribution", lambda p: self.plot_scores(save_path=p, show_plot=False))]

    def get_summary(
        self,
        models: Optional[List[AnomalyResult]] = None,
        save_prefix: Optional[str] = None,
        show_plot: bool = True,
    ):
        """Runs the full reporting pipeline: table, ranking chart and scatter plots."""
        models = self._resolve_models(models)
        df = self.show_tabular_report(models, return_df=True)
        if save_prefix:
            if df is not None:
                self.save_report(df, f"{save_prefix}_report.csv")
            self.plot_comparison(models, save_path=f"{save_prefix}_comparison.png", show_plot=show_plot)
            self.plot_anomalies(models, save_path=f"{save_prefix}_anomalies.png", show_plot=show_plot)
        else:
            self.plot_comparison(models, show_plot=show_plot)
            self.plot_anomalies(models, show_plot=show_plot)
