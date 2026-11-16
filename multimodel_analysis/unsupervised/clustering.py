"""Clustering: run many clustering algorithms on the same data and compare them.

Unlike the supervised analysers there is no target and no train/test split: every model is fitted on
the whole dataset and judged with *internal* scores (Silhouette, Calinski-Harabasz, Davies-Bouldin).
If you happen to know the true groups you can pass them as ``y`` and the report also shows *external*
scores (Adjusted Rand Index, ...), which is handy for learning and benchmarking.
"""
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import (
    DBSCAN,
    OPTICS,
    AffinityPropagation,
    AgglomerativeClustering,
    Birch,
    KMeans,
    MeanShift,
    MiniBatchKMeans,
    SpectralClustering,
    estimate_bandwidth,
)
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    completeness_score,
    davies_bouldin_score,
    homogeneity_score,
    normalized_mutual_info_score,
    silhouette_samples,
    silhouette_score,
    v_measure_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from ..core.mixins import _ReportingMixin
from ..core.persistence import TrainedModel
from ..core.plotting import comparison_figure
from ..core.preprocessing import build_preprocessor, to_dataframe, transform_to_frame
from ..core.registry import BuildContext, ModelRegistry, ModelSpec
from ..core.utils import _apply_plot_style, _safe_print, _validate_features

_K = list(range(2, 11))  # candidate numbers of clusters used by tune_models / find_optimal_k
_PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#17becf",
            "#bcbd22", "#7f7f7f"]
_NOISE = "#b0b0b0"

# Metrics where a LOWER value is better.
_LOWER = ("Davies-Bouldin", "Fit Time (s)")
_INTERNAL = ("Silhouette", "Calinski-Harabasz", "Davies-Bouldin")
_EXTERNAL = ("ARI", "NMI", "V-measure", "Homogeneity", "Completeness")

# Models that need O(n^2) memory/time: skipped on bigger data unless ``allow_large=True``.
_MAX_SAMPLES = {"spectral": 10_000, "agglomerative": 25_000, "agglomerative_average": 25_000,
                "agglomerative_complete": 25_000, "affinity_propagation": 5_000, "mean_shift": 20_000,
                "optics": 50_000}


# --------------------------------------------------------------------------- model factories
def _kmeans(ctx: BuildContext, **p):
    return KMeans(random_state=ctx.random_state, **p)


def _minibatch(ctx: BuildContext, **p):
    return MiniBatchKMeans(random_state=ctx.random_state, **p)


def _bisecting(ctx: BuildContext, **p):
    from sklearn.cluster import BisectingKMeans
    return BisectingKMeans(random_state=ctx.random_state, **p)


def _agglomerative(ctx: BuildContext, **p):
    return AgglomerativeClustering(**p)


def _gmm(ctx: BuildContext, n_clusters: int = 3, **p):
    return GaussianMixture(n_components=n_clusters, random_state=ctx.random_state, **p)


def _birch(ctx: BuildContext, **p):
    return Birch(**p)


def _spectral(ctx: BuildContext, **p):
    return SpectralClustering(random_state=ctx.random_state, n_jobs=ctx.n_jobs, **p)


def _dbscan(ctx: BuildContext, **p):
    return DBSCAN(n_jobs=ctx.n_jobs, **p)


def _hdbscan(ctx: BuildContext, **p):
    from sklearn.cluster import HDBSCAN
    return HDBSCAN(n_jobs=ctx.n_jobs, **p)


def _optics(ctx: BuildContext, **p):
    return OPTICS(n_jobs=ctx.n_jobs, **p)


def _mean_shift(ctx: BuildContext, **p):
    return MeanShift(n_jobs=ctx.n_jobs, **p)


def _affinity(ctx: BuildContext, **p):
    return AffinityPropagation(random_state=ctx.random_state, **p)


CLUSTERER_REGISTRY = ModelRegistry([
    # ---- standard set (run by run_all_models()) --------------------------------------------------
    ModelSpec("kmeans", "KMeans", _kmeans, {"n_clusters": 3, "n_init": 10}, {"n_clusters": _K}),
    ModelSpec("minibatch_kmeans", "MiniBatch KMeans", _minibatch,
              {"n_clusters": 3, "n_init": 3, "batch_size": 256}, {"n_clusters": _K}),
    ModelSpec("agglomerative", "Agglomerative (Ward)", _agglomerative, {"n_clusters": 3, "linkage": "ward"},
              {"n_clusters": _K, "linkage": ["ward", "average", "complete", "single"]}),
    ModelSpec("gaussian_mixture", "Gaussian Mixture", _gmm, {"n_clusters": 3, "covariance_type": "full", "n_init": 3},
              {"n_clusters": _K, "covariance_type": ["full", "tied", "diag", "spherical"]}),
    ModelSpec("birch", "Birch", _birch, {"n_clusters": 3},
              {"n_clusters": _K, "threshold": [0.3, 0.5, 0.8, 1.2]}),
    ModelSpec("spectral", "Spectral Clustering", _spectral,
              {"n_clusters": 3, "affinity": "nearest_neighbors", "n_neighbors": 10},
              {"n_clusters": _K, "n_neighbors": [5, 10, 20]}),
    ModelSpec("dbscan", "DBSCAN", _dbscan, {"min_samples": 5}, {"min_samples": [3, 5, 10]}),
    # ---- extra models (include=[...] or include='all') ---------------------------------------------
    ModelSpec("bisecting_kmeans", "Bisecting KMeans", _bisecting, {"n_clusters": 3, "n_init": 3},
              {"n_clusters": _K}, default=False),
    ModelSpec("agglomerative_average", "Agglomerative (Average)", _agglomerative,
              {"n_clusters": 3, "linkage": "average"}, {"n_clusters": _K}, default=False),
    ModelSpec("agglomerative_complete", "Agglomerative (Complete)", _agglomerative,
              {"n_clusters": 3, "linkage": "complete"}, {"n_clusters": _K}, default=False),
    ModelSpec("hdbscan", "HDBSCAN", _hdbscan, {"min_cluster_size": 5},
              {"min_cluster_size": [5, 10, 20, 40], "min_samples": [3, 5, 10]}, default=False),
    ModelSpec("optics", "OPTICS", _optics, {"min_samples": 5},
              {"min_samples": [3, 5, 10, 20], "xi": [0.03, 0.05, 0.1]}, default=False),
    ModelSpec("mean_shift", "Mean Shift", _mean_shift, {"bin_seeding": True}, {}, default=False),
    ModelSpec("affinity_propagation", "Affinity Propagation", _affinity, {"damping": 0.9, "max_iter": 400},
              {"damping": [0.6, 0.75, 0.9]}, default=False),
])


@dataclass
class ClusterResult:
    """What one clustering model produced. ``labels`` uses -1 for noise (DBSCAN, HDBSCAN, OPTICS)."""
    name: str
    labels: np.ndarray = field(repr=False)
    n_clusters: int
    n_noise: int
    silhouette: float
    calinski_harabasz: float
    davies_bouldin: float
    external: Dict[str, float] = field(default_factory=dict)
    fit_time: float = float("nan")


class MultiModelClusterer(_ReportingMixin):
    """Benchmarks clustering algorithms on one dataset.

    Parameters
    ----------
    X              features (DataFrame or array). Numeric and finite unless ``preprocess=True``.
    y              optional TRUE group labels. Never used for fitting - only to add external scores
                   (ARI, NMI, ...) to the report, so you can see how close each model gets.
    n_clusters     number of clusters for the models that need one (KMeans, Ward, GMM, ...), or ``'auto'``
                   to pick the best k by silhouette (see ``find_optimal_k``).
    scaled_data    standardise the features first (recommended: distance-based models are scale-sensitive).
    preprocess     impute missing values, one-hot encode text columns and scale (like the supervised analysers).

    Typical use::

        mm = MultiModelClusterer(X, n_clusters=3)
        models = mm.run_all_models()
        mm.show_tabular_report(models)
        mm.plot_clusters(models)
    """

    registry: ModelRegistry = CLUSTERER_REGISTRY
    task = "clustering"

    def __init__(
        self,
        X: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        n_clusters: Union[int, str] = 3,
        scaled_data: bool = True,
        random_state: Optional[int] = 42,
        n_jobs: Optional[int] = -1,
        verbose: bool = True,
        preprocess: bool = False,
    ):
        if X is None:
            raise ValueError("Features 'X' must be provided to initialize MultiModelClusterer.")
        if np.ndim(X) != 2:
            raise ValueError("X must be 2-dimensional (rows = samples, columns = features).")
        _validate_features(X, preprocess)
        n_samples = len(X)
        if n_samples < 4:
            raise ValueError(f"At least 4 rows are needed to cluster, got {n_samples}.")
        if n_clusters != "auto":
            if not isinstance(n_clusters, (int, np.integer)) or isinstance(n_clusters, bool) or n_clusters < 2:
                raise ValueError("n_clusters must be an integer >= 2 or 'auto'.")
            if n_clusters >= n_samples:
                raise ValueError(f"n_clusters ({n_clusters}) must be smaller than the number of rows ({n_samples}).")

        self.y_true_ = None
        if y is not None:
            y_arr = np.asarray(y).ravel()
            if len(y_arr) != n_samples:
                raise ValueError(f"X and y have different lengths: X has {n_samples} rows but y has {len(y_arr)}.")
            if pd.isnull(y_arr).any():
                raise ValueError("y contains missing values. Drop or fill them first.")
            self.y_true_ = pd.factorize(y_arr)[0]

        self.X = X
        self.n_samples_ = n_samples
        self.n_clusters = n_clusters
        self.n_clusters_ = None if n_clusters == "auto" else int(n_clusters)
        self.scaled_data = scaled_data
        self.preprocess = preprocess
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose

        self.models_: List[ClusterResult] = []
        self.failed_models_: Dict[str, str] = {}
        self.warnings_: Dict[str, List[str]] = {}
        self.estimators_: Dict[str, Any] = {}
        self.fit_times_: Dict[str, float] = {}
        self.best_params_: Dict[str, Dict[str, Any]] = {}
        self.tuning_results_: Optional[pd.DataFrame] = None
        self.auto_params_: Dict[str, Dict[str, Any]] = {}
        self.k_search_: Optional[pd.DataFrame] = None
        self.optimal_k_: Optional[int] = None
        self._results: Dict[str, ClusterResult] = {}
        self._last_df_report: Optional[pd.DataFrame] = None

        self._prepare_features()

    # ------------------------------------------------------------------ data preparation
    def _prepare_features(self) -> None:
        """Scales / preprocesses X once; the models are fitted on the result (``X_scaled``)."""
        X_all = to_dataframe(self.X) if self.preprocess else self.X
        self.input_columns_ = [str(c) for c in X_all.columns] if isinstance(X_all, pd.DataFrame) else None
        self.transformer_ = None
        if self.preprocess:
            self.transformer_ = build_preprocessor(X_all, scale=self.scaled_data)
            self.X_scaled = transform_to_frame(self.transformer_, X_all, fit=True)
        elif self.scaled_data:
            self.transformer_ = StandardScaler()
            values = self.transformer_.fit_transform(X_all)
            self.X_scaled = (pd.DataFrame(values, columns=X_all.columns, index=X_all.index)
                             if isinstance(X_all, pd.DataFrame) else values)
        else:
            self.X_scaled = X_all
        self.feature_names_ = ([str(c) for c in self.X_scaled.columns]
                               if isinstance(self.X_scaled, pd.DataFrame) else None)
        self._values = np.asarray(self.X_scaled, dtype=float)

    def _build_context(self, random_state: Optional[int] = None) -> BuildContext:
        return BuildContext(random_state=random_state if random_state is not None else self.random_state,
                            n_jobs=self.n_jobs, n_train=self.n_samples_)

    def _k(self) -> int:
        """Number of clusters to use; resolves ``'auto'`` on first use."""
        if self.n_clusters_ is None:
            self.find_optimal_k(verbose=False)
            self.n_clusters_ = self.optimal_k_
            if self.verbose:
                _safe_print(f"[INFO] n_clusters='auto' -> using k={self.n_clusters_} (best silhouette).")
        return self.n_clusters_

    # ------------------------------------------------------------------ choosing k
    def find_optimal_k(self, k_range: Optional[List[int]] = None, verbose: Optional[bool] = None) -> pd.DataFrame:
        """Fits KMeans for each k and scores it. The best k by Silhouette is stored in ``optimal_k_``.

        Look at ``plot_k_search()`` for the elbow curve. Returns a DataFrame with Inertia, Silhouette,
        Calinski-Harabasz and Davies-Bouldin per k.
        """
        ks = [int(k) for k in (k_range if k_range is not None else _K) if 2 <= int(k) < self.n_samples_]
        if not ks:
            raise ValueError("k_range must contain integers between 2 and n_samples - 1.")
        rows = []
        with self._track_warnings("find_optimal_k"):
            for k in ks:
                km = KMeans(n_clusters=k, n_init=10, random_state=self.random_state).fit(self._values)
                sil, ch, db = self._internal_scores(self._values, km.labels_)
                rows.append({"k": k, "Inertia": float(km.inertia_), "Silhouette": sil,
                             "Calinski-Harabasz": ch, "Davies-Bouldin": db})
        df = pd.DataFrame(rows)
        self.k_search_ = df
        self.optimal_k_ = int(df.loc[df["Silhouette"].idxmax(), "k"])
        if self.verbose if verbose is None else verbose:
            _safe_print(f"\n{'='*60}\n CHOOSING THE NUMBER OF CLUSTERS (KMeans) \n{'='*60}")
            _safe_print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
            _safe_print(f"* Best k by Silhouette: {self.optimal_k_}\n{'='*60}\n")
        return df

    def plot_k_search(self, save_path: Optional[str] = None, show_plot: bool = True) -> None:
        """Elbow curve (Inertia) and Silhouette against k, with the best k marked."""
        if self.k_search_ is None:
            self.find_optimal_k(verbose=False)
        df, best = self.k_search_, self.optimal_k_
        with _apply_plot_style():
            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            for ax, col, title in zip(axes, ("Inertia", "Silhouette"),
                                      ("Inertia  (look for the elbow)", "Silhouette  (higher is better)")):
                ax.plot(df["k"], df[col], marker="o", color="#1f5f8b", lw=2)
                ax.axvline(best, color="#d62728", ls="--", lw=1.2)
                ax.set_title(title, loc="left", fontweight="bold")
                ax.set_xlabel("Number of clusters (k)")
                ax.set_xticks(df["k"])
                for side in ("top", "right"):
                    ax.spines[side].set_visible(False)
            axes[1].annotate(f"best k = {best}", (best, df.loc[df["k"] == best, "Silhouette"].iloc[0]),
                             textcoords="offset points", xytext=(8, -14), color="#d62728", fontsize=10)
            fig.tight_layout()
            self._save_and_show(fig, save_path=save_path, show_plot=show_plot)

    # ------------------------------------------------------------------ scoring
    def _internal_scores(self, values: np.ndarray, labels: np.ndarray) -> Tuple[float, float, float]:
        """(Silhouette, Calinski-Harabasz, Davies-Bouldin); NaN when fewer than 2 clusters exist."""
        uniq = np.unique(labels)
        if len(uniq) < 2 or len(uniq) > len(labels) - 1:
            return float("nan"), float("nan"), float("nan")
        sample_size = 5000 if len(labels) > 5000 else None
        return (float(silhouette_score(values, labels, sample_size=sample_size, random_state=self.random_state)),
                float(calinski_harabasz_score(values, labels)),
                float(davies_bouldin_score(values, labels)))

    def _score_labels(self, labels: np.ndarray) -> Tuple[int, int, float, float, float]:
        """(n clusters, n noise points, Silhouette, CH, DB). Noise (-1) is left out of the internal scores."""
        mask = labels != -1
        k = len(np.unique(labels[mask]))
        sil, ch, db = self._internal_scores(self._values[mask], labels[mask]) if mask.sum() > 2 else (np.nan,) * 3
        return k, int((~mask).sum()), sil, ch, db

    def _external_scores(self, labels: np.ndarray) -> Dict[str, float]:
        if self.y_true_ is None:
            return {}
        t = self.y_true_
        return {"ARI": float(adjusted_rand_score(t, labels)), "NMI": float(normalized_mutual_info_score(t, labels)),
                "V-measure": float(v_measure_score(t, labels)), "Homogeneity": float(homogeneity_score(t, labels)),
                "Completeness": float(completeness_score(t, labels))}

    # ------------------------------------------------------------------ fitting
    def _fit_labels(self, model: Any) -> Tuple[np.ndarray, float]:
        start = time.perf_counter()
        labels = np.asarray(model.fit_predict(self.X_scaled)).astype(int)
        return labels, time.perf_counter() - start

    def _register(self, name: str, model: Any, labels: np.ndarray, seconds: float) -> ClusterResult:
        k, n_noise, sil, ch, db = self._score_labels(labels)
        if k < 2:
            self._warn(name, f"found {k} cluster(s); the internal scores need at least 2, so it is ranked last. "
                             "Try other parameters (see tune_models).")
        result = ClusterResult(name, labels, k, n_noise, sil, ch, db, self._external_scores(labels), seconds)
        self.estimators_[name] = model
        self.fit_times_[name] = seconds
        self._results[name] = result
        return result

    def _fit_and_evaluate(self, model: Any, name: str) -> ClusterResult:
        with self._track_warnings(name):
            labels, seconds = self._fit_labels(model)
            return self._register(name, model, labels, seconds)

    def _auto_params(self, spec: ModelSpec) -> Dict[str, Any]:
        """Data-driven defaults: the DBSCAN radius and the Mean Shift bandwidth."""
        if spec.key == "dbscan":
            m = spec.defaults.get("min_samples", 5)
            nn = NearestNeighbors(n_neighbors=min(m, self.n_samples_ - 1)).fit(self._values)
            kth = nn.kneighbors(self._values)[0][:, -1]
            return {"eps": float(np.percentile(kth, 90))}
        if spec.key == "mean_shift":
            bw = estimate_bandwidth(self._values, quantile=0.2, n_samples=min(500, self.n_samples_),
                                    random_state=self.random_state)
            return {"bandwidth": float(bw) if bw > 0 else None}
        return {}

    def _params_for(self, spec: ModelSpec, user: Dict[str, Any]) -> Dict[str, Any]:
        params = dict(self._auto_params(spec))
        if "n_clusters" in spec.defaults:
            params["n_clusters"] = self._k()
        params.update(user)
        if params and (spec.key in ("dbscan", "mean_shift")):
            self.auto_params_[spec.name] = {k: v for k, v in params.items() if k in ("eps", "bandwidth")}
        return params

    def _run_spec(self, ref: Any, random_state: Optional[int] = None, **params) -> ClusterResult:
        spec = self.registry.resolve(ref)
        model = spec.build(self._build_context(random_state), **self._params_for(spec, params))
        return self._fit_and_evaluate(model, spec.name)

    def run_model(self, name: str, random_state: Optional[int] = None, **params) -> ClusterResult:
        """Runs ONE built-in model by key or display name, e.g. ``run_model('dbscan', eps=0.4)``."""
        return self._run_spec(name, random_state=random_state, **params)

    def run_all_models(
        self,
        custom_models: Optional[Dict[str, Any]] = None,
        random_state: Optional[int] = None,
        include: Optional[Any] = None,
        exclude: Optional[List[str]] = None,
        allow_large: bool = False,
    ) -> List[ClusterResult]:
        """Runs every standard model (``include='all'`` for all installed ones) plus ``custom_models``.

        ``custom_models`` is ``{"display name": unfitted_estimator_with_fit_predict}``. Models that need
        O(n^2) memory (Spectral, Agglomerative, Affinity Propagation, ...) are skipped on big data unless
        ``allow_large=True``; skipped and failed models are listed in ``failed_models_``.
        """
        self.failed_models_, self.warnings_, self.estimators_, self.fit_times_ = {}, {}, {}, {}
        self._results = {}
        self._last_df_report = None
        runs: List[Tuple[str, Any]] = []
        for spec in self.registry.select(include=include, exclude=exclude):
            limit = _MAX_SAMPLES.get(spec.key)
            if limit is not None and self.n_samples_ > limit and not allow_large:
                self.failed_models_[spec.name] = (f"skipped: {self.n_samples_} rows is above {limit}; this model "
                                                  "needs O(n^2) memory. Use allow_large=True to run it anyway.")
                continue
            runs.append((spec.name, lambda spec=spec: self._run_spec(spec, random_state=random_state)))
        from sklearn.base import clone
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

    def tune_models(
        self,
        include: Optional[Any] = None,
        exclude: Optional[List[str]] = None,
        scoring: str = "Silhouette",
        param_grids: Optional[Dict[str, Dict[str, List[Any]]]] = None,
        random_state: Optional[int] = None,
        allow_large: bool = False,
    ) -> List[ClusterResult]:
        """Searches each model's parameters (a small grid, see the registry) for the best internal score.

        Clustering has no test set, so this simply tries every combination on the full data and keeps the
        one with the best ``scoring`` ('Silhouette', 'Calinski-Harabasz' or 'Davies-Bouldin'). It also
        tries different numbers of clusters. Pass ``param_grids={'dbscan': {'eps': [0.3, 0.5]}}`` to
        override a model's grid. Returns the usual results; ``best_params_`` and ``tuning_results_``
        summarise the search. NOTE: scores are optimistic (nothing is held out) - use them to compare.
        """
        from sklearn.model_selection import ParameterGrid

        if scoring not in _INTERNAL:
            raise ValueError(f"scoring must be one of {list(_INTERNAL)}, got '{scoring}'.")
        pick = {"Silhouette": 0, "Calinski-Harabasz": 1, "Davies-Bouldin": 2}[scoring]
        sign = -1.0 if scoring in _LOWER else 1.0
        overrides = {self.registry.resolve(k).key: v for k, v in (param_grids or {}).items()}
        ctx = self._build_context(random_state)

        self.failed_models_, self.warnings_, self.estimators_, self.fit_times_ = {}, {}, {}, {}
        self._results, self.best_params_, self._last_df_report = {}, {}, None
        rows, models = [], []
        for spec in self.registry.select(include=include, exclude=exclude):
            limit = _MAX_SAMPLES.get(spec.key)
            if limit is not None and self.n_samples_ > limit and not allow_large:
                self.failed_models_[spec.name] = f"skipped: {self.n_samples_} rows is above {limit} (use allow_large=True)."
                continue
            space = overrides.get(spec.key, self._search_space(spec))
            if not space:
                self.failed_models_[spec.name] = "no search space is defined for this model; pass param_grids={...}."
                continue
            try:
                start = time.perf_counter()
                best = None
                tried = 0
                with self._track_warnings(f"{spec.name} (tuning)"):
                    for combo in ParameterGrid({k: list(v) for k, v in space.items()}):
                        if combo.get("n_clusters", 2) >= self.n_samples_:
                            continue
                        tried += 1
                        try:
                            model = spec.build(ctx, **{**self._base_params(spec), **combo})
                            labels, seconds = self._fit_labels(model)
                            score = self._score_labels(labels)[2 + pick]
                        except Exception:
                            continue
                        if np.isfinite(score) and (best is None or sign * score > best[0]):
                            best = (sign * score, combo, model, labels, seconds, score)
                if best is None:
                    raise ValueError(f"none of the {tried} parameter combinations gave at least 2 clusters.")
                _, combo, model, labels, seconds, score = best
                self.best_params_[spec.name] = combo
                rows.append({"Model": spec.name, f"Best {scoring}": score, "Best Params": combo,
                             "Candidates": tried, "Search Time (s)": time.perf_counter() - start})
                with self._track_warnings(spec.name):
                    models.append(self._register(spec.name, model, labels, seconds))
            except Exception as e:
                self.failed_models_[spec.name] = f"{type(e).__name__}: {e}"
                _safe_print(f"[WARNING] Model '{spec.name}' failed to tune: {e}")

        self.models_ = models
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values(by=f"Best {scoring}", ascending=scoring in _LOWER).reset_index(drop=True)
        self.tuning_results_ = df
        if self.verbose and not df.empty:
            _safe_print(f"\n{'='*60}\n PARAMETER SEARCH (best {scoring}) \n{'='*60}")
            _safe_print(df.drop(columns=["Best Params"]).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
            for _, r in df.iterrows():
                _safe_print(f"   {r['Model']}: {r['Best Params']}")
            _safe_print(f"{'='*60}\n")
        return models

    def _base_params(self, spec: ModelSpec) -> Dict[str, Any]:
        """Parameters used during a search before the grid values are applied (k and data-driven defaults)."""
        params = dict(self._auto_params(spec))
        if "n_clusters" in spec.defaults:
            params["n_clusters"] = self._k()
        return params

    def _search_space(self, spec: ModelSpec) -> Dict[str, List[Any]]:
        """Static grid from the registry, plus data-scaled grids for DBSCAN (eps) and Mean Shift (bandwidth)."""
        space = dict(spec.search_space)
        auto = self._auto_params(spec)
        factors = [0.5, 0.75, 1.0, 1.5, 2.0]
        if spec.key == "dbscan":
            space["eps"] = [round(auto["eps"] * f, 4) for f in factors]
        elif spec.key == "mean_shift" and auto.get("bandwidth"):
            space["bandwidth"] = [round(auto["bandwidth"] * f, 4) for f in factors]
        return space

    # ------------------------------------------------------------------ reporting
    def _resolve_models(self, models: Optional[List[ClusterResult]] = None) -> List[ClusterResult]:
        if models:
            return models
        if self.models_:
            return self.models_
        _safe_print("[INFO] No models provided. Automatically running all models...")
        return self.run_all_models()

    def _report_df(self, models: List[ClusterResult], sort_by: str = "Silhouette") -> pd.DataFrame:
        rows = []
        for m in models:
            row = {"Model": m.name, "Clusters": m.n_clusters, "Noise Points": m.n_noise,
                   "Silhouette": m.silhouette, "Calinski-Harabasz": m.calinski_harabasz,
                   "Davies-Bouldin": m.davies_bouldin}
            row.update(m.external)
            row["Fit Time (s)"] = m.fit_time
            rows.append(row)
        df = pd.DataFrame(rows)
        if sort_by not in df.columns or sort_by in ("Model", "Clusters", "Noise Points"):
            valid = [c for c in df.columns if c not in ("Model", "Clusters", "Noise Points")]
            raise ValueError(f"sort_by must be one of {valid}, got '{sort_by}'.")
        return df.sort_values(by=sort_by, ascending=sort_by in _LOWER, na_position="last",
                              kind="stable").reset_index(drop=True)

    def show_tabular_report(
        self,
        models: Optional[List[ClusterResult]] = None,
        return_df: bool = False,
        sort_by: str = "Silhouette",
    ) -> Optional[pd.DataFrame]:
        """Prints every model's scores and recommends the best one.

        ``sort_by`` can be 'Silhouette', 'Calinski-Harabasz', 'Davies-Bouldin' (lower is better), 'Fit Time (s)'
        or - when you passed ``y`` - 'ARI', 'NMI', 'V-measure', 'Homogeneity', 'Completeness'.
        """
        models = self._resolve_models(models)
        if not models:
            _safe_print("No models were successfully evaluated.")
            self._print_failures()
            return None
        df = self._report_df(models, sort_by)
        self._last_df_report = df

        _safe_print(f"\n{'='*70}\n CLUSTERING MODEL COMPARISON \n{'='*70}")
        _safe_print(df.to_string(index=False, na_rep="N/A", float_format=lambda v: f"{v:.4f}"))
        _safe_print(f"{'='*70}\n")
        self._print_failures()
        best = df.iloc[0]
        _safe_print(f"* BEST MODEL (by {sort_by}): {best['Model']}  - {int(best['Clusters'])} clusters, "
                    f"Silhouette {best['Silhouette']:.4f}")
        _safe_print("  Silhouette: above 0.5 = clear groups, 0.25-0.5 = reasonable, below 0.25 = weak structure.")
        _safe_print("  Davies-Bouldin: lower is better. Noise points (label -1) are left out of the scores.")
        if best["Noise Points"] > 0:
            share = best["Noise Points"] / self.n_samples_ * 100
            _safe_print(f"  Note: {best['Model']} set {int(best['Noise Points'])} rows ({share:.1f}%) aside as noise, "
                        "which helps its score. Compare it with the models that label every row.")
        if self.y_true_ is not None:
            _safe_print("  ARI / NMI compare with the TRUE labels you passed as y (1.0 = perfect match).")
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

    def get_labels(self, model: Optional[str] = None) -> np.ndarray:
        """Cluster label of every row for ``model`` (default: the best one). -1 means noise."""
        return self._results[self._name(model)].labels

    def assign_clusters(self, model: Optional[str] = None) -> pd.DataFrame:
        """Your data with a ``cluster`` column added (default model: the best one)."""
        df = to_dataframe(self.X, self.input_columns_).copy()
        df["cluster"] = self.get_labels(model)
        return df

    def cluster_profile(self, model: Optional[str] = None) -> pd.DataFrame:
        """Size and average feature values of every cluster, in your ORIGINAL units - the quickest way to
        understand what each cluster is."""
        df = self.assign_clusters(model)
        numeric = [c for c in df.columns if c != "cluster" and pd.api.types.is_numeric_dtype(df[c])]
        out = df.groupby("cluster")[numeric].mean()
        sizes = df["cluster"].value_counts().sort_index()
        out.insert(0, "Share (%)", (sizes / len(df) * 100).round(1))
        out.insert(0, "Size", sizes)
        return out

    # ------------------------------------------------------------------ predicting new data
    def export_model(self, model: Optional[str] = None) -> TrainedModel:
        """Bundles a fitted model with its preprocessing so it can label NEW rows (default: the best model)."""
        if model is None:  # best-ranked model that is able to label new rows
            ranked = [str(n) for n in self._report_df(self.models_ or list(self._results.values()))["Model"]]
            able = [n for n in ranked if hasattr(self.estimators_.get(n), "predict")]
            if not able:
                raise TypeError("None of the fitted models can label new data (no predict method).")
            model = able[0]
        name = self._name(model)
        est = self.estimators_[name]
        if not hasattr(est, "predict"):
            can = [n for n, e in self.estimators_.items() if hasattr(e, "predict")]
            raise TypeError(f"'{name}' cannot label new data (it has no predict method). "
                            f"Models that can: {can}. Or assign new rows to the nearest cluster yourself.")
        from .. import __version__
        return TrainedModel(estimator=est, name=name, task=self.task, transformer=self.transformer_,
                            columns=self.input_columns_, feature_names=self.feature_names_,
                            metadata={"package_version": __version__, "random_state": self.random_state})

    def predict(self, X_new: Any, model: Optional[str] = None) -> np.ndarray:
        """Assigns NEW rows to the clusters of a fitted model (KMeans, MiniBatch KMeans, Gaussian Mixture,
        Birch, Mean Shift, Affinity Propagation). The training scaling is applied automatically."""
        return self.export_model(model).predict(X_new)

    def save_model(self, path: str, model: Optional[str] = None) -> str:
        """Saves a fitted model + its preprocessing (joblib). Reload with ``load_model(path)``."""
        return self.export_model(model).save(path)

    # ------------------------------------------------------------------ plots
    def _ranked(self, models: List[ClusterResult], metric: str = "Silhouette") -> List[ClusterResult]:
        order = {n: i for i, n in enumerate(self._report_df(models, metric)["Model"])}
        return sorted(models, key=lambda m: order[m.name])

    def plot_comparison(
        self,
        models: Optional[List[ClusterResult]] = None,
        save_path: Optional[str] = None,
        show_plot: bool = True,
        metric: str = "Silhouette",
    ):
        """Ranks all models on one metric (default 'Silhouette'; see ``show_tabular_report`` for the others)."""
        models = self._resolve_models(models)
        if not models:
            return
        df = self._report_df(models, metric)
        with _apply_plot_style():
            fig = comparison_figure(
                list(df["Model"]), list(df[metric]), metric=metric, title=f"Model comparison: {metric}",
                lower_is_better=metric in _LOWER,
                footnote="Internal scores on the full dataset; noise points are left out. No test split is used.",
            )
            self._save_and_show(fig, save_path=save_path, show_plot=show_plot)

    def _project_2d(self) -> Tuple[np.ndarray, List[str]]:
        """2-D coordinates for plotting: the raw features when there are 2, otherwise a PCA projection."""
        v = self._values
        if v.shape[1] < 2:
            raise ValueError("Plotting clusters needs at least 2 features.")
        if v.shape[1] == 2:
            names = self.feature_names_ or ["feature 1", "feature 2"]
            return v, [str(n) for n in names]
        pca = PCA(n_components=2, random_state=self.random_state).fit(v)
        ratio = pca.explained_variance_ratio_ * 100
        return pca.transform(v), [f"PC 1 ({ratio[0]:.0f}% of variance)", f"PC 2 ({ratio[1]:.0f}% of variance)"]

    def plot_clusters(
        self,
        models: Optional[List[ClusterResult]] = None,
        top: int = 6,
        save_path: Optional[str] = None,
        show_plot: bool = True,
    ):
        """Scatter plots of the best ``top`` models, coloured by cluster (grey crosses = noise).

        With more than 2 features the data is shown on its first two principal components.
        """
        models = self._resolve_models(models)
        if not models:
            return
        models = self._ranked(models)[:max(1, top)]
        xy, labels_xy = self._project_2d()
        ncols = min(3, len(models))
        nrows = math.ceil(len(models) / ncols)
        with _apply_plot_style():
            fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 4.0 * nrows), squeeze=False)
            for ax, m in zip(axes.ravel(), models):
                noise = m.labels == -1
                for i, c in enumerate(sorted(set(m.labels[~noise]))):
                    pts = xy[m.labels == c]
                    ax.scatter(pts[:, 0], pts[:, 1], s=18, color=_PALETTE[i % len(_PALETTE)], alpha=0.85,
                               edgecolor="white", linewidth=0.3)
                if noise.any():
                    ax.scatter(xy[noise, 0], xy[noise, 1], s=16, color=_NOISE, marker="x", linewidth=0.9)
                sil = "N/A" if np.isnan(m.silhouette) else f"{m.silhouette:.2f}"
                extra = f", {m.n_noise} noise" if m.n_noise else ""
                ax.set_title(f"{m.name}\n{m.n_clusters} clusters{extra} | silhouette {sil}", fontsize=11,
                             loc="left", fontweight="bold")
                ax.set_xticks([])
                ax.set_yticks([])
                for side in ("top", "right"):
                    ax.spines[side].set_visible(False)
            for ax in axes.ravel()[len(models):]:
                ax.axis("off")
            for ax in axes[-1]:
                ax.set_xlabel(labels_xy[0], fontsize=9)
            for ax in axes[:, 0]:
                ax.set_ylabel(labels_xy[1], fontsize=9)
            fig.tight_layout()
            self._save_and_show(fig, save_path=save_path, show_plot=show_plot)

    def plot_silhouette(self, model: Optional[str] = None, save_path: Optional[str] = None, show_plot: bool = True):
        """Silhouette of every single row, grouped by cluster: long bars past the dashed average line = well
        placed rows, short or negative bars = rows that may belong elsewhere."""
        name = self._name(model)
        labels = self._results[name].labels
        idx = np.where(labels != -1)[0]
        if len(np.unique(labels[idx])) < 2:
            raise ValueError(f"'{name}' found fewer than 2 clusters, so there is no silhouette to draw.")
        if len(idx) > 5000:
            idx = np.sort(np.random.default_rng(self.random_state).choice(idx, 5000, replace=False))
        vals = silhouette_samples(self._values[idx], labels[idx])
        lab = labels[idx]
        with _apply_plot_style():
            fig, ax = plt.subplots(figsize=(8, 5))
            y_low = 0
            for i, c in enumerate(sorted(set(lab))):
                v = np.sort(vals[lab == c])
                ax.fill_betweenx(np.arange(y_low, y_low + len(v)), 0, v, color=_PALETTE[i % len(_PALETTE)], alpha=0.85)
                ax.text(-0.04, y_low + len(v) / 2, str(c), va="center", ha="right", fontsize=10)
                y_low += len(v) + max(1, len(idx) // 50)
            ax.axvline(vals.mean(), color="#d62728", ls="--", lw=1.2)
            ax.set_title(f"Silhouette per row - {name}  (average {vals.mean():.2f})", loc="left", fontweight="bold")
            ax.set_xlabel("Silhouette value")
            ax.set_ylabel("Cluster")
            ax.set_yticks([])
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            fig.tight_layout()
            self._save_and_show(fig, save_path=save_path, show_plot=show_plot)

    def get_summary(
        self,
        models: Optional[List[ClusterResult]] = None,
        save_prefix: Optional[str] = None,
        show_plot: bool = True,
    ):
        """Runs the full reporting pipeline: table, ranking chart and cluster scatter plots."""
        models = self._resolve_models(models)
        df = self.show_tabular_report(models, return_df=True)
        if save_prefix:
            if df is not None:
                self.save_report(df, f"{save_prefix}_report.csv")
            self.plot_comparison(models, save_path=f"{save_prefix}_comparison.png", show_plot=show_plot)
            self.plot_clusters(models, save_path=f"{save_prefix}_clusters.png", show_plot=show_plot)
        else:
            self.plot_comparison(models, show_plot=show_plot)
            self.plot_clusters(models, show_plot=show_plot)
