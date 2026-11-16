"""Dimensionality reduction: squeeze many features into a few, and compare how well each method does it.

Each method turns the feature matrix into ``n_components`` new columns. Because there is no single "right" answer,
every embedding is scored on how well it keeps the structure of the original data:

* **Trustworthiness**  - are rows that are close in the embedding also close in the original data? (0 to 1)
* **Continuity**       - are rows that are close in the original data still close in the embedding? (0 to 1)
* **Global Structure** - do distances between far-apart rows survive? (Spearman correlation, -1 to 1)
* **Variance Kept (%)**- how much of the data can be rebuilt from the components (methods that can reverse the mapping)
* **Class Separation** - when you pass ``y``: cross-validated 5-NN accuracy on the embedding
"""
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.decomposition import PCA, FactorAnalysis, FastICA, KernelPCA, TruncatedSVD
from sklearn.manifold import MDS, TSNE, Isomap, LocallyLinearEmbedding, SpectralEmbedding, trustworthiness
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.random_projection import GaussianRandomProjection

from ..core.persistence import TrainedModel
from ..core.plotting import comparison_figure
from ..core.registry import BuildContext, ModelRegistry, ModelSpec
from ..core.utils import _apply_plot_style, _safe_print
from ._base import _UnsupervisedBase

_PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#17becf",
            "#bcbd22", "#7f7f7f"]
_LOWER = ("Fit Time (s)",)
_METRICS = ("Trustworthiness", "Continuity", "Global Structure", "Variance Kept (%)", "Class Separation")
_SCORE_ROWS = 2000    # rows used to score local structure (it needs an n x n distance matrix)
_GLOBAL_ROWS = 500    # rows used for the global distance correlation

# Models that are slow or memory hungry on big data: skipped unless ``allow_large=True``.
_MAX_SAMPLES = {"tsne": 10_000, "kernel_pca": 8_000, "isomap": 8_000, "lle": 10_000, "mds": 1_500,
                "spectral_embedding": 8_000}


# --------------------------------------------------------------------------- model factories
def _pca(ctx: BuildContext, **p):
    return PCA(random_state=ctx.random_state, **p)


def _truncated_svd(ctx: BuildContext, **p):
    return TruncatedSVD(random_state=ctx.random_state, **p)


def _kernel_pca(ctx: BuildContext, **p):
    return KernelPCA(random_state=ctx.random_state, n_jobs=ctx.n_jobs, **p)


class _StableFastICA(FastICA):
    """FastICA that ignores constant columns while fitting.

    A constant column (e.g. a pixel that is always 0) makes the whitening step rank-deficient, which can make
    the underlying LAPACK call fail on some machines. The fitted weights are zero-padded back to the full
    width, so ``transform`` and ``components_`` still work on the original columns.
    """

    def fit_transform(self, X, y=None):
        X = np.asarray(X, dtype=float)
        keep = X.std(axis=0) > 1e-10
        if keep.all():
            return super().fit_transform(X, y)
        sources = super().fit_transform(X[:, keep], y)
        width = X.shape[1]
        comp = np.zeros((self.components_.shape[0], width))
        comp[:, keep] = self.components_
        self.components_ = comp
        mix = np.zeros((width, self.mixing_.shape[1]))
        mix[keep] = self.mixing_
        self.mixing_ = mix
        if getattr(self, "mean_", None) is not None:
            mean = np.zeros(width)
            mean[keep] = self.mean_
            self.mean_ = mean
        self.n_features_in_ = width
        return sources

    def fit(self, X, y=None):
        self.fit_transform(X, y)
        return self


def _fastica(ctx: BuildContext, **p):
    return _StableFastICA(random_state=ctx.random_state, **p)


def _factor(ctx: BuildContext, **p):
    return FactorAnalysis(random_state=ctx.random_state, **p)


def _isomap(ctx: BuildContext, n_neighbors: int = 10, **p):
    return Isomap(n_neighbors=max(2, min(n_neighbors, ctx.n_train - 1)), n_jobs=ctx.n_jobs, **p)


def _lle(ctx: BuildContext, n_neighbors: int = 12, n_components: int = 2, **p):
    return LocallyLinearEmbedding(n_neighbors=max(n_components + 1, min(n_neighbors, ctx.n_train - 1)),
                                  n_components=n_components, random_state=ctx.random_state, n_jobs=ctx.n_jobs, **p)


def _tsne(ctx: BuildContext, n_components: int = 2, perplexity: float = 30.0, **p):
    perplexity = max(2.0, min(perplexity, (ctx.n_train - 1) / 3))
    method = "barnes_hut" if n_components < 4 else "exact"  # Barnes-Hut only supports up to 3 components
    return TSNE(n_components=n_components, perplexity=perplexity, init="pca", learning_rate="auto",
                method=method, random_state=ctx.random_state, **p)


def _mds(ctx: BuildContext, **p):
    return MDS(random_state=ctx.random_state, n_init=1, **p)


def _spectral_embedding(ctx: BuildContext, **p):
    return SpectralEmbedding(random_state=ctx.random_state, n_jobs=ctx.n_jobs, **p)


def _random_projection(ctx: BuildContext, **p):
    return GaussianRandomProjection(random_state=ctx.random_state, **p)


def _umap(ctx: BuildContext, **p):
    import umap
    return umap.UMAP(random_state=ctx.random_state, **p)


REDUCER_REGISTRY = ModelRegistry([
    # ---- standard set -----------------------------------------------------------------------------
    ModelSpec("pca", "PCA", _pca, {}, {}),
    ModelSpec("kernel_pca", "Kernel PCA", _kernel_pca, {"kernel": "rbf"}),
    ModelSpec("fastica", "FastICA", _fastica, {"whiten": "unit-variance", "max_iter": 1000}),
    ModelSpec("factor_analysis", "Factor Analysis", _factor, {}),
    ModelSpec("isomap", "Isomap", _isomap, {"n_neighbors": 10}),
    ModelSpec("lle", "Locally Linear Embedding", _lle, {"n_neighbors": 12}),
    ModelSpec("tsne", "t-SNE", _tsne, {"perplexity": 30.0}),
    # ---- extra ------------------------------------------------------------------------------------
    ModelSpec("truncated_svd", "Truncated SVD", _truncated_svd, {}, default=False),
    ModelSpec("spectral_embedding", "Spectral Embedding", _spectral_embedding, {}, default=False),
    ModelSpec("mds", "MDS", _mds, {}, default=False),
    ModelSpec("random_projection", "Gaussian Random Projection", _random_projection, {}, default=False),
    ModelSpec("umap", "UMAP", _umap, {"n_neighbors": 15}, requires="umap", default=False),
])


@dataclass
class ReductionResult:
    """What one method produced. ``embedding`` has one row per input row and ``n_components`` columns."""
    name: str
    embedding: np.ndarray = field(repr=False)
    n_components: int
    metrics: Dict[str, float] = field(default_factory=dict)
    fit_time: float = float("nan")
    can_transform: bool = False  # can it map NEW rows? (t-SNE, MDS, Spectral Embedding cannot)


class MultiModelReducer(_UnsupervisedBase):
    """Benchmarks dimensionality-reduction methods on one dataset.

    Parameters
    ----------
    X              features (DataFrame or array), numeric and finite unless ``preprocess=True``.
    y              optional labels, used ONLY to colour the plots and to add a 'Class Separation' score.
    n_components   how many new columns to create (default 2, which can be plotted). Use ``find_n_components``
                   to see how many PCA components keep e.g. 95% of the variance.
    scaled_data    standardise the features first (recommended).
    preprocess     impute missing values, one-hot encode text columns and scale.

    Typical use::

        red = MultiModelReducer(X, y=y, n_components=2)
        models = red.run_all_models()
        red.show_tabular_report(models)
        red.plot_embeddings(models)
    """

    registry: ModelRegistry = REDUCER_REGISTRY
    task = "dimensionality reduction"

    def __init__(
        self,
        X: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        n_components: int = 2,
        scaled_data: bool = True,
        random_state: Optional[int] = 42,
        n_jobs: Optional[int] = -1,
        verbose: bool = True,
        preprocess: bool = False,
    ):
        n_samples = self._check_features(X, preprocess, 10, "reduce dimensions")
        n_features = np.shape(X)[1]
        if not isinstance(n_components, (int, np.integer)) or isinstance(n_components, bool) or n_components < 1:
            raise ValueError("n_components must be a positive integer.")
        if n_features < 2:
            raise ValueError("X needs at least 2 features to reduce.")

        self.y_ = None
        self.y_is_class_ = True
        if y is not None:
            y_arr = np.asarray(y).ravel()
            if len(y_arr) != n_samples:
                raise ValueError(f"X and y have different lengths: X has {n_samples} rows but y has {len(y_arr)}.")
            if pd.isnull(y_arr).any():
                raise ValueError("y contains missing values. Drop or fill them first.")
            numeric = np.issubdtype(y_arr.dtype, np.number)
            n_unique = len(np.unique(y_arr))
            if not numeric and n_unique > 15:
                raise ValueError("y has more than 15 different text labels; pass numbers or fewer groups.")
            self.y_is_class_ = (not numeric) or n_unique <= 15
            self.y_ = y_arr

        self.X = X
        self.n_samples_ = n_samples
        self.n_features_in_ = n_features
        self.n_components = int(n_components)
        self.scaled_data = scaled_data
        self.preprocess = preprocess
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose

        self.models_: List[ReductionResult] = []
        self.failed_models_: Dict[str, str] = {}
        self.warnings_: Dict[str, List[str]] = {}
        self.estimators_: Dict[str, Any] = {}
        self.fit_times_: Dict[str, float] = {}
        self.variance_table_: Optional[pd.DataFrame] = None
        self._results: Dict[str, ReductionResult] = {}
        self._last_df_report: Optional[pd.DataFrame] = None

        self._prepare_features()
        # a model cannot create more components than there are columns once the data is encoded
        if self.n_components > min(self._values.shape[1], n_samples - 1):
            raise ValueError(f"n_components ({self.n_components}) must be at most {min(self._values.shape[1], n_samples - 1)} "
                             "(the number of features after preprocessing / rows - 1).")

    # ------------------------------------------------------------------ scoring
    def _rng_rows(self, m: int, seed_shift: int = 0) -> np.ndarray:
        n = self.n_samples_
        if n <= m:
            return np.arange(n)
        return np.sort(np.random.default_rng((self.random_state or 0) + seed_shift).choice(n, m, replace=False))

    def _scores(self, est: Any, Z: np.ndarray) -> Dict[str, float]:
        from scipy.spatial.distance import pdist
        from scipy.stats import spearmanr

        X = self._values
        idx = self._rng_rows(_SCORE_ROWS)
        k = max(1, min(10, len(idx) // 2 - 1))
        out: Dict[str, float] = {
            "Trustworthiness": float(trustworthiness(X[idx], Z[idx], n_neighbors=k)),
            "Continuity": float(trustworthiness(Z[idx], X[idx], n_neighbors=k)),
        }
        g = self._rng_rows(_GLOBAL_ROWS, 1)
        a, b = pdist(X[g]), pdist(Z[g])
        out["Global Structure"] = float(spearmanr(a, b)[0]) if a.std() > 0 and b.std() > 0 else float("nan")

        out["Variance Kept (%)"] = float("nan")
        if hasattr(est, "inverse_transform"):
            try:
                rebuilt = np.asarray(est.inverse_transform(Z), dtype=float)
                total = float(((X - X.mean(axis=0)) ** 2).sum())
                if total > 0 and rebuilt.shape == X.shape:
                    out["Variance Kept (%)"] = max(0.0, 1.0 - float(((X - rebuilt) ** 2).sum()) / total) * 100
            except Exception:
                pass

        if self.y_ is not None and self.y_is_class_:
            codes = pd.factorize(self.y_)[0]
            smallest = np.bincount(codes).min()
            cv = int(min(5, smallest))
            out["Class Separation"] = (float(cross_val_score(KNeighborsClassifier(5), Z, codes, cv=cv).mean())
                                       if cv >= 2 else float("nan"))
        return out

    # ------------------------------------------------------------------ fitting
    def _params_for(self, spec: ModelSpec, user: Dict[str, Any]) -> Dict[str, Any]:
        params = {"n_components": self.n_components}
        params.update(user)
        return params

    def _fit_and_evaluate(self, model: Any, name: str) -> ReductionResult:
        with self._track_warnings(name):
            start = time.perf_counter()
            Z = np.asarray(model.fit_transform(self.X_scaled), dtype=float)
            seconds = time.perf_counter() - start
            if Z.ndim != 2 or len(Z) != self.n_samples_:
                raise ValueError("the model did not return one row per input row.")
            if not np.all(np.isfinite(Z)):
                raise ValueError("the embedding contains non-finite values.")
            metrics = self._scores(model, Z)
        result = ReductionResult(name, Z, Z.shape[1], metrics, seconds, can_transform=hasattr(model, "transform"))
        self.estimators_[name] = model
        self.fit_times_[name] = seconds
        self._results[name] = result
        return result

    def _run_spec(self, ref: Any, random_state: Optional[int] = None, **params) -> ReductionResult:
        spec = self.registry.resolve(ref)
        model = spec.build(self._build_context(random_state), **self._params_for(spec, params))
        return self._fit_and_evaluate(model, spec.name)

    def run_model(self, name: str, random_state: Optional[int] = None, **params) -> ReductionResult:
        """Runs ONE built-in method by key or display name, e.g. ``run_model('tsne', perplexity=15)``."""
        return self._run_spec(name, random_state=random_state, **params)

    def run_all_models(
        self,
        custom_models: Optional[Dict[str, Any]] = None,
        random_state: Optional[int] = None,
        include: Optional[Any] = None,
        exclude: Optional[List[str]] = None,
        allow_large: bool = False,
    ) -> List[ReductionResult]:
        """Runs every standard method (``include='all'`` for all installed ones) plus ``custom_models``
        (``{"name": unfitted_transformer_with_fit_transform}``). Slow methods are skipped on big data unless
        ``allow_large=True``; skipped and failed methods are listed in ``failed_models_``."""
        self.failed_models_, self.warnings_, self.estimators_, self.fit_times_ = {}, {}, {}, {}
        self._results = {}
        self._last_df_report = None
        runs: List[Tuple[str, Any]] = []
        for spec in self.registry.select(include=include, exclude=exclude):
            limit = _MAX_SAMPLES.get(spec.key)
            if limit is not None and self.n_samples_ > limit and not allow_large:
                self.failed_models_[spec.name] = (f"skipped: {self.n_samples_} rows is above {limit}; this method is "
                                                  "slow on big data. Use allow_large=True to run it anyway.")
                continue
            runs.append((spec.name, lambda spec=spec: self._run_spec(spec, random_state=random_state)))
        for cname, inst in (custom_models or {}).items():
            runs.append((cname, lambda cname=cname, inst=inst: self._fit_and_evaluate(clone(inst), cname)))

        models = []
        for name, run in runs:
            try:
                models.append(run())
            except Exception as e:  # keep going: one broken method must not stop the benchmark
                self.failed_models_[name] = f"{type(e).__name__}: {e}"
                _safe_print(f"[WARNING] Model '{name}' failed: {e}")
        self.models_ = models
        return models

    # ------------------------------------------------------------------ how many components?
    def find_n_components(self, variance: float = 0.95, verbose: Optional[bool] = None) -> int:
        """Smallest number of PCA components that keeps at least ``variance`` (default 95%) of the variance.

        The full table (per-component and cumulative variance) is stored in ``variance_table_``; see
        ``plot_variance()``.
        """
        if not 0 < variance <= 1:
            raise ValueError("variance must be between 0 and 1.")
        with self._track_warnings("find_n_components"):
            pca = PCA(random_state=self.random_state).fit(self._values)
        ratio = pca.explained_variance_ratio_
        cum = np.cumsum(ratio)
        self.variance_table_ = pd.DataFrame({"Component": np.arange(1, len(ratio) + 1),
                                             "Variance (%)": ratio * 100, "Cumulative (%)": cum * 100})
        self._variance_target = variance
        n = int(np.searchsorted(cum, variance - 1e-12) + 1)
        n = min(n, len(ratio))
        if self.verbose if verbose is None else verbose:
            _safe_print(f"\n{'='*60}\n HOW MANY COMPONENTS? (PCA) \n{'='*60}")
            _safe_print(self.variance_table_.head(15).to_string(index=False, float_format=lambda v: f"{v:.2f}"))
            _safe_print(f"* {n} component(s) keep {cum[n - 1] * 100:.1f}% of the variance (target {variance * 100:.0f}%).\n{'='*60}\n")
        return n

    def plot_variance(self, save_path: Optional[str] = None, show_plot: bool = True) -> None:
        """Scree plot: variance of every PCA component (bars) and the running total (line)."""
        if self.variance_table_ is None:
            self.find_n_components(verbose=False)
        target = getattr(self, "_variance_target", 0.95) * 100
        reached = int(np.searchsorted(self.variance_table_["Cumulative (%)"].to_numpy(), target - 1e-9)) + 1
        df = self.variance_table_.head(min(len(self.variance_table_), max(10, reached + 3)))  # show where the target is reached
        with _apply_plot_style():
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.bar(df["Component"], df["Variance (%)"], color="#8fb3c9", label="each component")
            ax.plot(df["Component"], df["Cumulative (%)"], marker="o", color="#1f5f8b", lw=2, label="running total")
            ax.axhline(target, color="#d62728", ls="--", lw=1.2)
            ax.text(df["Component"].max(), target + 1.5, f"{target:.0f}%", color="#d62728", ha="right", fontsize=9)
            if len(df) <= 25:
                ax.set_xticks(df["Component"])
            ax.set_ylim(0, 105)
            ax.set_xlabel("PCA component")
            ax.set_ylabel("Share of variance (%)")
            ax.set_title("How much variance each component keeps", loc="left", fontweight="bold")
            ax.legend(frameon=False, loc="center right")
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            fig.tight_layout()
            self._save_and_show(fig, save_path=save_path, show_plot=show_plot)

    # ------------------------------------------------------------------ reporting
    def _resolve_models(self, models: Optional[List[ReductionResult]] = None) -> List[ReductionResult]:
        if models:
            return models
        if self.models_:
            return self.models_
        _safe_print("[INFO] No models provided. Automatically running all models...")
        return self.run_all_models()

    def _report_df(self, models: List[ReductionResult], sort_by: str = "Trustworthiness") -> pd.DataFrame:
        rows = []
        for m in models:
            row = {"Model": m.name, "Components": m.n_components}
            row.update(m.metrics)
            row["Can Transform New Data"] = "yes" if m.can_transform else "no"
            row["Fit Time (s)"] = m.fit_time
            rows.append(row)
        df = pd.DataFrame(rows)
        valid = [c for c in df.columns if c in _METRICS + _LOWER]
        if sort_by not in valid:
            raise ValueError(f"sort_by must be one of {valid}, got '{sort_by}'.")
        return df.sort_values(by=sort_by, ascending=sort_by in _LOWER, na_position="last",
                              kind="stable").reset_index(drop=True)

    def show_tabular_report(
        self,
        models: Optional[List[ReductionResult]] = None,
        return_df: bool = False,
        sort_by: str = "Trustworthiness",
    ) -> Optional[pd.DataFrame]:
        """Prints every method's quality scores. ``sort_by`` can be any score column, e.g. 'Continuity',
        'Global Structure', 'Variance Kept (%)', 'Class Separation' (with ``y``) or 'Fit Time (s)'."""
        models = self._resolve_models(models)
        if not models:
            _safe_print("No models were successfully evaluated.")
            self._print_failures()
            return None
        df = self._report_df(models, sort_by)
        self._last_df_report = df

        _safe_print(f"\n{'='*78}\n DIMENSIONALITY REDUCTION COMPARISON \n{'='*78}")
        _safe_print(df.to_string(index=False, na_rep="-", float_format=lambda v: f"{v:.4f}"))
        _safe_print(f"{'='*78}\n")
        self._print_failures()
        _safe_print(f"* BEST MODEL (by {sort_by}): {df.iloc[0]['Model']}")
        _safe_print("  Trustworthiness / Continuity: 1.0 = neighbourhoods are perfectly kept. Global Structure: how well")
        _safe_print("  far-apart rows stay far apart. Variance Kept: only for methods that can be reversed (PCA, ...).")
        _safe_print("  Local scores favour t-SNE / Isomap; PCA keeps the global shape and is the easiest to explain.")
        _safe_print(f"{'='*78}\n")
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

    # ------------------------------------------------------------------ results
    def get_embedding(self, model: Optional[str] = None) -> pd.DataFrame:
        """The new columns for every row (``Component 1``, ``Component 2``, ...), indexed like your data."""
        res = self._results[self._name(model)]
        index = self.X.index if isinstance(self.X, pd.DataFrame) else None
        return pd.DataFrame(res.embedding, index=index,
                            columns=[f"Component {i + 1}" for i in range(res.n_components)])

    def get_loadings(self, model: str = "PCA") -> pd.DataFrame:
        """How strongly each (scaled) feature contributes to each component - the way to *interpret* components.

        Available for methods with ``components_`` (PCA, Truncated SVD, Factor Analysis, FastICA, Random Projection).
        """
        name = self._name(model)
        est = self.estimators_[name]
        comps = getattr(est, "components_", None)
        if comps is None:
            raise TypeError(f"'{name}' has no linear components to show. Try PCA, Truncated SVD or Factor Analysis.")
        cols = self.feature_names_ or [f"feature {i + 1}" for i in range(np.asarray(comps).shape[1])]
        return pd.DataFrame(np.asarray(comps).T, index=cols,
                            columns=[f"Component {i + 1}" for i in range(np.asarray(comps).shape[0])])

    # ------------------------------------------------------------------ new data
    def export_model(self, model: Optional[str] = None) -> TrainedModel:
        """Bundles a fitted method with its preprocessing so it can transform NEW rows (default: best that can)."""
        if model is None:
            ranked = [str(n) for n in self._report_df(self.models_ or list(self._results.values()))["Model"]]
            able = [n for n in ranked if self._results[n].can_transform]
            if not able:
                raise TypeError("None of the fitted methods can transform new data.")
            model = able[0]
        name = self._name(model)
        if not self._results[name].can_transform:
            can = [n for n, r in self._results.items() if r.can_transform]
            raise TypeError(f"'{name}' cannot transform new data (it only embeds the rows it was fitted on). "
                            f"Methods that can: {can}.")
        from .. import __version__
        return TrainedModel(estimator=self.estimators_[name], name=name, task=self.task,
                            transformer=self.transformer_, columns=self.input_columns_,
                            feature_names=self.feature_names_,
                            metadata={"package_version": __version__, "n_components": self.n_components})

    def transform(self, X_new: Any, model: Optional[str] = None) -> pd.DataFrame:
        """Maps NEW rows to the components of a fitted method (PCA, Kernel PCA, FastICA, Isomap, ...)."""
        out = self.export_model(model).transform(X_new)
        return pd.DataFrame(out, columns=[f"Component {i + 1}" for i in range(out.shape[1])])

    def save_model(self, path: str, model: Optional[str] = None) -> str:
        """Saves a fitted method + its preprocessing (joblib). Reload with ``load_model(path)``."""
        return self.export_model(model).save(path)

    # ------------------------------------------------------------------ plots
    def plot_comparison(
        self,
        models: Optional[List[ReductionResult]] = None,
        save_path: Optional[str] = None,
        show_plot: bool = True,
        metric: str = "Trustworthiness",
    ):
        """Ranks all methods on one score (default 'Trustworthiness')."""
        models = self._resolve_models(models)
        if not models:
            return
        df = self._report_df(models, metric)
        with _apply_plot_style():
            fig = comparison_figure(list(df["Model"]), list(df[metric]), metric=metric,
                                    title=f"Method comparison: {metric}", lower_is_better=metric in _LOWER,
                                    footnote="Scores measure how well the original structure survives; see the report for details.")
            self._save_and_show(fig, save_path=save_path, show_plot=show_plot)

    def plot_embeddings(
        self,
        models: Optional[List[ReductionResult]] = None,
        top: int = 6,
        save_path: Optional[str] = None,
        show_plot: bool = True,
    ):
        """Scatter plots of the first two components of the best ``top`` methods. Coloured by ``y`` when given."""
        models = self._resolve_models(models)
        if not models:
            return
        order = {n: i for i, n in enumerate(self._report_df(models)["Model"])}
        models = [m for m in sorted(models, key=lambda m: order[m.name]) if m.n_components >= 2][:max(1, top)]
        if not models:
            raise ValueError("Plotting needs at least 2 components (n_components >= 2).")
        ncols = min(3, len(models))
        nrows = math.ceil(len(models) / ncols)
        with _apply_plot_style():
            fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 4.2 * nrows), squeeze=False)
            for ax, m in zip(axes.ravel(), models):
                Z = m.embedding
                if self.y_ is None:
                    ax.scatter(Z[:, 0], Z[:, 1], s=16, color="#1f5f8b", alpha=0.8, edgecolor="white", linewidth=0.3)
                elif self.y_is_class_:
                    for i, c in enumerate(pd.unique(self.y_)):
                        sel = self.y_ == c
                        ax.scatter(Z[sel, 0], Z[sel, 1], s=16, color=_PALETTE[i % len(_PALETTE)], alpha=0.85,
                                   edgecolor="white", linewidth=0.3, label=str(c))
                else:
                    ax.scatter(Z[:, 0], Z[:, 1], s=16, c=self.y_, cmap="viridis", alpha=0.85)
                tw = m.metrics.get("Trustworthiness", float("nan"))
                ax.set_title(f"{m.name}\ntrustworthiness {tw:.2f}", fontsize=11, loc="left", fontweight="bold")
                ax.set_xticks([])
                ax.set_yticks([])
                for side in ("top", "right"):
                    ax.spines[side].set_visible(False)
            for ax in axes.ravel()[len(models):]:
                ax.axis("off")
            if self.y_ is not None and self.y_is_class_:
                handles, labels = axes.ravel()[0].get_legend_handles_labels()
                fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 8), frameon=False,
                           markerscale=1.5, fontsize=9)
                fig.tight_layout(rect=(0, 0.05, 1, 1))
            else:
                fig.tight_layout()
            self._save_and_show(fig, save_path=save_path, show_plot=show_plot)

    def _report_subtitle(self) -> str:
        return (f"Dimensionality reduction - {self.n_samples_} rows, {self._values.shape[1]} features to "
                f"{self.n_components} components, {len(self._results)} methods")

    def _report_plots(self):
        plots = [("Method comparison", lambda p: self.plot_comparison(save_path=p, show_plot=False))]
        if self.n_components >= 2:
            plots.append(("Embeddings", lambda p: self.plot_embeddings(save_path=p, show_plot=False)))
        return plots

    def get_summary(
        self,
        models: Optional[List[ReductionResult]] = None,
        save_prefix: Optional[str] = None,
        show_plot: bool = True,
    ):
        """Runs the full reporting pipeline: table, ranking chart and embedding plots."""
        models = self._resolve_models(models)
        df = self.show_tabular_report(models, return_df=True)
        if save_prefix:
            if df is not None:
                self.save_report(df, f"{save_prefix}_report.csv")
            self.plot_comparison(models, save_path=f"{save_prefix}_comparison.png", show_plot=show_plot)
            if self.n_components >= 2:
                self.plot_embeddings(models, save_path=f"{save_prefix}_embeddings.png", show_plot=show_plot)
        else:
            self.plot_comparison(models, show_plot=show_plot)
            if self.n_components >= 2:
                self.plot_embeddings(models, show_plot=show_plot)
