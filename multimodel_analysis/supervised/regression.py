"""Supervised regression: ``MultiModelRegressor``."""
from typing import Any, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import BayesianRidge, ElasticNet, HuberRegressor, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR, LinearSVR
from sklearn.tree import DecisionTreeRegressor

from ..core.base import _BaseMultiModel
from ..core.plotting import comparison_figure
from ..core.registry import ModelRegistry, ModelSpec
from ..core.utils import _LOWER_IS_BETTER, _apply_plot_style, _safe_print, save_report


def _knn_reg(ctx, n_neighbors=None, **kw):
    n = n_neighbors if n_neighbors is not None else min(5, max(1, ctx.n_train - 1))
    return KNeighborsRegressor(n_neighbors=n, n_jobs=ctx.n_jobs, **kw)


def _target_scaled(estimator):
    """Wraps ``estimator`` so the target is standardised internally (needed by SVR, LinearSVR and MLP)."""
    return TransformedTargetRegressor(regressor=estimator, transformer=StandardScaler())


def _svr(ctx, **p):
    # SVR is sensitive to the target scale, so the target is standardised internally.
    return _target_scaled(SVR(**p))


def _xgb_reg(ctx, **p):
    from xgboost import XGBRegressor
    return XGBRegressor(random_state=ctx.random_state, n_jobs=ctx.n_jobs, verbosity=0, **p)


def _lgbm_reg(ctx, **p):
    from lightgbm import LGBMRegressor
    return LGBMRegressor(random_state=ctx.random_state, n_jobs=ctx.n_jobs, verbose=-1, **p)


def _cat_reg(ctx, **p):
    from catboost import CatBoostRegressor
    return CatBoostRegressor(random_seed=ctx.random_state, thread_count=ctx.n_jobs, verbose=0,
                             allow_writing_files=False, **p)


REGRESSOR_REGISTRY = ModelRegistry([
    ModelSpec("linear_regression", "Linear Regression", lambda ctx, **p: LinearRegression(n_jobs=ctx.n_jobs, **p),
              search_space={"fit_intercept": [True, False]}),
    ModelSpec("lasso", "Lasso Regression",
              lambda ctx, **p: Lasso(random_state=ctx.random_state, **p), defaults={"alpha": 0.1},
              search_space={"alpha": [0.001, 0.01, 0.1, 1, 10]}),
    ModelSpec("ridge", "Ridge Regression",
              lambda ctx, **p: Ridge(random_state=ctx.random_state, **p), defaults={"alpha": 1.0},
              search_space={"alpha": [0.01, 0.1, 1, 10, 100]}),
    ModelSpec("svr", "SVR", _svr, defaults={"kernel": "rbf"}, param_prefix="regressor__",
              search_space={"C": [0.1, 1, 10, 100], "epsilon": [0.01, 0.1, 0.5], "gamma": ["scale", "auto"]}),
    ModelSpec("knn", "KNN Regressor", _knn_reg, defaults={"n_neighbors": None},
              search_space={"n_neighbors": [3, 5, 7, 9, 11, 15], "weights": ["uniform", "distance"], "p": [1, 2]}),
    ModelSpec("decision_tree", "Decision Tree Regressor",
              lambda ctx, **p: DecisionTreeRegressor(random_state=ctx.random_state, **p),
              search_space={"max_depth": [None, 3, 5, 8, 12], "min_samples_split": [2, 5, 10],
                            "min_samples_leaf": [1, 2, 4]}),
    ModelSpec("random_forest", "Random Forest Regressor",
              lambda ctx, **p: RandomForestRegressor(random_state=ctx.random_state, n_jobs=ctx.n_jobs, **p),
              defaults={"n_estimators": 100},
              search_space={"n_estimators": [100, 200, 300], "max_depth": [None, 5, 10, 20],
                            "min_samples_split": [2, 5, 10], "min_samples_leaf": [1, 2, 4],
                            "max_features": [1.0, "sqrt", "log2"]}),
    ModelSpec("gradient_boosting", "Gradient Boosting Regressor",
              lambda ctx, **p: GradientBoostingRegressor(random_state=ctx.random_state, **p),
              defaults={"n_estimators": 100},
              search_space={"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.05, 0.1, 0.2],
                            "max_depth": [2, 3, 4, 5], "subsample": [0.7, 0.85, 1.0]}),
    ModelSpec("adaboost", "AdaBoost Regressor",
              lambda ctx, **p: AdaBoostRegressor(random_state=ctx.random_state, **p),
              defaults={"n_estimators": 50},
              search_space={"n_estimators": [25, 50, 100, 200], "learning_rate": [0.01, 0.1, 0.5, 1.0]}),

    # ---- extra models (Phase 5): run with include=[...] or include='all' ------------------------
    ModelSpec("elastic_net", "Elastic Net",
              lambda ctx, **p: ElasticNet(random_state=ctx.random_state, **p),
              defaults={"alpha": 0.1, "l1_ratio": 0.5}, default=False,
              search_space={"alpha": [0.001, 0.01, 0.1, 1, 10], "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]}),
    ModelSpec("bayesian_ridge", "Bayesian Ridge", lambda ctx, **p: BayesianRidge(**p), default=False,
              search_space={"alpha_1": [1e-7, 1e-6, 1e-5], "alpha_2": [1e-7, 1e-6, 1e-5],
                            "lambda_1": [1e-7, 1e-6, 1e-5], "lambda_2": [1e-7, 1e-6, 1e-5]}),
    ModelSpec("huber", "Huber Regressor", lambda ctx, **p: HuberRegressor(**p),
              defaults={"max_iter": 500}, default=False,
              search_space={"epsilon": [1.1, 1.35, 1.5, 2.0], "alpha": [1e-5, 1e-4, 1e-3, 1e-2]}),
    ModelSpec("linear_svr", "Linear SVR",
              lambda ctx, **p: _target_scaled(LinearSVR(random_state=ctx.random_state, **p)),
              defaults={"max_iter": 5000, "dual": "auto"}, param_prefix="regressor__", default=False,
              search_space={"C": [0.1, 1, 10, 100], "epsilon": [0.0, 0.1, 0.5]}),
    ModelSpec("extra_trees", "Extra Trees Regressor",
              lambda ctx, **p: ExtraTreesRegressor(random_state=ctx.random_state, n_jobs=ctx.n_jobs, **p),
              defaults={"n_estimators": 200}, default=False,
              search_space={"n_estimators": [100, 200, 300], "max_depth": [None, 5, 10, 20],
                            "min_samples_split": [2, 5, 10], "min_samples_leaf": [1, 2, 4]}),
    ModelSpec("hist_gradient_boosting", "Hist Gradient Boosting Regressor",
              lambda ctx, **p: HistGradientBoostingRegressor(random_state=ctx.random_state, **p), default=False,
              search_space={"learning_rate": [0.03, 0.1, 0.2], "max_iter": [100, 200, 300],
                            "max_depth": [None, 3, 6], "l2_regularization": [0.0, 0.1, 1.0]}),
    ModelSpec("mlp", "Neural Network Regressor (MLP)",
              lambda ctx, **p: _target_scaled(MLPRegressor(random_state=ctx.random_state, **p)),
              defaults={"hidden_layer_sizes": (64,), "max_iter": 500}, param_prefix="regressor__", default=False,
              search_space={"hidden_layer_sizes": [(32,), (64,), (64, 32)], "alpha": [1e-5, 1e-4, 1e-3, 1e-2],
                            "learning_rate_init": [1e-3, 1e-2]}),

    # ---- optional boosters: pip install xgboost / lightgbm / catboost -----------------------------
    ModelSpec("xgboost", "XGBoost Regressor", _xgb_reg, requires="xgboost", default=False,
              defaults={"n_estimators": 200, "learning_rate": 0.1},
              search_space={"n_estimators": [100, 200, 400], "learning_rate": [0.03, 0.1, 0.2],
                            "max_depth": [3, 4, 6, 8], "subsample": [0.7, 0.85, 1.0],
                            "colsample_bytree": [0.7, 0.85, 1.0]}),
    ModelSpec("lightgbm", "LightGBM Regressor", _lgbm_reg, requires="lightgbm", default=False,
              defaults={"n_estimators": 200, "learning_rate": 0.1},
              search_space={"n_estimators": [100, 200, 400], "learning_rate": [0.03, 0.1, 0.2],
                            "num_leaves": [15, 31, 63], "subsample": [0.7, 0.85, 1.0],
                            "colsample_bytree": [0.7, 0.85, 1.0]}),
    ModelSpec("catboost", "CatBoost Regressor", _cat_reg, requires="catboost", default=False,
              defaults={"iterations": 300, "learning_rate": 0.1},
              search_space={"iterations": [200, 300, 500], "learning_rate": [0.03, 0.1, 0.2],
                            "depth": [4, 6, 8]}),
])


class MultiModelRegressor(_BaseMultiModel):
    registry = REGRESSOR_REGISTRY
    task = "regression"
    default_scoring = "r2"
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
    preprocess : automatically impute missing values and one-hot encode text columns (fit on the
        training split only). Default ``False`` keeps the strict "numeric, no NaN" input check.
    """

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
        super().__init__(
            X=X, y=y, test_size=test_size, scaled_data=scaled_data,
            random_state=random_state, n_jobs=n_jobs, verbose=verbose, preprocess=preprocess
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

    # Each wrapper keeps its historical signature; they all delegate to the registry (see run_model()).
    def LinearRegression_model(self, **kwargs):
        return self._run_spec("linear_regression", **kwargs)

    def Lasso_model(self, alpha: float = 0.1, random_state: Optional[int] = None, **kwargs):
        return self._run_spec("lasso", random_state=random_state, alpha=alpha, **kwargs)

    def Ridge_model(self, alpha: float = 1.0, random_state: Optional[int] = None, **kwargs):
        return self._run_spec("ridge", random_state=random_state, alpha=alpha, **kwargs)

    def SVR_model(self, kernel: str = "rbf", **kwargs):
        return self._run_spec("svr", kernel=kernel, **kwargs)

    def KNN_model(self, n_neighbors: Optional[int] = None, **kwargs):
        return self._run_spec("knn", n_neighbors=n_neighbors, **kwargs)

    def DecisionTree_model(self, random_state: Optional[int] = None, **kwargs):
        return self._run_spec("decision_tree", random_state=random_state, **kwargs)

    def RandomForest_model(self, n_estimators: int = 100, random_state: Optional[int] = None, **kwargs):
        return self._run_spec("random_forest", random_state=random_state, n_estimators=n_estimators, **kwargs)

    def GradientBoosting_model(self, n_estimators: int = 100, random_state: Optional[int] = None, **kwargs):
        return self._run_spec("gradient_boosting", random_state=random_state, n_estimators=n_estimators, **kwargs)

    def AdaBoost_model(self, n_estimators: int = 50, random_state: Optional[int] = None, **kwargs):
        return self._run_spec("adaboost", random_state=random_state, n_estimators=n_estimators, **kwargs)

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
                'R2 Score': r2,
                'Fit Time (s)': self.fit_times_.get(name, np.nan)
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
        show_plot: bool = True,
        metric: str = 'R2 Score'
    ):
        """Plots all models ranked on one metric (default 'R2 Score'; or 'MAE', 'MSE', 'RMSE')."""
        models = self._resolve_models(models)
        if not models:
            return

        columns = {'R2 Score': 4, 'MAE': 1, 'MSE': 2, 'RMSE': 3}
        if metric not in columns:
            raise ValueError(f"metric must be one of {list(columns)}, got '{metric}'.")

        with _apply_plot_style():
            fig = comparison_figure(
                [m[0] for m in models], [m[columns[metric]] for m in models], metric=metric,
                title=f"Model comparison: {metric}", lower_is_better=metric in _LOWER_IS_BETTER,
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
            self.plot_true_vs_predicted(models, save_path=f"{save_prefix}_true_vs_pred.png", show_plot=show_plot)
            self.plot_comparison(models, save_path=f"{save_prefix}_comparison.png", show_plot=show_plot)
        else:
            self.plot_true_vs_predicted(models, show_plot=show_plot)
            self.plot_comparison(models, show_plot=show_plot)
