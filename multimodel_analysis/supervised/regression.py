"""Supervised regression: ``MultiModelRegressor``."""
from typing import Any, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from ..core.base import _BaseMultiModel
from ..core.registry import ModelRegistry, ModelSpec
from ..core.utils import _LOWER_IS_BETTER, _apply_plot_style, _safe_print, save_report


def _knn_reg(ctx, n_neighbors=None, **kw):
    n = n_neighbors if n_neighbors is not None else min(5, max(1, ctx.n_train - 1))
    return KNeighborsRegressor(n_neighbors=n, n_jobs=ctx.n_jobs, **kw)


def _svr(ctx, **p):
    # SVR is sensitive to the target scale, so the target is standardised internally.
    return TransformedTargetRegressor(regressor=SVR(**p), transformer=StandardScaler())


REGRESSOR_REGISTRY = ModelRegistry([
    ModelSpec("linear_regression", "Linear Regression", lambda ctx, **p: LinearRegression(n_jobs=ctx.n_jobs, **p)),
    ModelSpec("lasso", "Lasso Regression",
              lambda ctx, **p: Lasso(random_state=ctx.random_state, **p), defaults={"alpha": 0.1}),
    ModelSpec("ridge", "Ridge Regression",
              lambda ctx, **p: Ridge(random_state=ctx.random_state, **p), defaults={"alpha": 1.0}),
    ModelSpec("svr", "SVR", _svr, defaults={"kernel": "rbf"}),
    ModelSpec("knn", "KNN Regressor", _knn_reg, defaults={"n_neighbors": None}),
    ModelSpec("decision_tree", "Decision Tree Regressor",
              lambda ctx, **p: DecisionTreeRegressor(random_state=ctx.random_state, **p)),
    ModelSpec("random_forest", "Random Forest Regressor",
              lambda ctx, **p: RandomForestRegressor(random_state=ctx.random_state, n_jobs=ctx.n_jobs, **p),
              defaults={"n_estimators": 100}),
    ModelSpec("gradient_boosting", "Gradient Boosting Regressor",
              lambda ctx, **p: GradientBoostingRegressor(random_state=ctx.random_state, **p),
              defaults={"n_estimators": 100}),
    ModelSpec("adaboost", "AdaBoost Regressor",
              lambda ctx, **p: AdaBoostRegressor(random_state=ctx.random_state, **p),
              defaults={"n_estimators": 50}),
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
