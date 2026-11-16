"""Model explanations for the supervised analysers: which features matter, and how a feature changes the prediction.

Both tools work on ANY fitted model (no extra packages) and use the held-out test split, so they describe what the
model does on data it has not seen.
"""
from typing import Any, List, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from .utils import _apply_plot_style

_BAR = "#1f5f8b"
_BAR_LIGHT = "#8fb3c9"


class _ExplainMixin:
    """Needs ``X_test_scaled``, ``y_test``, ``feature_names_``, ``task``, ``default_scoring`` and ``get_model``."""

    def _explain_target(self, model: Optional[str]):
        if not self.estimators_:
            raise RuntimeError("Fit at least one model first (run_all_models / run_model).")
        name = model if model is not None else self._best_model_name()
        estimator = self.get_model(name)
        if name not in self.estimators_:
            name = self.registry.resolve(name).name
        return name, estimator

    def _explain_names(self) -> List[str]:
        X = self.X_test_scaled
        if isinstance(X, pd.DataFrame):
            return [str(c) for c in X.columns]
        return [f"x{i}" for i in range(np.asarray(X).shape[1])]

    def feature_importance(self, model: Optional[str] = None, n_repeats: int = 10, scoring: Optional[str] = None,
                           max_rows: int = 2000) -> pd.DataFrame:
        """Permutation importance: how much the test score drops when one feature is shuffled.

        Returns a DataFrame (Feature, Importance, Std) sorted from most to least important; it is also stored
        in ``importances_[model name]``. Importance is in score units (accuracy for classifiers, R2 for
        regressors, unless ``scoring`` says otherwise), so a value of 0.10 means "shuffling this feature costs
        10 points". Values near zero (or negative) mean the model does not use the feature.
        Text columns that were one-hot encoded by ``preprocess=True`` appear as separate columns.
        """
        name, estimator = self._explain_target(model)
        X, y = self.X_test_scaled, np.asarray(self.y_test)
        if len(y) > max_rows:  # keep it fast on big data
            idx = np.random.RandomState(self.random_state).choice(len(y), max_rows, replace=False)
            X = X.iloc[idx] if isinstance(X, pd.DataFrame) else np.asarray(X)[idx]
            y = y[idx]
        result = permutation_importance(
            estimator, X, y, scoring=scoring or self.default_scoring, n_repeats=n_repeats,
            random_state=self.random_state, n_jobs=1,
        )
        df = (pd.DataFrame({"Feature": self._explain_names(), "Importance": result.importances_mean,
                            "Std": result.importances_std})
              .sort_values("Importance", ascending=False).reset_index(drop=True))
        if not hasattr(self, "importances_"):
            self.importances_ = {}
        self.importances_[name] = df
        return df

    def plot_feature_importance(self, model: Optional[str] = None, top: int = 15, n_repeats: int = 10,
                                scoring: Optional[str] = None, save_path: Optional[str] = None,
                                show_plot: bool = True):
        """Horizontal bars of the permutation importance (most important feature on top, +/- 1 std as whiskers)."""
        name, _ = self._explain_target(model)
        df = self.feature_importance(name, n_repeats=n_repeats, scoring=scoring).head(top)
        df = df.iloc[::-1]  # barh draws the first row at the bottom
        with _apply_plot_style():
            fig, ax = plt.subplots(figsize=(8, max(2.5, 0.4 * len(df) + 1.2)))
            colors = [_BAR if i == len(df) - 1 else _BAR_LIGHT for i in range(len(df))]
            ax.barh(df["Feature"], df["Importance"], xerr=df["Std"], color=colors,
                    error_kw={"ecolor": "#555555", "lw": 1, "capsize": 2})
            ax.set_xlabel(f"Score drop when the feature is shuffled ({scoring or self.default_scoring})")
            ax.set_title(f"Feature importance - {name}")
            ax.grid(axis="y", visible=False)
        return self._save_and_show(fig, save_path, show_plot)

    def partial_dependence(self, feature: Union[str, int], model: Optional[str] = None, grid_points: int = 20,
                           class_label: Any = None, max_rows: int = 500) -> pd.DataFrame:
        """Average prediction as ONE feature sweeps over its range, all other columns left as they are.

        Classification uses the predicted probability of ``class_label`` (default: the last class, i.e. the
        "positive" class of a two-class problem); regression uses the predicted value. Values are on the
        scale the model sees (standardised features when ``scaled_data=True``).
        """
        name, estimator = self._explain_target(model)
        names = self._explain_names()
        if isinstance(feature, (int, np.integer)):
            col = int(feature)
        elif feature in names:
            col = names.index(feature)
        else:
            raise KeyError(f"Unknown feature '{feature}'. Choose from: {names}")

        X = self.X_test_scaled
        frame = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X), columns=names)
        if len(frame) > max_rows:
            frame = frame.sample(max_rows, random_state=self.random_state)
        column = frame.iloc[:, col].to_numpy(dtype=float)
        grid = np.unique(np.quantile(column, np.linspace(0.02, 0.98, grid_points)))

        use_proba = self.task == "classification" and hasattr(estimator, "predict_proba")
        target_idx = None
        if self.task == "classification":
            classes = list(self.classes_)
            if class_label is None:
                target_idx = len(classes) - 1
            elif class_label in classes:
                target_idx = classes.index(class_label)
            else:
                raise KeyError(f"Unknown class '{class_label}'. Classes: {classes}")

        values = []
        for g in grid:
            tmp = frame.copy()
            tmp.iloc[:, col] = g
            if use_proba:
                proba = estimator.predict_proba(tmp)
                est_classes = getattr(estimator, "classes_", np.arange(proba.shape[1]))
                pos = np.where(np.asarray(est_classes) == target_idx)[0]
                values.append(float(proba[:, pos[0]].mean()) if len(pos) else 0.0)
            elif self.task == "classification":
                values.append(float((np.asarray(estimator.predict(tmp)) == target_idx).mean()))
            else:
                values.append(float(np.mean(estimator.predict(tmp))))
        label = "Probability" if use_proba else ("Share predicted" if self.task == "classification" else "Prediction")
        return pd.DataFrame({"Feature value": grid, label: values}).assign(
            Feature=names[col], Model=name)[["Feature", "Feature value", label, "Model"]]

    def plot_partial_dependence(self, features: Union[str, int, Sequence[Union[str, int]], None] = None,
                                model: Optional[str] = None, class_label: Any = None, grid_points: int = 20,
                                save_path: Optional[str] = None, show_plot: bool = True):
        """One line per feature: how the average prediction moves as that feature changes.

        Without ``features`` the three most important features (by permutation importance) are shown.
        """
        name, _ = self._explain_target(model)
        if features is None:
            features = list(self.feature_importance(name)["Feature"].head(3))
        elif isinstance(features, (str, int, np.integer)):
            features = [features]
        features = list(features)
        curves = [self.partial_dependence(f, name, grid_points=grid_points, class_label=class_label)
                  for f in features]
        ylabel = curves[0].columns[2]
        if self.task == "classification" and class_label is None:
            class_label = list(self.classes_)[-1]
        with _apply_plot_style():
            fig, axes = plt.subplots(1, len(curves), figsize=(4.2 * len(curves), 3.6), sharey=True, squeeze=False)
            for ax, c in zip(axes[0], curves):
                ax.plot(c["Feature value"], c[ylabel], color=_BAR, lw=2)
                ax.set_xlabel(str(c["Feature"].iloc[0]))
                ax.grid(axis="x", visible=False)
            suffix = f" (class {class_label})" if self.task == "classification" else ""
            axes[0][0].set_ylabel(ylabel + suffix)
            fig.suptitle(f"Partial dependence - {name}")
            fig.tight_layout()
        return self._save_and_show(fig, save_path, show_plot)
