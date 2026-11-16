"""Backwards-compatibility module.

The implementation now lives in ``multimodel_analysis.core`` and ``multimodel_analysis.supervised``.
Imports such as ``from multimodel_analysis.multimodel import MultiModelClassifier`` keep working.
"""
from .core.base import _BaseMultiModel
from .core.utils import _apply_plot_style, _safe_print, _validate_inputs, save_report
from .supervised.classification import MultiModelClassifier
from .supervised.regression import MultiModelRegressor

# Historical misspelling of the class name, kept for old code.
MultiModelRegressior = MultiModelRegressor

__all__ = [
    "MultiModelClassifier",
    "MultiModelRegressor",
    "MultiModelRegressior",
    "save_report",
    "_BaseMultiModel",
    "_apply_plot_style",
    "_safe_print",
    "_validate_inputs",
]


if __name__ == "__main__":
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
