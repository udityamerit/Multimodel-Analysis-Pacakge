"""Shared helpers: safe printing, plot style, input validation and report saving."""
import contextlib
import os
from typing import Any, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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


def _validate_inputs(X: Any, y: Any, test_size: Union[int, float], preprocess: bool = False) -> np.ndarray:
    """Validates features / target / test_size and returns the flattened target array.

    With ``preprocess=True`` missing values and text columns in ``X`` are allowed (they get imputed / encoded).

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

    # --- features: numeric, finite (skipped when automatic preprocessing will handle them) ---
    if preprocess:
        if isinstance(X, pd.DataFrame):
            numeric = X.select_dtypes(include="number")
            n_inf = int(np.isinf(numeric.to_numpy(dtype=float)).sum()) if numeric.shape[1] else 0
            if n_inf:
                raise ValueError(f"X contains {n_inf} infinite value(s). Replace or drop them first.")
        return y_flat

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
