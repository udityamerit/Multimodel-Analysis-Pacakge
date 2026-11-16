"""Automatic preprocessing: impute missing values, one-hot encode text columns, scale numbers.

The transformer is always fitted on the TRAINING split only, so nothing leaks from the test split.
"""
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def to_dataframe(X: Any, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Returns ``X`` as a DataFrame (numpy arrays get ``columns`` or ``x0, x1, ...`` names)."""
    if isinstance(X, pd.DataFrame):
        return X
    arr = np.asarray(X, dtype=object) if not isinstance(X, np.ndarray) else X
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if columns is None or len(columns) != arr.shape[1]:
        columns = [f"x{i}" for i in range(arr.shape[1])]
    return pd.DataFrame(arr, columns=columns).infer_objects()


def split_columns(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Splits column names into (numeric, categorical)."""
    numeric = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c]) and not pd.api.types.is_bool_dtype(X[c])]
    categorical = [c for c in X.columns if c not in numeric]
    return numeric, categorical


def build_preprocessor(X: pd.DataFrame, scale: bool = True) -> ColumnTransformer:
    """Builds an UNFITTED ``ColumnTransformer``.

    numeric columns      -> median imputation (+ standardisation when ``scale``)
    other columns        -> most-frequent imputation + one-hot encoding (unseen categories are ignored)
    """
    numeric, categorical = split_columns(X)

    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))

    transformers = []
    if numeric:
        transformers.append(("num", Pipeline(num_steps), numeric))
    if categorical:
        cat = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("cat", cat, categorical))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def transform_to_frame(transformer: ColumnTransformer, X: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
    """Runs ``fit_transform`` / ``transform`` and returns a DataFrame with readable column names."""
    values = transformer.fit_transform(X) if fit else transformer.transform(X)
    names = list(transformer.get_feature_names_out())
    return pd.DataFrame(np.asarray(values, dtype=float), columns=names, index=X.index)
