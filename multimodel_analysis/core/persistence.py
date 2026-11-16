"""Saving and loading a trained model together with everything needed to predict on new data."""
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


class TrainedModel:
    """A fitted estimator bundled with its preprocessing (scaler / encoder / label decoder).

    Create one with ``analyser.save_model(...)`` or ``analyser.export_model(...)`` and reload it
    later with :func:`load_model`. SECURITY: files are Python pickles (via joblib) - only load
    files you created or trust.
    """

    def __init__(
        self,
        estimator: Any,
        name: str,
        task: str,
        transformer: Any = None,
        label_encoder: Any = None,
        columns: Optional[List[str]] = None,
        feature_names: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.estimator = estimator
        self.name = name
        self.task = task  # 'classification' | 'regression' | 'clustering' | ...
        self.transformer = transformer  # fitted StandardScaler / ColumnTransformer / None
        self.label_encoder = label_encoder
        self.columns = columns  # raw input columns expected by predict()
        self.feature_names = feature_names  # column names of the matrix the estimator was fitted on (or None)
        self.metadata = metadata or {}

    # ------------------------------------------------------------------ prediction
    def _prepare(self, X: Any):
        from .preprocessing import to_dataframe

        if self.columns is not None and not isinstance(X, pd.DataFrame):
            X = to_dataframe(X, self.columns)
        if self.columns is not None and isinstance(X, pd.DataFrame):
            missing = [c for c in self.columns if c not in X.columns]
            if missing:
                raise ValueError(f"New data is missing column(s) the model was trained on: {missing[:10]}")
            X = X[self.columns]
        if self.transformer is None:
            return X
        out = self.transformer.transform(X)
        if self.feature_names is not None:  # the estimator was trained on a DataFrame
            return pd.DataFrame(np.asarray(out, dtype=float), columns=self.feature_names)
        return out

    def predict(self, X: Any) -> np.ndarray:
        """Predicts for new rows. Classification returns the ORIGINAL labels (strings stay strings)."""
        pred = self.estimator.predict(self._prepare(X))
        if self.label_encoder is not None:
            return self.label_encoder.inverse_transform(np.asarray(pred, dtype=int))
        return np.asarray(pred)

    def transform(self, X: Any) -> np.ndarray:
        """Maps new rows to the new feature space (dimensionality-reduction models)."""
        if not hasattr(self.estimator, "transform"):
            raise TypeError(f"'{self.name}' has no transform method.")
        return np.asarray(self.estimator.transform(self._prepare(X)))

    def predict_proba(self, X: Any) -> pd.DataFrame:
        """Class probabilities as a DataFrame with one column per class (classification only)."""
        if self.task != "classification":
            raise AttributeError("predict_proba is only available for classification models.")
        if not hasattr(self.estimator, "predict_proba"):
            raise AttributeError(f"'{self.name}' does not provide probabilities.")
        proba = self.estimator.predict_proba(self._prepare(X))
        classes = self.label_encoder.inverse_transform(np.asarray(self.estimator.classes_, dtype=int))
        return pd.DataFrame(proba, columns=[str(c) for c in classes])

    # ------------------------------------------------------------------ persistence
    def save(self, path: str) -> str:
        import joblib

        joblib.dump(self, path)
        return path

    def __repr__(self) -> str:
        return f"TrainedModel(name={self.name!r}, task={self.task!r}, estimator={type(self.estimator).__name__})"


def load_model(path: str) -> TrainedModel:
    """Loads a file written by ``save_model``. Only load files you trust (pickle-based)."""
    import joblib

    obj = joblib.load(path)
    if not isinstance(obj, TrainedModel):
        raise ValueError(f"'{path}' does not contain a multimodel_analysis TrainedModel.")
    return obj
