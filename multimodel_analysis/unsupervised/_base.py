"""Shared set-up for the unsupervised analysers (clustering, anomaly detection, dimensionality reduction):
validate the input, scale / preprocess it once and offer a 2-D projection for plotting."""
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from ..core.mixins import _ReportingMixin
from ..core.preprocessing import build_preprocessor, to_dataframe, transform_to_frame
from ..core.registry import BuildContext
from ..core.utils import _validate_features


class _UnsupervisedBase(_ReportingMixin):
    """Needs ``X``, ``scaled_data``, ``preprocess``, ``random_state`` and ``n_jobs`` to be set before
    ``_prepare_features`` is called."""

    def _check_features(self, X, preprocess: bool, min_rows: int, purpose: str) -> int:
        if X is None:
            raise ValueError(f"Features 'X' must be provided to initialize {type(self).__name__}.")
        if np.ndim(X) != 2:
            raise ValueError("X must be 2-dimensional (rows = samples, columns = features).")
        _validate_features(X, preprocess)
        n_samples = len(X)
        if n_samples < min_rows:
            raise ValueError(f"At least {min_rows} rows are needed to {purpose}, got {n_samples}.")
        return n_samples

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

    def _project_2d(self) -> Tuple[np.ndarray, List[str]]:
        """2-D coordinates for plotting: the features themselves when there are 2, otherwise a PCA projection."""
        v = self._values
        if v.shape[1] < 2:
            raise ValueError("This plot needs at least 2 features.")
        if v.shape[1] == 2:
            names = self.feature_names_ or ["feature 1", "feature 2"]
            return v, [str(n) for n in names]
        pca = PCA(n_components=2, random_state=self.random_state).fit(v)
        ratio = pca.explained_variance_ratio_ * 100
        return pca.transform(v), [f"PC 1 ({ratio[0]:.0f}% of variance)", f"PC 2 ({ratio[1]:.0f}% of variance)"]
