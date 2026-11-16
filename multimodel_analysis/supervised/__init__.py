"""Supervised learning analysers (classification and regression)."""
from .classification import CLASSIFIER_REGISTRY, MultiModelClassifier
from .regression import REGRESSOR_REGISTRY, MultiModelRegressor

__all__ = ["CLASSIFIER_REGISTRY", "REGRESSOR_REGISTRY", "MultiModelClassifier", "MultiModelRegressor"]
