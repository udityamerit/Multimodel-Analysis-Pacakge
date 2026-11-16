# Entry level code and import for Multi-Model Analysis

from .core.persistence import TrainedModel, load_model
from .core.utils import save_report
from .multimodel import MultiModelRegressior
from .supervised import MultiModelClassifier, MultiModelRegressor
from .unsupervised import MultiModelAnomalyDetector, MultiModelClusterer, MultiModelReducer

__version__ = '0.6.1'
__author__ = 'Uditya Narayan Tiwari'

__all__ = [
    'TrainedModel',
    'load_model',
    'MultiModelClassifier',
    'MultiModelRegressor',
    'MultiModelClusterer',
    'MultiModelAnomalyDetector',
    'MultiModelReducer',
    'MultiModelRegressior',
    'save_report',
]
