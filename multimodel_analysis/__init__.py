# Entry level code and import for Multi-Model Analysis

from .core.persistence import TrainedModel, load_model
from .core.utils import save_report
from .multimodel import MultiModelRegressior
from .supervised import MultiModelClassifier, MultiModelRegressor
from .unsupervised import MultiModelClusterer

__version__ = '0.5.0'
__author__ = 'Uditya Narayan Tiwari'

__all__ = [
    'TrainedModel',
    'load_model',
    'MultiModelClassifier',
    'MultiModelRegressor',
    'MultiModelClusterer',
    'MultiModelRegressior',
    'save_report',
]
