# Entry level code and import for Multi-Model Analysis

from .core.persistence import TrainedModel, load_model
from .core.utils import save_report
from .multimodel import MultiModelRegressior
from .supervised import MultiModelClassifier, MultiModelRegressor

__version__ = '0.2.0'
__author__ = 'Uditya Narayan Tiwari'

__all__ = [
    'TrainedModel',
    'load_model',
    'MultiModelClassifier',
    'MultiModelRegressor',
    'MultiModelRegressior',
    'save_report',
]
