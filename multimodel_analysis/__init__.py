# Entry level code and import for Multi-Model Analysis

from .core.utils import save_report
from .multimodel import MultiModelRegressior
from .supervised import MultiModelClassifier, MultiModelRegressor

__version__ = '0.2.0.dev0'
__author__ = 'Uditya Narayan Tiwari'

__all__ = [
    'MultiModelClassifier',
    'MultiModelRegressor',
    'MultiModelRegressior',
    'save_report',
]
