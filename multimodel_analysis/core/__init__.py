"""Shared building blocks (base class, registry, helpers)."""
from .persistence import TrainedModel, load_model
from .registry import BuildContext, ModelRegistry, ModelSpec
from .utils import save_report

__all__ = ["BuildContext", "ModelRegistry", "ModelSpec", "TrainedModel", "load_model", "save_report"]
