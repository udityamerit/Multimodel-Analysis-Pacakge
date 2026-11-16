"""Shared building blocks (base class, registry, helpers)."""
from .registry import BuildContext, ModelRegistry, ModelSpec
from .utils import save_report

__all__ = ["BuildContext", "ModelRegistry", "ModelSpec", "save_report"]
