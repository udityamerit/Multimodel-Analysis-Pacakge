"""Model registry: one ``ModelSpec`` per built-in model so adding a model is a one-line change."""
import importlib.util
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional


@dataclass(frozen=True)
class BuildContext:
    """Information a model factory may need that is not a hyper-parameter."""
    random_state: Optional[int]
    n_jobs: Optional[int]
    n_train: int
    class_weight: Optional[str] = None


@dataclass(frozen=True)
class ModelSpec:
    """Describes how to build one estimator.

    key           machine-friendly id, e.g. ``'random_forest'``.
    name          display name used in reports, e.g. ``'Random Forest'``.
    factory       ``factory(ctx, **params) -> unfitted estimator``.
    defaults      default hyper-parameters (user ``**params`` override them).
    search_space  ``{param: [candidate values]}`` used by ``tune_models`` (may be empty).
    param_prefix  prefix needed to reach the estimator's parameters when the factory wraps it
                  (e.g. ``'regressor__'`` for a TransformedTargetRegressor); keys in ``search_space``
                  and ``**params`` stay un-prefixed.
    requires      optional third-party package needed, if any (e.g. ``'xgboost'``).
    default       ``True`` = part of the standard ``run_all_models()`` set; ``False`` = extra model that
                  runs only when asked for with ``include=[...]`` or ``include='all'``.
    """
    key: str
    name: str
    factory: Callable[..., Any]
    defaults: Dict[str, Any] = field(default_factory=dict)
    search_space: Dict[str, Any] = field(default_factory=dict)
    requires: Optional[str] = None
    param_prefix: str = ""
    default: bool = True

    @property
    def available(self) -> bool:
        """``False`` when the optional package this model needs is not installed."""
        return self.requires is None or importlib.util.find_spec(self.requires) is not None

    def build(self, ctx: BuildContext, **params):
        if not self.available:
            raise ImportError(
                f"'{self.name}' needs the optional package '{self.requires}'. "
                f"Install it with: pip install {self.requires}"
            )
        return self.factory(ctx, **{**self.defaults, **params})


def _norm(text: str) -> str:
    return str(text).strip().lower().replace(" ", "_").replace("-", "_")


class ModelRegistry:
    """Ordered collection of ``ModelSpec`` objects, looked up by key or display name."""

    def __init__(self, specs: Iterable[ModelSpec] = ()):
        self._specs: Dict[str, ModelSpec] = {}
        for spec in specs:
            self.register(spec)

    def register(self, spec: ModelSpec) -> None:
        self._specs[spec.key] = spec

    def __iter__(self):
        return iter(self._specs.values())

    def __len__(self):
        return len(self._specs)

    def __contains__(self, item) -> bool:
        try:
            self.resolve(item)
            return True
        except KeyError:
            return False

    def keys(self) -> List[str]:
        return list(self._specs)

    def names(self) -> List[str]:
        return [s.name for s in self._specs.values()]

    def resolve(self, ref) -> ModelSpec:
        """Finds a spec from a ``ModelSpec``, key (``'random_forest'``) or display name (``'Random Forest'``)."""
        if isinstance(ref, ModelSpec):
            return ref
        wanted = _norm(ref)
        for spec in self._specs.values():
            if wanted in (_norm(spec.key), _norm(spec.name)):
                return spec
        raise KeyError(f"Unknown model '{ref}'. Available models: {self.keys()}")

    def select(self, include=None, exclude: Optional[Iterable] = None) -> List[ModelSpec]:
        """Returns specs filtered by ``include`` and ``exclude``, keeping registry order.

        ``include=None``   the standard set (``default=True`` models).
        ``include='all'``  every model whose optional package is installed.
        ``include=[...]``  exactly these models (a missing optional package is reported when it runs).
        """
        specs = list(self._specs.values())
        if include is None:
            specs = [s for s in specs if s.default]
        elif isinstance(include, str) and _norm(include) == "all":
            specs = [s for s in specs if s.available]
        else:
            if isinstance(include, (str, ModelSpec)):
                include = [include]
            wanted = [self.resolve(r).key for r in include]
            specs = [s for s in specs if s.key in wanted]
        if exclude is not None:
            dropped = {self.resolve(r).key for r in exclude}
            specs = [s for s in specs if s.key not in dropped]
        return specs
