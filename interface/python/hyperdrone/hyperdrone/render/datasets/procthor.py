"""Scene catalogs defined in RLtools headers; accessing metadata loads only the native core."""

from .._component import load_core

__all__ = ["Train200"]


def __getattr__(name):
    if name in __all__:
        return getattr(load_core().procthor, name)
    raise AttributeError(f"module 'hyperdrone.render.datasets.procthor' has no attribute {name}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
