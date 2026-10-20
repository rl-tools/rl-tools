"""hyperdrone — drone simulation stack for RLtools.

hyperdrone.HyperDrone is the RL environment: the C++ HyperDrone behind the batch verbs, compiled per hyperdrone.Config. hyperdrone.articulate writes the joints of a body/prop_* drone asset for the renderer and hyperdrone.Rotorcraft maps rotor phases onto them.

Subpackages:
  hyperdrone.render   — raytracing renderer (self-contained, no drone dynamics)
  hyperdrone.dynamics — L2F multirotor simulator (vectorized, stateful Sim; pure dynamics)
  hyperdrone.gym      — optional Gymnasium VectorEnv adapter (pip install "hyperdrone[gym]")

Render and dynamics are independent peers coupled by DLPack camera bases; HyperDrone integrates them through the C++ environment. Native components are JIT-compiled per set of compile-time constants and cached under ~/.cache/hyperdrone (override with HYPERDRONE_CACHE_DIR). Importing hyperdrone (or any subpackage) is cheap; the first build happens on first use. set_verbosity(level) selects the native log level (error|warning|info|debug, default info).
"""
import importlib

__all__ = ["Config", "HyperDrone", "Rotorcraft", "articulate", "set_verbosity", "verbosity"]

__version__ = "0.1.7"

_SUBPACKAGES = ("render", "dynamics", "gym")
_VERBOSITY_LEVELS = ("error", "warning", "info", "debug")
_verbosity = "info"
_logging_extensions = []


def set_verbosity(level):
    """Set the level for subsequent native diagnostic checks in loaded and future extensions."""
    global _verbosity
    if level not in _VERBOSITY_LEVELS:
        raise ValueError(f"hyperdrone: invalid verbosity {level!r} ({'|'.join(_VERBOSITY_LEVELS)})")
    _verbosity = level
    for extension in _logging_extensions:
        extension._set_verbosity(_VERBOSITY_LEVELS.index(level))


def verbosity():
    return _verbosity


def _register_extension(extension):
    if hasattr(extension, "_set_verbosity"):
        extension._set_verbosity(_VERBOSITY_LEVELS.index(_verbosity))
        _logging_extensions.append(extension)


def __getattr__(name):
    if name in _SUBPACKAGES:
        return importlib.import_module(f".{name}", __name__)
    if name in ("Config", "HyperDrone"):
        return getattr(importlib.import_module("._environment", __name__), name)
    if name in ("Rotorcraft", "articulate"):
        return getattr(importlib.import_module("._rotorcraft", __name__), name)
    raise AttributeError(f"module 'hyperdrone' has no attribute {name}")


def __dir__():
    return sorted(list(globals()) + list(_SUBPACKAGES) + ["Config", "HyperDrone", "Rotorcraft", "articulate"])
