"""hyperdrone — drone simulation stack for RLtools, built on the HyperLux raytracer.

hyperdrone.HyperDrone is the RL environment: the C++ HyperDrone behind the batch verbs, compiled per hyperdrone.Config. hyperdrone.articulate writes the joints of a body/prop_* drone asset for the renderer and hyperdrone.Rotorcraft maps rotor phases onto them.

Subpackages:
  hyperdrone.dynamics — L2F multirotor simulator (vectorized, stateful Sim; pure dynamics)
  hyperdrone.gym      — optional Gymnasium VectorEnv adapter (pip install "hyperdrone[gym]")

Rendering is the hyperlux package: hyperlux.HyperLux renders, and HyperDrone integrates renderer and dynamics through the C++ environment. hyperlux also owns the JIT build cache (HYPERLUX_CACHE_DIR, HYPERLUX_BACKEND) and the native log level (hyperlux.set_verbosity) of every native module, hyperdrone's included. Importing hyperdrone is cheap; the first build happens on first use.
"""
import importlib

import hyperlux

__all__ = ["Config", "HyperDrone", "ImageObservation", "Rotorcraft", "articulate"]

__version__ = "0.1.8"

if hyperlux.__version__ != __version__.split(".post")[0]:
    raise ImportError(f"hyperdrone {__version__} requires hyperlux {__version__.split('.post')[0]}, found hyperlux {hyperlux.__version__}: install the matching hyperlux release")

_SUBPACKAGES = ("dynamics", "gym")


def __getattr__(name):
    if name in _SUBPACKAGES:
        return importlib.import_module(f".{name}", __name__)
    if name in ("Config", "HyperDrone"):
        return getattr(importlib.import_module("._environment", __name__), name)
    if name in ("Rotorcraft", "articulate"):
        return getattr(importlib.import_module("._rotorcraft", __name__), name)
    if name == "ImageObservation":
        return importlib.import_module(".camera", __name__).ImageObservation
    raise AttributeError(f"module 'hyperdrone' has no attribute {name}")


def __dir__():
    return sorted(list(globals()) + list(_SUBPACKAGES) + __all__)
