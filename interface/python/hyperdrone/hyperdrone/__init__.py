"""hyperdrone — drone simulation stack for RLtools.

Subpackages:
  hyperdrone.render   — raytracing renderer (self-contained, no drone dynamics)
  hyperdrone.dynamics — L2F multirotor simulator (vectorized, stateful Sim; pure dynamics)
  hyperdrone.env      — the RL environment: the C++ MultiEnvironment behind the batch verbs
  hyperdrone.gym      — optional Gymnasium VectorEnv adapter (pip install "hyperdrone[gym]")
  hyperdrone.jit      — shared compile-and-cache infrastructure
  hyperdrone.cuda     — CUDA staging helpers (DLPack tensor sets)

The three domain packages are independent: render and dynamics couple only through a
DLPack tensor of camera bases; env owns no Python-side semantics — it drives the C++
environment. Native components are JIT-compiled per set of compile-time constants and
cached under ~/.cache/hyperdrone (override with HYPERDRONE_CACHE_DIR). Importing
hyperdrone (or any subpackage) is cheap; the first build happens on first use.
"""
import importlib

__version__ = "1.0.0"

_SUBPACKAGES = ("render", "dynamics", "env", "gym", "jit", "cuda")


def __getattr__(name):
    if name in _SUBPACKAGES:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module 'hyperdrone' has no attribute {name}")


def __dir__():
    return sorted(list(globals()) + list(_SUBPACKAGES))
