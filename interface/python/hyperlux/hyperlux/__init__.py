"""hyperlux — the HyperLux raytracer, the rendering engine behind HyperDrone.

Scene assembly (GLB loading, procedural meshes, lights, asset pools) is runtime data and
available directly. HyperLux instances are JIT-compiled per set of compile-time constants and
cached under ~/.cache/hyperlux (override with HYPERLUX_CACHE_DIR); the backend is selected at
first build via HYPERLUX_BACKEND (OPTIX|METAL|VULKAN|WEBGPU|GENERIC; AUTO default: first
available of OptiX, Metal, Vulkan, WebGPU, Generic). Importing hyperlux is cheap; the first
build happens on first use. set_verbosity(level) controls build progress and the native log level
(error|warning|info|debug, default info) of every loaded and future native module, including
those of packages built on hyperlux's JIT such as hyperdrone.
"""

__version__ = "0.1.9"

_VERBOSITY_LEVELS = ("error", "warning", "info", "debug")
_verbosity = "info"
_logging_extensions = []


def set_verbosity(level):
    """Set build progress verbosity and the native log level in loaded and future extensions."""
    global _verbosity
    if level not in _VERBOSITY_LEVELS:
        raise ValueError(f"hyperlux: invalid verbosity {level!r} ({'|'.join(_VERBOSITY_LEVELS)})")
    _verbosity = level
    for extension in _logging_extensions:
        extension._set_verbosity(_VERBOSITY_LEVELS.index(level))


def verbosity():
    return _verbosity


def _register_extension(extension):
    if hasattr(extension, "_set_verbosity"):
        extension._set_verbosity(_VERBOSITY_LEVELS.index(_verbosity))
        _logging_extensions.append(extension)
    _cuda.register_extension(extension)


from . import _cuda
from ._outputs import FlowStatus
from . import datasets, primitives
from ._component import _backend as backend, component, load_core
from ._config import FIDELITY, Config
from ._renderer import OverlayPlacement, PlacementBatch, Pose, HyperLux
from ._sampling import FreeSpace
from ._scene import (
    compose_transforms,
    load_assembly,
    load_object,
    load_scene,
    make_camera,
    make_transform,
)

__all__ = [
    "AGENT_SEGMENTATION_CLASS",
    "AssetPool",
    "FIDELITY",
    "FlowStatus",
    "Config",
    "FreeSpace",
    "FreeSpaceParameters",
    "HyperLux",
    "Mesh",
    "Object",
    "ObjectAssembly",
    "OverlayPlacement",
    "PlacementBatch",
    "Pose",
    "Scene",
    "SceneLight",
    "backend",
    "compose_transforms",
    "datasets",
    "load_assembly",
    "load_object",
    "load_scene",
    "make_camera",
    "make_transform",
    "primitives",
    "set_verbosity",
    "verbosity",
]

def __getattr__(name):
    # bound classes come from the lazily-built core module so that importing hyperlux stays
    # cheap and the first compile happens on first use
    if name in ("Scene", "Object", "ObjectAssembly", "AssetPool", "Mesh", "SceneLight", "FreeSpaceParameters", "AGENT_SEGMENTATION_CLASS"):
        return getattr(load_core(), name)
    raise AttributeError(f"module 'hyperlux' has no attribute {name}")
