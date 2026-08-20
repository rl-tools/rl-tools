"""hyperdrone.render — Python interface to the RLtools raytracing renderer.

Self-contained: no dependency on drone dynamics. Scene assembly (GLB loading, procedural
meshes, lights, asset pools) is runtime data and available directly. Renderer
instantiations are JIT-compiled per set of compile-time constants and cached; the backend
is selected at first build via HYPERDRONE_RENDER_BACKEND (OPTIX|METAL|VULKAN|WEBGPU|GENERIC;
AUTO default: Metal on macOS, OptiX elsewhere).
"""
from .. import cuda as _cuda
from ._component import backend, component, load_core
from ._config import FIDELITY, OUTPUT_MODE, RendererConfig
from ._renderer import Renderer
from ._scene import (
    compose_transforms,
    load_assembly,
    load_object,
    load_scene,
    make_camera,
    make_transform,
)

__all__ = [
    "AssetPool",
    "FIDELITY",
    "Mesh",
    "Object",
    "ObjectAssembly",
    "Renderer",
    "RendererConfig",
    "Scene",
    "SceneLight",
    "backend",
    "compose_transforms",
    "load_assembly",
    "load_object",
    "load_scene",
    "make_camera",
    "make_transform",
]

_cuda.register_provider(load_core)


def __getattr__(name):
    # bound classes come from the lazily-built core module so that importing
    # hyperdrone.render stays cheap and the first compile happens on first use
    if name in ("Scene", "Object", "ObjectAssembly", "AssetPool", "Mesh", "SceneLight", "JitRenderer"):
        return getattr(load_core(), name)
    raise AttributeError(f"module 'hyperdrone.render' has no attribute {name}")
