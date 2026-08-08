"""hypert — Hyper Ray Tracer: Python interface to the RLtools raytracing renderer.

Scene assembly (GLB loading, procedural meshes, lights, asset pools) is runtime data and
available directly. Renderer instantiations are JIT-compiled per set of compile-time
constants and cached under ~/.cache/hypert (override with HYPERT_CACHE_DIR). The backend
is selected at first build via HYPERT_BACKEND (OPTIX|METAL|VULKAN|GENERIC; AUTO default).
"""
import math

from ._build import backend, build_dir, load_core
from ._config import OUTPUT_MODE, SHADING, RendererConfig
from ._cuda import CudaTensorSet, cuda_upload
from ._renderer import Renderer

__all__ = [
    "AssetPool",
    "Mesh",
    "Object",
    "ObjectAssembly",
    "Renderer",
    "RendererConfig",
    "Scene",
    "SceneLight",
    "CudaTensorSet",
    "backend",
    "build_dir",
    "compose_transforms",
    "cuda_upload",
    "load_assembly",
    "load_object",
    "load_scene",
    "make_camera",
    "make_transform",
]


def _shading_id(shading):
    return SHADING[shading.lower()] if isinstance(shading, str) else int(shading)


def __getattr__(name):
    # bound classes come from the lazily-built core module so that importing hypert stays
    # cheap and the first compile happens on first use
    if name in ("Scene", "Object", "ObjectAssembly", "AssetPool", "Mesh", "SceneLight", "JitRenderer"):
        return getattr(load_core(), name)
    raise AttributeError(f"module 'hypert' has no attribute {name}")


def load_scene(path, shading="high", rgb=True):
    scene = load_core().Scene()
    scene.load(str(path), _shading_id(shading), bool(rgb))
    return scene


def load_object(path, shading="high", rgb=True):
    return load_core().load_object(str(path), _shading_id(shading), bool(rgb))


def load_assembly(path, shading="high", rgb=True):
    return load_core().load_assembly(str(path), _shading_id(shading), bool(rgb))


def make_camera(position, look_at, up=(0.0, 0.0, 1.0), fov=math.radians(60.0), aspect=1.0):
    return load_core().make_camera(tuple(position), tuple(look_at), tuple(up), float(fov), float(aspect))


def make_transform(position=(0.0, 0.0, 0.0), orientation_wxyz=(1.0, 0.0, 0.0, 0.0)):
    return load_core().make_transform(tuple(position), tuple(orientation_wxyz))


def compose_transforms(a, b):
    return load_core().compose_transforms(a, b)
