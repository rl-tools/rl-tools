import os
import platform
import shutil
import sys

from . import _cuda
from . import _jit as jit
from ._jit._toolchain import cuda_available

BACKENDS = ("OPTIX", "METAL", "VULKAN", "WEBGPU", "GENERIC")


def resolve_auto_backend():
    # mirrors the CMake AUTO probe: OPTIX > METAL > VULKAN > WEBGPU > GENERIC
    if sys.platform != "win32" and cuda_available():
        return "OPTIX"
    if sys.platform == "darwin":
        return "METAL"
    if shutil.which("glslangValidator") is not None:
        return "VULKAN"
    if sys.platform.startswith("linux") and platform.machine() in ("x86_64", "AMD64"):
        return "WEBGPU"
    return "GENERIC"


def _backend():
    value = os.environ.get("HYPERLUX_BACKEND", "AUTO").upper()
    if value == "AUTO":
        value = resolve_auto_backend()
    if value not in BACKENDS:
        raise ValueError(
            f"hyperlux: invalid HYPERLUX_BACKEND {value} (OPTIX|METAL|VULKAN|WEBGPU|GENERIC)"
        )
    return value.lower()


def component():
    resolved = _backend()
    return jit.Component(
        name="renderer",
        variant=resolved.lower(),
        core_target="hyperlux_core",
        needs_cuda=resolved == "optix",
    )


def load_core():
    core = jit.core(component())
    # the renderer becomes a CUDA staging provider once it is actually loaded, so a package that
    # only needs dynamics never builds the renderer to stage its buffers
    _cuda.register_provider(load_core)
    return core


def load_renderer(config):
    return jit.module(component(), config)
