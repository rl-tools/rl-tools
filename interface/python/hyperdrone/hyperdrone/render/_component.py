import os
import platform
import shutil
import sys

from .. import jit
from ..dynamics._component import _cuda_available

BACKENDS = ("OPTIX", "METAL", "VULKAN", "WEBGPU", "GENERIC")


def resolve_auto_backend():
    # mirrors the CMake AUTO probe: OPTIX > METAL > VULKAN > WEBGPU > GENERIC
    if sys.platform != "win32" and _cuda_available():
        return "OPTIX"
    if sys.platform == "darwin":
        return "METAL"
    if shutil.which("glslangValidator") is not None:
        return "VULKAN"
    if sys.platform.startswith("linux") and platform.machine() in ("x86_64", "AMD64"):
        return "WEBGPU"
    return "GENERIC"


def backend():
    value = os.environ.get("HYPERDRONE_RENDER_BACKEND", "AUTO").upper()
    if value == "AUTO":
        value = resolve_auto_backend()
    if value not in BACKENDS:
        raise jit.BuildError(
            f"hyperdrone: invalid HYPERDRONE_RENDER_BACKEND {value} (OPTIX|METAL|VULKAN|WEBGPU|GENERIC)"
        )
    return value


def component():
    resolved = backend()
    return jit.Component(
        name="render",
        variant=resolved.lower(),
        core_target="hyperdrone_render_core",
        needs_cuda=resolved == "OPTIX",
    )


def load_core():
    return jit.core(component())


def ensure_renderer_library(config):
    return jit.ensure(component(), config)
