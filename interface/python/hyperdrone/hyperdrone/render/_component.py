import os
import sys

from .. import jit

BACKENDS = ("OPTIX", "METAL", "VULKAN", "GENERIC")


def backend():
    value = os.environ.get("HYPERDRONE_RENDER_BACKEND", "AUTO").upper()
    if value == "AUTO":
        value = "METAL" if sys.platform == "darwin" else "OPTIX"
    if value not in BACKENDS:
        raise jit.BuildError(
            f"hyperdrone: invalid HYPERDRONE_RENDER_BACKEND {value} (OPTIX|METAL|VULKAN|GENERIC)"
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
