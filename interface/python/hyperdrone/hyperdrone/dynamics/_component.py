from .. import _jit as jit
from .._jit._toolchain import cuda_available

DEVICES = ("cpu", "cuda")


def _resolve_device(device):
    value = str(device).lower()
    if value == "auto":
        value = "cuda" if cuda_available() else "cpu"
    if value not in DEVICES:
        raise ValueError(f"hyperdrone: invalid dynamics device {value} (CPU|CUDA|AUTO)")
    return value


def component(device):
    resolved = _resolve_device(device)
    return jit.Component(
        name="dynamics",
        variant=resolved.lower(),
        core_target="hyperdrone_dynamics_core",
        needs_cuda=resolved == "cuda",
    )


def load_core(device):
    return jit.core(component(device))


def ensure_sim_library(config, device):
    return jit.ensure(component(device), config)
