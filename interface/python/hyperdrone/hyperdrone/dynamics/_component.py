import os

from .. import jit

DEVICES = ("CPU", "CUDA")


def resolve_device(device="auto"):
    value = (device or "auto").upper()
    if value == "AUTO":
        value = os.environ.get("HYPERDRONE_DYNAMICS_DEVICE", "AUTO").upper()
    if value == "AUTO":
        value = "CUDA" if _cuda_available() else "CPU"
    if value not in DEVICES:
        raise jit.BuildError(f"hyperdrone: invalid dynamics device {value} (CPU|CUDA|AUTO)")
    return value


def _cuda_available():
    from pathlib import Path
    import shutil
    return Path("/usr/local/cuda/bin/nvcc").exists() or shutil.which("nvcc") is not None


def component(device="auto"):
    resolved = resolve_device(device)
    return jit.Component(
        name="dynamics",
        variant=resolved.lower(),
        core_target="hyperdrone_dynamics_core",
        needs_cuda=resolved == "CUDA",
    )


def load_core(device="auto"):
    return jit.core(component(device))


def ensure_sim_library(config, device="auto"):
    return jit.ensure(component(device), config)
