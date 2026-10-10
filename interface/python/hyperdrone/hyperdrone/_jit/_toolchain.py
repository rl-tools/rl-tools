import os
import shutil
from pathlib import Path


def cuda_available():
    return Path("/usr/local/cuda/bin/nvcc").exists() or shutil.which("nvcc") is not None


def environment(component):
    env = os.environ.copy()
    if component.needs_cuda and "CUDACXX" not in env and "CMAKE_CUDA_COMPILER" not in env:
        for candidate in ("/usr/local/cuda/bin/nvcc", shutil.which("nvcc")):
            if candidate and Path(candidate).exists():
                env["CUDACXX"] = str(candidate)
                break
    return env
