import hashlib
import os
import platform
import subprocess
import sys
import sysconfig
from pathlib import Path


class BuildError(RuntimeError):
    pass


def cache_root():
    env = os.environ.get("HYPERDRONE_CACHE_DIR")
    if env:
        return Path(env)
    return Path.home() / ".cache" / "hyperdrone"


def package_root():
    return Path(__file__).resolve().parents[1]


def native_root():
    return package_root() / "_native"


def source_root():
    """rl-tools source root: env override, enclosing repository checkout, vendored tree."""
    env = os.environ.get("HYPERDRONE_RLTOOLS_ROOT")
    if env:
        root = Path(env).resolve()
        if not (root / "include" / "rl_tools").is_dir():
            raise BuildError(
                f"hyperdrone: HYPERDRONE_RLTOOLS_ROOT={root} is not an rl-tools source root "
                "(include/rl_tools missing)"
            )
        return root
    candidate = package_root()
    for _ in range(8):
        if (candidate / "include" / "rl_tools").is_dir():
            return candidate
        candidate = candidate.parent
    vendored = package_root() / "_vendor" / "rl-tools"
    if (vendored / "include" / "rl_tools").is_dir():
        return vendored
    raise BuildError(
        "hyperdrone: rl-tools sources not found (no enclosing repository checkout and no "
        "vendored tree; set HYPERDRONE_RLTOOLS_ROOT to an rl-tools source root)"
    )


def dependencies_root():
    """FetchContent sources and build state live under the configured cache root.

    Source checkouts and installed packages may be read-only; neither should accumulate
    generated files. Keeping dependencies beside the component build trees also makes
    HYPERDRONE_CACHE_DIR a complete, relocatable build cache.
    """
    return cache_root() / ".dependencies"


def host_target():
    try:
        with open("/proc/cpuinfo") as cpuinfo:
            for line in cpuinfo:
                name, separator, value = line.partition(":")
                if separator and name.strip() in ("flags", "Features"):
                    return platform.machine() + "|" + " ".join(sorted(value.split()))
    except OSError:
        pass
    return platform.machine() + "|" + platform.processor()


def cuda_target():
    if os.environ.get("CUDAARCHS"):
        return os.environ["CUDAARCHS"]
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"], capture_output=True, text=True, check=True, timeout=5)
        return ";".join(sorted(set(result.stdout.split()))) + ("|visible=" + os.environ["CUDA_VISIBLE_DEVICES"] if "CUDA_VISIBLE_DEVICES" in os.environ else "")
    except (OSError, subprocess.SubprocessError) as error:
        raise BuildError("hyperdrone: cannot identify the native CUDA target; set CUDAARCHS explicitly") from error


def workspace_tag(component=None):
    from .. import __version__
    fingerprint = f"{source_root()}|{os.path.abspath(sys.executable)}|{sysconfig.get_config_var('SOABI')}|{__version__}|{host_target()}"
    if component is not None and component.needs_cuda:
        fingerprint += "|" + cuda_target()
    return hashlib.sha1(fingerprint.encode()).hexdigest()[:8]


def build_dir(component):
    return cache_root() / f"{component.name}-{component.variant}-{workspace_tag(component)}"
