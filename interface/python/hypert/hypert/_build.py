import hashlib
import importlib.util
import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

_core_module = None

BUILD_JOBS = "5"


class BuildError(RuntimeError):
    pass


def repo_root():
    env = os.environ.get("HYPERT_RLTOOLS_ROOT")
    if env:
        root = Path(env).resolve()
    else:
        root = Path(__file__).resolve().parents[4]
    if not (root / "include" / "rl_tools").is_dir():
        raise BuildError(
            f"hypert: rl-tools repository not found at {root} "
            "(set HYPERT_RLTOOLS_ROOT to the repository root)"
        )
    return root


def backend():
    value = os.environ.get("HYPERT_BACKEND", "AUTO").upper()
    if value == "AUTO":
        value = "METAL" if sys.platform == "darwin" else "OPTIX"
    if value not in ("OPTIX", "METAL", "VULKAN", "GENERIC"):
        raise BuildError(f"hypert: invalid HYPERT_BACKEND {value} (OPTIX|METAL|VULKAN|GENERIC)")
    return value


def cache_root():
    env = os.environ.get("HYPERT_CACHE_DIR")
    if env:
        return Path(env)
    return Path.home() / ".cache" / "hypert"


def build_dir():
    tag = hashlib.sha1(
        f"{repo_root()}|{sysconfig.get_config_var('SOABI')}".encode()
    ).hexdigest()[:8]
    return cache_root() / f"{backend().lower()}-{tag}"


def _seed_dependencies(build_dir_name):
    # FetchContent sources land in <repo>/.dependencies/<build-dir-basename>. Seed them from
    # the main build's copies when available: the OWL tree there carries a local
    # cudaFreeHost patch (second renderer lifecycle crashes without it) and re-cloning
    # would silently drop it.
    root = repo_root()
    source_base = root / ".dependencies" / "build"
    target_base = root / ".dependencies" / build_dir_name
    seeded_owl = False
    if source_base.is_dir():
        for name in ("owl-src", "stb-src", "nlohmann_json-src"):
            source = source_base / name
            target = target_base / name
            if source.is_dir() and not target.exists():
                target_base.mkdir(parents=True, exist_ok=True)
                shutil.copytree(source, target, symlinks=True)
            if name == "owl-src" and target.exists():
                seeded_owl = True
    return seeded_owl


def _patch_owl():
    # OWL's PinnedHostMem frees cudaMallocHost memory with cudaFree (dtor + resize); the
    # second renderer lifecycle in one process dies with "invalid argument" without this.
    # Idempotent; drop once fixed upstream (NVIDIA/OWL owl/DeviceMemory.h).
    path = repo_root() / ".dependencies" / build_dir().name / "owl-src" / "owl" / "DeviceMemory.h"
    if not path.exists():
        return
    text = path.read_text()
    patched = text.replace("if (ptr) cudaFree(ptr);", "if (ptr) cudaFreeHost(ptr);")
    if patched != text:
        path.write_text(patched)


def _run(arguments, environment=None):
    process = subprocess.run(
        arguments,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=environment,
    )
    if process.returncode != 0:
        tail = "\n".join(process.stdout.splitlines()[-60:])
        raise BuildError(
            f"hypert: command failed: {' '.join(str(argument) for argument in arguments)}\n{tail}"
        )
    return process.stdout


def _configure_environment():
    environment = os.environ.copy()
    if backend() == "OPTIX" and "CUDACXX" not in environment and "CMAKE_CUDA_COMPILER" not in environment:
        for candidate in ("/usr/local/cuda/bin/nvcc",):
            if Path(candidate).exists():
                environment["CUDACXX"] = candidate
                break
    return environment


def ensure_configured():
    directory = build_dir()
    if (directory / "CMakeCache.txt").exists():
        _patch_owl()
        return directory
    directory.mkdir(parents=True, exist_ok=True)
    seeded_owl = _seed_dependencies(directory.name)
    package_root = Path(__file__).resolve().parents[1]
    arguments = [
        "cmake",
        "-S", str(package_root / "cpp"),
        "-B", str(directory),
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DRLTOOLS_ROOT={repo_root()}",
        f"-DHYPERT_BACKEND={backend()}",
        f"-DPython_EXECUTABLE={sys.executable}",
    ]
    if seeded_owl:
        arguments.append("-DRL_TOOLS_OFFLINE_BUILD=ON")
    try:
        _run(arguments, _configure_environment())
    except BuildError:
        if not seeded_owl:
            raise
        # seeded sources may be incomplete for this configuration; retry with the network
        _run([argument for argument in arguments if argument != "-DRL_TOOLS_OFFLINE_BUILD=ON"], _configure_environment())
    _patch_owl()
    return directory


def _build_target(target):
    directory = ensure_configured()
    if os.environ.get("HYPERT_SKIP_BUILD"):
        return directory
    _run(["cmake", "--build", str(directory), "--target", target, "-j", BUILD_JOBS], _configure_environment())
    return directory


def load_core():
    global _core_module
    if _core_module is not None:
        return _core_module
    directory = _build_target("hypert_core")
    matches = sorted(directory.glob("hypert_core*.so")) + sorted(directory.glob("hypert_core*.dylib"))
    if not matches:
        raise BuildError(f"hypert: hypert_core module not found in {directory}")
    specification = importlib.util.spec_from_file_location("hypert_core", matches[0])
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    sys.modules["hypert_core"] = module
    _core_module = module
    return module


def ensure_renderer_library(config):
    directory = ensure_configured()
    config_file = directory / "jit_configs.txt"
    line = config.config_line()
    existing = config_file.read_text().splitlines() if config_file.exists() else []
    if line not in existing:
        with open(config_file, "a") as handle:
            handle.write(line + "\n")
    library = directory / "jit" / f"hypert_renderer_{config.key()}.so"
    if not library.exists():
        # make cannot resolve a target that only exists after regeneration, so reconfigure
        # explicitly before building a configuration for the first time
        _run(["cmake", str(directory)], _configure_environment())
    _build_target(f"hypert_renderer_{config.key()}")
    if not library.exists():
        raise BuildError(f"hypert: JIT renderer library missing after build: {library}")
    return library
