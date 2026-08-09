"""hyperdrone.jit — shared compile-and-cache infrastructure.

A *component* is a native code unit under hyperdrone/_native/<name>/: an optional nanobind
core module built once per build tree, plus zero or more JIT artifacts — shared libraries
compiled per compile-time configuration behind a stable extern "C" ABI.

Configs implement the JitConfig protocol: defines() -> {NAME: value} compile definitions,
canonical() -> stable human-readable string (baked into the artifact and compared on load),
key() -> short hash naming the artifact. Config files land as configs/<key>.txt inside the
build tree, one NAME=VALUE line each; the shared hyperdrone_jit.cmake turns each file into
a MODULE target named <component>_<key>.

Build trees live at <cache>/<component>-<variant>-<tag> (one per component and variant so
switching variants never invalidates another's cache); all configure/build steps run under
a per-tree file lock so concurrent processes cannot race one CMake tree.
"""
import hashlib
import importlib.util
import sys
from dataclasses import dataclass, field

from . import _build
from ._workspace import BuildError, build_dir, cache_root, native_root, source_root, workspace_tag

__all__ = [
    "BuildError",
    "Component",
    "build_dir",
    "cache_root",
    "canonical_key",
    "core",
    "ensure",
    "native_root",
    "source_root",
    "workspace_tag",
]

_core_modules = {}


@dataclass(frozen=True)
class Component:
    name: str
    variant: str
    core_target: str = None
    needs_cuda: bool = False
    cmake_defines: tuple = field(default_factory=tuple)
    source_dir: str = None  # default: hyperdrone/_native/<name>


def canonical_key(canonical):
    return hashlib.sha1(canonical.encode()).hexdigest()[:16]


def _write_config_file(directory, config):
    import os
    config_dir = directory / "configs"
    config_dir.mkdir(exist_ok=True)
    config_file = config_dir / f"{config.key()}.txt"
    if config_file.exists():
        return False
    lines = "".join(f"{name}={value}\n" for name, value in sorted(config.defines().items()))
    # per-process staging name: concurrent writers of the same config must not race on one
    # staging path; the final rename is atomic and idempotent (identical content)
    staging = config_dir / f"{config.key()}.{os.getpid()}.tmp"
    staging.write_text(lines)
    staging.rename(config_file)
    return True


def ensure(component, config):
    """Build (if necessary) and return the path of the JIT artifact for this config."""
    directory = _build.ensure_configured(component)
    fresh = _write_config_file(directory, config)
    artifact = directory / "jit" / f"{component.name}_{config.key()}.so"
    if fresh or not artifact.exists():
        # the target only exists after the configs/ glob is re-run
        _build.reconfigure(component)
    _build.build_target(component, f"{component.name}_{config.key()}")
    if not artifact.exists():
        raise BuildError(f"hyperdrone: JIT artifact missing after build: {artifact}")
    return artifact


def core(component):
    """Build (if necessary), load, and cache the component's nanobind core module.

    Extension modules are identified by name process-wide, so one component core can only
    be loaded for one variant per process (e.g. dynamics on cpu OR cuda, not both)."""
    if component.core_target is None:
        raise BuildError(f"hyperdrone: component {component.name} declares no core module")
    cached = _core_modules.get(component.core_target)
    if cached is not None:
        variant, module = cached
        if variant != component.variant:
            raise BuildError(
                f"hyperdrone: {component.core_target} is already loaded for variant "
                f"'{variant}'; one process can use only one {component.name} variant"
            )
        return module
    directory = _build.build_target(component, component.core_target)
    matches = sorted(directory.glob(f"{component.core_target}*.so"))
    matches += sorted(directory.glob(f"{component.core_target}*.dylib"))
    if not matches:
        raise BuildError(f"hyperdrone: {component.core_target} module not found in {directory}")
    specification = importlib.util.spec_from_file_location(component.core_target, matches[0])
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    sys.modules[component.core_target] = module
    _core_modules[component.core_target] = (component.variant, module)
    return module
