"""hyperlux._jit — shared compile-and-cache infrastructure.

A *component* is a native code unit under hyperlux/_native/<name>/: an optional nanobind
core module built once per build tree, plus JIT artifacts compiled per configuration:
Python extension modules or shared libraries with a C ABI.

Configs implement the JitConfig protocol: defines() -> {NAME: value} compile definitions,
canonical() -> stable human-readable string of the configuration, key() -> its short hash,
which names the artifact. Config files land as configs/<key>.txt inside the
build tree, one NAME=VALUE line each; the shared hyperlux_jit.cmake turns each file into
a MODULE target named <component>_<key>. A config with camera_config() -> C++ header text
also gets configs/<key>/camera_config.h, included through HYPERLUX_CAMERA_CONFIG_HEADER.

Build trees live at <cache>/<component>-<variant>-<tag> (one per component and variant so
switching variants never invalidates another's cache); all configure/build steps run under
a per-tree file lock so concurrent processes cannot race one CMake tree.
"""
import hashlib
import importlib.util
import os
import sys
from dataclasses import dataclass, field

from .. import _register_extension
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
    "module",
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
    package: str = "hyperlux"  # prefixes the extension module names; a package built on this JIT passes its own
    cmake_defines: tuple = field(default_factory=tuple)
    source_dir: str = None  # default: hyperlux/_native/<name>


def canonical_key(canonical):
    return hashlib.sha1(canonical.encode()).hexdigest()[:16]


def _replace_if_changed(path, contents):
    if path.exists() and path.read_text() == contents:
        return False
    # per-process staging name: concurrent writers of the same config must not race on one
    # staging path; the final rename is atomic and idempotent (identical content)
    staging = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    staging.write_text(contents)
    staging.replace(path)
    return True


def _write_config_file(directory, config):
    config_dir = directory / "configs"
    config_dir.mkdir(exist_ok=True)
    defines = dict(config.defines())
    changed = False
    if hasattr(config, "camera_config"):
        header = config_dir / config.key() / "camera_config.h"
        header.parent.mkdir(exist_ok=True)
        changed = _replace_if_changed(header, config.camera_config())
        defines["HYPERLUX_CAMERA_CONFIG_HEADER"] = f'"{config.key()}/camera_config.h"'
    lines = "".join(f"{name}={value}\n" for name, value in sorted(defines.items()))
    return _replace_if_changed(config_dir / f"{config.key()}.txt", lines) or changed


def ensure(component, config):
    """Build (if necessary) and return the path of the JIT artifact for this config."""
    directory = _build.ensure_configured(component)
    with _build.lock(component):
        fresh = _write_config_file(directory, config)
    artifact = directory / "jit" / f"{component.name}_{config.key()}.so"
    if fresh or not artifact.exists():
        # the target only exists after the configs/ glob is re-run
        _build.reconfigure(component)
    _build.build_target(component, f"{component.name}_{config.key()}")
    return artifact


def core(component):
    """Build (if necessary), load, and cache the component's nanobind core module.

    Extension modules are identified by name process-wide, so one component core can only
    be loaded for one variant per process (e.g. dynamics on cpu OR cuda, not both)."""
    if component.core_target is None:
        raise BuildError(f"hyperlux: component {component.name} declares no core module")
    cached = _core_modules.get(component.core_target)
    if cached is not None:
        variant, module = cached
        if variant != component.variant:
            raise BuildError(
                f"hyperlux: {component.core_target} is already loaded for variant "
                f"'{variant}'; one process can use only one {component.name} variant"
            )
        return module
    directory = build_dir(component)
    if (directory / "CMakeCache.txt").exists() and not sorted(directory.glob(f"{component.core_target}*.so")):
        # a core target added since this workspace was configured exists only after a reconfigure
        _build.reconfigure(component)
    directory = _build.build_target(component, component.core_target)
    matches = sorted(directory.glob(f"{component.core_target}*.so"))
    if not matches:
        raise BuildError(f"hyperlux: {component.core_target} module not found in {directory}")
    extension = _load_extension(component.core_target, matches[0])
    _core_modules[component.core_target] = (component.variant, extension)
    return extension


def _load_extension(name, path):
    cached = sys.modules.get(name)
    if cached is not None:
        return cached
    specification = importlib.util.spec_from_file_location(name, path)
    extension = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(extension)
    sys.modules[name] = extension
    _register_extension(extension)
    return extension


def module(component, config):
    """Build and import a configuration's Python extension, after its shared core."""
    if component.core_target is not None:
        core(component)
    artifact = ensure(component, config)
    return _load_extension(f"{component.package}_{component.name}_{component.variant}_{config.key()}", artifact)
