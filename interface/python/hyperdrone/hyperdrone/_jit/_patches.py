"""Reuse dependencies from sibling JIT build caches.

Git sources use FETCHCONTENT_SOURCE_DIR overrides, with patch scripts applied explicitly.
Assimp reuses downloaded archives so CMake verifies URL_HASH before extracting them.
"""
import re
import shutil
import subprocess

from ._workspace import source_root

# FetchContent name -> source directory basename
SEEDED_SOURCES = {
    "OWL": "owl-src",
    "STB": "stb-src",
    "NLOHMANN_JSON": "nlohmann_json-src",
}

# FetchContent name -> cmake file declaring its GIT_TAG; seeds are validated against the
# declared pin so a repin upstream can never be shadowed by a stale seeded checkout
PINNED_SOURCES = {
    "OWL": "cmake/optional/optix.cmake",
}


def declared_pin(fetch_name):
    relative = PINNED_SOURCES.get(fetch_name)
    if relative is None:
        return None
    declaration = source_root() / relative
    if not declaration.exists():
        return None
    match = re.search(
        r"FetchContent_Declare\(\s*" + fetch_name + r"\b.*?GIT_TAG\s+([0-9a-fA-F]{7,40})",
        declaration.read_text(), re.IGNORECASE | re.DOTALL,
    )
    return match.group(1) if match else None


def checkout_head(tree):
    result = subprocess.run(
        ["git", "-C", str(tree), "rev-parse", "HEAD"], capture_output=True, text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def matches_pin(tree, pin):
    return pin is None or checkout_head(tree) == pin

# cmake -P scripts under <source_root>/cmake/patches, applied to the named FetchContent
# source dir; each script is idempotent and documents its upstream-removal condition.
# (Empty since the OWL PinnedHostMem fix landed upstream with the 978a6664 repin.)
PATCH_SCRIPTS = {}


def apply_patch_scripts(seeded):
    for fetch_name, (script_name, root_variable) in PATCH_SCRIPTS.items():
        tree = seeded.get(fetch_name)
        script = source_root() / "cmake" / "patches" / script_name
        if tree is not None and script.exists():
            subprocess.run(
                ["cmake", f"-D{root_variable}={tree}", "-P", str(script)],
                check=True, capture_output=True,
            )


def seed_dependencies(build_dir_name):
    """Copy FetchContent sources from a populated sibling; returns {FETCH_NAME: path} for
    every seeded (or already present) source directory."""
    from ._workspace import dependencies_root
    dependency_root = dependencies_root()
    target_base = dependency_root / build_dir_name
    donors = sorted(
        candidate for candidate in dependency_root.glob("*")
        if candidate.is_dir() and candidate.name != build_dir_name
    ) if dependency_root.is_dir() else []
    seeded = {}
    for donor in donors:
        for archive in (donor / "assimp-download").glob("*.tar.gz"):
            target = target_base / "assimp-download" / archive.name
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(archive, target)
    for fetch_name, source_name in SEEDED_SOURCES.items():
        pin = declared_pin(fetch_name)
        target = target_base / source_name
        if target.exists() and not matches_pin(target, pin):
            shutil.rmtree(target)
        if not target.exists():
            for donor in donors:
                source = donor / source_name
                if source.is_dir() and matches_pin(source, pin):
                    target_base.mkdir(parents=True, exist_ok=True)
                    shutil.copytree(source, target, symlinks=True)
                    break
        if target.is_dir():
            seeded[fetch_name] = target
    apply_patch_scripts(seeded)
    return seeded
