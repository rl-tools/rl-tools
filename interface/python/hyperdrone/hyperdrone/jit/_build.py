import os
import subprocess
from pathlib import Path

from ._lock import FileLock
from ._patches import seed_dependencies
from ._toolchain import environment
from ._workspace import BuildError, build_dir, dependencies_root, native_root, source_root

BUILD_JOBS = os.environ.get("HYPERDRONE_BUILD_JOBS", "5")


def lock(component):
    directory = build_dir(component)
    return FileLock(directory.parent / f"{directory.name}.lock")


def run(arguments, env):
    process = subprocess.run(
        [str(argument) for argument in arguments],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env,
    )
    if process.returncode != 0:
        tail = "\n".join(process.stdout.splitlines()[-60:])
        raise BuildError(
            f"hyperdrone: command failed: {' '.join(str(argument) for argument in arguments)}\n{tail}"
        )
    return process.stdout


def source_dir(component):
    if component.source_dir is not None:
        from pathlib import Path
        return Path(component.source_dir)
    return native_root() / component.name


def fetchcontent_base(directory):
    return (dependencies_root() / directory.name).resolve()


def configured_fetchcontent_base(directory):
    cache_file = directory / "CMakeCache.txt"
    if not cache_file.exists():
        return None
    prefix = "FETCHCONTENT_BASE_DIR:"
    for line in cache_file.read_text().splitlines():
        if line.startswith(prefix):
            return Path(line.split("=", 1)[1]).resolve()
    return None


def configure(component, directory):
    import sys
    seeded = seed_dependencies(directory.name)
    arguments = [
        "cmake",
        "-S", source_dir(component),
        "-B", directory,
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DRLTOOLS_ROOT={source_root()}",
        f"-DHYPERDRONE_VARIANT={component.variant.upper()}",
        f"-DHYPERDRONE_NATIVE_ROOT={native_root()}",
        f"-DPython_EXECUTABLE={sys.executable}",
        # fully qualified: the rl_tools root CMakeLists cannot append the build-dir
        # basename to an externally-passed value (non-FORCE cache set)
        f"-DFETCHCONTENT_BASE_DIR={fetchcontent_base(directory)}",
    ]
    for fetch_name, path in sorted(seeded.items()):
        arguments.append(f"-DFETCHCONTENT_SOURCE_DIR_{fetch_name}={path}")
    for name, value in component.cmake_defines:
        arguments.append(f"-D{name}={value}")
    if os.environ.get("HYPERDRONE_OFFLINE"):
        arguments.append("-DRL_TOOLS_OFFLINE_BUILD=ON")
    run(arguments, environment(component))


def ensure_configured(component):
    directory = build_dir(component)
    expected_fetchcontent_base = fetchcontent_base(directory)
    if configured_fetchcontent_base(directory) == expected_fetchcontent_base:
        return directory
    with lock(component):
        if configured_fetchcontent_base(directory) != expected_fetchcontent_base:
            fresh = not (directory / "CMakeCache.txt").exists()
            directory.mkdir(parents=True, exist_ok=True)
            try:
                configure(component, directory)
            except BaseException:
                # a failed configure must not leave a tree that looks configured; the
                # FetchContent state lives in .dependencies/ and survives for the retry
                if fresh:
                    import shutil
                    shutil.rmtree(directory, ignore_errors=True)
                raise
    return directory


def build_target(component, target):
    directory = ensure_configured(component)
    if os.environ.get("HYPERDRONE_SKIP_BUILD"):
        return directory
    with lock(component):
        run(["cmake", "--build", directory, "--target", target, "-j", BUILD_JOBS],
            environment(component))
    return directory


def reconfigure(component):
    directory = ensure_configured(component)
    with lock(component):
        run(["cmake", directory], environment(component))
    return directory
