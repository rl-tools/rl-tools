"""hyperlux._jit unit tests against the toy component: configure, per-config build,
cache hits, artifact validation, and multi-process lock contention — all in seconds,
without any rendering or dynamics dependency."""
import ctypes
import sys
from dataclasses import dataclass
from pathlib import Path

from hyperlux import _jit as jit

TOY_SOURCE = Path(__file__).resolve().parent / "toy"


@dataclass(frozen=True)
class ToyConfig:
    value: int

    def canonical(self):
        return f"value={self.value}"

    def key(self):
        return jit.canonical_key(self.canonical())

    def defines(self):
        return {"HYPERLUX_TOY_VALUE": self.value}


def toy_component():
    return jit.Component(name="toy", variant="cpu", source_dir=str(TOY_SOURCE))


def load_toy(artifact):
    library = ctypes.CDLL(str(artifact))
    library.hyperlux_toy_value.restype = ctypes.c_int
    library.hyperlux_toy_config_string.restype = ctypes.c_char_p
    library.hyperlux_toy_iface_version.restype = ctypes.c_int
    return library


def test_build_and_load():
    config = ToyConfig(value=42)
    artifact = jit.ensure(toy_component(), config)
    assert artifact.exists()
    library = load_toy(artifact)
    assert library.hyperlux_toy_value() == 42
    assert library.hyperlux_toy_config_string().decode() == config.canonical()
    assert library.hyperlux_toy_iface_version() == 1


def test_distinct_configs_get_distinct_artifacts():
    first = jit.ensure(toy_component(), ToyConfig(value=1))
    second = jit.ensure(toy_component(), ToyConfig(value=2))
    assert first != second
    assert load_toy(first).hyperlux_toy_value() == 1
    assert load_toy(second).hyperlux_toy_value() == 2


def test_camera_config_header_is_compiled_and_tracked():
    @dataclass(frozen=True)
    class GeneratedConfig(ToyConfig):
        def defines(self):
            return {"HYPERLUX_TOY_VALUE": "hyperlux_toy_config::VALUE"}

        def camera_config(self):
            return f"namespace hyperlux_toy_config {{ constexpr int VALUE = {self.value}; }}\n"

        def canonical(self):
            return super().canonical() + self.camera_config()

    config = GeneratedConfig(value=91)
    artifact = jit.ensure(toy_component(), config)
    assert load_toy(artifact).hyperlux_toy_value() == 91
    header = jit.build_dir(toy_component()) / "configs" / config.key() / "camera_config.h"
    assert header.read_text() == config.camera_config()
    modification_time = header.stat().st_mtime_ns
    jit.ensure(toy_component(), config)
    assert header.stat().st_mtime_ns == modification_time


def test_cache_hit_does_not_rebuild():
    config = ToyConfig(value=7)
    artifact = jit.ensure(toy_component(), config)
    modification_time = artifact.stat().st_mtime_ns
    assert jit.ensure(toy_component(), config) == artifact
    assert artifact.stat().st_mtime_ns == modification_time


def test_existing_tree_moves_fetchcontent_state_into_cache():
    from hyperlux._jit import _workspace

    component = toy_component()
    config = ToyConfig(value=8)
    jit.ensure(component, config)
    cache_file = jit.build_dir(component) / "CMakeCache.txt"
    expected = (_workspace.dependencies_root() / jit.build_dir(component).name).resolve()
    stale = (jit.cache_root().parent / "old-source-checkout" / ".dependencies").resolve()
    contents = cache_file.read_text()
    expected_entry = next(
        line for line in contents.splitlines()
        if line.startswith("FETCHCONTENT_BASE_DIR:")
    )
    assert expected_entry.split("=", 1)[1] == str(expected)
    stale_entry = f"{expected_entry.split('=', 1)[0]}={stale}"
    contents = contents.replace(expected_entry, stale_entry)
    cache_file.write_text(contents)

    jit.ensure(component, config)

    migrated_entry = next(
        line for line in cache_file.read_text().splitlines()
        if line.startswith("FETCHCONTENT_BASE_DIR:")
    )
    assert migrated_entry.split("=", 1)[1] == str(expected)


def test_concurrent_builds_do_not_race():
    # two fresh processes race ensure() on the same new config; the per-tree lock must
    # serialize them and both must come back with a valid artifact
    import os

    value = 20000 + os.getpid() % 10000  # unique per run: always exercises the fresh path
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "from test_jit import ToyConfig, toy_component, load_toy; "
        "from hyperlux import _jit as jit; "
        "artifact = jit.ensure(toy_component(), ToyConfig(value=int(sys.argv[2]))); "
        "assert load_toy(artifact).hyperlux_toy_value() == int(sys.argv[2]); "
        "print(artifact)"
    )
    import subprocess

    workers = [
        subprocess.Popen(
            [sys.executable, "-c", code, str(Path(__file__).parent), str(value)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        for _ in range(2)
    ]
    outputs = [worker.communicate()[0] for worker in workers]
    assert all(worker.returncode == 0 for worker in workers), outputs
    assert outputs[0].strip() == outputs[1].strip()


def test_workspace_layout():
    from hyperlux._jit import _workspace

    component = toy_component()
    directory = jit.build_dir(component)
    assert directory.name == f"toy-cpu-{jit.workspace_tag(component)}"
    assert directory.parent == jit.cache_root()
    assert _workspace.dependencies_root() == jit.cache_root() / ".dependencies"
    # the same component from another checkout gets its own tree: CMake keeps the source directory it was configured with
    elsewhere = jit.Component(name="toy", variant="cpu", source_dir=str(TOY_SOURCE.parent))
    assert jit.build_dir(elsewhere) != directory


def test_python_environments_have_separate_workspaces(tmp_path, monkeypatch):
    executable = Path(sys.executable).resolve()
    first = tmp_path / "first" / "bin" / "python3"
    second = tmp_path / "second" / "bin" / "python3"
    for interpreter in (first, second):
        interpreter.parent.mkdir(parents=True)
        interpreter.symlink_to(executable)
    assert first.resolve() == second.resolve()

    component = toy_component()
    monkeypatch.setattr(sys, "executable", str(first))
    first_directory = jit.build_dir(component)
    monkeypatch.setattr(sys, "executable", str(second))
    assert jit.build_dir(component) != first_directory
    monkeypatch.setattr(sys, "executable", str(first))
    assert jit.build_dir(component) == first_directory


def test_native_host_targets_have_separate_workspaces(monkeypatch):
    from hyperlux._jit import _workspace

    monkeypatch.setattr(_workspace, "host_target", lambda: "x86_64|avx avx2")
    newer = _workspace.workspace_tag()
    monkeypatch.setattr(_workspace, "host_target", lambda: "x86_64|avx")
    assert _workspace.workspace_tag() != newer
    component = jit.Component(name="dynamics", variant="cuda", needs_cuda=True)
    monkeypatch.setattr(_workspace, "cuda_target", lambda: "8.9")
    older_gpu = _workspace.workspace_tag(component)
    monkeypatch.setattr(_workspace, "cuda_target", lambda: "12.0")
    assert _workspace.workspace_tag(component) != older_gpu


def test_source_root_prefers_override_then_vendored_tree_then_enclosing_checkout(tmp_path, monkeypatch):
    checkout = tmp_path / "checkout"
    package = checkout / ".venv" / "site-packages" / "hyperlux"
    (checkout / "include" / "rl_tools").mkdir(parents=True)
    package.mkdir(parents=True)
    monkeypatch.delenv("HYPERLUX_RLTOOLS_ROOT", raising=False)
    monkeypatch.setattr(jit._workspace, "package_root", lambda: package)
    assert jit.source_root() == checkout
    (package / "_vendor" / "rl-tools" / "include" / "rl_tools").mkdir(parents=True)
    assert jit.source_root() == package / "_vendor" / "rl-tools"
    override = tmp_path / "override"
    (override / "include" / "rl_tools").mkdir(parents=True)
    monkeypatch.setenv("HYPERLUX_RLTOOLS_ROOT", str(override))
    assert jit.source_root() == override.resolve()
