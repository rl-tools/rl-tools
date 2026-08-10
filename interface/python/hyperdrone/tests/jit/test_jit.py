"""hyperdrone.jit unit tests against the toy component: configure, per-config build,
cache hits, artifact validation, and multi-process lock contention — all in seconds,
without any rendering or dynamics dependency."""
import ctypes
import sys
from dataclasses import dataclass
from pathlib import Path

from hyperdrone import jit

TOY_SOURCE = Path(__file__).resolve().parent / "toy"


@dataclass(frozen=True)
class ToyConfig:
    value: int

    def canonical(self):
        return f"value={self.value}"

    def key(self):
        return jit.canonical_key(self.canonical())

    def defines(self):
        return {"HYPERDRONE_TOY_VALUE": self.value}


def toy_component():
    return jit.Component(name="toy", variant="cpu", source_dir=str(TOY_SOURCE))


def load_toy(artifact):
    library = ctypes.CDLL(str(artifact))
    library.hyperdrone_toy_value.restype = ctypes.c_int
    library.hyperdrone_toy_config_string.restype = ctypes.c_char_p
    library.hyperdrone_toy_iface_version.restype = ctypes.c_int
    return library


def test_build_and_load():
    config = ToyConfig(value=42)
    artifact = jit.ensure(toy_component(), config)
    assert artifact.exists()
    library = load_toy(artifact)
    assert library.hyperdrone_toy_value() == 42
    assert library.hyperdrone_toy_config_string().decode() == config.canonical()
    assert library.hyperdrone_toy_iface_version() == 1


def test_distinct_configs_get_distinct_artifacts():
    first = jit.ensure(toy_component(), ToyConfig(value=1))
    second = jit.ensure(toy_component(), ToyConfig(value=2))
    assert first != second
    assert load_toy(first).hyperdrone_toy_value() == 1
    assert load_toy(second).hyperdrone_toy_value() == 2


def test_cache_hit_does_not_rebuild():
    config = ToyConfig(value=7)
    artifact = jit.ensure(toy_component(), config)
    modification_time = artifact.stat().st_mtime_ns
    assert jit.ensure(toy_component(), config) == artifact
    assert artifact.stat().st_mtime_ns == modification_time


def test_existing_tree_moves_fetchcontent_state_into_cache():
    from hyperdrone.jit import _workspace

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
        "from hyperdrone import jit; "
        "artifact = jit.ensure(toy_component(), ToyConfig(value=int(sys.argv[2]))); "
        "assert load_toy(artifact).hyperdrone_toy_value() == int(sys.argv[2]); "
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
    from hyperdrone.jit import _workspace

    component = toy_component()
    directory = jit.build_dir(component)
    assert directory.name == f"toy-cpu-{jit.workspace_tag()}"
    assert directory.parent == jit.cache_root()
    assert _workspace.dependencies_root() == jit.cache_root() / ".dependencies"
