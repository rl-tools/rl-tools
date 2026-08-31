"""Protocol tests for hyperdrone.conta, mirroring tests/src/utils/conta.cpp: the Python
client must implement the exact cache layout and download protocol of include/conta/conta.h
so both languages interoperate on one cache. All network cases run offline via file:// URLs.
"""
import os
import subprocess
from pathlib import Path

import pytest

from hyperdrone import conta
from hyperdrone.jit import source_root

ABC_SHA1 = "a9993e364706816aba3e25717850c26c9cd0d89d"
EMPTY_SHA1 = "da39a3ee5e6b4b0d3255bfef95601890afd80709"


def config(tmp_path, store=None, root=None, url_base=None):
    result = conta.Config()
    if root is not None:
        result.root = str(root)
    else:
        result.cache = str(tmp_path / "cache")
    if url_base is not None:
        result.url_base = url_base
    elif store is not None:
        result.url_base = store.as_uri() + "/"
    else:
        result.url_base = "http://127.0.0.1:1/"
    return result


def write_store_blob(store, sha1, content):
    store.mkdir(parents=True, exist_ok=True)
    (store / sha1).write_bytes(content)


def test_sha1_known_vectors(tmp_path):
    (tmp_path / "empty").write_bytes(b"")
    (tmp_path / "abc").write_bytes(b"abc")
    assert conta.sha1_file(tmp_path / "empty") == EMPTY_SHA1
    assert conta.sha1_file(tmp_path / "abc") == ABC_SHA1


def test_hash_validation(tmp_path):
    assert conta._reference_hash(ABC_SHA1) == (ABC_SHA1, ABC_SHA1)
    assert conta._reference_hash(ABC_SHA1.upper()) == (ABC_SHA1, ABC_SHA1)
    assert conta._reference_hash("conta:" + ABC_SHA1) == (ABC_SHA1, ABC_SHA1)
    for invalid in (ABC_SHA1[:39], ABC_SHA1 + "0", "g" + ABC_SHA1[1:], "../../../../../../../../etc/passwd00000000"):
        with pytest.raises(conta.ContaError, match="invalid hash"):
            conta._reference_hash(invalid)
    with pytest.raises(conta.ContaError, match="invalid hash"):
        conta.resolve("conta:not-a-hash", config(tmp_path))


def test_manifest_entry_reference(tmp_path):
    store = tmp_path / "store"
    write_store_blob(store, ABC_SHA1, b"abc")
    entry = {"description": "docs/abc.txt", "hash": ABC_SHA1}
    path = conta.resolve(entry, config(tmp_path, store=store))
    assert path.read_bytes() == b"abc"
    with pytest.raises(conta.ContaError, match="no \"hash\" key"):
        conta.resolve({"description": "no hash"}, config(tmp_path))


def test_root_hit(tmp_path):
    root = tmp_path / "root"
    (root / "data").mkdir(parents=True)
    (root / "data" / ABC_SHA1).write_bytes(b"abc")
    path = conta.resolve(ABC_SHA1, config(tmp_path, root=root))
    assert path == root / "data" / ABC_SHA1


def test_root_miss(tmp_path):
    root = tmp_path / "root"
    with pytest.raises(conta.ContaError, match="CONTA_ROOT") as error:
        conta.resolve(ABC_SHA1, config(tmp_path, root=root))
    assert str(root / "data" / ABC_SHA1) in str(error.value)


def test_root_lfs_pointer(tmp_path):
    root = tmp_path / "root"
    (root / "data").mkdir(parents=True)
    (root / "data" / ABC_SHA1).write_bytes(
        b"version https://git-lfs.github.com/spec/v1\n"
        b"oid sha256:0000000000000000000000000000000000000000000000000000000000000000\nsize 123\n"
    )
    with pytest.raises(conta.ContaError, match="git lfs pull"):
        conta.resolve(ABC_SHA1, config(tmp_path, root=root))


def test_cache_hit_no_network(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / ABC_SHA1).write_bytes(b"abc")
    path = conta.resolve(ABC_SHA1, config(tmp_path))
    assert path == cache / ABC_SHA1


def test_download_file_url(tmp_path):
    store = tmp_path / "store"
    write_store_blob(store, ABC_SHA1, b"abc")
    path = conta.resolve(ABC_SHA1, config(tmp_path, store=store))
    assert path == tmp_path / "cache" / ABC_SHA1
    assert conta.sha1_file(path) == ABC_SHA1
    assert not list((tmp_path / "cache").glob("*.partial.*"))


def test_download_hash_mismatch(tmp_path):
    store = tmp_path / "store"
    write_store_blob(store, ABC_SHA1, b"not abc")
    with pytest.raises(conta.ContaError, match="hash mismatch"):
        conta.resolve(ABC_SHA1, config(tmp_path, store=store))
    assert not (tmp_path / "cache" / ABC_SHA1).exists()
    assert not list((tmp_path / "cache").glob("*.partial.*"))


def test_batch_all_or_nothing(tmp_path):
    store = tmp_path / "store"
    write_store_blob(store, ABC_SHA1, b"abc")
    with pytest.raises(conta.ContaError):
        conta.resolve_all([ABC_SHA1, EMPTY_SHA1], config(tmp_path, store=store))
    paths = conta.resolve_all([ABC_SHA1], config(tmp_path, store=store))
    assert len(paths) == 1


def test_default_cache_path_parity(monkeypatch, tmp_path):
    # the fallback chain is pinned by include/conta/conta.h::config_from_environment
    for variable in ("CONTA_ROOT", "CONTA_CACHE", "CONTA_URL", "XDG_CACHE_HOME"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert conta.config_from_environment().cache == str(tmp_path / ".cache" / "rl_tools" / "conta")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert conta.config_from_environment().cache == str(tmp_path / "xdg" / "rl_tools" / "conta")
    monkeypatch.setenv("CONTA_CACHE", str(tmp_path / "explicit"))
    assert conta.config_from_environment().cache == str(tmp_path / "explicit")
    monkeypatch.setenv("CONTA_URL", "https://example.com/store")
    assert conta.config_from_environment().url_base == "https://example.com/store/"


def conta_cli():
    candidates = sorted(source_root().glob("build*/src/conta/conta"))
    if not candidates:
        pytest.skip("conta CLI not built (cmake --build build --target conta)")
    return candidates[0]


def cli_environment(cache):
    environment = os.environ.copy()
    environment.pop("CONTA_ROOT", None)
    environment["CONTA_CACHE"] = str(cache)
    environment["CONTA_URL"] = "http://127.0.0.1:1/"
    return environment


def test_cache_interop_python_to_cpp(tmp_path):
    cli = conta_cli()
    store = tmp_path / "store"
    write_store_blob(store, ABC_SHA1, b"abc")
    python_path = conta.resolve(ABC_SHA1, config(tmp_path, store=store))
    result = subprocess.run(
        [str(cli), ABC_SHA1], capture_output=True, text=True,
        env=cli_environment(tmp_path / "cache"),
    )
    assert result.returncode == 0, result.stderr
    assert Path(result.stdout.strip()) == python_path


def test_cache_interop_cpp_to_python(tmp_path):
    cli = conta_cli()
    store = tmp_path / "store"
    write_store_blob(store, ABC_SHA1, b"abc")
    environment = cli_environment(tmp_path / "cache")
    environment["CONTA_URL"] = store.as_uri() + "/"
    result = subprocess.run([str(cli), ABC_SHA1], capture_output=True, text=True, env=environment)
    assert result.returncode == 0, result.stderr
    path = conta.resolve(ABC_SHA1, config(tmp_path))
    assert path == Path(result.stdout.strip())
    assert conta.sha1_file(path) == ABC_SHA1


def test_network_download(tmp_path):
    if os.environ.get("RL_TOOLS_TEST_CONTA_NETWORK") != "1":
        pytest.skip("set RL_TOOLS_TEST_CONTA_NETWORK=1 to enable")
    blob = "a38b2994e7674f467fe81e86d2f21c45bfd965a0"
    path = conta.resolve(blob, config(tmp_path, url_base=conta.DEFAULT_URL_BASE))
    assert conta.sha1_file(path) == blob
