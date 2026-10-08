"""Protocol tests for conta, mirroring tests/src/utils/conta.cpp: the Python
client must implement the exact cache layout and download protocol of include/conta/conta.h
so both languages interoperate on one cache. All network cases run offline via file:// URLs.
"""
import hashlib
import http.server
import os
import subprocess
import threading
from pathlib import Path

import pytest

import conta

REPOSITORY = Path(__file__).resolve().parents[4]

ABC_SHA1 = "a9993e364706816aba3e25717850c26c9cd0d89d"
EMPTY_SHA1 = "da39a3ee5e6b4b0d3255bfef95601890afd80709"
ABC_SHA256 = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
HELLO_SHA1 = "aaf4c61ddcc5e8a2dabede0f3b482cd9aea9434d"
HELLO_SHA256 = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"


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
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    monkeypatch.setattr(conta, "_WINDOWS", False)
    assert conta.config_from_environment().cache == str(tmp_path / ".cache" / "conta")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert conta.config_from_environment().cache == str(tmp_path / "xdg" / "conta")
    monkeypatch.setattr(conta, "_WINDOWS", True)
    assert conta.config_from_environment().cache == str(tmp_path / "local" / "conta")
    monkeypatch.delenv("LOCALAPPDATA")
    with pytest.raises(conta.ContaError, match="set CONTA_CACHE or LOCALAPPDATA"):
        conta.resolve(ABC_SHA1, conta.config_from_environment())
    monkeypatch.setenv("CONTA_CACHE", str(tmp_path / "explicit"))
    assert conta.config_from_environment().cache == str(tmp_path / "explicit")
    monkeypatch.setenv("CONTA_URL", "https://example.com/store")
    assert conta.config_from_environment().url_base == "https://example.com/store/"


def conta_cli():
    candidates = sorted(REPOSITORY.glob("build*/src/conta/conta"))
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


def test_default_cache_path_cpp(tmp_path):
    cli = conta_cli()
    store = tmp_path / "store"
    write_store_blob(store, ABC_SHA1, b"abc")
    environment = {key: value for key, value in os.environ.items() if not key.startswith("CONTA_") and key != "XDG_CACHE_HOME"}
    environment["HOME"] = str(tmp_path)
    environment["LOCALAPPDATA"] = str(tmp_path / "local")
    environment["CONTA_URL"] = store.as_uri() + "/"
    result = subprocess.run([str(cli), ABC_SHA1], capture_output=True, text=True, env=environment)
    assert result.returncode == 0, result.stderr
    expected = tmp_path / "local" / "conta" if os.name == "nt" else tmp_path / ".cache" / "conta"
    assert Path(result.stdout.strip()) == expected / ABC_SHA1


def test_network_download(tmp_path):
    if os.environ.get("RL_TOOLS_TEST_CONTA_NETWORK") != "1":
        pytest.skip("set RL_TOOLS_TEST_CONTA_NETWORK=1 to enable")
    blob = "a38b2994e7674f467fe81e86d2f21c45bfd965a0"
    path = conta.resolve(blob, config(tmp_path, url_base="https://huggingface.co/datasets/rl-tools/conta/resolve/main/data/"))
    assert conta.sha1_file(path) == blob


def file_url(path):
    return path.as_uri()


def entry(sha1, sha256, shards='"local"'):
    return f'{{"sha1": "{sha1}", "sha256": "{sha256}", "shards": [{shards}], "description": "test blob"}}'


def index_text(shards, entries):
    return f'{{\n"shards": {{\n{shards}\n}},\n"entries": [\n{entries}\n]\n}}\n'


def local_shard(store):
    return f'"local": {{"type": "http", "url": "{file_url(store)}/{{sha1}}"}}'


def publish(site, index, listed_sha256=None, location="generated/index.json"):
    (site / "version" / "1" / "generated").mkdir(parents=True, exist_ok=True)
    (site / "version" / "1" / "generated" / "index.json").write_text(index)
    sha256 = listed_sha256 or hashlib.sha256(index.encode()).hexdigest()
    (site / "version" / "1" / "meta.json").write_text(f'{{"format": 1, "index": {{"location": "{location}", "sha256": "{sha256}"}}}}\n')
    return file_url(site / "version" / "1" / "meta.json")


def index_config(tmp_path, entry_points):
    return conta.Config(cache=str(tmp_path / "cache"), entry_points=entry_points)


def test_sha256_digests(tmp_path):
    (tmp_path / "hello").write_bytes(b"hello")
    assert conta._digests(tmp_path / "hello") == (HELLO_SHA1, HELLO_SHA256)


def test_references(tmp_path):
    root = tmp_path / "root"
    (root / "data").mkdir(parents=True)
    (root / "data" / ABC_SHA1).write_bytes(b"abc")
    assert conta.resolve("conta:" + ABC_SHA1, config(tmp_path, root=root)) == root / "data" / ABC_SHA1
    assert conta.resolve(ABC_SHA1.upper(), config(tmp_path, root=root)) == root / "data" / ABC_SHA1
    with pytest.raises(conta.ContaError, match="is a sha256"):
        conta.resolve("conta:" + ABC_SHA256, config(tmp_path, root=root))


def test_index_json_tolerance():
    text = ('{"future": {"nested": [1, 2.5e3, -0.5, true, false, null, {"deep": []}]},\n'
            '"shards": {"a": {"type": "future-type", "url": "https://a.example/{sha1}", "license": {"spdx": "CC0-1.0"}}, "no-url": {"type": "torrent"}},\n'
            '"entries": [\n'
            f'{{"\\u0073ha1": "{ABC_SHA1}", "sha256": "{ABC_SHA256}", "shards": ["unknown", 7, "a"], "description": "caf\\u00e9 \\ud83d\\ude00", "size": 3, "license": ["x"]}},\n'
            f'{{"sha1": "{ABC_SHA1}", "sha256": "{EMPTY_SHA256}", "shards": ["a"], "description": "duplicate"}},\n'
            f'{{"sha1": "{EMPTY_SHA1}", "sha256": "not-hex", "shards": ["a"]}},\n'
            '"not an object",\n'
            f'{{"sha1": "{EMPTY_SHA1}", "sha256": "{EMPTY_SHA256}", "shards": "wrong type", "description": 5}}\n'
            ']}')
    index = conta._parse_index(text.encode(), {ABC_SHA1, EMPTY_SHA1}, "test")
    assert index.entry_count == 3
    assert index.shard_urls == {"a": "https://a.example/{sha1}"}
    assert index.entries[ABC_SHA1].sha256 == ABC_SHA256
    assert index.entries[ABC_SHA1].shards == ["unknown", "a"]
    assert index.entries[ABC_SHA1].description == "café \U0001F600"
    assert index.entries[EMPTY_SHA1].shards == []
    assert index.entries[EMPTY_SHA1].description == ""
    assert conta._parse_index(b'{"shards": {}}', set(), "test") is None
    assert conta._parse_index(text.encode() + b" trailing", set(), "test") is None
    meta = f'{{"format": 1, "index": {{"location": "generated/index.json", "sha256": "{ABC_SHA256}", "size": 1}}, "future": {{"x": [1]}}}}'
    assert conta._parse_meta(meta.encode()) == ("generated/index.json", ABC_SHA256)
    for format in ("2", "1.0", '"1"', "true"):
        assert conta._parse_meta(meta.replace('"format": 1', f'"format": {format}').encode()) is None
    assert conta._parse_meta(b'{"format": 1, "x": "raw\ncontrol"}') is None


def test_index_resolve_and_cache(tmp_path):
    write_store_blob(tmp_path / "store", ABC_SHA1, b"abc")
    index = index_text(local_shard(tmp_path / "store"), entry(ABC_SHA1, ABC_SHA256))
    settings = index_config(tmp_path, [publish(tmp_path / "site", index)])
    assert conta.resolve(ABC_SHA1, settings) == tmp_path / "cache" / ABC_SHA1
    assert (tmp_path / "cache" / conta.CACHED_INDEX_NAME).read_text() == index
    (tmp_path / "cache" / ABC_SHA1).unlink()
    settings.entry_points = [file_url(tmp_path / "missing" / "meta.json")]
    assert conta.resolve(ABC_SHA1, settings) == tmp_path / "cache" / ABC_SHA1


def test_entry_point_fallback(tmp_path):
    write_store_blob(tmp_path / "store", ABC_SHA1, b"abc")
    (tmp_path / "future").mkdir()
    (tmp_path / "future" / "meta.json").write_text(f'{{"format": 2, "index": {{"location": "index.json", "sha256": "{EMPTY_SHA256}"}}}}')
    good = publish(tmp_path / "site", index_text(local_shard(tmp_path / "store"), entry(ABC_SHA1, ABC_SHA256)))
    settings = index_config(tmp_path, [file_url(tmp_path / "missing" / "meta.json"), file_url(tmp_path / "future" / "meta.json"), good])
    assert conta.resolve(ABC_SHA1, settings) == tmp_path / "cache" / ABC_SHA1


def test_index_sha256_mismatch_is_not_cached(tmp_path):
    write_store_blob(tmp_path / "store", ABC_SHA1, b"abc")
    index = index_text(local_shard(tmp_path / "store"), entry(ABC_SHA1, ABC_SHA256))
    conta.resolve(ABC_SHA1, index_config(tmp_path, [publish(tmp_path / "site", index, EMPTY_SHA256)]))
    assert not (tmp_path / "cache" / conta.CACHED_INDEX_NAME).exists()


def test_entry_sha256_is_verified(tmp_path):
    write_store_blob(tmp_path / "store", ABC_SHA1, b"abc")
    index = index_text(local_shard(tmp_path / "store"), entry(ABC_SHA1, EMPTY_SHA256))
    with pytest.raises(conta.ContaError, match="sha256 mismatch"):
        conta.resolve(ABC_SHA1, index_config(tmp_path, [publish(tmp_path / "site", index)]))
    assert not (tmp_path / "cache" / ABC_SHA1).exists()
    assert not list((tmp_path / "cache").glob("*.partial.*"))


def test_unindexed_report(tmp_path):
    write_store_blob(tmp_path / "store", ABC_SHA1, b"abc")
    settings = index_config(tmp_path, [publish(tmp_path / "site", index_text(local_shard(tmp_path / "store"), ""))])
    with pytest.raises(conta.ContaError, match="no reachable index lists") as error:
        conta.resolve(ABC_SHA1, settings)
    assert file_url(tmp_path / "site" / "version" / "1" / "generated" / "index.json") + ": does not list it" in str(error.value)


def test_older_index_does_not_replace_cache(tmp_path):
    write_store_blob(tmp_path / "store", ABC_SHA1, b"abc")
    unreachable_shard = '"local": {"type": "http", "url": "http://127.0.0.1:1/{sha1}"}'
    cached = index_text(unreachable_shard, entry(ABC_SHA1, ABC_SHA256) + ",\n" + entry(HELLO_SHA1, HELLO_SHA256))
    (tmp_path / "cache").mkdir()
    (tmp_path / "cache" / conta.CACHED_INDEX_NAME).write_text(cached)
    older = index_text(local_shard(tmp_path / "store"), entry(ABC_SHA1, ABC_SHA256))
    settings = index_config(tmp_path, [publish(tmp_path / "older", older)])
    conta.resolve(ABC_SHA1, settings)
    assert (tmp_path / "cache" / conta.CACHED_INDEX_NAME).read_text() == cached
    (tmp_path / "cache" / ABC_SHA1).unlink()
    newer = index_text(local_shard(tmp_path / "store"), ",\n".join([entry(ABC_SHA1, ABC_SHA256), entry(HELLO_SHA1, HELLO_SHA256), entry(EMPTY_SHA1, EMPTY_SHA256)]))
    settings.entry_points = [publish(tmp_path / "newer", newer)]
    conta.resolve(ABC_SHA1, settings)
    assert (tmp_path / "cache" / conta.CACHED_INDEX_NAME).read_text() == newer


def test_unsafe_locations_are_skipped(tmp_path):
    shards = ('"unc": {"url": "file:////server/share/{sha1}"},\n'
              '"percent": {"url": "https://example.invalid/%HOME%/{sha1}"},\n'
              '"placeholder": {"url": "https://example.invalid/{sha1}/{size}"},\n'
              '"quote": {"url": "https://example.invalid/\'{sha1}"}')
    index = index_text(shards, entry(ABC_SHA1, ABC_SHA256, '"unc", "percent", "placeholder", "quote"'))
    with pytest.raises(conta.ContaError) as error:
        conta.resolve(ABC_SHA1, index_config(tmp_path, [publish(tmp_path / "site", index)]))
    for shard in ("unc", "percent", "placeholder", "quote"):
        assert f"skipped shard {shard}" in str(error.value)


def test_unsupported_index_location(tmp_path):
    write_store_blob(tmp_path / "store", ABC_SHA1, b"abc")
    index = index_text(local_shard(tmp_path / "store"), entry(ABC_SHA1, ABC_SHA256))
    with pytest.raises(conta.ContaError, match="unsupported index.location"):
        conta.resolve(ABC_SHA1, index_config(tmp_path, [publish(tmp_path / "site", index, location="../escape/index.json")]))


def test_batch_through_index(tmp_path):
    write_store_blob(tmp_path / "store", ABC_SHA1, b"abc")
    write_store_blob(tmp_path / "store", HELLO_SHA1, b"hello")
    index = index_text(local_shard(tmp_path / "store"), entry(ABC_SHA1, ABC_SHA256) + ",\n" + entry(HELLO_SHA1, HELLO_SHA256))
    paths = conta.resolve_all([ABC_SHA1, {"description": "hello.txt", "hash": "conta:" + HELLO_SHA1}], index_config(tmp_path, [publish(tmp_path / "site", index)]))
    assert paths == [tmp_path / "cache" / ABC_SHA1, tmp_path / "cache" / HELLO_SHA1]


def test_permanent_redirect(tmp_path):
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/old/"):
                self.send_response(308)
                self.send_header("Location", "/new/" + self.path[len("/old/"):])
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Length", "3")
            self.end_headers()
            self.wfile.write(b"abc")

        def log_message(self, *arguments):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        path = conta.resolve(ABC_SHA1, config(tmp_path, url_base=f"http://127.0.0.1:{server.server_port}/old/"))
        assert path.read_bytes() == b"abc"
    finally:
        server.shutdown()


def test_cached_index_interop_python_to_cpp(tmp_path):
    cli = conta_cli()
    write_store_blob(tmp_path / "store", ABC_SHA1, b"abc")
    index = index_text(local_shard(tmp_path / "store"), entry(ABC_SHA1, ABC_SHA256))
    conta.resolve(ABC_SHA1, index_config(tmp_path, [publish(tmp_path / "site", index)]))
    (tmp_path / "cache" / ABC_SHA1).unlink()
    environment = cli_environment(tmp_path / "cache")
    environment.pop("CONTA_URL")
    result = subprocess.run([str(cli), ABC_SHA1], capture_output=True, text=True, env=environment)
    assert result.returncode == 0, result.stderr
    assert Path(result.stdout.strip()) == tmp_path / "cache" / ABC_SHA1
    assert "refreshing the index" not in result.stderr


def test_network_index(tmp_path):
    if os.environ.get("RL_TOOLS_TEST_CONTA_NETWORK") != "1":
        pytest.skip("set RL_TOOLS_TEST_CONTA_NETWORK=1 to enable")
    blob = "b6bd082c60fcb27f42a2b5f2f635587d2b60d170"
    path = conta.resolve(blob, conta.Config(cache=str(tmp_path / "cache")))
    assert conta.sha1_file(path) == blob
    assert (tmp_path / "cache" / conta.CACHED_INDEX_NAME).exists()
