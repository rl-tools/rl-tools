"""Protocol tests for conta, mirroring tests/src/utils/conta.cpp: the Python
client must implement the exact cache layout and download protocol of include/conta/conta.h
so both languages interoperate on one cache. All network cases run offline via file:// URLs.
"""
import gzip
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
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
    if os.environ.get("CONTA_TEST_CLI"):
        return Path(os.environ["CONTA_TEST_CLI"])
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


def entry(sha1, sha256, shards='{"id": "local"}'):
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
            f'{{"\\u0073ha1": "{ABC_SHA1}", "sha256": "{ABC_SHA256}", "shards": [{{"id": "unknown"}}, 7, "legacy", {{"id": "a", "compressed": ["gz", "future", 7]}}], "description": "caf\\u00e9 \\ud83d\\ude00", "size": 3, "license": ["x"]}},\n'
            f'{{"sha1": "{ABC_SHA1}", "sha256": "{EMPTY_SHA256}", "shards": [{{"id": "a"}}], "description": "duplicate"}},\n'
            f'{{"sha1": "{EMPTY_SHA1}", "sha256": "not-hex", "shards": [{{"id": "a"}}]}},\n'
            '"not an object",\n'
            f'{{"sha1": "{EMPTY_SHA1}", "sha256": "{EMPTY_SHA256}", "shards": "wrong type", "description": 5}}\n'
            ']}')
    index = conta._parse_index(text.encode(), {ABC_SHA1, EMPTY_SHA1}, "test")
    assert index.entry_count == 3
    assert index.shard_urls == {"a": "https://a.example/{sha1}"}
    assert index.entries[ABC_SHA1].sha256 == ABC_SHA256
    assert index.entries[ABC_SHA1].shards == [conta._Shard("unknown", []), conta._Shard("a", ["gz", "future"])]
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
    index = index_text(shards, entry(ABC_SHA1, ABC_SHA256, '{"id": "unc"}, {"id": "percent"}, {"id": "placeholder"}, {"id": "quote"}'))
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


def gzip_store(tmp_path, content=b"abc", compressed=("gz",), sha256=ABC_SHA256):
    store = tmp_path / "store"
    store.mkdir()
    (store / f"{ABC_SHA1}.gz").write_bytes(gzip.compress(content, compresslevel=9, mtime=0))
    shards = json.dumps({"id": "local", "compressed": list(compressed)})
    index = index_text(local_shard(store), entry(ABC_SHA1, sha256, shards))
    return index_config(tmp_path, [publish(tmp_path / "site", index)])


@pytest.mark.parametrize("api", [conta.load, conta.resolve])
def test_plain_cache_precedes_gzip(tmp_path, monkeypatch, api):
    settings = config(tmp_path)
    write_store_blob(tmp_path / "cache", ABC_SHA1, b"abc")
    (tmp_path / "cache" / f"{ABC_SHA1}.gz").write_bytes(b"corrupt")
    monkeypatch.setattr(conta, "_stream", lambda *args, **kwargs: pytest.fail("cache hit used network"))
    result = api(ABC_SHA1, settings)
    assert result == (b"abc" if api is conta.load else tmp_path / "cache" / ABC_SHA1)


def test_gzip_cache_load_and_resolve(tmp_path, monkeypatch):
    settings = config(tmp_path)
    cache = tmp_path / "cache"
    cache.mkdir()
    compressed = cache / f"{ABC_SHA1}.gz"
    compressed.write_bytes(gzip.compress(b"abc", mtime=0))
    monkeypatch.setattr(conta, "_stream", lambda *args, **kwargs: pytest.fail("cache hit used network"))
    assert conta.load(ABC_SHA1, settings) == b"abc"
    assert not (cache / ABC_SHA1).exists()
    path = conta.resolve(ABC_SHA1, settings)
    assert path.read_bytes() == b"abc"
    assert compressed.exists()
    monkeypatch.setattr(conta, "_decode_gzip", lambda *args, **kwargs: pytest.fail("plain cache was decoded again"))
    assert conta.load(ABC_SHA1, settings) == b"abc"
    assert conta.resolve(ABC_SHA1, settings) == path
    assert not list(cache.glob("*.partial.*"))


def test_gzip_load_plain_policy_materializes_once(tmp_path, monkeypatch):
    settings = config(tmp_path)
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / f"{ABC_SHA1}.gz").write_bytes(gzip.compress(b"abc", mtime=0))
    assert conta.load(ABC_SHA1, settings, compressed_cache=False) == b"abc"
    assert (cache / ABC_SHA1).read_bytes() == b"abc"
    monkeypatch.setattr(conta, "_decode_gzip", lambda *args, **kwargs: pytest.fail("plain cache was decoded again"))
    assert conta.load(ABC_SHA1, settings, compressed_cache=False) == b"abc"


@pytest.mark.parametrize("api,keep_gzip", [(conta.load, True), (conta.resolve, False), (lambda reference, settings: conta.load(reference, settings, compressed_cache=False), False)])
def test_gzip_download_storage(tmp_path, monkeypatch, api, keep_gzip):
    settings = gzip_store(tmp_path)
    decode = conta._decode_gzip
    decodes = []

    def record_decode(*args, **kwargs):
        decodes.append(args[0])
        return decode(*args, **kwargs)

    monkeypatch.setattr(conta, "_decode_gzip", record_decode)
    result = api({"hash": ABC_SHA1, "description": "abc"}, settings)
    assert (result.read_bytes() if isinstance(result, Path) else result) == b"abc"
    assert len(decodes) == 1
    cache = tmp_path / "cache"
    assert (cache / f"{ABC_SHA1}.gz").exists() == keep_gzip
    assert (cache / ABC_SHA1).exists() != keep_gzip
    assert not list(cache.glob("*.partial.*"))


def test_load_raw_direct_and_index_fallback(tmp_path):
    settings = gzip_store(tmp_path, compressed=("future", "gzip"))
    write_store_blob(tmp_path / "store", ABC_SHA1, b"abc")
    assert conta.load(ABC_SHA1, settings) == b"abc"
    cache = tmp_path / "cache"
    assert (cache / ABC_SHA1).read_bytes() == b"abc"
    assert not (cache / f"{ABC_SHA1}.gz").exists()
    (cache / ABC_SHA1).unlink()
    assert conta.load(ABC_SHA1, config(tmp_path, store=tmp_path / "store")) == b"abc"
    assert not (cache / f"{ABC_SHA1}.gz").exists()


@pytest.mark.parametrize("payload", [b"invalid", b"", gzip.compress(b"abc", mtime=0)[:-4], gzip.compress(b"wrong", mtime=0)])
def test_bad_gzip_download_falls_back_to_raw(tmp_path, payload):
    settings = gzip_store(tmp_path)
    (tmp_path / "store" / f"{ABC_SHA1}.gz").write_bytes(payload)
    write_store_blob(tmp_path / "store", ABC_SHA1, b"abc")
    assert conta.load(ABC_SHA1, settings) == b"abc"
    cache = tmp_path / "cache"
    assert (cache / ABC_SHA1).exists()
    assert not (cache / f"{ABC_SHA1}.gz").exists()
    assert not list(cache.glob("*.partial.*"))


def test_gzip_sha256_mismatch_is_not_published(tmp_path):
    settings = gzip_store(tmp_path, sha256=EMPTY_SHA256)
    with pytest.raises(conta.ContaError, match="sha256 mismatch"):
        conta.load(ABC_SHA1, settings)
    cache = tmp_path / "cache"
    assert not (cache / ABC_SHA1).exists()
    assert not (cache / f"{ABC_SHA1}.gz").exists()
    assert not list(cache.glob("*.partial.*"))


@pytest.mark.parametrize("in_memory", [True, False])
def test_corrupt_cached_gzip_recovers_without_deleting_it(tmp_path, in_memory):
    cache = tmp_path / "cache"
    write_store_blob(cache, ABC_SHA1 + ".gz", gzip.compress(b"wrong", mtime=0))
    write_store_blob(tmp_path / "store", ABC_SHA1, b"abc")
    settings = config(tmp_path, store=tmp_path / "store")
    assert conta.load(ABC_SHA1, settings, compressed_cache=in_memory) == b"abc"
    assert (cache / ABC_SHA1).read_bytes() == b"abc"
    assert gzip.decompress((cache / f"{ABC_SHA1}.gz").read_bytes()) == b"wrong"


def test_cached_gzip_checks_cached_index_sha256(tmp_path):
    settings = gzip_store(tmp_path, sha256=EMPTY_SHA256)
    cache = tmp_path / "cache"
    write_store_blob(cache, ABC_SHA1 + ".gz", gzip.compress(b"abc", mtime=0))
    cached_index = (tmp_path / "site" / "version" / "1" / "generated" / "index.json").read_bytes()
    (cache / conta.CACHED_INDEX_NAME).write_bytes(cached_index)
    settings.entry_points = []
    with pytest.raises(conta.ContaError, match="sha256 mismatch"):
        conta.load(ABC_SHA1, settings)
    assert not (cache / ABC_SHA1).exists()


def test_compressed_policy_precedence(tmp_path, monkeypatch):
    settings = config(tmp_path)
    cache = tmp_path / "cache"
    write_store_blob(cache, ABC_SHA1 + ".gz", gzip.compress(b"abc", mtime=0))
    monkeypatch.delenv("CONTA_ROOT", raising=False)
    monkeypatch.setenv("CONTA_CACHE", str(cache))
    monkeypatch.setenv("CONTA_URL", "file:///missing/")
    monkeypatch.setenv("CONTA_COMPRESSED_CACHE", "0")
    assert not conta.config_from_environment().compressed_cache
    assert conta.load(ABC_SHA1, settings) == b"abc"
    assert not (cache / ABC_SHA1).exists()
    assert conta.load(ABC_SHA1, compressed_cache=True) == b"abc"
    assert not (cache / ABC_SHA1).exists()
    assert conta.load(ABC_SHA1) == b"abc"
    assert (cache / ABC_SHA1).exists()
    (cache / ABC_SHA1).unlink()
    settings.compressed_cache = False
    assert conta.load(ABC_SHA1, settings, compressed_cache=True) == b"abc"
    assert not (cache / ABC_SHA1).exists()
    assert not settings.compressed_cache
    assert conta.load(ABC_SHA1, settings) == b"abc"
    assert (cache / ABC_SHA1).exists()
    monkeypatch.setenv("CONTA_COMPRESSED_CACHE", "1")
    assert conta.config_from_environment().compressed_cache
    monkeypatch.delenv("CONTA_COMPRESSED_CACHE")
    assert conta.config_from_environment().compressed_cache


def test_root_gzip_is_read_only_offline_and_reuses_materialization(tmp_path, monkeypatch):
    root = tmp_path / "root"
    compressed = root / "data" / f"{ABC_SHA1}.gz"
    write_store_blob(root / "data", ABC_SHA1 + ".gz", gzip.compress(b"abc", mtime=0))
    settings = config(tmp_path, root=root)
    monkeypatch.setattr(conta, "_stream", lambda *args, **kwargs: pytest.fail("root used network"))
    assert conta.load(ABC_SHA1, settings) == b"abc"
    assert not (tmp_path / "cache").exists()
    assert conta.resolve(ABC_SHA1, settings) == tmp_path / "cache" / ABC_SHA1
    assert set((root / "data").iterdir()) == {compressed}
    monkeypatch.setattr(conta, "_decode_gzip", lambda *args, **kwargs: pytest.fail("materialized root was decoded again"))
    assert conta.load(ABC_SHA1, settings, compressed_cache=False) == b"abc"
    compressed.unlink()
    with pytest.raises(conta.ContaError, match="CONTA_ROOT"):
        conta.load(ABC_SHA1, settings)


def test_root_gzip_memory_needs_no_cache(tmp_path):
    root = tmp_path / "root"
    write_store_blob(root / "data", ABC_SHA1 + ".gz", gzip.compress(b"abc", mtime=0))
    settings = conta.Config(root=str(root))
    assert conta.load(ABC_SHA1, settings) == b"abc"
    with pytest.raises(conta.ContaError, match="cannot determine cache directory"):
        conta.resolve(ABC_SHA1, settings)


def test_empty_gzip_blob(tmp_path):
    cache = tmp_path / "cache"
    write_store_blob(cache, EMPTY_SHA1 + ".gz", gzip.compress(b"", mtime=0))
    assert conta.load(EMPTY_SHA1, config(tmp_path)) == b""
    assert not (cache / EMPTY_SHA1).exists()
    assert conta.resolve(EMPTY_SHA1, config(tmp_path)).read_bytes() == b""


def test_interrupted_gzip_download_leaves_no_partial(tmp_path, monkeypatch):
    settings = gzip_store(tmp_path)
    stream = conta._stream

    def interrupted(url, output, digests=()):
        if url.endswith(".gz"):
            output.write(b"partial")
            raise OSError("interrupted")
        return stream(url, output, digests)

    monkeypatch.setattr(conta, "_stream", interrupted)
    with pytest.raises(conta.ContaError, match="interrupted"):
        conta.load(ABC_SHA1, settings)
    cache = tmp_path / "cache"
    assert not (cache / ABC_SHA1).exists()
    assert not (cache / f"{ABC_SHA1}.gz").exists()
    assert not list(cache.glob("*.partial.*"))


def test_concurrent_memory_and_path_publication(tmp_path):
    settings = gzip_store(tmp_path)
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda api: api(ABC_SHA1, settings), [conta.load, conta.resolve, conta.load, conta.resolve]))
    assert results[0] == results[2] == b"abc"
    assert results[1].read_bytes() == results[3].read_bytes() == b"abc"
    cache = tmp_path / "cache"
    if (cache / f"{ABC_SHA1}.gz").exists():
        assert gzip.decompress((cache / f"{ABC_SHA1}.gz").read_bytes()) == b"abc"
    assert not list(cache.glob("*.partial.*"))


@pytest.mark.parametrize("suffix", ["?token=value", "?token=value#part", "#part"])
def test_gzip_url_suffix_precedes_query_and_raw_fallback_stays_in_shard(tmp_path, monkeypatch, suffix):
    shards = '"first": {"url": "https://first.example/{sha1}' + suffix + '"}, "second": {"url": "https://second.example/{sha1}"}'
    locations = '{"id": "first", "compressed": ["unknown", "gz"]}, {"id": "second", "compressed": ["gz"]}'
    index = index_text(shards, entry(ABC_SHA1, ABC_SHA256, locations))
    settings = index_config(tmp_path, [publish(tmp_path / "site", index)])
    stream = conta._stream
    attempts = []

    def local_stream(url, output, digests=()):
        if not url.startswith("https:"):
            return stream(url, output, digests)
        attempts.append(url)
        if "first.example" in url:
            raise OSError("unavailable")
        output.write(gzip.compress(b"abc", mtime=0))

    monkeypatch.setattr(conta, "_stream", local_stream)
    assert conta.load(ABC_SHA1, settings) == b"abc"
    assert attempts == [f"https://first.example/{ABC_SHA1}.gz{suffix}", f"https://first.example/{ABC_SHA1}{suffix}", f"https://second.example/{ABC_SHA1}.gz"]


@pytest.mark.parametrize("api", [conta.load, conta.resolve])
def test_failed_cache_publication_removes_own_temporary_files(tmp_path, monkeypatch, api):
    settings = gzip_store(tmp_path)
    replace = os.replace

    def fail_blob_publication(source, target):
        if Path(target).name in (ABC_SHA1, ABC_SHA1 + ".gz"):
            raise OSError("publication failed")
        return replace(source, target)

    monkeypatch.setattr(os, "replace", fail_blob_publication)
    with pytest.raises(conta.ContaError, match="publication failed"):
        api(ABC_SHA1, settings)
    cache = tmp_path / "cache"
    assert not (cache / ABC_SHA1).exists()
    assert not (cache / f"{ABC_SHA1}.gz").exists()
    assert not list(cache.glob("*.partial.*"))


@pytest.mark.parametrize("representation,api", [("raw", conta.load), ("gz", conta.load), ("gz", conta.resolve)])
def test_cache_publication_tolerates_concurrent_winner(tmp_path, monkeypatch, representation, api):
    settings = gzip_store(tmp_path, compressed=("gz",) if representation == "gz" else ())
    write_store_blob(tmp_path / "store", ABC_SHA1, b"abc")
    cache = tmp_path / "cache"
    target = cache / (ABC_SHA1 + ".gz" if representation == "gz" and api is conta.load else ABC_SHA1)
    winning_bytes = gzip.compress(b"abc", mtime=0) if target.suffix == ".gz" else b"abc"
    replace = os.replace

    def concurrent_publication(source, destination):
        if Path(destination) == target:
            target.write_bytes(winning_bytes)
            raise PermissionError("a concurrent publisher has already created the file")
        return replace(source, destination)

    monkeypatch.setattr(os, "replace", concurrent_publication)
    result = api(ABC_SHA1, settings)
    assert (result.read_bytes() if isinstance(result, Path) else result) == b"abc"
    assert target.read_bytes() == winning_bytes
    assert not list(cache.glob("*.partial.*"))


@pytest.mark.parametrize("representation,api", [("raw", conta.load), ("gz", conta.load), ("gz", conta.resolve)])
def test_cache_publication_rejects_corrupt_concurrent_target(tmp_path, monkeypatch, representation, api):
    settings = gzip_store(tmp_path, compressed=("gz",) if representation == "gz" else ())
    if representation == "raw":
        write_store_blob(tmp_path / "store", ABC_SHA1, b"abc")
    cache = tmp_path / "cache"
    target = cache / (ABC_SHA1 + ".gz" if representation == "gz" and api is conta.load else ABC_SHA1)
    replace = os.replace

    def conflicting_publication(source, destination):
        if Path(destination) == target:
            target.write_bytes(b"x" * Path(source).stat().st_size)
            raise PermissionError("publication failed")
        return replace(source, destination)

    monkeypatch.setattr(os, "replace", conflicting_publication)
    with pytest.raises(conta.ContaError, match="publication failed"):
        api(ABC_SHA1, settings)
    assert target.read_bytes() == b"x" * target.stat().st_size
    assert not list(cache.glob("*.partial.*"))


def test_empty_gzip_download(tmp_path):
    store = tmp_path / "store"
    write_store_blob(store, EMPTY_SHA1 + ".gz", gzip.compress(b"", mtime=0))
    index = index_text(local_shard(store), entry(EMPTY_SHA1, EMPTY_SHA256, '{"id": "local", "compressed": ["gz"]}'))
    settings = index_config(tmp_path, [publish(tmp_path / "site", index)])
    assert conta.load(EMPTY_SHA1, settings) == b""
    assert (tmp_path / "cache" / f"{EMPTY_SHA1}.gz").exists()
    assert not (tmp_path / "cache" / EMPTY_SHA1).exists()


def test_gzip_cache_interop_python_to_cpp(tmp_path):
    cli = conta_cli()
    settings = gzip_store(tmp_path)
    assert conta.load(ABC_SHA1, settings) == b"abc"
    cache = tmp_path / "cache"
    assert not (cache / ABC_SHA1).exists()
    result = subprocess.run([str(cli), ABC_SHA1], capture_output=True, text=True, env=cli_environment(cache))
    assert result.returncode == 0, result.stderr
    assert Path(result.stdout.strip()) == cache / ABC_SHA1
    assert (cache / ABC_SHA1).read_bytes() == b"abc"
    assert gzip.decompress((cache / f"{ABC_SHA1}.gz").read_bytes()) == b"abc"
    assert "downloading" not in result.stderr
    assert not list(cache.glob("*.partial.*"))
