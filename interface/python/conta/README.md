## Conta: Content Addressed Blob Storage

Use `load()` when you want the blob's bytes in memory. It keeps downloaded gzip copies compressed in the cache by default, saving disk space. Use `resolve()` when a library needs a filename, or for seeking, mmap, and blobs too large to load into memory.

```python
import conta

data = conta.load(reference)                          # bytes
path = conta.resolve(reference)                       # pathlib.Path, plain file
data = conta.load(reference, compressed_cache=False)  # bytes, keep a plain cache
```

`reference` is a SHA-1 or a manifest reference accepted by the existing path API. `resolve_all(references)` and the `conta` CLI still return plain filesystem paths.

Both APIs share the same cache directory: `<sha1>` contains the original bytes and `<sha1>.gz` contains gzip bytes. Existing plain files take precedence. When only gzip is cached, `load()` decodes directly into memory; `resolve()` and `load(..., compressed_cache=False)` materialize a verified plain file for subsequent calls. Existing alternate copies are preserved. A cold gzip download retains only the requested cache representation; raw-only sources remain raw, with no local recompression or automatic cache migration.

To avoid repeated decompression while keeping the memory API, set `CONTA_COMPRESSED_CACHE=0` (default: `1`), or set `Config.compressed_cache=False`. Python's `compressed_cache` keyword overrides the supplied config. Without an explicit config, both APIs use `config_from_environment()`; an explicit config takes precedence over environment settings. The setting changes `load()`'s cache policy, never its returned bytes, and does not change `resolve()`.

```python
config = conta.config_from_environment()
config.compressed_cache = False
data = conta.load(reference, config)
```

`CONTA_CACHE` selects the cache directory. `CONTA_ROOT` selects an authoritative offline store: plain root files are used directly, and gzip root files can be loaded into memory or expanded into `CONTA_CACHE` for path access. Conta never writes into the root; if neither representation exists there, the lookup fails without network access. `CONTA_URL` remains a direct raw-blob URL override that bypasses index discovery.

The v1 index advertises per-location alternatives as `{"id": "primary", "compressed": ["gz"]}`. Clients try supported compressed formats before raw at each location and ignore unknown extensions. Downloads and gzip decoding verify the original content's SHA-1 and, when available, SHA-256 before returning data or publishing cache files atomically. Python uses its standard-library gzip decoder.

### C++

```cpp
#include <conta/conta.h>
#include <stdexcept>

std::vector<std::uint8_t> data;
std::string path, error;
if(!conta::load(sha1, data, error)) {
    throw std::runtime_error(error);
}
if(!conta::resolve(sha1, path, error)) {
    throw std::runtime_error(error);
}

auto config = conta::config_from_environment();
config.compressed_cache = false;
if(!conta::load(config, sha1, data, error)) {
    throw std::runtime_error(error);
}
```

The C++ API uses SHA-1 strings and returns success as `bool`, with diagnostics in `error`. RLtools enables gzip through its existing optional zlib dependency. Standalone header consumers can define `CONTA_ENABLE_ZLIB` and link zlib; without it, online lookups use raw copies, and a gzip-only offline store reports that decoding requires zlib.
