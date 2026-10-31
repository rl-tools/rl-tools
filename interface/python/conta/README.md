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

Downloads and gzip decoding are verified as the index contract below requires; Python uses its standard-library gzip decoder.

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

### Index contract (format 1)

Clients hardcode the entry points of the format they implement, tried in order:

1. `https://conta.rl.tools/version/1/meta.json`
2. `https://raw.githubusercontent.com/rl-tools/conta-index/main/version/1/meta.json`

Blobs are immutable and verified by hash, so the index is trusted for availability only. A usable cached blob never touches the network; the cache rules above are part of the contract. On a blob-cache miss the client consults its cached index. A lookup matches an entry by `sha1` or `sha256`; for example, one blob might have these locations:

```json
{
  "shards": [
    {"id": "primary", "size": {"raw": 1048576, "gz": 210000}},
    {"id": "mirror", "size": {"raw": 1048576}}
  ]
}
```

For each location in order, look up its `id` in the top-level shard table and substitute the entry's digest into that shard's `url`. The keys in `size` advertise available formats; their order does not express a preference or a sequence of compression layers. Clients select supported extensions in their own preference order and ignore unknown extensions. Test key presence, not whether its size is truthy: zero bytes is valid. Every location includes `raw`; if there are no supported compressed formats, try raw directly. We own the extension namespace: `gz` means gzip, and future tokens will define their decoding algorithms without changing the index schema.

`raw` uses the URL unchanged. Each other key is the literal file extension without the dot. Append `"." + extension` to the URL's path, preserving any query string or fragment: `https://host/data/<sha1>?download=1` becomes `https://host/data/<sha1>.gz?download=1` for `gz`. The decoding algorithm is implicit in our namespace; the index needs no separate algorithm or suffix mapping, compressed URL template, or preliminary `HEAD` request to obtain sizes.

Hash the original bytes while streaming decompression or the raw response. Verify the requested digest and the entry's SHA-1 and SHA-256 before returning decoded bytes or atomically publishing a cache file. For a cold gzip download, `load()` retains the verified gzip file and returns the bytes from the same decoding pass; path access or a disabled compressed cache retains only the plain file. A failed download, decompression, or hash check discards operation-owned temporary files and returns no partial data. Try another supported advertised format after a compressed download fails, then raw at the same shard, then proceed to the next shard after a raw failure. The example above tries `primary/<sha1>.gz`, `primary/<sha1>`, then `mirror/<sha1>`.

If the hash is absent or every location fails, refresh once: fetch `meta.json`, and only when `index.sha256` differs from the cached index fetch `index.location` (relative to `meta.json`), verify its SHA-256 (a CDN can briefly serve `meta.json` and the index from different commits; on mismatch retry or try the next entry point), replace the index cache and retry the lookup. Report failure if that retry also fails.

### Maintaining the index

`conta index {ingest,scan,build,check}` maintains a checkout of [conta-index](https://github.com/rl-tools/conta-index), the repository the entry points serve, and runs from its root:

- `version/<n>/` — one directory per contract version. Version 1 has no released consumers yet and can change in place. Once released, every version stays published and keeps being generated from the shared shard files for as long as the store exists, so a client hardcodes its version; extensions then go into a new version. `conta index` has one generator per version and refuses a version directory it has no generator for.
- `version/<n>/meta.json` — hand-written except `index.sha256`: `format` (equal to `<n>`), `index` (`location`, and `sha256` of the generated index written by `build`).
- `version/<n>/shards.json` — hand-written shard table, id → `{type, url, repo}` in order of preference. `url` is the http transport, a template with `{sha1}` or `{sha256}`, so a changed download location is an edit here and no client update. `type` pre-empts transports where inserting a hash into a url is not enough; a new type needs a client that knows it. `repo` is informational.
- `shards/<id>.json` — source of truth per shard, shared by all versions: entries `{sha1, sha256, description, size}`, sorted by description with numbers compared by value (`Train-2` before `Train-10`), one per line; `scan` restores the order. `size` maps available formats to byte counts, for example `{"raw": 1048576, "gz": 210000}`. Run `conta index scan <id> <checkout>` to verify and measure every `data/<sha1>`, verify any `data/<sha1>.gz` decompresses to the same SHA-1 and SHA-256, record its compressed byte count as `size.gz`, take descriptions from the checkout's optional `metadata.json`, and rebuild. Scanning discovers companions; it does not create them. Currently it verifies raw and gzip copies; size metadata for other extensions already in a record is preserved, but their files require tooling that understands those formats to verify them.
- `version/<n>/generated/index.json` — written by `conta index build`, never edited: the shard table plus `entries` `{sha1, sha256, shards, description}` merged across shards. Each entry's `shards` is an ordered list of `{id, size}` locations, with each location's own format availability and byte counts. Committed so the raw GitHub URL serves it; `conta index check` fails when it or `index.sha256` does not match the sources. `build` and `check` validate stored source metadata without reading shard blob files.

Every `size` map requires `raw`; its value is the original file's size in bytes. Other keys are file extensions without a leading dot (non-empty lowercase alphanumeric tokens), and their values are the sizes of those files on the host. Use `gz` for gzip. Every value must be a non-negative integer; `0` is a valid size for an empty raw file. A missing extension means that format is unavailable. Raw sizes must agree across all shard copies of the same blob; compressed sizes can differ between shards.

A shard stores raw blobs named by their SHA-1, optionally accompanied by `<sha1>.gz`, in the `CONTA_ROOT` layout, so `CONTA_ROOT=<checkout>` serves a shard checkout offline. Both digests always identify the original, uncompressed bytes. Never delete or rewrite a raw blob or an advertised compressed companion; the index is append-only, so a newer index never has fewer entries or advertised copies. For raw and gzip copies, `scan` rejects corrupt files, orphan gzip companions, missing previously indexed copies, and changes to recorded byte counts.

Shard entries may also have `first_indexed_at`, the UTC time when that blob was first registered through that shard in the index repository, in `YYYY-MM-DDTHH:MM:SSZ` format (for example, `2026-10-10T18:42:00Z`). `scan` and `ingest` set it once for new shard entries. Rescanning, changing descriptions, and adding compressed copies preserve existing timestamps; older entries without a known date stay undated. This records registration there, not when the blob was produced or published.

Generated entries use the same `first_indexed_at` field for the earliest date across all copies of the blob in the shared `shards/*.json` sources, including shards outside that version's shard table. If any copy has an unknown date, `first_indexed_at` is omitted. `build` and `check` only read and aggregate stored dates; neither assigns the current time.

To ingest files into a new shard `<name>`, add `<name>` to `version/1/shards.json` with the url template under which `<checkout>/data` will be hosted, then run from the repository root:

```
conta index ingest <name> <checkout> <files>
```

It copies each file to `<checkout>/data/<sha1>` and creates `<checkout>/data/<sha1>.gz` by default. Compression streams at gzip level 9 with no embedded filename or timestamp. New files are verified and published atomically; existing copies are verified and never overwritten. It appends the file's path as given (just the file name for an absolute path) as its description to `<checkout>/metadata.json`, records the raw and gzip byte counts in `shards/<name>.json`, and rebuilds the index. A blob already described in `metadata.json` keeps its description. Re-ingesting it creates a missing gzip companion, so the same command can backfill previously ingested files.

To ingest raw copies only, use `conta index ingest --no-gzip <name> <checkout> <files>`. This still verifies and advertises any existing gzip companions. Files that have not been backfilled remain raw-only; there is no shard-wide assumption that gzip is available.

Upload `<checkout>/data` and `<checkout>/metadata.json` to the host before pushing the index. Serve `.gz` companions as gzip files (for example, `Content-Type: application/gzip`), without `Content-Encoding: gzip`; clients explicitly decompress them.
