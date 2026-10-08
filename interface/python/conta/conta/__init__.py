"""Client for the sha1 content-addressed asset store; index contract: https://github.com/rl-tools/conta-index (format 1).

Protocol twin of the C++ client (include/conta/conta.h): the same lookup order (<cache>/<sha1>, the
cached index <cache>/index-1.json, then the index entry points), the same
environment contract (CONTA_ROOT read-only store without network, CONTA_URL direct download without
the index, CONTA_CACHE cache directory), and the same download protocol (<name>.partial.<pid>-<random>,
verify the sha1 and the sha256 the index lists, atomic rename), so Python and C++ processes share one
cache safely. Failures raise ContaError.

A reference is a bare 40-hex sha1, a "conta:<sha1>" string, or a store manifest entry
{"description": ..., "hash": ...}; the description is used in log messages only.
"""

import hashlib
import io
import json
import os
import secrets
import socket
import ssl
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

ENTRY_POINTS = (
    "https://conta.rl.tools/version/1/meta.json",
    "https://raw.githubusercontent.com/rl-tools/conta-index/main/version/1/meta.json",
)
CACHED_INDEX_NAME = "index-1.json"
_LFS_POINTER_PREFIX = b"version https://git-lfs"
_DOWNLOAD_ATTEMPTS = 3
_CONNECT_TIMEOUT = 15
_STALL_SECONDS = 60
_STALL_BYTES_PER_SECOND = 1024
_HEX = set("0123456789abcdef")
_ALPHANUMERIC = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
_URL_CHARACTERS = _ALPHANUMERIC | set("._~:/?&=+,;@-")
_LOCATION_CHARACTERS = _ALPHANUMERIC | set("._~-/")


class ContaError(RuntimeError):
    pass


@dataclass
class Config:
    root: str = ""
    cache: str = ""
    url_base: str = ""
    entry_points: list = field(default_factory=lambda: list(ENTRY_POINTS))


def config_from_environment():
    config = Config()
    config.root = os.environ.get("CONTA_ROOT", "")
    config.cache = os.environ.get("CONTA_CACHE", "")
    if not config.cache:
        xdg_cache_home = os.environ.get("XDG_CACHE_HOME", "")
        home = os.environ.get("HOME", "")
        if xdg_cache_home:
            config.cache = str(Path(xdg_cache_home) / "conta")
        elif home:
            config.cache = str(Path(home) / ".cache" / "conta")
        elif os.name == "nt":
            local_app_data = os.environ.get("LOCALAPPDATA", "")
            if local_app_data:
                config.cache = str(Path(local_app_data) / "conta")
    config.url_base = os.environ.get("CONTA_URL", "")
    if config.url_base and not config.url_base.endswith("/"):
        config.url_base += "/"
    return config


def _is_hex(text, length):
    return isinstance(text, str) and len(text) == length and set(text) <= _HEX


def _reference_hash(reference):
    label = None
    if isinstance(reference, dict):
        if "hash" not in reference:
            raise ContaError(f"conta: reference has no \"hash\" key: {reference!r}")
        label = reference.get("description")
        reference = reference["hash"]
    reference = str(reference)
    normalized = (reference[len("conta:"):] if reference.startswith("conta:") else reference).lower()
    if _is_hex(normalized, 40):
        return normalized, label if label is not None else normalized
    if _is_hex(normalized, 64):
        raise ContaError(f"conta: \"{reference}\" is a sha256, but conta references are sha1 (40 hexadecimal characters); the sha256 in the index only verifies downloads")
    raise ContaError(f"conta: invalid hash \"{reference}\": expected a sha1 (40 hexadecimal characters)")


def _read_chunks(path):
    with open(path, "rb") as file:
        while True:
            chunk = file.read(64 * 1024)
            if not chunk:
                return
            yield chunk


def sha1_file(path):
    sha1 = hashlib.sha1(usedforsecurity=False)
    for chunk in _read_chunks(path):
        sha1.update(chunk)
    return sha1.hexdigest()


def _digests(path):
    sha1, sha256 = hashlib.sha1(usedforsecurity=False), hashlib.sha256()
    for chunk in _read_chunks(path):
        sha1.update(chunk)
        sha256.update(chunk)
    return sha1.hexdigest(), sha256.hexdigest()


def _is_lfs_pointer(path):
    try:
        if os.path.getsize(path) >= 1024:
            return False
        with open(path, "rb") as file:
            return file.read(len(_LFS_POINTER_PREFIX)) == _LFS_POINTER_PREFIX
    except OSError:
        return False


def _printable(text):
    return "".join(character if " " <= character <= "~" else "?" for character in str(text)[:4096])


def _unique_suffix():
    return f"{os.getpid()}-{secrets.token_hex(4)}"


class _RedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, new_url):  # urllib before Python 3.11 refuses 308
        return super().redirect_request(request, fp, 307 if code == 308 else code, message, headers, new_url)

    http_error_308 = urllib.request.HTTPRedirectHandler.http_error_302


_OPENER = urllib.request.build_opener(_RedirectHandler)


def _stream(url, file, digests=()):
    request = urllib.request.Request(url, headers={"User-Agent": "conta"})
    with _OPENER.open(request, timeout=_CONNECT_TIMEOUT) as response:
        read = getattr(response, "read1", response.read)
        window_start, window_bytes = time.monotonic(), 0
        while True:
            chunk = read(1 << 20)
            if not chunk:
                return
            for digest in digests:
                digest.update(chunk)
            file.write(chunk)
            window_bytes += len(chunk)
            now = time.monotonic()
            if now - window_start >= _STALL_SECONDS:  # the socket timeout only catches silence, not a trickle
                if window_bytes < _STALL_SECONDS * _STALL_BYTES_PER_SECOND:
                    raise TimeoutError(f"transfer below {_STALL_BYTES_PER_SECOND} bytes/s for {_STALL_SECONDS} s")
                window_start, window_bytes = now, 0


def _transient(error):
    if isinstance(error, urllib.error.HTTPError):
        return error.code in (408, 429) or error.code >= 500
    return isinstance(getattr(error, "reason", error), (TimeoutError, socket.timeout))


def _failure(error):
    reason = getattr(error, "reason", error)
    if isinstance(error, urllib.error.HTTPError):
        return f"HTTP {error.code}" + (", the host does not serve it" if error.code in (404, 410) else "")
    if isinstance(reason, ssl.SSLError):
        return f"TLS failed ({reason}); in an old environment point SSL_CERT_FILE at a current CA bundle"
    if isinstance(reason, FileNotFoundError):
        return "file not found"
    if isinstance(reason, (TimeoutError, socket.timeout)):
        return f"host unreachable or transfer stalled ({reason})"
    return str(reason)


def _attempt(action):
    for attempt in range(_DOWNLOAD_ATTEMPTS):
        try:
            action()
            return None
        except (urllib.error.URLError, OSError, ValueError) as error:
            if attempt + 1 == _DOWNLOAD_ATTEMPTS or not _transient(error):
                return _failure(error)
            retry_after = error.headers.get("Retry-After", "") if isinstance(error, urllib.error.HTTPError) else ""
            time.sleep(min(int(retry_after), 60) if retry_after.isdigit() else 1 << attempt)


def _fetch_bytes(url):
    buffer = io.BytesIO()

    def action():
        buffer.seek(0)
        buffer.truncate()
        _stream(url, buffer)

    failure = _attempt(action)
    return (None, failure) if failure else (buffer.getvalue(), None)


def _fetch_blob(url, sha1, sha256, description, target):
    temporary = Path(f"{target}.partial.{_unique_suffix()}")
    described = f"{sha1} ({_printable(description)})" if description else sha1
    print(f"conta: downloading {described} from {url}", file=sys.stderr)
    digests = []

    def action():
        digests[:] = [hashlib.sha1(usedforsecurity=False), hashlib.sha256()]
        with open(temporary, "wb") as file:
            _stream(url, file, digests)

    try:
        failure = _attempt(action)
        if failure:
            return failure
        actual_sha1, actual_sha256 = (digest.hexdigest() for digest in digests)
        if actual_sha1 != sha1:
            return f"hash mismatch: the content hashes to {actual_sha1}, not caching"
        if sha256 and actual_sha256 != sha256:
            return f"sha256 mismatch: the index lists {sha256} but the content hashes to {actual_sha256}, not caching"
        try:
            os.replace(temporary, target)
        except OSError as error:
            if not target.exists():  # a concurrent download of the same content won the race
                return f"failed to move {temporary} to {target}: {error}"
        return None
    finally:
        temporary.unlink(missing_ok=True)


def _parse_json(data):
    try:
        return json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, RecursionError):
        return None


def _parse_meta(data):
    document = _parse_json(data)
    if not isinstance(document, dict) or type(document.get("format")) is not int or document["format"] != 1:
        return None
    index = document.get("index")
    if not isinstance(index, dict) or not isinstance(index.get("location"), str) or not _is_hex(index.get("sha256"), 64):
        return None
    return index["location"], index["sha256"]


@dataclass
class _Entry:
    sha256: str
    shards: list
    description: str
    notice: str


@dataclass
class _Index:
    source: str
    sha256: str
    entry_count: int
    shard_urls: dict
    entries: dict


def _text(value):
    return value if isinstance(value, str) else ""


def _parse_index(data, wanted, source):
    document = _parse_json(data)
    if not isinstance(document, dict) or not isinstance(document.get("shards"), dict) or not isinstance(document.get("entries"), list):
        return None
    shard_urls = {shard: spec["url"] for shard, spec in document["shards"].items() if isinstance(spec, dict) and isinstance(spec.get("url"), str) and spec["url"]}
    entries, entry_count = {}, 0
    for item in document["entries"]:
        if not isinstance(item, dict) or not _is_hex(item.get("sha1"), 40) or not _is_hex(item.get("sha256"), 64):
            continue
        entry_count += 1
        sha1 = item["sha1"]
        if sha1 in wanted and sha1 not in entries:
            shards = item.get("shards") if isinstance(item.get("shards"), list) else []
            entries[sha1] = _Entry(item["sha256"], [shard for shard in shards if isinstance(shard, str) and shard], _text(item.get("description")), _text(item.get("notice")))
    return _Index(source, hashlib.sha256(data).hexdigest(), entry_count, shard_urls, entries)


def _is_document_url(url):
    for prefix in ("https://", "http://", "file:///"):
        if url.startswith(prefix):
            rest = url[len(prefix):]
            return bool(rest) and rest[0] in _ALPHANUMERIC and set(rest) <= _URL_CHARACTERS
    return False


def _resolve_location(entry_point, location):
    if "://" in location:
        return location if _is_document_url(location) else None
    if not location or location.startswith("/") or ".." in location or not set(location) <= _LOCATION_CHARACTERS or "/" not in entry_point:
        return None
    return entry_point[:entry_point.rfind("/") + 1] + location


def _shard_location(url_template, sha1, sha256):
    url = url_template.replace("{sha1}", sha1).replace("{sha256}", sha256)
    if "{" in url or "}" in url or not _is_document_url(url):
        return None
    return url


@dataclass
class _Session:
    wanted: set
    indexes: list = field(default_factory=list)
    attempted_urls: set = field(default_factory=set)
    next_entry_point: int = 0
    cached_index_loaded: bool = False
    cached_index_sha256: str = ""
    cached_index_entry_count: int = 0


def _find_entry(session, sha1):
    return next((index.entries[sha1] for index in session.indexes if sha1 in index.entries), None)


def _try_index(session, index, sha1, target, report):
    entry = index.entries.get(sha1)
    if entry is None:
        report.append(f"{index.source}: does not list it")
        return False
    if not entry.shards:
        report.append(f"{index.source}: lists no location")
    for shard in entry.shards:
        if shard not in index.shard_urls:
            report.append(f"{index.source}: shard {shard} is not in its shard table")
            continue
        url = _shard_location(index.shard_urls[shard], sha1, entry.sha256)
        if url is None:
            report.append(f"{index.source}: skipped shard {shard}, unsupported url {index.shard_urls[shard]}")
            continue
        if url in session.attempted_urls:
            continue
        session.attempted_urls.add(url)
        failure = _fetch_blob(url, sha1, entry.sha256, entry.description, target)
        if failure is None:
            return True
        report.append(f"{url}: {failure}")
    return False


def _load_cached_index(config, session):
    session.cached_index_loaded = True
    path = Path(config.cache) / CACHED_INDEX_NAME
    try:
        data = path.read_bytes()
    except OSError:
        return
    index = _parse_index(data, session.wanted, str(path))
    if index is None:
        return
    session.cached_index_sha256 = index.sha256
    session.cached_index_entry_count = index.entry_count
    session.indexes.append(index)


def _visit_entry_point(config, session, sha1, target, report):
    entry_point = config.entry_points[session.next_entry_point]
    session.next_entry_point += 1
    print(f"conta: refreshing the index from {entry_point}", file=sys.stderr)
    meta_data, failure = _fetch_bytes(entry_point)
    if failure:
        report.append(f"{entry_point}: {failure}")
        return False
    meta = _parse_meta(meta_data)
    if meta is None:
        report.append(f"{entry_point}: not a format 1 meta.json")
        return False
    location, listed_sha256 = meta
    if any(index.sha256 == listed_sha256 for index in session.indexes):
        report.append(f"{entry_point}: serves the index already searched")
        return False
    index_url = _resolve_location(entry_point, location)
    if index_url is None:
        report.append(f"{entry_point}: unsupported index.location {location}")
        return False
    data, failure = _fetch_bytes(index_url)
    if failure:
        report.append(f"{index_url}: {failure}")
        return False
    index = _parse_index(data, session.wanted, index_url)
    if index is None:
        report.append(f"{index_url}: not a format 1 index")
        return False
    if index.sha256 != listed_sha256:
        report.append(f"{index_url}: differs from the sha256 in meta.json (a publish may still be propagating), used for this call only")
    elif not session.cached_index_sha256 or index.entry_count >= session.cached_index_entry_count:  # entries are never removed, so a smaller index is older and must not replace the cached one
        temporary = Path(config.cache) / f"{CACHED_INDEX_NAME}.partial.{_unique_suffix()}"
        try:
            temporary.write_bytes(data)
            os.replace(temporary, Path(config.cache) / CACHED_INDEX_NAME)
            session.cached_index_sha256 = index.sha256
            session.cached_index_entry_count = index.entry_count
        except OSError:
            temporary.unlink(missing_ok=True)
    session.indexes.append(index)
    return _try_index(session, index, sha1, target, report)


def _failure_message(session, sha1, report):
    entry = _find_entry(session, sha1)
    message = f"conta: cannot resolve {sha1}"
    if entry is None:
        message += ", which no reachable index lists"
    elif entry.description:
        message += f" ({_printable(entry.description)})"
    if entry is not None and entry.notice:
        message += f"\nnotice: {_printable(entry.notice)}"
    for line in report:
        message += f"\n  {_printable(line)}"
    message += f"\nA copy you already have works with CONTA_URL=file:///<directory>/ holding it as {sha1} (verified), or with CONTA_ROOT=<directory> holding it as data/{sha1} (no network)."
    return message


def _resolve_one(config, session, reference):
    sha1, label = _reference_hash(reference)
    if config.root:
        path = Path(config.root) / "data" / sha1
        if not path.exists():
            raise ContaError(
                f"conta: {sha1} not found at {path}. CONTA_ROOT points to a read-only "
                f"store, so downloading is disabled; populate the store or unset CONTA_ROOT to "
                f"enable the download cache."
            )
        if _is_lfs_pointer(path):
            raise ContaError(f"conta: {path} is a git-lfs pointer file, not the blob. Run \"git lfs pull\" in {config.root}.")
        return path
    if not config.cache:
        raise ContaError("conta: cannot determine cache directory: set CONTA_CACHE, XDG_CACHE_HOME, or HOME")
    target = Path(config.cache) / sha1
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    if config.url_base:
        url = config.url_base + sha1
        failure = _fetch_blob(url, sha1, "", label if label != sha1 else "", target)
        if failure:
            raise ContaError(f"conta: download of {sha1} from {url} failed: {failure}")
        return target
    report = []
    if not session.cached_index_loaded:
        _load_cached_index(config, session)
    for index in list(session.indexes):
        if _try_index(session, index, sha1, target, report):
            return target
    while session.next_entry_point < len(config.entry_points):
        if _visit_entry_point(config, session, sha1, target, report):
            return target
    raise ContaError(_failure_message(session, sha1, report))


def _wanted(references):
    wanted = set()
    for reference in references:
        try:
            wanted.add(_reference_hash(reference)[0])
        except ContaError:
            pass
    return wanted


def resolve(reference, config=None):
    """Resolve one reference to a local file path, downloading into the cache if required."""
    if config is None:
        config = config_from_environment()
    return _resolve_one(config, _Session(_wanted([reference])), reference)


def resolve_all(references, config=None):
    """Resolve references in order with one walk over the index entry points; the first failure raises."""
    if config is None:
        config = config_from_environment()
    session = _Session(_wanted(references))
    return [_resolve_one(config, session, reference) for reference in references]


def main(arguments=None):
    arguments = sys.argv[1:] if arguments is None else arguments
    if not arguments:
        print("Usage: conta <sha1|conta:sha1> [more hashes ...]", file=sys.stderr)
        print("Resolves content-addressed blobs to local filesystem paths (downloading into the cache through the conta index if required) and prints one path per line.", file=sys.stderr)
        print("Environment: CONTA_ROOT (read-only store, disables the network), CONTA_CACHE (cache directory), CONTA_URL (fetch <url>/<sha1> directly, without the index).", file=sys.stderr)
        return 1
    try:
        paths = resolve_all(arguments)
    except ContaError as error:
        print(error, file=sys.stderr)
        return 1
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
