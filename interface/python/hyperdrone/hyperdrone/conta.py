"""Client for the sha1 content-addressed asset store (https://huggingface.co/datasets/rl-tools/conta).

Protocol twin of the C++ client (include/conta/conta.h): the same environment contract
(CONTA_ROOT read-only store, CONTA_CACHE writable cache, CONTA_URL download base), the
same cache layout (flat <sha1> files), and the same download protocol
(<sha1>.partial.<pid>, verify, atomic rename), so Python and C++ processes share one
cache safely. Failures raise ContaError instead of aborting.

A reference is a bare 40-hex sha1, a "conta:<sha1>" string, or a store manifest entry
{"description": ..., "hash": ...}; the description is used in log messages only.
"""

import hashlib
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

DEFAULT_URL_BASE = "https://huggingface.co/datasets/rl-tools/conta/resolve/main/data/"
_LFS_POINTER_PREFIX = b"version https://git-lfs"
_DOWNLOAD_ATTEMPTS = 3
_CONNECT_TIMEOUT = 15


class ContaError(RuntimeError):
    pass


@dataclass
class Config:
    root: str = ""
    cache: str = ""
    url_base: str = DEFAULT_URL_BASE


def config_from_environment():
    config = Config()
    config.root = os.environ.get("CONTA_ROOT", "")
    config.cache = os.environ.get("CONTA_CACHE", "")
    if not config.cache:
        xdg_cache_home = os.environ.get("XDG_CACHE_HOME", "")
        home = os.environ.get("HOME", "")
        if xdg_cache_home:
            config.cache = str(Path(xdg_cache_home) / "rl_tools" / "conta")
        elif home:
            config.cache = str(Path(home) / ".cache" / "rl_tools" / "conta")
        elif os.name == "nt":
            local_app_data = os.environ.get("LOCALAPPDATA", "")
            if local_app_data:
                config.cache = str(Path(local_app_data) / "rl_tools" / "conta")
    config.url_base = os.environ.get("CONTA_URL", "") or DEFAULT_URL_BASE
    if not config.url_base.endswith("/"):
        config.url_base += "/"
    return config


def _reference_hash(reference):
    label = None
    if isinstance(reference, dict):
        if "hash" not in reference:
            raise ContaError(f"conta: reference has no \"hash\" key: {reference!r}")
        label = reference.get("description")
        reference = reference["hash"]
    reference = str(reference)
    if reference.startswith("conta:"):
        reference = reference[len("conta:"):]
    normalized = reference.lower()
    if len(normalized) != 40 or any(character not in "0123456789abcdef" for character in normalized):
        raise ContaError(f"conta: invalid hash \"{reference}\": expected 40 hexadecimal characters")
    return normalized, label if label is not None else normalized


def sha1_file(path):
    sha1 = hashlib.sha1()
    with open(path, "rb") as file:
        while True:
            chunk = file.read(64 * 1024)
            if not chunk:
                break
            sha1.update(chunk)
    return sha1.hexdigest()


def _is_lfs_pointer(path):
    try:
        if os.path.getsize(path) >= 1024:
            return False
        with open(path, "rb") as file:
            return file.read(len(_LFS_POINTER_PREFIX)) == _LFS_POINTER_PREFIX
    except OSError:
        return False


def _download(config, normalized_hash, label, target):
    temporary = Path(str(target) + f".partial.{os.getpid()}")
    url = config.url_base + normalized_hash
    described = normalized_hash if label == normalized_hash else f"{label} ({normalized_hash})"
    print(f"conta: downloading {described} from {url}", file=sys.stderr)
    error = None
    for attempt in range(_DOWNLOAD_ATTEMPTS):
        try:
            sha1 = hashlib.sha1()
            with urllib.request.urlopen(url, timeout=_CONNECT_TIMEOUT) as response, open(temporary, "wb") as file:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    sha1.update(chunk)
                    file.write(chunk)
            actual_hash = sha1.hexdigest()
            if actual_hash != normalized_hash:
                temporary.unlink(missing_ok=True)
                raise ContaError(
                    f"conta: hash mismatch for {normalized_hash}: downloaded content hashes to "
                    f"{actual_hash}; not caching: {url}"
                )
            try:
                os.replace(temporary, target)
            except OSError:
                if not target.exists():  # a concurrent download of the same content won the race
                    raise
                temporary.unlink(missing_ok=True)
            return
        except urllib.error.HTTPError as http_error:
            error = http_error
            if http_error.code < 500:
                break
        except (urllib.error.URLError, OSError, TimeoutError) as transient:
            error = transient
    temporary.unlink(missing_ok=True)
    raise ContaError(f"conta: download of {normalized_hash} failed ({error}): {url}. Check network connectivity.")


def resolve(reference, config=None):
    """Resolve one reference to a local file path, downloading into the cache if required."""
    if config is None:
        config = config_from_environment()
    normalized_hash, label = _reference_hash(reference)
    if config.root:
        path = Path(config.root) / "data" / normalized_hash
        if not path.exists():
            raise ContaError(
                f"conta: {normalized_hash} not found at {path}. CONTA_ROOT points to a read-only "
                f"store, so downloading is disabled; populate the store or unset CONTA_ROOT to "
                f"enable the download cache."
            )
        if _is_lfs_pointer(path):
            raise ContaError(
                f"conta: {path} is a git-lfs pointer file, not the blob. "
                f"Run \"git lfs pull\" in {config.root}."
            )
        return path
    if not config.cache:
        raise ContaError("conta: cannot determine cache directory: set CONTA_CACHE, XDG_CACHE_HOME, or HOME")
    target = Path(config.cache) / normalized_hash
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    _download(config, normalized_hash, label, target)
    return target


def resolve_all(references, config=None):
    if config is None:
        config = config_from_environment()
    return [resolve(reference, config) for reference in references]


def main(arguments=None):
    arguments = sys.argv[1:] if arguments is None else arguments
    if not arguments:
        print("Usage: python -m hyperdrone.conta <sha1|conta:sha1> [more hashes ...]", file=sys.stderr)
        print("Resolves content-addressed blobs to local filesystem paths (downloading into the cache if required) and prints one path per line.", file=sys.stderr)
        print("Environment: CONTA_ROOT (read-only store, disables downloads), CONTA_CACHE (cache directory), CONTA_URL (download base URL).", file=sys.stderr)
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
