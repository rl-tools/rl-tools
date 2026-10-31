"""Maintain the conta-index checkout at <index_path>.

ingest: store files as <local_checkout_path>/data/<sha1> and <sha1>.gz by default, append descriptions to metadata.json, then scan.
scan: verify a shard checkout's blobs and optional gzip companions, update shards/<shard_name>.json, then build.
build: for every version/<n>/, combine its shards.json with the shared shards/*.json into version/<n>/generated/index.json and write its sha256 into version/<n>/meta.json.
check: validate everything and fail if any generated file is stale, hand-edited or re-encoded."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import zlib
from pathlib import Path

from conta import ContaError, _LFS_POINTER_PREFIX, _is_hex

INDEX_LOCATION = "generated/index.json"
VERSION_NAME = re.compile(r"^[1-9][0-9]*$")
FORMAT = re.compile(r"^[a-z0-9]+$")
FIRST_INDEXED_AT = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z")
INDEX_HELP = "path to the conta-index checkout containing version/ and shards/ (e.g. ../conta-index/)"


def warn(message):
    print(f"warning: {message}", file=sys.stderr)


def read_json(path):
    return json.loads(path.read_bytes().decode("utf-8"))


def lines_json(items):
    return "[\n" + ",\n".join(json.dumps(item, ensure_ascii=False) for item in items) + "\n]"


def load_meta(directory, version):
    path = directory / "meta.json"
    meta = read_json(path)
    if type(meta.get("format")) is not int or meta["format"] != version or (meta.get("index") or {}).get("location") != INDEX_LOCATION:
        raise ContaError(f"{path}: expected format {version} and index.location \"{INDEX_LOCATION}\"")
    return meta


def load_shard_table(directory):
    path = directory / "shards.json"
    shards = read_json(path)
    if not shards:
        raise ContaError(f"{path}: no shards")
    for shard, spec in shards.items():
        url = spec.get("url", "")
        if not spec.get("type") or ("{sha1}" not in url and "{sha256}" not in url):
            raise ContaError(f"{path}: shard {shard} needs a type and a url with {{sha1}} or {{sha256}}")
    return shards


def load_shard(root, shard):
    path = root / "shards" / f"{shard}.json"
    entries = read_json(path) if path.exists() else []
    seen = set()
    for entry in entries:
        sha1 = entry.get("sha1", "")
        if not _is_hex(sha1, 40) or not _is_hex(entry.get("sha256", ""), 64):
            raise ContaError(f"{path}: bad digests in {entry}")
        if sha1 in seen:
            raise ContaError(f"{path}: {sha1} is listed twice")
        for legacy_field in ("gzip", "compression", "compressed"):
            if legacy_field in entry:
                raise ContaError(f"{path}: replace {legacy_field} with a size map for {sha1}")
        size = entry.get("size")
        if (not isinstance(size, dict) or "raw" not in size
                or any(not FORMAT.fullmatch(format) for format in size)
                or any(type(value) is not int or value < 0 for value in size.values())):
            raise ContaError(f"{path}: size must be a map of lowercase alphanumeric formats to non-negative integer bytes, including raw, for {sha1}")
        if "gzip" in size:
            raise ContaError(f"{path}: use gz instead of gzip in size for {sha1}")
        if "first_indexed_at" in entry:
            timestamp = entry["first_indexed_at"]
            if not isinstance(timestamp, str) or not FIRST_INDEXED_AT.fullmatch(timestamp):
                raise ContaError(f"{path}: first_indexed_at must be a UTC timestamp YYYY-MM-DDTHH:MM:SSZ for {sha1}")
            try:
                datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ")
            except ValueError:
                raise ContaError(f"{path}: first_indexed_at is not a valid date and time for {sha1}") from None
        seen.add(sha1)
    return entries


def load_shards(root):
    return {path.stem: load_shard(root, path.stem) for path in sorted((root / "shards").glob("*.json"))}


def shard_order(entry):
    description = entry.get("description", "")
    parts = re.split(r"([0-9]+)", description)
    return [(1, int(part)) if part.isdigit() else (0, part.casefold()) for part in parts], description, entry["sha1"]


def save_shard(root, shard, entries):
    entries.sort(key=shard_order)
    path = root / "shards" / f"{shard}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_bytes((lines_json(entries) + "\n").encode("utf-8"))


def first_indexed_dates(sources):
    """Validate blob identities and derive dates without hiding unknown history."""
    identities = {}
    earliest = {}
    unknown = set()
    for shard, entries in sources.items():
        for entry in entries:
            sha1 = entry["sha1"]
            sha256, raw_size, previous_shard = identities.setdefault(sha1, (entry["sha256"], entry["size"]["raw"], shard))
            if entry["sha256"] != sha256:
                raise ContaError(f"{sha1}: sha256 differs between shards {previous_shard} and {shard}")
            if entry["size"]["raw"] != raw_size:
                raise ContaError(f"{sha1}: raw size differs between shards {previous_shard} and {shard}")
            if "first_indexed_at" not in entry:
                unknown.add(sha1)
                continue
            # Canonical UTC timestamps sort chronologically.
            timestamp = entry["first_indexed_at"]
            if sha1 not in earliest or timestamp < earliest[sha1]:
                earliest[sha1] = timestamp
    return {sha1: timestamp for sha1, timestamp in earliest.items() if sha1 not in unknown}


def combine(root, shards):
    # A version's download locations may omit the shard that first indexed a
    # blob. Its historical metadata still comes from all shared shard sources.
    sources = load_shards(root)
    dates = first_indexed_dates(sources)
    combined = {}
    for shard in shards:
        for entry in sources.get(shard, []):
            location = {"id": shard, "size": entry["size"]}
            existing = combined.get(entry["sha1"])
            if existing is None:
                combined[entry["sha1"]] = {"sha1": entry["sha1"], "sha256": entry["sha256"], "shards": [location], "description": entry.get("description", "")}
            else:
                if existing["sha256"] != entry["sha256"]:
                    raise ContaError(f"{entry['sha1']}: sha256 differs between shards {existing['shards'][0]['id']} and {shard}")
                if entry.get("description") and existing["description"] and entry["description"] != existing["description"]:
                    warn(f"{entry['sha1']}: description differs between shards ({existing['description']!r} from {existing['shards'][0]['id']}, {entry['description']!r} from {shard})")
                existing["shards"].append(location)
                existing["description"] = existing["description"] or entry.get("description", "")
    entries = [combined[sha1] for sha1 in sorted(combined)]
    for entry in entries:
        if entry["sha1"] in dates:
            entry["first_indexed_at"] = dates[entry["sha1"]]
        if not entry["description"]:
            warn(f"{entry['sha1']}: no description")
    table = ",\n".join(f"{json.dumps(shard)}: {json.dumps(spec, ensure_ascii=False)}" for shard, spec in shards.items())
    return entries, "{\n\"shards\": {\n" + table + "\n},\n\"entries\": " + lines_json(entries) + "\n}\n"


def generate_format_1(root, directory):
    meta = load_meta(directory, 1)
    entries, text = combine(root, load_shard_table(directory))
    index_bytes = text.encode("utf-8")
    meta["index"]["sha256"] = hashlib.sha256(index_bytes).hexdigest()
    return entries, index_bytes, (json.dumps(meta, indent=2) + "\n").encode("utf-8"), meta["index"]["sha256"]


GENERATORS = {1: generate_format_1}


def versions(root):
    directories = root / "version"
    if not directories.is_dir():
        raise ContaError(f"{root} is not a conta-index checkout: {directories} is not a directory")
    found = {}
    for directory in sorted(directories.iterdir()):
        if directory.name.startswith("."):
            continue
        if not directory.is_dir() or not VERSION_NAME.match(directory.name):
            raise ContaError(f"{directory}: everything in version/ is a directory named by a positive integer")
        found[int(directory.name)] = directory
    if not found:
        raise ContaError(f"{directories} contains no versions")
    for version in found:
        if version not in GENERATORS:
            raise ContaError(f"{found[version]}: this conta has no generator for version {version}, upgrade conta")
    return dict(sorted(found.items()))


def build(root):
    for version, directory in versions(root).items():
        entries, index_bytes, meta_bytes, index_sha256 = GENERATORS[version](root, directory)
        index = directory / INDEX_LOCATION
        index.parent.mkdir(exist_ok=True)
        index.write_bytes(index_bytes)
        (directory / "meta.json").write_bytes(meta_bytes)
        print(f"{index}: {len(entries)} entries, sha256 {index_sha256[:12]}… written to {directory / 'meta.json'}")


def check(root):
    available = versions(root)
    for version, directory in available.items():
        entries, index_bytes, meta_bytes, index_sha256 = GENERATORS[version](root, directory)
        index = directory / INDEX_LOCATION
        if not index.exists() or index.read_bytes() != index_bytes:
            raise ContaError(f"{index} is stale, hand-edited or re-encoded (line endings, encoding): run conta index build")
        if (directory / "meta.json").read_bytes() != meta_bytes:
            raise ContaError(f"{directory / 'meta.json'}: index.sha256 is stale, or the file was re-encoded: run conta index build")
        print(f"ok: version {version}, {len(read_json(directory / 'shards.json'))} shards, {len(entries)} entries, index sha256 {index_sha256[:12]}…")
    current_path = root / "current.json"
    if current_path.exists():
        current = read_json(current_path).get("version")
        if type(current) is not int or current not in available:
            raise ContaError(f"{current_path}: version {current!r} is not one of {list(available)}")


def digests(path, *, compressed=False):
    """Hash and count original bytes, rejecting LFS pointers and invalid gzip."""
    sha1, sha256 = hashlib.sha1(usedforsecurity=False), hashlib.sha256()
    size = 0
    try:
        with open(path, "rb") as file:
            prefix = file.read(len(_LFS_POINTER_PREFIX))
        if prefix == _LFS_POINTER_PREFIX:
            raise ContaError(f'{path} is a git-lfs pointer, run "git lfs pull" in the shard checkout')
        if compressed and not prefix.startswith(b"\x1f\x8b"):
            raise ContaError(f"{path}: not a gzip file")
        with (gzip.open if compressed else open)(path, "rb") as file:
            for chunk in iter(lambda: file.read(1 << 20), b""):
                sha1.update(chunk)
                sha256.update(chunk)
                size += len(chunk)
    except (OSError, EOFError, zlib.error) as error:
        raise ContaError(f"{path}: cannot read {'gzip blob' if compressed else 'blob'}: {error}") from error
    return sha1.hexdigest(), sha256.hexdigest(), size


def verify_blob(path, expected, *, compressed=False):
    if digests(path, compressed=compressed) != expected:
        raise ContaError(f"{path}: {'decompressed content' if compressed else 'content'} does not match expected sha1/sha256/size {expected}")
    return path.stat().st_size


def store_blob(source, destination, expected, *, compressed=False):
    """Publish only complete, verified files; an existing blob is verified, not rewritten."""
    if destination.exists():
        verify_blob(destination, expected, compressed=compressed)
        return
    # Publish by renaming a sibling, not link + unlink: on NFS, unlinking a name that is still referenced (e.g. by virtiofsd) leaves it behind as a .nfs* hard link to the blob. A concurrent ingestion can only be replaced by a verified copy of the same content.
    descriptor, name = tempfile.mkstemp(prefix=".ingest-", dir=destination.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        if compressed:
            with open(source, "rb") as original, open(temporary, "wb") as output:
                with gzip.GzipFile(filename="", mode="wb", fileobj=output, compresslevel=9, mtime=0) as zipped:
                    shutil.copyfileobj(original, zipped, length=1 << 20)
        else:
            shutil.copyfile(source, temporary)
        verify_blob(temporary, expected, compressed=compressed)
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def descriptions_from_metadata(checkout):
    path = checkout / "metadata.json"
    if not path.exists():
        return {}
    descriptions = {}
    for entry in read_json(path):
        sha1 = str(entry.get("hash", "")).lower()
        if not _is_hex(sha1, 40):
            warn(f"{path}: invalid hash {entry.get('hash')!r}")
        elif sha1 in descriptions:
            warn(f"{path}: {sha1} listed twice ({descriptions[sha1]!r}, {entry.get('description')!r}); keeping the first")
        else:
            descriptions[sha1] = str(entry.get("description", ""))
    return descriptions


def require_listed(root, shard):
    tables = {directory / "shards.json": load_shard_table(directory) for directory in versions(root).values()}
    if not any(shard in table for table in tables.values()):
        registered = ", ".join(sorted(set().union(*tables.values())))
        raise ContaError(f"shard {shard} is not registered (registered: {registered}): add it to {' or '.join(map(str, tables))}")


def description_for(file):
    path = Path(file)
    return path.name if path.is_absolute() or ".." in path.parts else path.as_posix()


def append_metadata(path, additions):
    """Append entries in the file's one-per-line style, aligning "hash" with the last entry, so the hand-kept formatting survives."""
    text = path.read_bytes().decode("utf-8") if path.exists() else "[\n]\n"
    end = text.rfind("]")
    if end < 0:
        raise ContaError(f"{path}: not a JSON array")
    head = text[:end].rstrip()
    last_line = head.split("\n")[-1]
    hash_column = last_line.find('"hash"') if last_line.lstrip().startswith("{") else -1
    lines = []
    for description, sha1 in additions:
        prefix = '{"description": ' + json.dumps(description, ensure_ascii=False) + ","
        lines.append(prefix + " " * max(hash_column - len(prefix), 1) + '"hash": "' + sha1 + '"}')
    separator = "\n" if head.endswith("[") else ",\n"
    path.write_bytes((head + separator + ",\n".join(lines) + "\n" + text[end:]).encode("utf-8"))
    read_json(path)


def parallel_map(function, items, jobs, *, ordered=True):
    """Run file work concurrently, optionally yielding in completion order."""
    if jobs < 1:
        raise ValueError("jobs must be a positive integer")
    if jobs == 1:
        yield from map(function, items)
        return
    # Hashing and zlib release the GIL; threads can use multiple CPU cores.
    executor = ThreadPoolExecutor(max_workers=jobs)
    try:
        if ordered:
            yield from executor.map(function, items)
        else:
            futures = [executor.submit(function, item) for item in items]
            for future in as_completed(futures):
                yield future.result()
    finally:
        # On failure, cancel queued work and let active atomic writes finish.
        executor.shutdown(wait=True, cancel_futures=True)


def ingest(root, shard, checkout, files, *, compress=True, jobs=1, progress=True):
    require_listed(root, shard)
    checkout = Path(checkout).expanduser().resolve()
    data = checkout / "data"
    data.mkdir(parents=True, exist_ok=True)
    descriptions = descriptions_from_metadata(checkout)

    def ingest_file(file):
        if not Path(file).is_file():
            raise ContaError(f"{file} is not a file")
        expected = digests(file)
        sha1 = expected[0]
        blob = data / sha1
        store_blob(file, blob, expected)
        if compress:
            store_blob(blob, data / f"{sha1}.gz", expected, compressed=True)
        return file, sha1

    additions = []
    # Consume in argument order so duplicate inputs retain the first description
    # regardless of which worker finishes first. Only this thread writes metadata.
    for file, sha1 in parallel_map(ingest_file, files, jobs):
        description = description_for(file)
        if sha1 in descriptions:
            if descriptions[sha1] != description:
                warn(f"{file}: {sha1} is already described as {descriptions[sha1]!r}, keeping that")
            continue
        descriptions[sha1] = description
        additions.append((description, sha1))
    if additions:
        append_metadata(checkout / "metadata.json", additions)
    scan(root, shard, checkout, jobs=jobs, progress=progress)


def inspect_blob(path):
    """Verify a raw blob and its optional gzip companion without changing metadata."""
    sha1, sha256, raw_size = digests(path)
    if sha1 != path.name:
        raise ContaError(f"{path}: content hashes to {sha1}")
    size = {"raw": raw_size}
    companion = path.with_name(f"{path.name}.gz")
    if companion.exists():
        size["gz"] = verify_blob(companion, (sha1, sha256, raw_size), compressed=True)
    return path, sha256, size


class ScanProgress:
    """Render only from the coordinating thread, leaving redirected output clean."""

    def __init__(self, total, *, enabled=True):
        self.total = total
        self.completed = 0
        self.stream = sys.stderr
        self.enabled = enabled and self.stream.isatty()

    def render(self):
        if self.enabled:
            filled = 24 * self.completed // self.total if self.total else 24
            percent = 100 * self.completed // self.total if self.total else 100
            bar = "#" * filled + "-" * (24 - filled)
            print(f"\rScan [{bar}] {self.completed}/{self.total} blobs ({percent:3d}%)", end="", file=self.stream, flush=True)

    def advance(self):
        self.completed += 1
        self.render()

    def __enter__(self):
        self.render()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.enabled:
            print(file=self.stream, flush=True)


def scan(root, shard, checkout, *, jobs=1, progress=True):
    require_listed(root, shard)
    checkout = Path(checkout).expanduser().resolve()
    data = checkout / "data"
    if not data.is_dir():
        raise ContaError(f"{data} is not a directory")
    descriptions = descriptions_from_metadata(checkout)
    entries = {entry["sha1"]: entry for entry in load_shard(root, shard)}
    additions = []
    on_disk = set()
    paths = []
    for path in sorted(data.iterdir()):
        name = path.name
        if name.endswith(".gz") and _is_hex(name[:-3], 40):
            if not path.is_file() or not (data / name[:-3]).is_file():
                raise ContaError(f"{path}: gzip companions require a file and a matching raw blob")
            continue  # Verified together with the raw blob below.
        if not path.is_file() or not _is_hex(name, 40):
            warn(f"{path}: not a sha1-named blob, skipped")
            continue
        on_disk.add(name)
        paths.append(path)
    with ScanProgress(len(paths), enabled=progress) as bar:
        for path, sha256, measured_size in parallel_map(inspect_blob, paths, jobs, ordered=False):
            name = path.name
            raw_size = measured_size["raw"]
            if name not in entries:
                entries[name] = {"sha1": name, "sha256": sha256, "description": "", "size": {"raw": raw_size}}
                additions.append(entries[name])
            elif entries[name]["sha256"] != sha256:
                raise ContaError(f"{path}: sha256 differs from shards/{shard}.json")
            size = entries[name]["size"]
            if size["raw"] != raw_size:
                raise ContaError(f"{path}: raw size differs from shards/{shard}.json")
            companion = data / f"{name}.gz"
            if "gz" in measured_size:
                gz_size = measured_size["gz"]
                if "gz" in size and size["gz"] != gz_size:
                    raise ContaError(f"{companion}: gz size differs from shards/{shard}.json")
                size["gz"] = gz_size
            elif "gz" in size:
                raise ContaError(f"{companion}: advertised gzip is missing; shards are append-only")
            if name in descriptions:
                entries[name]["description"] = descriptions[name]
            bar.advance()
    for sha1 in sorted(set(descriptions) - on_disk):
        warn(f"metadata.json lists {sha1} ({descriptions[sha1]}) but {data / sha1} does not exist")
    for sha1 in sorted(set(entries) - on_disk):
        raise ContaError(f"{sha1} is in shards/{shard}.json but missing from {data}: shards are append-only")
    # Check mirror consistency before publishing the shard or generated files.
    sources = load_shards(root)
    sources[shard] = list(entries.values())
    first_indexed_dates(sources)
    if additions:
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        for entry in additions:
            entry["first_indexed_at"] = timestamp
    save_shard(root, shard, list(entries.values()))
    print(f"{root / 'shards' / f'{shard}.json'}: {len(entries)} entries")
    build(root)


def positive_jobs(value):
    try:
        jobs = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("jobs must be a positive integer") from None
    if jobs < 1:
        raise argparse.ArgumentTypeError("jobs must be a positive integer")
    return jobs


def main(arguments):
    parser = argparse.ArgumentParser(
        prog="conta index",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Arguments in <angle brackets> are placeholders; replace them with your values.\n\n"
               "Example scan:\n  conta index scan ../conta-index procthor ../conta-shards/procthor/",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    ingest_parser = commands.add_parser(
        "ingest", help="store raw blobs and gzip companions, then scan",
        description="Store files in the shard checkout, update its metadata, then scan and rebuild the index.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Arguments in <angle brackets> are placeholders; replace them with your values.\n\n"
               "Example:\n  conta index ingest ../conta-index procthor ../conta-shards/procthor/ house.glb",
    )
    ingest_parser.add_argument("index", metavar="<index_path>", help=INDEX_HELP)
    ingest_parser.add_argument("--no-gzip", action="store_true", help="do not create gzip companions (existing copies are still verified)")
    ingest_parser.add_argument("--no-progress", action="store_true", help="hide the final scan's terminal progress bar")
    ingest_parser.add_argument("-j", "--jobs", type=positive_jobs, default=1, help="number of parallel file workers, including the final scan (default: 1)")
    ingest_parser.add_argument("shard", metavar="<shard_name>", help="registered shard name from version/<n>/shards.json (e.g. procthor)")
    ingest_parser.add_argument("checkout", metavar="<local_checkout_path>", help="path to the local shard checkout (e.g. ../conta-shards/procthor/)")
    ingest_parser.add_argument("files", metavar="<file>", nargs="+", help="one or more source files to ingest")
    scan_parser = commands.add_parser(
        "scan", help="verify blobs and gzip companions, then build",
        description="Verify the checkout's raw blobs and gzip companions, update shards/<shard_name>.json, and rebuild the index.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Arguments in <angle brackets> are placeholders; replace them with your values.\n\n"
               "Example:\n  conta index scan ../conta-index procthor ../conta-shards/procthor/",
    )
    scan_parser.add_argument("index", metavar="<index_path>", help=INDEX_HELP)
    scan_parser.add_argument("shard", metavar="<shard_name>", help="registered shard name from version/<n>/shards.json (e.g. procthor)")
    scan_parser.add_argument("checkout", metavar="<local_checkout_path>", help="path to the local shard checkout containing data/ (e.g. ../conta-shards/procthor/)")
    scan_parser.add_argument("--no-progress", action="store_true", help="hide the terminal progress bar")
    scan_parser.add_argument("-j", "--jobs", type=positive_jobs, default=1, help="number of parallel verification workers (default: 1)")
    commands.add_parser("build", help="rebuild all index versions").add_argument("index", metavar="<index_path>", help=INDEX_HELP)
    commands.add_parser("check", help="validate generated indexes").add_argument("index", metavar="<index_path>", help=INDEX_HELP)
    args = parser.parse_args(arguments)
    root = Path(args.index).expanduser().resolve()
    try:
        if args.command == "ingest":
            ingest(root, args.shard, args.checkout, args.files, compress=not args.no_gzip, jobs=args.jobs, progress=not args.no_progress)
        elif args.command == "scan":
            scan(root, args.shard, args.checkout, jobs=args.jobs, progress=not args.no_progress)
        elif args.command == "build":
            build(root)
        else:
            check(root)
    except ContaError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0
