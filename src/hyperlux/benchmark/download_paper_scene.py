"""Fetch and verify the ProcTHOR asset pinned by the May 2026 paper benchmarks."""

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import urllib.request


REVISION = "d39a1ede85a505af01afd897aa51779f87db3cc1"
SHA256 = "76c463017f2a25683abaea3b29786294602bd2629b2f0725b7215dabf58f35aa"
SIZE = 32034928
URL = (
    "https://huggingface.co/datasets/rl-tools/test-data/resolve/"
    f"{REVISION}/ProcTHOR-Train-1.glb"
)
DEFAULT_DESTINATION = Path(__file__).resolve().parents[3] / "build/paper-benchmark/ProcTHOR-Train-1.glb"


def identity(path):
    sha256 = hashlib.sha256()
    sha1 = hashlib.sha1()
    size = 0
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            size += len(chunk)
            sha256.update(chunk)
            sha1.update(chunk)
    return {"size": size, "sha256": sha256.hexdigest(), "sha1": sha1.hexdigest()}


def matches(found):
    return found["size"] == SIZE and found["sha256"] == SHA256


def download(destination):
    if destination.is_file():
        found = identity(destination)
        if matches(found):
            return found
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".download", delete=False) as output:
            temporary = Path(output.name)
            with urllib.request.urlopen(URL, timeout=60) as response:
                size = 0
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    size += len(chunk)
                    if size > SIZE:
                        raise RuntimeError("Downloaded scene exceeds its pinned size")
                    output.write(chunk)
        found = identity(temporary)
        if not matches(found):
            raise RuntimeError(f"Downloaded scene does not match the May asset: {found}")
        temporary.replace(destination)
        return found
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    arguments = parser.parse_args()
    destination = arguments.destination.resolve()
    found = download(destination)
    print(json.dumps({"path": str(destination), "revision": REVISION, **found}))


if __name__ == "__main__":
    main()
