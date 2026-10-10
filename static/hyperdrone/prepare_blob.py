"""Pack the default scene, drone and RAPTOR checkpoint into the local conta overlay."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=Path)
    parser.add_argument("drone", type=Path)
    parser.add_argument("policy", type=Path)
    args = parser.parse_args()
    page = Path(__file__).resolve().parent
    registry = json.loads((page / "registry.json").read_text())
    pose = json.loads((page / "pose.json").read_text())
    assets = [(args.scene, pose), (args.drone, registry["drones"][0]), (args.policy, registry["policy"])]
    packed = []
    for source, asset in assets:
        sha1 = asset["reference"].removeprefix("conta:")
        data = source.read_bytes()
        if hashlib.sha1(data).hexdigest() != sha1:
            raise ValueError(f"{source} does not match {asset['reference']}")
        if "sha256" in asset and hashlib.sha256(data).hexdigest() != asset["sha256"]:
            raise ValueError(f"{source} does not match its pinned SHA-256")
        packed.append((sha1, gzip.compress(data, mtime=0)))
    blob = page / "blob"
    blob.mkdir(exist_ok=True)
    for sha1, data in packed:
        target = blob / f"{sha1}.gz"
        target.write_bytes(data)
        print(f"{target}: {len(data)} bytes")


if __name__ == "__main__":
    main()
