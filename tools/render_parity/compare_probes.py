#!/usr/bin/env python3
"""Compare two collision-probe binaries (save_probes format: int32 num_cameras, int32 num_probes,
then CollisionResult{float distance; int32 hit;}[]) or two depth binaries (save_depth format:
int32 num_cameras, int32 height, int32 width, then float[]). Hit flags must match exactly;
distances within a relative tolerance."""
import argparse
import struct
import sys


def read_probes(path):
    with open(path, "rb") as f:
        data = f.read()
    num_cameras, num_probes = struct.unpack("<ii", data[:8])
    count = num_cameras * num_probes
    results = struct.unpack(f"<{count * 2}f", data[8:8 + count * 8])
    distances = [results[i * 2] for i in range(count)]
    hits = [struct.unpack("<i", data[8 + i * 8 + 4:8 + i * 8 + 8])[0] for i in range(count)]
    return num_cameras, num_probes, distances, hits


def read_depth(path):
    with open(path, "rb") as f:
        data = f.read()
    num_cameras, height, width = struct.unpack("<iii", data[:12])
    count = num_cameras * height * width
    values = struct.unpack(f"<{count}f", data[12:12 + count * 4])
    return num_cameras, height, width, values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file_a")
    parser.add_argument("file_b")
    parser.add_argument("--mode", choices=["probes", "depth"], default="probes")
    parser.add_argument("--rel-tolerance", type=float, default=1e-3)
    args = parser.parse_args()

    if args.mode == "probes":
        nca, npa, dist_a, hits_a = read_probes(args.file_a)
        ncb, npb, dist_b, hits_b = read_probes(args.file_b)
        if (nca, npa) != (ncb, npb):
            print(f"FAIL: shape mismatch {nca}x{npa} vs {ncb}x{npb}")
            return 1
        hit_mismatches = sum(1 for a, b in zip(hits_a, hits_b) if a != b)
        max_rel = 0.0
        compared = 0
        for a_hit, b_hit, a_dist, b_dist in zip(hits_a, hits_b, dist_a, dist_b):
            if a_hit and b_hit:
                compared += 1
                rel = abs(a_dist - b_dist) / max(abs(a_dist), abs(b_dist), 1e-6)
                if rel > max_rel:
                    max_rel = rel
        print(f"probes: {nca * npa} total, hit mismatches: {hit_mismatches}, max rel distance diff over {compared} shared hits: {max_rel:.2e}")
        if hit_mismatches > 0 or max_rel > args.rel_tolerance:
            print("FAIL")
            return 1
    else:
        nca, ha, wa, values_a = read_depth(args.file_a)
        ncb, hb, wb, values_b = read_depth(args.file_b)
        if (nca, ha, wa) != (ncb, hb, wb):
            print(f"FAIL: shape mismatch {nca}x{ha}x{wa} vs {ncb}x{hb}x{wb}")
            return 1
        max_rel = 0.0
        for a, b in zip(values_a, values_b):
            rel = abs(a - b) / max(abs(a), abs(b), 1e-6)
            if rel > max_rel:
                max_rel = rel
        print(f"depth: {nca * ha * wa} values, max rel diff: {max_rel:.2e}")
        if max_rel > args.rel_tolerance:
            print("FAIL")
            return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
