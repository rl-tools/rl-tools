#!/usr/bin/env python3
"""Display the n-th image from the ImageNet binary format."""
import argparse
import struct
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binfile", type=Path, help="Binary file (e.g. train.bin)")
    parser.add_argument("index", type=int, help="0-based image index")
    args = parser.parse_args()

    with open(args.binfile, "rb") as f:
        num_images, = struct.unpack("<Q", f.read(8))
        if not (0 <= args.index < num_images):
            sys.exit(f"Index {args.index} out of range [0, {num_images})")
        f.seek(8 + args.index * 16)
        offset, size, label = struct.unpack("<QII", f.read(16))
        f.seek(offset)
        jpeg = f.read(size)

    out = Path(f"image_{args.index}_label_{label}.jpg")
    out.write_bytes(jpeg)
    print(f"Image {args.index}/{num_images}  label={label}  size={size} bytes -> {out}")

if __name__ == "__main__":
    main()
