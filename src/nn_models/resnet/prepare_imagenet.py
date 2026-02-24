#!/usr/bin/env python3
"""
Prepare ImageNet-1k parquet data for C++ training.

Reads HuggingFace parquet files from --input-dir and produces a compact binary
dataset that the C++ ImageNet training code can consume efficiently.

Binary format (one file per split):
  Header:  [num_images: uint64]
  Index:   [offset: uint64, size: uint32, label: uint32] * num_images
  Data:    concatenated JPEG bytes

Usage:
  python prepare_imagenet.py --input-dir ~/git/imagenet-1k --output-dir ./imagenet_bin
"""

import argparse
import io
import struct
import sys
from pathlib import Path

import numpy as np

try:
    import pyarrow.parquet as pq
except ImportError:
    sys.exit("pyarrow is required: pip install pyarrow")

try:
    from PIL import Image
except ImportError:
    sys.exit("Pillow is required: pip install Pillow")


def find_parquet_files(input_dir: Path, split: str):
    cache_dir = input_dir / ".cache" / "huggingface" / "download" / "data"
    if cache_dir.exists():
        files = sorted(cache_dir.glob(f"{split}-*.parquet"))
        if files:
            return files
    files = sorted(input_dir.glob(f"data/{split}-*.parquet"))
    if files:
        return files
    files = sorted(input_dir.glob(f"{split}-*.parquet"))
    if files:
        return files
    files = sorted(input_dir.glob(f"**/{split}-*.parquet"))
    return files


def image_to_jpeg_bytes(image_data) -> bytes:
    if isinstance(image_data, bytes):
        return image_data
    if isinstance(image_data, dict) and "bytes" in image_data:
        return image_data["bytes"]
    if isinstance(image_data, Image.Image):
        buf = io.BytesIO()
        image_data.save(buf, format="JPEG", quality=95)
        return buf.getvalue()
    raise ValueError(f"Unexpected image data type: {type(image_data)}")


def process_split(input_dir: Path, output_path: Path, split: str):
    parquet_files = find_parquet_files(input_dir, split)
    if not parquet_files:
        print(f"  WARNING: No parquet files found for split '{split}'")
        return 0

    print(f"  Found {len(parquet_files)} parquet files for '{split}'")

    images = []
    labels = []
    for i, pf in enumerate(parquet_files):
        table = pq.read_table(pf)
        for row_idx in range(len(table)):
            img_col = table.column("image")
            lbl_col = table.column("label")
            img_data = img_col[row_idx].as_py()
            label = int(lbl_col[row_idx].as_py())
            jpeg_bytes = image_to_jpeg_bytes(img_data)
            images.append(jpeg_bytes)
            labels.append(label)
        if (i + 1) % 10 == 0 or i == len(parquet_files) - 1:
            print(f"    Processed {i+1}/{len(parquet_files)} files ({len(images)} images)")

    num_images = len(images)
    print(f"  Writing {num_images} images to {output_path}")

    index_start = 8
    index_size = num_images * (8 + 4 + 4)
    data_start = index_start + index_size

    with open(output_path, "wb") as f:
        f.write(struct.pack("<Q", num_images))

        offset = data_start
        for jpeg_bytes, label in zip(images, labels):
            size = len(jpeg_bytes)
            f.write(struct.pack("<QII", offset, size, label))
            offset += size

        for jpeg_bytes in images:
            f.write(jpeg_bytes)

    file_size = output_path.stat().st_size
    print(f"  Written {file_size / (1024**3):.2f} GB")
    return num_images


def main():
    parser = argparse.ArgumentParser(description="Prepare ImageNet-1k for C++ training")
    parser.add_argument("--input-dir", type=Path, default=Path.home() / "git" / "imagenet-1k",
                        help="Path to HuggingFace imagenet-1k dataset")
    parser.add_argument("--output-dir", type=Path, default=Path("imagenet_bin"),
                        help="Output directory for binary files")
    args = parser.parse_args()

    if not args.input_dir.exists():
        sys.exit(f"Input directory does not exist: {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Preparing ImageNet-1k binary dataset")
    print(f"  Input:  {args.input_dir}")
    print(f"  Output: {args.output_dir}")

    for split in ["train", "validation"]:
        print(f"\nProcessing {split}...")
        out_name = "val.bin" if split == "validation" else f"{split}.bin"
        n = process_split(args.input_dir, args.output_dir / out_name, split)
        if n > 0:
            print(f"  {split}: {n} images")

    print("\nDone!")


if __name__ == "__main__":
    main()
