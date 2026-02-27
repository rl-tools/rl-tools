#!/usr/bin/env python3
"""
Prepare ImageNet-1k parquet data for C++ training.

Reads HuggingFace parquet files from --input-dir, decodes each image,
converts to RGB, and re-encodes as baseline JPEG at original resolution.
This ensures nvJPEG can decode every image without failures (no CMYK,
grayscale, progressive issues). RandomResizedCrop is done on GPU during
training.

Binary format (one file per split):
  Header:  [num_images: uint64]
  Index:   [offset: uint64, size: uint32, label: uint32] * num_images
  Data:    concatenated JPEG bytes

Usage:
  python prepare_imagenet.py --input-dir ~/git/imagenet-1k --output-dir /dev/shm/imagenet_bin --workers 200
"""

import argparse
import io
import os
import struct
import sys
import time
from pathlib import Path
from multiprocessing import Pool

try:
    import pyarrow.parquet as pq
except ImportError:
    sys.exit("pyarrow is required: pip install pyarrow")

try:
    from PIL import Image
except ImportError:
    sys.exit("Pillow is required: pip install Pillow")

JPEG_QUALITY = 95


def find_parquet_files(input_dir: Path, split: str):
    for search_dir in [
        input_dir / ".cache" / "huggingface" / "download" / "data",
        input_dir / "data",
        input_dir,
    ]:
        if search_dir.exists():
            files = sorted(search_dir.glob(f"{split}-*.parquet"))
            files = [f for f in files if ".metadata" not in f.name]
            if files:
                return files
    return sorted(f for f in input_dir.rglob(f"{split}-*.parquet") if ".metadata" not in f.name)


def extract_image_bytes(image_data) -> bytes:
    if isinstance(image_data, dict) and "bytes" in image_data:
        return image_data["bytes"]
    if isinstance(image_data, bytes):
        return image_data
    raise ValueError(f"Unexpected image data type: {type(image_data)}")


def process_image(args):
    """Decode, convert to RGB, re-encode as baseline JPEG at original resolution."""
    raw_bytes, label = args
    try:
        img = Image.open(io.BytesIO(raw_bytes))
        img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=JPEG_QUALITY, progressive=False)
        return buf.getvalue(), label
    except Exception as e:
        return None, label


def process_split(input_dir: Path, output_path: Path, split: str, num_workers: int):
    parquet_files = find_parquet_files(input_dir, split)
    if not parquet_files:
        print(f"  WARNING: No parquet files found for split '{split}'")
        return 0

    print(f"  Found {len(parquet_files)} parquet files for '{split}'")

    total_rows = sum(pq.read_metadata(str(f)).num_rows for f in parquet_files)
    print(f"  Total samples: {total_rows}")

    INDEX_ENTRY_SIZE = 8 + 4 + 4
    header_size = 8
    index_size = total_rows * INDEX_ENTRY_SIZE
    data_start = header_size + index_size

    entries = []

    with open(output_path, "wb") as f:
        f.write(struct.pack("<Q", total_rows))
        f.write(b"\x00" * index_size)
        assert f.tell() == data_start

        written = 0
        skipped = 0
        t0 = time.time()

        with Pool(num_workers) as pool:
            for fi, pf in enumerate(parquet_files):
                table = pq.read_table(str(pf))
                img_col = table.column("image")
                lbl_col = table.column("label")

                batch = []
                for row_idx in range(len(table)):
                    img_data = img_col[row_idx].as_py()
                    raw_bytes = extract_image_bytes(img_data)
                    label = int(lbl_col[row_idx].as_py())
                    batch.append((raw_bytes, label))

                chunksize = max(1, len(batch) // (num_workers * 4))
                for jpeg_bytes, label in pool.imap(process_image, batch, chunksize=chunksize):
                    if jpeg_bytes is None:
                        skipped += 1
                        continue
                    offset = f.tell()
                    f.write(jpeg_bytes)
                    entries.append((offset, len(jpeg_bytes), label))
                    written += 1

                elapsed = time.time() - t0
                rate = written / elapsed if elapsed > 0 else 0
                print(f"\r    [{fi+1}/{len(parquet_files)}] {written}/{total_rows} images "
                      f"({rate:.0f} img/s, {skipped} skipped)", end="", flush=True)

        print()
        if skipped > 0:
            print(f"  WARNING: {skipped} images failed to decode and were skipped")

        f.seek(0)
        f.write(struct.pack("<Q", written))
        f.seek(header_size)
        for offset, size, label in entries:
            f.write(struct.pack("<QII", offset, size, label))

    file_size = output_path.stat().st_size
    elapsed = time.time() - t0
    print(f"  Written {file_size / (1024**3):.2f} GB ({written} images) in {elapsed:.1f}s")
    return written


def main():
    parser = argparse.ArgumentParser(description="Prepare ImageNet-1k for C++ training")
    parser.add_argument("--input-dir", type=Path, default=Path.home() / "git" / "imagenet-1k",
                        help="Path to HuggingFace imagenet-1k dataset")
    parser.add_argument("--output-dir", type=Path, default=Path("imagenet_bin"),
                        help="Output directory for binary files")
    parser.add_argument("--workers", type=int, default=os.cpu_count(),
                        help="Number of parallel workers (default: all CPUs)")
    args = parser.parse_args()

    if not args.input_dir.exists():
        sys.exit(f"Input directory does not exist: {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Preparing ImageNet-1k binary dataset")
    print(f"  Input:   {args.input_dir}")
    print(f"  Output:  {args.output_dir}")
    print(f"  Workers: {args.workers}")
    print(f"  Re-encode: RGB baseline JPEG, quality {JPEG_QUALITY}")

    for split in ["train", "validation"]:
        print(f"\nProcessing {split}...")
        out_name = "val.bin" if split == "validation" else f"{split}.bin"
        n = process_split(args.input_dir, args.output_dir / out_name, split, args.workers)
        if n > 0:
            print(f"  {split}: {n} images")

    print("\nDone!")


if __name__ == "__main__":
    main()
