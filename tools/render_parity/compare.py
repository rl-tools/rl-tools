#!/usr/bin/env python3
"""Compare two renderer output PNGs (e.g. Metal vs OptiX goldens) with a mean-abs-diff threshold."""
import argparse
import struct
import sys
import zlib


def read_png_rgba(path):
    with open(path, "rb") as f:
        data = f.read()
    assert data[:8] == b"\x89PNG\r\n\x1a\n", f"{path}: not a PNG"
    pos = 8
    width = height = bit_depth = color_type = None
    idat = b""
    while pos < len(data):
        length, chunk_type = struct.unpack(">I4s", data[pos:pos + 8])
        chunk = data[pos + 8:pos + 8 + length]
        if chunk_type == b"IHDR":
            width, height, bit_depth, color_type = struct.unpack(">IIBB", chunk[:10])
            interlace = chunk[12]
            assert bit_depth == 8, f"{path}: unsupported bit depth {bit_depth}"
            assert color_type in (2, 6), f"{path}: unsupported color type {color_type}"
            assert interlace == 0, f"{path}: interlaced PNGs are unsupported"
        elif chunk_type == b"IDAT":
            idat += chunk
        pos += 12 + length
    channels = 4 if color_type == 6 else 3
    raw = zlib.decompress(idat)
    stride = width * channels
    pixels = bytearray(height * stride)
    previous = bytearray(stride)
    offset = 0
    for y in range(height):
        filter_type = raw[offset]
        offset += 1
        line = bytearray(raw[offset:offset + stride])
        offset += stride
        if filter_type == 1:
            for i in range(channels, stride):
                line[i] = (line[i] + line[i - channels]) & 0xFF
        elif filter_type == 2:
            for i in range(stride):
                line[i] = (line[i] + previous[i]) & 0xFF
        elif filter_type == 3:
            for i in range(stride):
                left = line[i - channels] if i >= channels else 0
                line[i] = (line[i] + ((left + previous[i]) >> 1)) & 0xFF
        elif filter_type == 4:
            for i in range(stride):
                left = line[i - channels] if i >= channels else 0
                up = previous[i]
                up_left = previous[i - channels] if i >= channels else 0
                p = left + up - up_left
                pa, pb, pc = abs(p - left), abs(p - up), abs(p - up_left)
                if pa <= pb and pa <= pc:
                    predictor = left
                elif pb <= pc:
                    predictor = up
                else:
                    predictor = up_left
                line[i] = (line[i] + predictor) & 0xFF
        else:
            assert filter_type == 0, f"{path}: unsupported filter {filter_type}"
        pixels[y * stride:(y + 1) * stride] = line
        previous = line
    return width, height, channels, pixels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_a")
    parser.add_argument("image_b")
    parser.add_argument("--mad-threshold", type=float, default=2.0, help="max allowed mean abs diff in 8-bit levels (default 2.0)")
    parser.add_argument("--report", action="store_true", help="print per-channel statistics")
    args = parser.parse_args()

    wa, ha, ca, a = read_png_rgba(args.image_a)
    wb, hb, cb, b = read_png_rgba(args.image_b)
    if (wa, ha) != (wb, hb):
        print(f"FAIL: dimension mismatch {wa}x{ha} vs {wb}x{hb}")
        return 1
    channels = min(ca, cb)

    total = 0
    max_diff = [0] * channels
    sum_diff = [0] * channels
    num_pixels = wa * ha
    for i in range(num_pixels):
        for channel in range(channels):
            diff = abs(a[i * ca + channel] - b[i * cb + channel])
            sum_diff[channel] += diff
            if diff > max_diff[channel]:
                max_diff[channel] = diff
            total += diff

    mad = total / (num_pixels * channels)
    if args.report:
        for channel, name in enumerate("RGBA"[:channels]):
            print(f"  {name}: mean={sum_diff[channel] / num_pixels:.4f} max={max_diff[channel]}")
    print(f"mean abs diff: {mad:.4f} (threshold {args.mad_threshold})")
    if mad > args.mad_threshold:
        print("FAIL")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
