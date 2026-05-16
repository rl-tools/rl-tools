#!/usr/bin/env python3
import argparse
import csv
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[4]
DEFAULT_CHECKPOINT = SCRIPT_DIR / "checkpoints" / "checkpoint_512examples.h5"
DEFAULT_CAPTURE_DIR = SCRIPT_DIR / "captures" / "test"


def h5_attr_str(value):
    return value.decode() if isinstance(value, bytes) else value


def split_top_level_once(s):
    depth = 0
    for i, ch in enumerate(s):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            return s[:i].strip(), s[i + 1:].strip()
    return s.strip(), ""


def parse_call_token(token):
    pos = token.find("(")
    if pos < 0:
        return token, []
    end = token.rfind(")")
    if end < pos:
        return token, []
    name = token[:pos]
    args = [a.strip() for a in token[pos + 1:end].split(",")]
    return name, args


def camera_config_from_checkpoint(path):
    with h5py.File(path, "r") as f:
        meta_raw = h5_attr_str(f["actor"].attrs.get("meta"))
        if not meta_raw:
            raise RuntimeError("checkpoint actor has no meta attribute")
        meta = json.loads(meta_raw)
        obs = meta["environment"]["observation"]
        input_shape = tuple(int(x) for x in f["example/inputs/0"].shape)

    image_obs, _ = split_top_level_once(obs)
    name, args = parse_call_token(image_obs)
    if name != "CameraRGBStackedWithTarget":
        raise RuntimeError("expected CameraRGBStackedWithTarget, got %r" % image_obs)
    if len(args) < 5:
        raise RuntimeError("malformed camera observation: %r" % image_obs)
    height = int(args[1])
    width = int(args[2])
    stride = int(args[3])
    stack = int(args[4])
    channels = input_shape[-1]
    logical_channels = (stack + 1) * 3
    if channels < logical_channels:
        raise RuntimeError(
            "checkpoint input channels %d < logical channels %d" %
            (channels, logical_channels)
        )
    return {
        "observation": obs,
        "fov": float(args[0]),
        "height": height,
        "width": width,
        "stride": stride,
        "stack": stack,
        "channels": channels,
        "logical_channels": logical_channels,
    }


def default_tflite_path(checkpoint, quantize):
    stem = checkpoint.with_suffix("")
    if quantize == "int8":
        return stem.with_name(stem.name + ".host.int8.tflite")
    return stem.with_name(stem.name + ".host.tflite")


def ensure_tflite(checkpoint, quantize, tflite_path, example_bin_limit, no_convert):
    if tflite_path.exists():
        return tflite_path
    if no_convert:
        raise FileNotFoundError("missing tflite: %s" % tflite_path)

    converter = REPO_ROOT / "tools" / "hdf5_to_tflite.py"
    float_out = default_tflite_path(checkpoint, "none")
    cmd = [
        sys.executable,
        str(converter),
        "--quantize", quantize,
        "--no-split-image-input",
        "--example-bin-limit", str(example_bin_limit),
        "-o", str(float_out),
        str(checkpoint),
    ]
    print("converting checkpoint to TFLite:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)
    if not tflite_path.exists():
        raise FileNotFoundError("converter did not create expected tflite: %s" % tflite_path)
    return tflite_path


def load_frames(capture_dir, width, height):
    paths = sorted(capture_dir.glob("frame_*.jpg"))
    if not paths:
        raise RuntimeError("no frame_*.jpg files in %s" % capture_dir)
    frames = []
    for path in paths:
        img = Image.open(path).convert("RGB")
        if img.size != (width, height):
            img = img.resize((width, height), Image.Resampling.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        frames.append(arr)
    return paths, frames


def build_combined(frames, frame_i, target_i, cfg):
    h = cfg["height"]
    w = cfg["width"]
    c = cfg["channels"]
    stack = cfg["stack"]
    x = np.zeros((h, w, c), dtype=np.float32)
    for hist_i in range(stack):
        src_i = frame_i - hist_i
        if src_i < 0:
            src_i = 0
        x[:, :, hist_i * 3:(hist_i + 1) * 3] = frames[src_i]
    target_start = stack * 3
    x[:, :, target_start:target_start + 3] = frames[target_i]
    return x


def keras_index_from_tflite_name(name):
    m = re.search(r"in_(\d+)", name)
    if not m:
        return None
    return int(m.group(1))


def ordered_input_details(interpreter):
    details = interpreter.get_input_details()
    indexed = [(keras_index_from_tflite_name(d["name"]), d) for d in details]
    if all(i is not None for i, _ in indexed):
        return [d for _, d in sorted(indexed, key=lambda item: item[0])]
    return details


def quantize_input(x, detail):
    dtype = detail["dtype"]
    if dtype == np.float32:
        return x.astype(np.float32)[None, ...]
    scale, zp = detail["quantization"]
    if scale <= 0:
        raise RuntimeError("quantized input has invalid scale %r" % (scale,))
    q = np.round(x / scale + zp)
    info = np.iinfo(dtype)
    q = np.clip(q, info.min, info.max).astype(dtype)
    return q[None, ...]


def dequantize_output(y, detail):
    y = np.asarray(y)
    if y.dtype == np.float32:
        return y.astype(np.float32)
    scale, zp = detail["quantization"]
    if scale <= 0:
        return y.astype(np.float32)
    return (y.astype(np.float32) - zp) * scale


def run_model(tflite_path, combined_inputs, cfg):
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    input_details = ordered_input_details(interpreter)
    output_detail = interpreter.get_output_details()[0]
    outputs = []

    for x in combined_inputs:
        if len(input_details) == 1:
            interpreter.set_tensor(input_details[0]["index"], quantize_input(x, input_details[0]))
        else:
            chunks = [
                x[:, :, i * 3:(i + 1) * 3]
                for i in range(cfg["stack"] + 1)
            ]
            if len(input_details) != len(chunks):
                raise RuntimeError(
                    "tflite has %d inputs, expected 1 or %d" %
                    (len(input_details), len(chunks))
                )
            for detail, chunk in zip(input_details, chunks):
                interpreter.set_tensor(detail["index"], quantize_input(chunk, detail))
        interpreter.invoke()
        y_raw = interpreter.get_tensor(output_detail["index"])[0]
        y = dequantize_output(y_raw, output_detail).reshape(-1)
        outputs.append(y)
    return np.asarray(outputs, dtype=np.float32)


def yaw_from_output(y):
    c = float(y[0])
    s = float(y[1])
    norm = math.sqrt(c * c + s * s)
    if norm > 1.0e-6:
        c /= norm
        s /= norm
    else:
        c = 1.0
        s = 0.0
    yaw_rad = math.atan2(s, c)
    return c, s, norm, yaw_rad, yaw_rad * 180.0 / math.pi


def text_size(draw, text, font):
    if hasattr(draw, "textbbox"):
        box = draw.textbbox((0, 0), text, font=font)
        return box[2] - box[0], box[3] - box[1]
    return draw.textsize(text, font=font)


def draw_cross(draw, x, y, r, color, width=1):
    draw.line((x - r, y, x + r, y), fill=color, width=width)
    draw.line((x, y - r, x, y + r), fill=color, width=width)


def draw_projection_tile(frame, row, yaw_rad, yaw_deg, cfg, scale, sign, font):
    src_w, src_h = frame.size
    tile = frame.resize((src_w * scale, src_h * scale), Image.Resampling.BILINEAR)
    draw = ImageDraw.Draw(tile)

    cx = (src_w - 1) * 0.5
    cy = (src_h - 1) * 0.5
    fx = (src_w * 0.5) / math.tan(cfg["fov"] * 0.5)
    u = cx + sign * fx * math.tan(yaw_rad)
    v = cy
    inside = 0 <= u <= src_w - 1 and 0 <= v <= src_h - 1
    marker_u = min(max(u, 0), src_w - 1)
    marker_v = min(max(v, 0), src_h - 1)

    center = (cx * scale, cy * scale)
    marker = (marker_u * scale, marker_v * scale)
    draw.line((center[0], center[1], marker[0], marker[1]),
              fill=(255, 220, 0), width=max(1, scale // 2))
    draw_cross(draw, center[0], center[1], 3 * scale,
               (0, 220, 255), max(1, scale // 2))

    r = 3 * scale
    draw.ellipse((marker[0] - r, marker[1] - r, marker[0] + r, marker[1] + r),
                 outline=(255, 40, 40), width=max(1, scale // 2))
    draw_cross(draw, marker[0], marker[1], r + scale,
               (255, 40, 40), max(1, scale // 2))
    if not inside:
        edge = "left" if u < 0 else "right"
        label = "off %s" % edge
        tw, th = text_size(draw, label, font)
        tx = 2 if u < 0 else tile.width - tw - 2
        draw.rectangle((tx - 1, tile.height - th - 5,
                        tx + tw + 1, tile.height - 1), fill=(0, 0, 0))
        draw.text((tx, tile.height - th - 4), label,
                  fill=(255, 80, 80), font=font)

    label = "#%d  yaw=%+.2f deg" % (row["frame"], yaw_deg)
    tw, th = text_size(draw, label, font)
    draw.rectangle((0, 0, tw + 5, th + 5), fill=(0, 0, 0))
    draw.text((3, 2), label, fill=(255, 255, 255), font=font)
    return tile, u, v, inside


def write_projection_compound(paths, rows, cfg, output, target_index, columns, scale, sign, anchor_target):
    if not rows:
        raise RuntimeError("no yaw predictions to project")
    if not (0 <= target_index < len(rows)):
        raise RuntimeError("target index %d out of range for %d frames" %
                           (target_index, len(rows)))

    target_yaw = rows[target_index]["yaw_rad"] if anchor_target else 0.0
    font = ImageFont.load_default()
    tiles = []
    projections = []
    for path, row in zip(paths, rows):
        yaw_rad = row["yaw_rad"] - target_yaw
        yaw_deg = yaw_rad * 180.0 / math.pi
        frame = Image.open(path).convert("RGB")
        if frame.size != (cfg["width"], cfg["height"]):
            frame = frame.resize((cfg["width"], cfg["height"]), Image.Resampling.BILINEAR)
        tile, u, v, inside = draw_projection_tile(
            frame, row, yaw_rad, yaw_deg, cfg, scale, sign, font
        )
        tiles.append(tile)
        projections.append((row["frame"], u, v, inside, yaw_deg))

    columns = max(1, min(columns, len(tiles)))
    row_count = (len(tiles) + columns - 1) // columns
    gap = 8
    title_h = 38
    legend_h = 32
    tile_w, tile_h = tiles[0].size
    out_w = columns * tile_w + (columns + 1) * gap
    out_h = title_h + row_count * tile_h + (row_count + 1) * gap + legend_h
    compound = Image.new("RGB", (out_w, out_h), (24, 24, 24))
    draw = ImageDraw.Draw(compound)

    title = "Target principal point projected into captured frames"
    subtitle = "target frame=%d, fov=%.5f rad, u = cx %s fx*tan(yaw)" % (
        target_index,
        cfg["fov"],
        "-" if sign < 0 else "+",
    )
    draw.text((gap, 6), title, fill=(255, 255, 255), font=font)
    draw.text((gap, 21), subtitle, fill=(210, 210, 210), font=font)

    y0 = title_h + gap
    for i, tile in enumerate(tiles):
        col = i % columns
        row = i // columns
        x = gap + col * (tile_w + gap)
        y = y0 + row * (tile_h + gap)
        compound.paste(tile, (x, y))

    legend_y = out_h - legend_h + 7
    draw_cross(draw, gap + 7, legend_y + 6, 6, (0, 220, 255), 2)
    draw.text((gap + 20, legend_y), "cyan=center", fill=(225, 225, 225), font=font)
    draw.ellipse((gap + 105, legend_y + 1, gap + 117, legend_y + 13),
                 outline=(255, 40, 40), width=2)
    draw.text((gap + 125, legend_y), "red=projected target principal point",
              fill=(225, 225, 225), font=font)

    output.parent.mkdir(parents=True, exist_ok=True)
    compound.save(output)
    return projections


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--captures", type=Path, default=DEFAULT_CAPTURE_DIR)
    ap.add_argument("--tflite", type=Path, default=None)
    ap.add_argument("--quantize", choices=["none", "int8"], default="none")
    ap.add_argument("--target-index", type=int, default=0)
    ap.add_argument("--example-bin-limit", type=int, default=2)
    ap.add_argument("--no-convert", action="store_true")
    ap.add_argument("--output-csv", type=Path, default=None)
    ap.add_argument("--output-image", type=Path, default=None)
    ap.add_argument("--no-output-image", action="store_true")
    ap.add_argument("--projection-columns", type=int, default=5)
    ap.add_argument("--projection-scale", type=int, default=6)
    ap.add_argument("--projection-sign", type=float, default=1.0,
                    help="projection sign in u = cx + sign*fx*tan(yaw)")
    ap.add_argument("--no-anchor-target", action="store_true",
                    help="do not subtract the target frame yaw estimate before projecting")
    args = ap.parse_args()

    checkpoint = args.checkpoint.resolve()
    captures = args.captures.resolve()
    tflite_path = args.tflite.resolve() if args.tflite else default_tflite_path(checkpoint, args.quantize)
    output_csv = args.output_csv or (captures / "yaw_predictions.csv")
    output_image = args.output_image or (captures / "yaw_projection_compound.png")

    cfg = camera_config_from_checkpoint(checkpoint)
    print("checkpoint:", checkpoint)
    print("observation:", cfg["observation"])
    print("tflite:", tflite_path)
    print("captures:", captures)
    print("interpreting adjacent captured frames as stride-%d history samples" % cfg["stride"])

    ensure_tflite(checkpoint, args.quantize, tflite_path, args.example_bin_limit, args.no_convert)
    paths, frames = load_frames(captures, cfg["width"], cfg["height"])
    if not (0 <= args.target_index < len(frames)):
        raise RuntimeError("target index %d out of range for %d frames" %
                           (args.target_index, len(frames)))

    combined = [
        build_combined(frames, i, args.target_index, cfg)
        for i in range(len(frames))
    ]
    outputs = run_model(tflite_path, combined, cfg)

    rows = []
    print("frame,path,raw_cos,raw_sin,norm,yaw_deg")
    for i, (path, y) in enumerate(zip(paths, outputs)):
        c, s, norm, yaw_rad, yaw_deg = yaw_from_output(y)
        row = {
            "frame": i,
            "path": path.name,
            "raw_cos": float(y[0]),
            "raw_sin": float(y[1]),
            "norm": norm,
            "yaw_rad": yaw_rad,
            "yaw_deg": yaw_deg,
            "norm_cos": c,
            "norm_sin": s,
        }
        rows.append(row)
        print("%d,%s,%+.6f,%+.6f,%.6f,%+.2f" %
              (i, path.name, row["raw_cos"], row["raw_sin"], norm, yaw_deg))

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("wrote", output_csv)

    if not args.no_output_image:
        projections = write_projection_compound(
            paths,
            rows,
            cfg,
            output_image,
            args.target_index,
            args.projection_columns,
            args.projection_scale,
            args.projection_sign,
            not args.no_anchor_target,
        )
        print("wrote", output_image)
        for frame_i, u, v, inside, yaw_deg in projections:
            print("projection frame=%d yaw_deg=%+.3f xy=(%.2f, %.2f) inside=%d" %
                  (frame_i, yaw_deg, u, v, 1 if inside else 0))


if __name__ == "__main__":
    main()
