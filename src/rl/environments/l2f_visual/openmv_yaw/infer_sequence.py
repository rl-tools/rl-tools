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


def wrap_rad(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def deg(angle):
    return angle * 180.0 / math.pi


def parse_int_field(row, key):
    value = row.get(key)
    if value is None or value == "":
        return None
    return int(value)


def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.asarray((
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ), dtype=np.float64)


def quat_from_rotvec(rot):
    angle = float(np.linalg.norm(rot))
    if angle < 1.0e-12:
        return np.asarray((1.0, 0.5 * rot[0], 0.5 * rot[1], 0.5 * rot[2]), dtype=np.float64)
    axis = rot / angle
    half = 0.5 * angle
    s = math.sin(half)
    return np.asarray((math.cos(half), axis[0] * s, axis[1] * s, axis[2] * s), dtype=np.float64)


def yaw_from_quat(q):
    w, x, y, z = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def load_frame_time_us(capture_dir, frame_count):
    index_path = capture_dir / "index.csv"
    if not index_path.exists():
        return None

    frame_times = [None] * frame_count
    with open(index_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame_i = parse_int_field(row, "frame")
            if frame_i is None or frame_i < 0 or frame_i >= frame_count:
                continue
            frame_t = parse_int_field(row, "frame_mid_us")
            if frame_t is None:
                frame_start = parse_int_field(row, "frame_start_us")
                snapshot_done = parse_int_field(row, "snapshot_done_us")
                if frame_start is not None and snapshot_done is not None:
                    frame_t = (frame_start + snapshot_done) // 2
            if frame_t is None:
                ticks = parse_int_field(row, "ticks_us")
                snapshot_us = parse_int_field(row, "snapshot_us")
                if ticks is not None and snapshot_us is not None:
                    frame_t = ticks + snapshot_us // 2
                else:
                    frame_t = ticks
            frame_times[frame_i] = frame_t

    if any(t is None for t in frame_times):
        return None
    return np.asarray(frame_times, dtype=np.float64)


def load_gyro_groundtruth(capture_dir, frame_count, target_index, gyro_csv):
    gyro_path = gyro_csv if gyro_csv is not None else capture_dir / "gyro.csv"
    if not gyro_path.exists():
        return None

    frame_times_us = load_frame_time_us(capture_dir, frame_count)
    if frame_times_us is None:
        return None

    samples = []
    with open(gyro_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = parse_int_field(row, "ticks_us")
            if t is None:
                t = parse_int_field(row, "rel_us")
            gx = row.get("gx_rad_s")
            gy = row.get("gy_rad_s")
            gz = row.get("gz_rad_s")
            if t is None or gz is None or gz == "":
                continue
            if gx is None or gx == "":
                gx = 0.0
            if gy is None or gy == "":
                gy = 0.0
            samples.append((float(t), float(gx), float(gy), float(gz)))

    samples.sort(key=lambda item: item[0])
    times = []
    omega_values = []
    last_t = None
    for t, gx, gy, gz in samples:
        if last_t is not None and t <= last_t:
            continue
        times.append(t)
        omega_values.append((gx, gy, gz))
        last_t = t

    if len(times) < 2:
        return None

    gyro_t_s = (np.asarray(times, dtype=np.float64) - times[0]) * 1.0e-6
    omega = np.asarray(omega_values, dtype=np.float64)
    yaw = np.zeros_like(gyro_t_s)
    q = np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float64)
    for i in range(1, len(gyro_t_s)):
        dt = gyro_t_s[i] - gyro_t_s[i - 1]
        dq = quat_from_rotvec(0.5 * (omega[i - 1] + omega[i]) * dt)
        q = quat_mul(q, dq)
        q /= np.linalg.norm(q)
        yaw[i] = yaw_from_quat(q)

    frame_t_s = (frame_times_us - times[0]) * 1.0e-6
    yaw = np.unwrap(yaw)
    frame_yaw = np.interp(frame_t_s, gyro_t_s, yaw, left=yaw[0], right=yaw[-1])
    target_yaw = frame_yaw[target_index]
    rel_yaw = np.asarray([wrap_rad(y - target_yaw) for y in frame_yaw], dtype=np.float64)
    return {
        "path": gyro_path,
        "sample_count": len(times),
        "coverage_start_s": float(gyro_t_s[0]),
        "coverage_end_s": float(gyro_t_s[-1]),
        "frame_start_s": float(frame_t_s[0]),
        "frame_end_s": float(frame_t_s[-1]),
        "yaw_rad": rel_yaw,
    }


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
        yaw_rad = wrap_rad(row["yaw_rad"] - target_yaw)
        yaw_deg = deg(yaw_rad)
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
    ap.add_argument("--gyro-csv", type=Path, default=None)
    ap.add_argument("--no-gyro-groundtruth", action="store_true")
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
    gyro_csv = args.gyro_csv.resolve() if args.gyro_csv else None
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

    target_yaw = rows[args.target_index]["yaw_rad"]
    for row in rows:
        pred_target_yaw = wrap_rad(row["yaw_rad"] - target_yaw)
        row["pred_target_yaw_rad"] = pred_target_yaw
        row["pred_target_yaw_deg"] = deg(pred_target_yaw)

    gyro_truth = None
    if not args.no_gyro_groundtruth:
        gyro_truth = load_gyro_groundtruth(captures, len(rows), args.target_index, gyro_csv)
        if gyro_truth is None:
            print("gyro ground truth: unavailable")
        else:
            errors = []
            for i, row in enumerate(rows):
                gyro_yaw = float(gyro_truth["yaw_rad"][i])
                err = wrap_rad(row["pred_target_yaw_rad"] - gyro_yaw)
                row["gyro_yaw_rad"] = gyro_yaw
                row["gyro_yaw_deg"] = deg(gyro_yaw)
                row["gyro_error_rad"] = err
                row["gyro_error_deg"] = deg(err)
                errors.append(err)
            errors = np.asarray(errors, dtype=np.float64)
            print(
                "gyro ground truth: csv=%s samples=%d coverage=%.3f..%.3fs frames=%.3f..%.3fs "
                "mae=%.3fdeg rmse=%.3fdeg max=%.3fdeg" %
                (
                    gyro_truth["path"], gyro_truth["sample_count"],
                    gyro_truth["coverage_start_s"], gyro_truth["coverage_end_s"],
                    gyro_truth["frame_start_s"], gyro_truth["frame_end_s"],
                    deg(float(np.mean(np.abs(errors)))),
                    deg(float(np.sqrt(np.mean(errors * errors)))),
                    deg(float(np.max(np.abs(errors)))),
                )
            )

    if gyro_truth is None:
        print("frame,path,raw_cos,raw_sin,norm,yaw_deg,pred_target_yaw_deg")
        for row in rows:
            print("%d,%s,%+.6f,%+.6f,%.6f,%+.2f,%+.2f" %
                  (row["frame"], row["path"], row["raw_cos"], row["raw_sin"],
                   row["norm"], row["yaw_deg"], row["pred_target_yaw_deg"]))
    else:
        print("frame,path,raw_cos,raw_sin,norm,yaw_deg,pred_target_yaw_deg,gyro_yaw_deg,gyro_error_deg")
        for row in rows:
            print("%d,%s,%+.6f,%+.6f,%.6f,%+.2f,%+.2f,%+.2f,%+.2f" %
                  (row["frame"], row["path"], row["raw_cos"], row["raw_sin"],
                   row["norm"], row["yaw_deg"], row["pred_target_yaw_deg"],
                   row["gyro_yaw_deg"], row["gyro_error_deg"]))

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
