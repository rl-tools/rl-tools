import gc
import json
import math
import os
import time

import csi
import image
import machine
import ml
from ulab import numpy as np


VISION_HZ = 100
VISION_TICK_US = 1_000_000 // VISION_HZ
DIAG_PRINT_EVERY = VISION_HZ

CAMERA_FRAMEBUFFERS = 3
CAMERA_SENSOR_FPS = 200
CAMERA_EXPOSURE_US = 4000
CAMERA_GAIN_DB = 20

UART_BRIDGE_PORT = 4
UART_BRIDGE_BAUD = 115200
UART_RX_LINE_MAX = 160

FRAME_START_MASK = 0x80
FRAME_TYPE_MASK = 0x7F
FRAME_TYPE_VISUAL_YAW = 0x10
FRAME_TYPE_TARGET_CAPTURE = 0x11

VISUAL_YAW_FLAG_TARGET_VALID = 0x01
VISUAL_YAW_FLAG_PREDICTION_VALID = 0x02
VISUAL_YAW_FLAG_TARGET_CAPTURE_ACK = 0x04

TARGET_CAPTURE_COMMAND_CURRENT = 0x01
TARGET_CAPTURE_COMMAND_RECORD_START = 0x02
TARGET_CAPTURE_COMMAND_RECORD_STOP_SAVE = 0x03
TARGET_CAPTURE_COMMAND_RECORD_ABORT = 0x04

CAPTURE_ENABLE = True
CAPTURE_OUTPUT_ROOT = "/flash/capture"
CAPTURE_MAX_FRAMES = 240
CAPTURE_RECORD_HZ = 10
CAPTURE_RECORD_TICK_US = 1_000_000 // CAPTURE_RECORD_HZ
CAPTURE_JPEG_QUALITY = 85
CAPTURE_TARGET_NAME = "target.jpg"
CAPTURE_SYNC_SETTLE_MS = 500

VISUAL_YAW_RAW_BYTES = 8
VISUAL_YAW_DATA_BYTES = 10
VISUAL_YAW_FRAME_BYTES = 1 + VISUAL_YAW_DATA_BYTES
VISUAL_YAW_CRC_PAYLOAD_BYTES = 7
TARGET_CAPTURE_RAW_BYTES = 6
TARGET_CAPTURE_DATA_BYTES = 7
TARGET_CAPTURE_CRC_PAYLOAD_BYTES = 5
YAW_Q_SCALE = 10000.0

IMAGE_U8_SCALE = 1.0 / 255.0
SELF_CHECK_FP32_MAX_ERR = 1.0e-3
SELF_CHECK_INT8_MAX_ERR = 0.5
SELF_CHECK_INT8_FAIL_LSBS = 100


def print_mem(label):
    gc.collect()
    print("%-24s free=%d alloc=%d" % (label, gc.mem_free(), gc.mem_alloc()))


def ensure_dir(path):
    try:
        os.stat(path)
        return True
    except OSError:
        try:
            os.mkdir(path)
            return True
        except OSError as e:
            print("mkdir failed:", path, e)
            return False


def prepare_capture_dir():
    if not ensure_dir(CAPTURE_OUTPUT_ROOT):
        raise RuntimeError("no capture output directory available")

    for name in os.listdir(CAPTURE_OUTPUT_ROOT):
        if (name.startswith("frame_") or
                name == CAPTURE_TARGET_NAME or
                name == "index.csv" or
                name == "gyro.csv" or
                name == "meta.txt" or
                name == "stats.txt"):
            path = CAPTURE_OUTPUT_ROOT + "/" + name
            try:
                os.remove(path)
            except OSError as e:
                print("remove failed:", path, e)
    gc.collect()
    return CAPTURE_OUTPUT_ROOT


def write_text(path, text):
    f = open(path, "w")
    try:
        f.write(text)
    finally:
        f.close()


def write_jpeg(path, data):
    f = open(path, "wb")
    try:
        f.write(data)
    finally:
        f.close()


def sync_filesystem():
    sync_fn = getattr(os, "sync", None)
    if sync_fn is not None:
        try:
            sync_fn()
        except OSError as e:
            print("sync failed:", e)
    time.sleep_ms(CAPTURE_SYNC_SETTLE_MS)


def compress_jpeg(img):
    try:
        return img.compress(copy=True, quality=CAPTURE_JPEG_QUALITY)
    except TypeError:
        tmp = img.copy()
        try:
            return tmp.compress(quality=CAPTURE_JPEG_QUALITY)
        except TypeError:
            return tmp.compress()


def jpeg_bytes(img):
    try:
        return bytes(img.bytearray())
    except Exception:
        return bytes(img)


def capture_frame_name(frame_i):
    return "frame_%06d.jpg" % frame_i


class BlueLed:
    def __init__(self):
        self.led = None
        for arg in ("LED_BLUE", "blue", 3):
            try:
                self.led = machine.LED(arg)
                break
            except Exception:
                pass
        if self.led is None:
            try:
                self.led = machine.Pin("LED_BLUE", machine.Pin.OUT)
            except Exception:
                pass
        self.off()

    def on(self):
        if self.led is None:
            return
        try:
            self.led.on()
        except Exception:
            try:
                self.led.value(1)
            except Exception:
                pass

    def off(self):
        if self.led is None:
            return
        try:
            self.led.off()
        except Exception:
            try:
                self.led.value(0)
            except Exception:
                pass


class YawFrameCapture:
    def __init__(self, runtime, width, height, storage_led=None):
        self.runtime = runtime
        self.width = width
        self.height = height
        self.storage_led = storage_led
        self.save_count = 0
        self.clear_episode()

    def clear_episode(self):
        self.active = False
        self.frames = []
        self.rows = []
        self.target_data = None
        self.target_seq = 0
        self.target_ticks_us = 0
        self.target_jpeg_us = 0
        self.target_bias = 0.0
        self.start_us = 0
        self.stop_us = 0
        self.start_seq = 0
        self.stop_seq = 0
        self.start_reason = 0
        self.stop_reason = 0
        self.next_record_us = 0
        self.overflow = False
        self.dropped = 0
        self.sum_jpeg_us = 0
        self.max_jpeg_us = 0
        self.sum_capture_us = 0
        self.max_capture_us = 0
        self.max_capture_end_lag_us = -2147483648
        gc.collect()

    def start(self, seq, reason):
        if not CAPTURE_ENABLE:
            return
        self.clear_episode()
        self.active = True
        self.start_us = time.ticks_us()
        self.next_record_us = self.start_us
        self.start_seq = seq
        self.start_reason = reason
        print("record start seq=%d reason=%d hz=%d max_frames=%d" %
              (seq, reason, CAPTURE_RECORD_HZ, CAPTURE_MAX_FRAMES))

    def abort(self, seq, reason):
        if not CAPTURE_ENABLE:
            return
        n = len(self.frames)
        self.clear_episode()
        print("record abort seq=%d reason=%d discarded=%d" %
              (seq, reason, n))

    def capture_target(self, img, target_seq, ticks_us):
        if not self.active:
            return
        t0 = time.ticks_us()
        try:
            data = jpeg_bytes(compress_jpeg(img))
        except Exception as e:
            self.dropped += 1
            print("record target jpeg failed:", e)
            gc.collect()
            return
        self.target_data = data
        self.target_seq = target_seq
        self.target_ticks_us = ticks_us
        self.target_jpeg_us = time.ticks_diff(time.ticks_us(), t0)
        print("record target seq=%d bytes=%d jpeg_us=%d" %
              (target_seq, len(data), self.target_jpeg_us))

    def set_target_bias(self, target_bias):
        if self.active:
            self.target_bias = target_bias

    def append_frame(self, img, tx_seq, target_seq, flags, t0, t_snapshot,
                     t_resize, t_predict, next_deadline, age_ms,
                     yaw_raw_rad, target_bias, yaw_rad, yaw_norm):
        if not self.active:
            return
        if time.ticks_diff(t_snapshot, self.next_record_us) < 0:
            return
        while time.ticks_diff(t_snapshot, self.next_record_us) >= 0:
            self.next_record_us = time.ticks_add(
                self.next_record_us, CAPTURE_RECORD_TICK_US)
        frame_i = len(self.frames)
        if frame_i >= CAPTURE_MAX_FRAMES:
            self.overflow = True
            self.dropped += 1
            return

        t_jpeg0 = time.ticks_us()
        try:
            data = jpeg_bytes(compress_jpeg(img))
        except Exception as e:
            self.dropped += 1
            print("record frame jpeg failed:", e)
            gc.collect()
            return
        t_end = time.ticks_us()
        jpeg_us = time.ticks_diff(t_end, t_jpeg0)
        snapshot_us = time.ticks_diff(t_snapshot, t0)
        resize_us = time.ticks_diff(t_resize, t_snapshot)
        predict_us = time.ticks_diff(t_predict, t_resize)
        capture_us = time.ticks_diff(t_end, t0)
        capture_end_lag_us = time.ticks_diff(t_end, next_deadline)
        if capture_end_lag_us > self.max_capture_end_lag_us:
            self.max_capture_end_lag_us = capture_end_lag_us
        if jpeg_us > self.max_jpeg_us:
            self.max_jpeg_us = jpeg_us
        if capture_us > self.max_capture_us:
            self.max_capture_us = capture_us
        self.sum_jpeg_us += jpeg_us
        self.sum_capture_us += capture_us

        name = capture_frame_name(frame_i)
        self.frames.append((name, data))
        self.rows.append((
            frame_i, name, t0, snapshot_us, resize_us, predict_us, jpeg_us,
            capture_us, capture_end_lag_us, len(data), age_ms,
            yaw_raw_rad, target_bias, yaw_rad, yaw_norm,
            target_seq, flags, tx_seq, t0, t_snapshot, t_predict
        ))

    def build_meta(self):
        bytes_total = 0
        for _, data in self.frames:
            bytes_total += len(data)
        if self.target_data is not None:
            bytes_total += len(self.target_data)
        return (
            "format=jpeg\n"
            "width=%d\n"
            "height=%d\n"
            "channels=3\n"
            "channel_order=RGB\n"
            "jpeg_quality=%d\n"
            "vision_hz=%d\n"
            "tick_us=%d\n"
            "capture_hz=%d\n"
            "capture_tick_us=%d\n"
            "max_frames=%d\n"
            "frame_count=%d\n"
            "dropped_frames=%d\n"
            "overflow=%d\n"
            "bytes_total=%d\n"
            "target_path=%s\n"
            "target_present=%d\n"
            "target_seq=%d\n"
            "target_ticks_us=%d\n"
            "target_jpeg_us=%d\n"
            "target_bias_rad=%.9g\n"
            "start_ticks_us=%d\n"
            "stop_ticks_us=%d\n"
            "start_seq=%d\n"
            "stop_seq=%d\n"
            "start_reason=%d\n"
            "stop_reason=%d\n"
            "model_path=%s\n"
            "checkpoint_name=%s\n"
            "camera_sensor_fps=%d\n"
            "camera_exposure_us=%d\n"
            "camera_gain_db=%d\n"
            "camera_pixformat=RGB565\n"
            "camera_framesize=QVGA\n"
            "camera_auto_rotation=false\n" %
            (
                self.width, self.height, CAPTURE_JPEG_QUALITY, VISION_HZ,
                VISION_TICK_US, CAPTURE_RECORD_HZ, CAPTURE_RECORD_TICK_US,
                CAPTURE_MAX_FRAMES, len(self.frames),
                self.dropped, 1 if self.overflow else 0, bytes_total,
                CAPTURE_TARGET_NAME, 1 if self.target_data is not None else 0,
                self.target_seq, self.target_ticks_us, self.target_jpeg_us,
                self.target_bias, self.start_us, self.stop_us,
                self.start_seq, self.stop_seq,
                self.start_reason, self.stop_reason,
                self.runtime.model_path, self.runtime.checkpoint_name,
                CAMERA_SENSOR_FPS, CAMERA_EXPOSURE_US, CAMERA_GAIN_DB
            )
        )

    def write_outputs(self, capture_dir):
        write_text(capture_dir + "/meta.txt", self.build_meta())
        sum_write_us = 0
        max_write_us = 0
        target_write_us = 0
        if self.target_data is not None:
            t0 = time.ticks_us()
            write_jpeg(capture_dir + "/" + CAPTURE_TARGET_NAME, self.target_data)
            target_write_us = time.ticks_diff(time.ticks_us(), t0)
            sum_write_us += target_write_us
            max_write_us = target_write_us

        write_uses = []
        for name, data in self.frames:
            t0 = time.ticks_us()
            write_jpeg(capture_dir + "/" + name, data)
            write_us = time.ticks_diff(time.ticks_us(), t0)
            write_uses.append(write_us)
            sum_write_us += write_us
            if write_us > max_write_us:
                max_write_us = write_us

        index = open(capture_dir + "/index.csv", "w")
        try:
            index.write(
                "frame,path,ticks_us,snapshot_us,resize_us,predict_us,jpeg_us,"
                "capture_us,capture_end_lag_us,encoded_bytes,age_ms,"
                "yaw_raw_rad,target_bias_rad,yaw_rad,yaw_norm,target_seq,"
                "flags,tx_seq,frame_start_us,snapshot_done_us,predict_done_us,"
                "write_us\n"
            )
            fmt = ("%d,%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,"
                   "%.9g,%.9g,%.9g,%.9g,%d,%d,%d,%d,%d,%d,%d\n")
            for i in range(len(self.rows)):
                index.write(fmt % (self.rows[i] + (write_uses[i],)))
        finally:
            index.close()

        n = len(self.frames)
        avg_jpeg_us = self.sum_jpeg_us // n if n else 0
        avg_capture_us = self.sum_capture_us // n if n else 0
        write_count = n + (1 if self.target_data is not None else 0)
        avg_write_us = sum_write_us // write_count if write_count else 0
        stats = (
            "frames=%d\n"
            "target_present=%d\n"
            "dropped_frames=%d\n"
            "overflow=%d\n"
            "avg_jpeg_us=%d\n"
            "max_jpeg_us=%d\n"
            "avg_capture_us=%d\n"
            "max_capture_us=%d\n"
            "max_capture_end_lag_us=%d\n"
            "target_write_us=%d\n"
            "avg_write_us=%d\n"
            "max_write_us=%d\n" %
            (
                n, 1 if self.target_data is not None else 0,
                self.dropped, 1 if self.overflow else 0,
                avg_jpeg_us, self.max_jpeg_us,
                avg_capture_us, self.max_capture_us,
                self.max_capture_end_lag_us if n else 0,
                target_write_us, avg_write_us, max_write_us
            )
        )
        write_text(capture_dir + "/stats.txt", stats)
        return sum_write_us, max_write_us

    def stop_and_save(self, seq, reason):
        if not CAPTURE_ENABLE:
            return
        if not self.active and not self.frames and self.target_data is None:
            print("record stop ignored seq=%d reason=%d inactive" %
                  (seq, reason))
            return
        self.active = False
        self.stop_us = time.ticks_us()
        self.stop_seq = seq
        self.stop_reason = reason
        if self.storage_led is not None:
            self.storage_led.on()
        try:
            capture_dir = prepare_capture_dir()
            print("record stop seq=%d reason=%d frames=%d; writing to %s" %
                  (seq, reason, len(self.frames), capture_dir))
            t0 = time.ticks_us()
            self.write_outputs(capture_dir)
            sync_filesystem()
            write_us = time.ticks_diff(time.ticks_us(), t0)
            self.save_count += 1
            print("record saved frames=%d target=%d write_us=%d dir=%s" %
                  (len(self.frames), 1 if self.target_data is not None else 0,
                   write_us, capture_dir))
            self.clear_episode()
        finally:
            if self.storage_led is not None:
                self.storage_led.off()


def pick_first(files, predicate, label):
    matches = [f for f in files if predicate(f)]
    if not matches:
        raise RuntimeError("no %s found; files=%r" % (label, files))
    return sorted(matches, key=len)[0]


def input_bin_index(fname):
    if not fname.endswith(".bin"):
        return -1
    stem = fname[:-4]
    marker = ".example_input."
    pos = stem.rfind(marker)
    if pos < 0:
        return -1
    idx_str = stem[pos + len(marker):]
    if not idx_str or not all("0" <= ch <= "9" for ch in idx_str):
        return -1
    return int(idx_str)


def autodetect_paths():
    files = [f for f in os.listdir(".") if not f.startswith(".")]
    int8_vela = [f for f in files if f.endswith("_vela.tflite") and ".int8" in f]
    int8_plain = [
        f for f in files
        if f.endswith(".int8.tflite") and not f.endswith("_vela.tflite")
    ]
    fp32_vela = [f for f in files if f.endswith("_vela.tflite") and ".int8" not in f]
    fp32_plain = [
        f for f in files
        if f.endswith(".tflite") and ".int8" not in f and not f.endswith("_vela.tflite")
    ]
    if int8_vela:
        model = sorted(int8_vela, key=len)[0]
    elif int8_plain:
        model = sorted(int8_plain, key=len)[0]
    elif fp32_vela:
        model = sorted(fp32_vela, key=len)[0]
    else:
        model = pick_first(files, lambda f: f in fp32_plain, ".tflite")

    indexed_inputs = [(input_bin_index(f), f) for f in files]
    indexed_inputs = [(i, f) for i, f in indexed_inputs if i >= 0]
    if not indexed_inputs:
        raise RuntimeError("no .example_input.<i>.bin found; files=%r" % (files,))
    indexed_inputs.sort()
    inputs = [f for _, f in indexed_inputs]
    output = pick_first(files, lambda f: f.endswith(".example_output.bin"), ".example_output.bin")
    meta = pick_first(files, lambda f: f.endswith(".example_meta.json"), ".example_meta.json")
    return model, inputs, output, meta


def shape_numel(shape):
    n = 1
    for d in shape:
        n *= int(d)
    return n


def normalize_shapes(shapes):
    if not isinstance(shapes, (list, tuple)):
        return [shapes]
    if len(shapes) > 0 and isinstance(shapes[0], (list, tuple)):
        return shapes
    return [shapes]


def dtype_char(dtype):
    return chr(dtype) if isinstance(dtype, int) else dtype


def model_values(model, name, n, default):
    values = getattr(model, name, None)
    if values is None:
        return [default] * n
    if not isinstance(values, (list, tuple)):
        return [values] * n
    out = []
    for i in range(n):
        out.append(values[i] if i < len(values) else default)
    return out


def first_value(values, default):
    if values is None:
        return default
    if isinstance(values, (list, tuple)):
        if len(values) == 0:
            return default
        return values[0]
    return values


def load_sample_float32(path, sample_index, num_floats):
    byte_count = num_floats * 4
    with open(path, "rb") as f:
        f.seek(sample_index * byte_count)
        buf = bytearray(byte_count)
        read = f.readinto(buf)
    if read != byte_count:
        raise RuntimeError("short read from %s: %d of %d" % (path, read, byte_count))
    return np.frombuffer(buf, dtype=np.float)


def max_abs_diff(a, b):
    err = 0.0
    for i in range(len(a)):
        d = float(a[i]) - float(b[i])
        if d < 0.0:
            d = -d
        if d > err:
            err = d
    return err


def q_byte(v, scale, zp, dtype):
    q = int(round(float(v) / scale + zp))
    if dtype == "B":
        if q < 0:
            q = 0
        elif q > 255:
            q = 255
        return q
    if q < -128:
        q = -128
    elif q > 127:
        q = 127
    return q & 0xFF


def q_bytes_from_float_sample(path, sample_index, numel, scale, zp, dtype):
    x = load_sample_float32(path, sample_index, numel)
    out = bytearray(numel)
    for i in range(numel):
        out[i] = q_byte(x[i], scale, zp, dtype)
    return out


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
    inner = token[pos + 1:end]
    args = [a.strip() for a in inner.split(",")]
    return name, args


def parse_camera_config(observation):
    image_obs, _ = split_top_level_once(observation)
    name, args = parse_call_token(image_obs)
    if name not in ("CameraRGB", "CameraRGBWithTarget",
                    "CameraRGBStacked", "CameraRGBStackedWithTarget"):
        raise RuntimeError("unsupported camera observation %r" % image_obs)
    if len(args) < 3:
        raise RuntimeError("camera observation missing H/W: %r" % image_obs)
    height = int(args[1])
    width = int(args[2])
    stride = 1
    stack = 1
    has_target = False
    if name == "CameraRGBWithTarget":
        has_target = True
    elif name == "CameraRGBStacked":
        stride = int(args[3])
        stack = int(args[4])
    elif name == "CameraRGBStackedWithTarget":
        stride = int(args[3])
        stack = int(args[4])
        has_target = True
    return {
        "image_observation": image_obs,
        "height": height,
        "width": width,
        "stride": stride,
        "stack": stack,
        "has_target": has_target,
        "num_image_inputs": stack + (1 if has_target else 0),
    }


def observation_from_meta(meta):
    actor_meta = meta.get("actor_meta")
    if not actor_meta:
        return ""
    env = actor_meta.get("environment")
    if not env:
        return ""
    return env.get("observation", "")


def checkpoint_name_from_artifact(path):
    suffixes = (
        ".int8_vela.tflite",
        ".int8.tflite",
        "_vela.tflite",
        ".tflite",
        ".example_meta.json",
    )
    for suffix in suffixes:
        if path.endswith(suffix):
            return path[:-len(suffix)]
    return path


def checkpoint_name_from_meta(meta, fallback_path):
    for key in ("source_hdf5_path", "checkpoint_name", "hdf5_checkpoint_name"):
        value = meta.get(key)
        if value:
            return value
    actor_meta = meta.get("actor_meta")
    if isinstance(actor_meta, dict):
        value = actor_meta.get("checkpoint_name")
        if value:
            return value
    return checkpoint_name_from_artifact(fallback_path)


def configure_camera(csi0):
    csi0.reset()
    csi0.pixformat(csi.RGB565)
    csi0.framesize(csi.QVGA)
    csi0.framerate(CAMERA_SENSOR_FPS)
    csi0.auto_exposure(False, exposure_us=CAMERA_EXPOSURE_US)
    csi0.auto_gain(False, gain_db=CAMERA_GAIN_DB)
    csi0.auto_rotation(False)
    csi0.hmirror(False)
    csi0.vflip(False)
    csi0.transpose(False)
    csi0.auto_blc(False)
    csi0.auto_whitebal(True)


class YawRuntime:
    def __init__(self):
        self.model_path, self.input_paths, self.output_path, self.meta_path = autodetect_paths()
        print("MODEL_PATH :", repr(self.model_path), "size:", os.stat(self.model_path)[6])
        for i, p in enumerate(self.input_paths):
            print("INPUT_%d    :" % i, repr(p), "size:", os.stat(p)[6])
        print("OUTPUT_PATH:", repr(self.output_path), "size:", os.stat(self.output_path)[6])
        print("META_PATH  :", repr(self.meta_path))
        with open(self.meta_path) as f:
            self.meta = json.load(f)
        self.checkpoint_name = checkpoint_name_from_meta(self.meta, self.model_path)
        print("CHECKPOINT_NAME:", self.checkpoint_name)

        self.model = ml.Model(self.model_path)
        print_mem("after ml.Model()")
        print(self.model)

        self.input_shapes = normalize_shapes(self.model.input_shape)
        self.output_shapes = normalize_shapes(self.model.output_shape)
        self.num_inputs = len(self.input_shapes)
        if len(self.input_paths) != self.num_inputs:
            raise RuntimeError("model has %d inputs but %d input bins" %
                               (self.num_inputs, len(self.input_paths)))
        self.input_numels = [shape_numel(s) for s in self.input_shapes]
        self.output_numel = shape_numel(self.output_shapes[0])
        if self.output_numel != 2:
            raise RuntimeError("yaw model output dim %d != 2" % self.output_numel)

        self.input_dtypes = [
            dtype_char(v) for v in model_values(self.model, "input_dtype", self.num_inputs, "?")
        ]
        self.output_dtype = dtype_char(first_value(getattr(self.model, "output_dtype", "?"), "?"))
        self.input_scales = [
            float(v) for v in model_values(self.model, "input_scale", self.num_inputs, 1.0)
        ]
        self.input_zps = [
            float(v) for v in model_values(self.model, "input_zero_point", self.num_inputs, 0.0)
        ]
        self.output_scale = float(first_value(getattr(self.model, "output_scale", None), 1.0))
        self.output_zp = float(first_value(getattr(self.model, "output_zero_point", None), 0.0))

        for i in range(self.num_inputs):
            if self.input_dtypes[i] not in ("b", "B"):
                raise RuntimeError("input %d dtype %r is not int8/uint8" %
                                   (i, self.input_dtypes[i]))
            if self.input_scales[i] <= 0.0:
                raise RuntimeError("input %d has invalid scale %g" %
                                   (i, self.input_scales[i]))
        scale0 = self.input_scales[0]
        zp0 = self.input_zps[0]
        dtype0 = self.input_dtypes[0]
        for i in range(1, self.num_inputs):
            if (abs(self.input_scales[i] - scale0) > 1e-9 or
                    abs(self.input_zps[i] - zp0) > 1e-6 or
                    self.input_dtypes[i] != dtype0):
                raise RuntimeError("image input %d quantization differs from input 0" % i)
        self.image_scale = scale0
        self.image_zp = zp0
        self.image_dtype = dtype0
        self.image_direct_s8 = (
            self.image_dtype == "b" and
            abs(self.image_scale - IMAGE_U8_SCALE) < 1e-6 and
            int(self.image_zp) == -128
        )

        self.tflite_to_keras = [None] * self.num_inputs
        self.keras_to_tflite = [None] * self.num_inputs
        for entry in self.meta.get("inputs", []):
            if "tflite_input_index" in entry and "keras_input_index" in entry:
                tflite_i = int(entry["tflite_input_index"])
                keras_i = int(entry["keras_input_index"])
                if 0 <= tflite_i < self.num_inputs and 0 <= keras_i < self.num_inputs:
                    self.tflite_to_keras[tflite_i] = keras_i
                    self.keras_to_tflite[keras_i] = tflite_i
        for i in range(self.num_inputs):
            if self.tflite_to_keras[i] is None:
                self.tflite_to_keras[i] = i
            if self.keras_to_tflite[i] is None:
                self.keras_to_tflite[i] = i

        observation = observation_from_meta(self.meta)
        print("observation:", observation)
        self.camera = parse_camera_config(observation)
        if not self.camera["has_target"]:
            raise RuntimeError("yaw runtime expects Camera...WithTarget observation")
        if self.camera["num_image_inputs"] != self.num_inputs:
            raise RuntimeError("camera inputs %d != model inputs %d" %
                               (self.camera["num_image_inputs"], self.num_inputs))

        self.int8_output_path = None
        int8_output = self.meta.get("int8_output")
        if int8_output:
            candidate = int8_output.get("path")
            if candidate:
                try:
                    os.stat(candidate)
                    self.int8_output_path = candidate
                except OSError:
                    print("meta references int8_output %r but file is absent" % candidate)
        self.predict_raw_output = False

        print("camera: %dx%d stack=%d stride=%d target=1 inputs=%d" %
              (self.camera["width"], self.camera["height"],
               self.camera["stack"], self.camera["stride"], self.num_inputs))
        print("tflite->keras:", self.tflite_to_keras)
        print("input dtype:", self.image_dtype,
              "scale=%g zp=%g direct_s8=%d" %
              (self.image_scale, self.image_zp, 1 if self.image_direct_s8 else 0))
        print("output dtype:", self.output_dtype,
              "scale=%g zp=%g" % (self.output_scale, self.output_zp))

    def output_values(self, y_flat):
        if not self.predict_raw_output:
            return y_flat
        out = [0.0] * len(y_flat)
        for i in range(len(y_flat)):
            out[i] = (float(y_flat[i]) - self.output_zp) * self.output_scale
        return out

    def self_check(self, sources_by_keras, feeders):
        ref_path = self.int8_output_path or self.output_path
        output_size = os.stat(ref_path)[6]
        if output_size == 0 or output_size % (self.output_numel * 4) != 0:
            raise RuntimeError("output bin size %d invalid" % output_size)
        n_check = output_size // (self.output_numel * 4)
        err_max = 0.0
        for sample_idx in range(n_check):
            for tflite_i in range(self.num_inputs):
                keras_i = self.tflite_to_keras[tflite_i]
                expected = self.input_numels[tflite_i] * 4 * n_check
                actual = os.stat(self.input_paths[tflite_i])[6]
                if actual < expected:
                    raise RuntimeError("input %d bin size %d < expected %d" %
                                       (tflite_i, actual, expected))
                sources_by_keras[keras_i] = q_bytes_from_float_sample(
                    self.input_paths[tflite_i], sample_idx, self.input_numels[tflite_i],
                    self.input_scales[tflite_i], self.input_zps[tflite_i],
                    self.input_dtypes[tflite_i]
                )
            y_raw = self.model.predict(feeders)[0].flatten()
            y_ref = load_sample_float32(ref_path, sample_idx, self.output_numel)
            if sample_idx == 0 and self.output_dtype in ("b", "B"):
                err_float = max_abs_diff(y_raw, y_ref)
                y_deq = [
                    (float(y_raw[i]) - self.output_zp) * self.output_scale
                    for i in range(len(y_raw))
                ]
                err_raw = max_abs_diff(y_deq, y_ref)
                self.predict_raw_output = err_raw < err_float
                if self.predict_raw_output:
                    print("output API returned raw int8 values")
            y_f32 = self.output_values(y_raw)
            err = max_abs_diff(y_f32, y_ref)
            if err > err_max:
                err_max = err
            print("[%3d] yaw_err=%.6g ref=%+.4f,%+.4f pred=%+.4f,%+.4f" %
                  (sample_idx, err, y_ref[0], y_ref[1],
                   float(y_f32[0]), float(y_f32[1])))

        if self.int8_output_path is not None:
            fail_tol = min(SELF_CHECK_INT8_FAIL_LSBS * self.output_scale,
                           SELF_CHECK_INT8_MAX_ERR)
            if fail_tol <= 0.0:
                fail_tol = SELF_CHECK_INT8_MAX_ERR
        else:
            fail_tol = SELF_CHECK_FP32_MAX_ERR
        print("self-check: yaw max_abs_err=%.6g fail_tol=%.6g over %d samples" %
              (err_max, fail_tol, n_check))
        if err_max > fail_tol:
            raise RuntimeError("yaw self-check err %.6g > %.6g" %
                               (err_max, fail_tol))


def make_feeder(runtime, sources_by_keras, tflite_i):
    keras_i = runtime.tflite_to_keras[tflite_i]

    def feeder(buf, shape, dtype):
        buf[:] = sources_by_keras[keras_i]

    return feeder


def normalize_yaw(c, s):
    n = math.sqrt(c * c + s * s)
    if n > 1.0e-6:
        return c / n, s / n, n
    return 1.0, 0.0, 0.0


def wrap_rad(a):
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def clamp_int(v, lo, hi):
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def q_s16(v, scale):
    q = int(round(float(v) * scale))
    return clamp_int(q, -32768, 32767)


def put_u16_be(dst, idx, v):
    dst[idx] = (v >> 8) & 0xFF
    dst[idx + 1] = v & 0xFF


def crc16_ccitt(buf, n):
    crc = 0xFFFF
    for i in range(n):
        crc ^= buf[i] << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


def pack7(raw, raw_len, out, out_offset):
    acc = 0
    nbits = 0
    w = out_offset
    for i in range(raw_len):
        acc = (acc << 8) | raw[i]
        nbits += 8
        while nbits >= 7:
            nbits -= 7
            out[w] = (acc >> nbits) & 0x7F
            w += 1
    if nbits > 0:
        out[w] = (acc << (7 - nbits)) & 0x7F


def unpack7(packed, packed_len, out, raw_len):
    acc = 0
    nbits = 0
    out_idx = 0
    for i in range(packed_len):
        acc = (acc << 7) | (packed[i] & 0x7F)
        nbits += 7
        if nbits >= 8 and out_idx < raw_len:
            nbits -= 8
            out[out_idx] = (acc >> nbits) & 0xFF
            out_idx += 1


def uart_text_byte(b):
    return b == 9 or b == 10 or b == 13 or 32 <= b <= 126


def build_visual_yaw_frame_into(frame, raw, crc_payload,
                                seq, target_seq, flags, age_ms, yaw_rad):
    start = FRAME_START_MASK | FRAME_TYPE_VISUAL_YAW
    yaw_q = q_s16(yaw_rad, YAW_Q_SCALE)

    raw[0] = seq & 0xFF
    raw[1] = target_seq & 0xFF
    raw[2] = flags & 0xFF
    raw[3] = clamp_int(int(age_ms), 0, 255)
    raw[4] = (yaw_q >> 8) & 0xFF
    raw[5] = yaw_q & 0xFF

    crc_payload[0] = start
    for i in range(6):
        crc_payload[i + 1] = raw[i]
    crc = crc16_ccitt(crc_payload, VISUAL_YAW_CRC_PAYLOAD_BYTES)
    put_u16_be(raw, 6, crc)

    frame[0] = start
    pack7(raw, VISUAL_YAW_RAW_BYTES, frame, 1)


class CfYawCommandReceiver:
    def __init__(self):
        self.data_buf = bytearray(TARGET_CAPTURE_DATA_BYTES)
        self.raw = bytearray(TARGET_CAPTURE_RAW_BYTES)
        self.crc_payload = bytearray(TARGET_CAPTURE_CRC_PAYLOAD_BYTES)
        self.rx_line_buf = bytearray()
        self.start_byte = 0
        self.data_idx = -1
        self.line_has_binary = False
        self.capture_requested = False
        self.record_start_requested = False
        self.record_stop_requested = False
        self.record_abort_requested = False
        self.target_seq = 0
        self.command_seq = 0
        self.record_start_seq = 0
        self.record_stop_seq = 0
        self.record_abort_seq = 0
        self.reason = 0
        self.record_start_reason = 0
        self.record_stop_reason = 0
        self.record_abort_reason = 0
        self.flags = 0
        self.record_start_flags = 0
        self.record_stop_flags = 0
        self.record_abort_flags = 0
        self.frames_ok = 0
        self.frames_bad_crc = 0
        self.frame_restarts = 0
        self.unknown_frames = 0
        self.dropped_binary_bytes = 0

    def poll(self, uart_bridge):
        self.capture_requested = False
        self.record_start_requested = False
        self.record_stop_requested = False
        self.record_abort_requested = False
        n_avail = uart_bridge.any()
        if not n_avail:
            return False
        chunk = uart_bridge.read(n_avail)
        if not chunk:
            return False
        for b in chunk:
            self.feed_byte(b)
        return (self.capture_requested or
                self.record_start_requested or
                self.record_stop_requested or
                self.record_abort_requested)

    def feed_byte(self, b):
        if self.data_idx >= 0:
            if b & FRAME_START_MASK:
                self.frame_restarts += 1
                self.start_frame_or_drop(b)
                return
            self.data_buf[self.data_idx] = b
            self.data_idx += 1
            if self.data_idx == TARGET_CAPTURE_DATA_BYTES:
                self.apply_target_capture_frame()
                self.data_idx = -1
            return

        if b & FRAME_START_MASK:
            self.start_frame_or_drop(b)
            return

        if uart_text_byte(b):
            self.feed_text_byte(b)
        else:
            self.line_has_binary = True
            self.dropped_binary_bytes += 1

    def start_frame_or_drop(self, b):
        frame_type = b & FRAME_TYPE_MASK
        if frame_type == FRAME_TYPE_TARGET_CAPTURE:
            self.start_byte = b
            self.data_idx = 0
        else:
            self.data_idx = -1
            self.unknown_frames += 1

    def apply_target_capture_frame(self):
        unpack7(self.data_buf, TARGET_CAPTURE_DATA_BYTES,
                self.raw, TARGET_CAPTURE_RAW_BYTES)
        rx_crc = (self.raw[4] << 8) | self.raw[5]
        self.crc_payload[0] = self.start_byte
        for i in range(4):
            self.crc_payload[i + 1] = self.raw[i]
        ex_crc = crc16_ccitt(self.crc_payload, TARGET_CAPTURE_CRC_PAYLOAD_BYTES)
        if rx_crc != ex_crc:
            self.frames_bad_crc += 1
            return

        self.command_seq = self.raw[0]
        command = self.raw[1]
        reason = self.raw[2]
        flags = self.raw[3]
        if command == TARGET_CAPTURE_COMMAND_CURRENT:
            self.target_seq = self.command_seq
            self.reason = reason
            self.flags = flags
            self.capture_requested = True
        elif command == TARGET_CAPTURE_COMMAND_RECORD_START:
            self.record_start_seq = self.command_seq
            self.record_start_reason = reason
            self.record_start_flags = flags
            self.record_start_requested = True
        elif command == TARGET_CAPTURE_COMMAND_RECORD_STOP_SAVE:
            self.record_stop_seq = self.command_seq
            self.record_stop_reason = reason
            self.record_stop_flags = flags
            self.record_stop_requested = True
        elif command == TARGET_CAPTURE_COMMAND_RECORD_ABORT:
            self.record_abort_seq = self.command_seq
            self.record_abort_reason = reason
            self.record_abort_flags = flags
            self.record_abort_requested = True
        else:
            self.unknown_frames += 1
            return
        self.frames_ok += 1

    def feed_text_byte(self, b):
        self.rx_line_buf.append(b)
        if b == 10:
            line = bytes(self.rx_line_buf[:-1]).rstrip(b"\r")
            self.rx_line_buf = bytearray()
            if self.line_has_binary:
                self.line_has_binary = False
                return
            self.handle_line(line)
        elif len(self.rx_line_buf) > UART_RX_LINE_MAX:
            self.rx_line_buf = bytearray()
            self.line_has_binary = False

    def handle_line(self, line):
        if b"[u1br] arm rising edge" in line:
            self.target_seq = (self.target_seq + 1) & 0xFF
            self.reason = 0
            self.flags = 0
            self.capture_requested = True
        try:
            print("[cf]", line.decode("utf-8"))
        except UnicodeError:
            print("[cf-bin]", line)


def run():
    print_mem("boot")
    print("cwd:", os.getcwd())
    print("listdir(cwd):", os.listdir("."))

    runtime = YawRuntime()
    img_h = runtime.camera["height"]
    img_w = runtime.camera["width"]
    frame_stack_n = runtime.camera["stack"]
    frame_stride = runtime.camera["stride"]
    target_keras_index = frame_stack_n
    frame_history_length = frame_stride * (frame_stack_n - 1) + 1

    frame_bytes = img_h * img_w * 3
    frame_history_q = [bytearray(frame_bytes) for _ in range(frame_history_length)]
    target_q = bytearray(frame_bytes)
    scaled_frame_u8 = None if runtime.image_direct_s8 else bytearray(frame_bytes)

    zero = q_byte(0.0, runtime.image_scale, runtime.image_zp, runtime.image_dtype)
    for buf in frame_history_q:
        for i in range(len(buf)):
            buf[i] = zero
    for i in range(len(target_q)):
        target_q[i] = zero

    sources_by_keras = [frame_history_q[0]] * runtime.num_inputs
    sources_by_keras[target_keras_index] = target_q
    feeders = [
        make_feeder(runtime, sources_by_keras, i)
        for i in range(runtime.num_inputs)
    ]

    runtime.self_check(sources_by_keras, feeders)

    csi0 = csi.CSI()
    configure_camera(csi0)
    csi0.framebuffers(CAMERA_FRAMEBUFFERS)
    sensor_w, sensor_h = csi0.width(), csi0.height()
    print("camera async framebuffers:", csi0.framebuffers())
    print("camera %dx%d fixed fps=%d exposure_us=%d gain_db=%d" %
          (sensor_w, sensor_h, CAMERA_SENSOR_FPS, CAMERA_EXPOSURE_US, CAMERA_GAIN_DB))
    if sensor_w * img_h != sensor_h * img_w:
        print("camera/model aspect mismatch: sensor=%dx%d model=%dx%d stretch-resize active" %
              (sensor_w, sensor_h, img_w, img_h))

    frame_draw_hint = image.BILINEAR | image.SCALE_ASPECT_IGNORE
    scaled_frame_rgb565 = image.Image(img_w, img_h, image.RGB565)
    target_frame_rgb565 = image.Image(img_w, img_h, image.RGB565)
    storage_led = BlueLed()
    capture = YawFrameCapture(runtime, img_w, img_h, storage_led)

    def resize_quantize_frame(src_img, dst_q):
        scaled_frame_rgb565.draw_image(src_img, 0, 0, hint=frame_draw_hint)
        if runtime.image_direct_s8:
            scaled_frame_rgb565.to_ndarray(dtype="b", buffer=dst_q)
            return
        scaled_frame_rgb565.to_ndarray(dtype="B", buffer=scaled_frame_u8)
        for i in range(len(dst_q)):
            dst_q[i] = q_byte(scaled_frame_u8[i] / 255.0,
                              runtime.image_scale, runtime.image_zp,
                              runtime.image_dtype)

    def update_sources(history_write_ptr):
        sources_by_keras[0] = frame_history_q[history_write_ptr]
        for f in range(1, frame_stack_n):
            slot = (history_write_ptr - f * frame_stride) % frame_history_length
            sources_by_keras[f] = frame_history_q[slot]
        sources_by_keras[target_keras_index] = target_q

    reset_button = machine.Pin("SW", machine.Pin.IN, machine.Pin.PULL_UP)
    reset_button_last = reset_button.value()
    uart_bridge = machine.UART(UART_BRIDGE_PORT, UART_BRIDGE_BAUD,
                               timeout=0, timeout_char=0)
    cf_rx = CfYawCommandReceiver()
    yaw_frame = bytearray(VISUAL_YAW_FRAME_BYTES)
    yaw_raw = bytearray(VISUAL_YAW_RAW_BYTES)
    yaw_crc_payload = bytearray(VISUAL_YAW_CRC_PAYLOAD_BYTES)
    yaw_seq = 0
    target_seq = 0
    target_capture_pending = True
    target_ack_pending = False
    target_bias = 0.0
    target_captured = False
    history_write_ptr = 0
    tick = 0
    camera_pending = csi0.snapshot(blocking=False)
    next_deadline = time.ticks_us()

    print("starting yaw loop vision=%dHz uart%d=%d" %
          (VISION_HZ, UART_BRIDGE_PORT, UART_BRIDGE_BAUD))
    while True:
        t0 = time.ticks_us()
        reset_button_cur = reset_button.value()
        if reset_button_last == 1 and reset_button_cur == 0:
            target_seq = (target_seq + 1) & 0xFF
            target_capture_pending = True
            target_ack_pending = False
            print("target reset (button) seq=%d" % target_seq)
        reset_button_last = reset_button_cur

        if cf_rx.poll(uart_bridge):
            if cf_rx.record_start_requested:
                capture.start(cf_rx.record_start_seq,
                              cf_rx.record_start_reason)
                if target_captured:
                    capture.capture_target(target_frame_rgb565, target_seq,
                                           time.ticks_us())
                    capture.set_target_bias(target_bias)
            if cf_rx.record_abort_requested:
                capture.abort(cf_rx.record_abort_seq,
                              cf_rx.record_abort_reason)
            if cf_rx.record_stop_requested:
                capture.stop_and_save(cf_rx.record_stop_seq,
                                      cf_rx.record_stop_reason)
            if cf_rx.capture_requested:
                target_seq = cf_rx.target_seq
                target_capture_pending = True
                target_ack_pending = True
                print("target reset (uart) seq=%d reason=%d flags=0x%02x" %
                      (target_seq, cf_rx.reason, cf_rx.flags))

        img = camera_pending
        camera_pending = None
        if img is None:
            img = csi0.snapshot(blocking=False)
        t_snapshot = time.ticks_us()
        if img is None:
            delay = time.ticks_diff(next_deadline, time.ticks_us())
            if delay > 0:
                time.sleep_us(delay)
            next_deadline = time.ticks_add(next_deadline, VISION_TICK_US)
            continue

        resize_quantize_frame(img, frame_history_q[history_write_ptr])
        t_resize = time.ticks_us()
        camera_pending = csi0.snapshot(blocking=False)
        captured_now = False
        if not target_captured or target_capture_pending:
            target_q[:] = frame_history_q[history_write_ptr]
            for i in range(frame_history_length):
                if i != history_write_ptr:
                    frame_history_q[i][:] = frame_history_q[history_write_ptr]
            target_frame_rgb565.draw_image(scaled_frame_rgb565, 0, 0)
            target_captured = True
            target_capture_pending = False
            captured_now = True
            capture.capture_target(target_frame_rgb565, target_seq, t_resize)
            print("target captured seq=%d" % target_seq)

        update_sources(history_write_ptr)
        if captured_now:
            y_bias_raw = runtime.model.predict(feeders)[0].flatten()
            y_bias = runtime.output_values(y_bias_raw)
            bias_cos, bias_sin, _ = normalize_yaw(float(y_bias[0]), float(y_bias[1]))
            target_bias = math.atan2(bias_sin, bias_cos)
            capture.set_target_bias(target_bias)

        y_raw = runtime.model.predict(feeders)[0].flatten()
        y = runtime.output_values(y_raw)
        yaw_cos, yaw_sin, yaw_norm = normalize_yaw(float(y[0]), float(y[1]))
        yaw_raw_rad = math.atan2(yaw_sin, yaw_cos)
        yaw_rad = wrap_rad(yaw_raw_rad - target_bias)
        yaw_deg = yaw_rad * 180.0 / math.pi
        t_predict = time.ticks_us()

        flags = VISUAL_YAW_FLAG_PREDICTION_VALID
        if target_captured:
            flags |= VISUAL_YAW_FLAG_TARGET_VALID
        if target_ack_pending:
            flags |= VISUAL_YAW_FLAG_TARGET_CAPTURE_ACK
        age_ms = time.ticks_diff(t_predict, t_snapshot) // 1000
        build_visual_yaw_frame_into(yaw_frame, yaw_raw, yaw_crc_payload,
                                    yaw_seq, target_seq, flags, age_ms, yaw_rad)
        uart_bridge.write(yaw_frame)
        capture.append_frame(scaled_frame_rgb565, yaw_seq, target_seq, flags,
                             t0, t_snapshot, t_resize, t_predict,
                             next_deadline, age_ms, yaw_raw_rad, target_bias,
                             yaw_rad, yaw_norm)
        yaw_seq = (yaw_seq + 1) & 0xFF
        target_ack_pending = False

        history_write_ptr = (history_write_ptr + 1) % frame_history_length

        if tick % DIAG_PRINT_EVERY == 0:
            elapsed_us = time.ticks_diff(t_predict, t0)
            snapshot_us = time.ticks_diff(t_snapshot, t0)
            resize_us = time.ticks_diff(t_resize, t_snapshot)
            predict_us = time.ticks_diff(t_predict, t_resize)
            delay_us = time.ticks_diff(next_deadline, time.ticks_us())
            print("us=%5d cam=%5d resize=%4d pred=%5d delay=%d yaw=%+.1fdeg "
                  "raw=%+.1fdeg bias=%+.1fdeg cos_sin=%+.3f,%+.3f norm=%.3f "
                  "target=%d seq=%d tx=%d" %
                  (elapsed_us, snapshot_us, resize_us, predict_us, delay_us,
                   yaw_deg, yaw_raw_rad * 180.0 / math.pi,
                   target_bias * 180.0 / math.pi, yaw_cos, yaw_sin, yaw_norm,
                   1 if target_captured else 0, target_seq, yaw_seq))

        tick += 1
        delay = time.ticks_diff(next_deadline, time.ticks_us())
        if delay > 0:
            time.sleep_us(delay)
        next_deadline = time.ticks_add(next_deadline, VISION_TICK_US)
        while time.ticks_diff(time.ticks_us(), next_deadline) >= 0:
            next_deadline = time.ticks_add(next_deadline, VISION_TICK_US)


if __name__ == "__main__":
    run()
