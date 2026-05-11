import gc
import json
import math
import os
import time

import csi
import image
import imu
import machine
import micropython
import ml
from ulab import numpy as np


CONTROL_HZ = 500
VISION_HZ = 100
CONTROL_SUBSTEPS = CONTROL_HZ // VISION_HZ
TICK_US = 1_000_000 // CONTROL_HZ
DIAG_PRINT_EVERY = CONTROL_HZ
INV_CONTROL_SUBSTEPS = 1.0 / CONTROL_SUBSTEPS

UART_BRIDGE_PORT = 4
UART_BRIDGE_BAUD = 115200
UART_RX_LINE_MAX = 256

FRAME_START_MASK = 0x80
FRAME_FLAGS = 0x00
FRAME_START_BYTE = FRAME_START_MASK | FRAME_FLAGS

IMU_CTRL1_XL = 0x10
IMU_CTRL2_G = 0x11
IMU_CTRL4_C = 0x13
IMU_CTRL6_C = 0x15

MG_TO_MPS2 = 9.80665e-3
MDPS_TO_RADPS = math.pi / (180.0 * 1000.0)

ACTION_DIM = 4
ACCEL_DIM = 3
SELF_CHECK_FAIL_TOL = 0.5
IMAGE_U8_SCALE = 1.0 / 255.0


def configure_imu():
    imu.__write_reg(IMU_CTRL2_G, (0b0111 << 4) | (0b11 << 2))
    imu.__write_reg(IMU_CTRL1_XL, (0b0111 << 4) | (0b11 << 2))
    imu.__write_reg(IMU_CTRL4_C, 0x02)
    imu.__write_reg(IMU_CTRL6_C, 0x01)


def print_mem(label):
    gc.collect()
    print("%-24s free=%d alloc=%d" % (label, gc.mem_free(), gc.mem_alloc()))


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
    visual = pick_first(
        files,
        lambda f: f.endswith(".visual.int8_vela.tflite") or f.endswith(".visual.int8.tflite"),
        "visual int8 tflite",
    )
    control = pick_first(
        files,
        lambda f: f.endswith(".control.int8_vela.tflite") or f.endswith(".control.int8.tflite"),
        "control int8 tflite",
    )
    indexed_inputs = [(input_bin_index(f), f) for f in files]
    indexed_inputs = [(i, f) for i, f in indexed_inputs if i >= 0]
    if not indexed_inputs:
        raise RuntimeError("no .example_input.<i>.bin found; files=%r" % (files,))
    indexed_inputs.sort()
    inputs = [f for _, f in indexed_inputs]
    meta = pick_first(files, lambda f: f.endswith(".example_meta.json"), ".example_meta.json")
    return visual, control, inputs, meta


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


def first_value(values, default):
    if values is None:
        return default
    if isinstance(values, (list, tuple)):
        if len(values) == 0:
            return default
        return values[0]
    return values


def model_scales(model, name, n):
    values = getattr(model, name, None)
    if values is None:
        return [1.0] * n
    if not isinstance(values, (list, tuple)):
        return [float(values)] * n
    return [float(values[i]) for i in range(n)]


def model_zps(model, name, n):
    values = getattr(model, name, None)
    if values is None:
        return [0.0] * n
    if not isinstance(values, (list, tuple)):
        return [float(values)] * n
    return [float(values[i]) for i in range(n)]


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


def q_byte(v, scale, zp):
    q = int(round(float(v) / scale + zp))
    if q < -128:
        q = -128
    elif q > 127:
        q = 127
    return q & 0xFF


def q_bytes_from_float_sample(path, sample_index, numel, scale, zp):
    x = load_sample_float32(path, sample_index, numel)
    out = bytearray(numel)
    for i in range(numel):
        out[i] = q_byte(x[i], scale, zp)
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


def split_dots_top_level(s):
    out = []
    depth = 0
    start = 0
    for i, ch in enumerate(s):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "." and depth == 0:
            out.append(s[start:i].strip())
            start = i + 1
    tail = s[start:].strip()
    if tail:
        out.append(tail)
    return out


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
    image_obs, dense_obs = split_top_level_once(observation)
    name, args = parse_call_token(image_obs)
    if name not in ("CameraRGB", "CameraRGBWithTarget",
                    "CameraRGBStacked", "CameraRGBStackedWithTarget"):
        raise RuntimeError("unsupported camera observation %r" % image_obs)
    if len(args) < 3:
        raise RuntimeError("camera observation missing H/W: %r" % image_obs)
    height = int(args[1])
    width = int(args[2])
    stride = 1
    n_stack = 1
    has_target = False
    if name == "CameraRGBWithTarget":
        has_target = True
    elif name == "CameraRGBStacked":
        stride = int(args[3])
        n_stack = int(args[4])
    elif name == "CameraRGBStackedWithTarget":
        stride = int(args[3])
        n_stack = int(args[4])
        has_target = True
    return {
        "image_observation": image_obs,
        "dense_observation": dense_obs,
        "height": height,
        "width": width,
        "stride": stride,
        "stack": n_stack,
        "has_target": has_target,
        "num_image_inputs": n_stack + (1 if has_target else 0),
    }


def parse_dense_layout(dense_obs, expected_dim):
    components = []
    offset = 0
    action_history = 0
    accel_history = 0
    for token in split_dots_top_level(dense_obs):
        name, args = parse_call_token(token)
        if name in ("OrientationWorldZ", "OrientationBodyZ"):
            dim = 3
            kind = "orientation_world_z"
            count = 1
        elif name == "AngularVelocity":
            dim = 3
            kind = "angular_velocity"
            count = 1
        elif name == "LinearAccelerationBodyFrame":
            dim = 3
            kind = "linear_acceleration"
            count = 1
        elif name == "LinearAccelerationBodyFrameHistory":
            count = int(args[0]) if args else 0
            dim = ACCEL_DIM * count
            kind = "linear_acceleration_history"
            accel_history = count
        elif name == "ActionHistory":
            count = int(args[0]) if args else 0
            dim = ACTION_DIM * count
            kind = "action_history"
            action_history = count
        else:
            raise RuntimeError("unsupported dense observation component %r" % token)
        components.append({
            "kind": kind,
            "offset": offset,
            "dim": dim,
            "count": count,
        })
        offset += dim
    if offset != expected_dim:
        raise RuntimeError("dense observation dim %d != control state dim %d" %
                           (offset, expected_dim))
    return components, action_history, accel_history


class MahonyFilter:
    def __init__(self, kp=2.0, ki=0.005, max_bias_rad_s=0.1,
                 accel_gate_lo_g=0.75, accel_gate_hi_g=1.25, g_ref=9.80665):
        self.q0 = 1.0
        self.q1 = 0.0
        self.q2 = 0.0
        self.q3 = 0.0
        self.bx = 0.0
        self.by = 0.0
        self.bz = 0.0
        self.kp = kp
        self.ki = ki
        self.max_bias = max_bias_rad_s
        self.g_ref = g_ref
        self.a_lo = accel_gate_lo_g * g_ref
        self.a_hi = accel_gate_hi_g * g_ref
        self.initialized = False

    def _seed_from_accel(self, ax, ay, az):
        norm = math.sqrt(ax * ax + ay * ay + az * az)
        if norm < 1e-9:
            return
        ux = ax / norm
        uy = ay / norm
        uz = az / norm
        if uz < -0.999999:
            self.q0, self.q1, self.q2, self.q3 = 0.0, 1.0, 0.0, 0.0
            return
        w = 1.0 + uz
        x = uy
        y = -ux
        z = 0.0
        qn = math.sqrt(w * w + x * x + y * y + z * z)
        self.q0 = w / qn
        self.q1 = x / qn
        self.q2 = y / qn
        self.q3 = z / qn

    def update(self, ax, ay, az, gx, gy, gz, dt):
        if not self.initialized:
            self._seed_from_accel(ax, ay, az)
            self.initialized = True

        a_norm = math.sqrt(ax * ax + ay * ay + az * az)
        use_accel = a_norm > 1e-6 and self.a_lo <= a_norm <= self.a_hi

        if use_accel:
            inv = 1.0 / a_norm
            ahx = ax * inv
            ahy = ay * inv
            ahz = az * inv

            vx = 2.0 * (self.q1 * self.q3 - self.q0 * self.q2)
            vy = 2.0 * (self.q2 * self.q3 + self.q0 * self.q1)
            vz = self.q0 * self.q0 - self.q1 * self.q1 - self.q2 * self.q2 + self.q3 * self.q3

            ex = ahy * vz - ahz * vy
            ey = ahz * vx - ahx * vz
            ez = ahx * vy - ahy * vx

            self.bx -= self.ki * ex * dt
            self.by -= self.ki * ey * dt
            self.bz -= self.ki * ez * dt
            self.bx = clamp(self.bx, -self.max_bias, self.max_bias)
            self.by = clamp(self.by, -self.max_bias, self.max_bias)
            self.bz = clamp(self.bz, -self.max_bias, self.max_bias)

            wx = gx - self.bx + self.kp * ex
            wy = gy - self.by + self.kp * ey
            wz = gz - self.bz + self.kp * ez
        else:
            wx = gx - self.bx
            wy = gy - self.by
            wz = gz - self.bz

        q0, q1, q2, q3 = self.q0, self.q1, self.q2, self.q3
        dq0 = 0.5 * (-q1 * wx - q2 * wy - q3 * wz)
        dq1 = 0.5 * ( q0 * wx + q2 * wz - q3 * wy)
        dq2 = 0.5 * ( q0 * wy - q1 * wz + q3 * wx)
        dq3 = 0.5 * ( q0 * wz + q1 * wy - q2 * wx)

        q0 += dq0 * dt
        q1 += dq1 * dt
        q2 += dq2 * dt
        q3 += dq3 * dt

        qn = math.sqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3)
        if qn > 1e-12:
            inv_q = 1.0 / qn
            self.q0 = q0 * inv_q
            self.q1 = q1 * inv_q
            self.q2 = q2 * inv_q
            self.q3 = q3 * inv_q
        else:
            self.q0, self.q1, self.q2, self.q3 = 1.0, 0.0, 0.0, 0.0

    def orientation_world_z(self):
        x = 2.0 * (self.q1 * self.q3 - self.q0 * self.q2)
        y = 2.0 * (self.q2 * self.q3 + self.q0 * self.q1)
        z = self.q0 * self.q0 - self.q1 * self.q1 - self.q2 * self.q2 + self.q3 * self.q3
        return x, y, z


def finite_reasonable(v):
    return v == v and -1.0e6 < v < 1.0e6


def clamp(v, lo, hi):
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


@micropython.viper
def copy_history_4(dst: ptr8, dst_offset: int, src: ptr8, write_ptr: int, n_slots: int):
    cur = write_ptr - 1
    if cur < 0:
        cur += n_slots
    i = 0
    while i < n_slots:
        s = cur * 4
        d = dst_offset + i * 4
        dst[d] = src[s]
        dst[d + 1] = src[s + 1]
        dst[d + 2] = src[s + 2]
        dst[d + 3] = src[s + 3]
        cur -= 1
        if cur < 0:
            cur += n_slots
        i += 1


@micropython.viper
def copy_history_3(dst: ptr8, dst_offset: int, src: ptr8, write_ptr: int, n_slots: int):
    cur = write_ptr - 1
    if cur < 0:
        cur += n_slots
    i = 0
    while i < n_slots:
        s = cur * 3
        d = dst_offset + i * 3
        dst[d] = src[s]
        dst[d + 1] = src[s + 1]
        dst[d + 2] = src[s + 2]
        cur -= 1
        if cur < 0:
            cur += n_slots
        i += 1


def pack3(dst, offset, scale, zp, a, b, c):
    dst[offset] = q_byte(a, scale, zp)
    dst[offset + 1] = q_byte(b, scale, zp)
    dst[offset + 2] = q_byte(c, scale, zp)


def build_frame_into(frame13, a0, a1, a2, a3):
    frame13[0] = FRAME_START_BYTE
    for idx, a in ((0, a0), (2, a1), (4, a2), (6, a3)):
        if not finite_reasonable(a):
            a = 0.0
        a = clamp(a, -1.0, 1.0)
        pwm = int((a + 1.0) * 32767.5 + 0.5)
        pwm = clamp(pwm, 0, 0xFFFF)
        raw_payload[idx] = (pwm >> 8) & 0xFF
        raw_payload[idx + 1] = pwm & 0xFF
    crc_payload[0] = frame13[0]
    for i in range(8):
        crc_payload[i + 1] = raw_payload[i]
    crc = crc16_ccitt(crc_payload, 9)
    raw_payload[8] = (crc >> 8) & 0xFF
    raw_payload[9] = crc & 0xFF
    pack7(raw_payload, 10, frame13, 1)


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


raw_payload = bytearray(10)
crc_payload = bytearray(9)


class SplitRuntime:
    def __init__(self):
        self.visual_path, self.control_path, self.input_paths, self.meta_path = autodetect_paths()
        print("VISUAL_MODEL :", repr(self.visual_path), "size:", os.stat(self.visual_path)[6])
        print("CONTROL_MODEL:", repr(self.control_path), "size:", os.stat(self.control_path)[6])
        for i, p in enumerate(self.input_paths):
            print("INPUT_%d      :" % i, repr(p), "size:", os.stat(p)[6])
        print("META_PATH    :", repr(self.meta_path))
        with open(self.meta_path) as f:
            self.meta = json.load(f)
        self.split = self.meta.get("openmv_visual_split")
        if not self.split or not self.split.get("enabled"):
            raise RuntimeError("meta JSON has no openmv_visual_split block")
        self.input_path_by_keras = {}
        for entry in self.meta.get("inputs", []):
            if "keras_input_index" in entry and "path" in entry:
                self.input_path_by_keras[int(entry["keras_input_index"])] = entry["path"]

        actor_meta = self.meta.get("actor_meta", {})
        env_meta = actor_meta.get("environment", {})
        observation = env_meta.get("observation", "")
        print("observation:", observation)
        self.camera = parse_camera_config(observation)

        self.visual_model = ml.Model(self.visual_path)
        self.control_model = ml.Model(self.control_path)
        print_mem("after ml.Model()")
        print(self.visual_model)
        print(self.control_model)

        self.visual_shapes = normalize_shapes(self.visual_model.input_shape)
        self.control_shapes = normalize_shapes(self.control_model.input_shape)
        self.visual_num_inputs = len(self.visual_shapes)
        self.control_num_inputs = len(self.control_shapes)
        self.visual_numels = [shape_numel(s) for s in self.visual_shapes]
        self.control_numels = [shape_numel(s) for s in self.control_shapes]
        self.visual_scales = model_scales(self.visual_model, "input_scale", self.visual_num_inputs)
        self.visual_zps = model_zps(self.visual_model, "input_zero_point", self.visual_num_inputs)
        self.control_scales = model_scales(self.control_model, "input_scale", self.control_num_inputs)
        self.control_zps = model_zps(self.control_model, "input_zero_point", self.control_num_inputs)
        self.visual_out_scale = float(first_value(getattr(self.visual_model, "output_scale", None), 1.0))
        self.visual_out_zp = float(first_value(getattr(self.visual_model, "output_zero_point", None), 0.0))
        self.control_out_scale = float(first_value(getattr(self.control_model, "output_scale", None), 1.0))
        self.control_out_zp = float(first_value(getattr(self.control_model, "output_zero_point", None), 0.0))
        self.visual_out_dtype = dtype_char(first_value(getattr(self.visual_model, "output_dtype", "?"), "?"))
        self.control_out_dtype = dtype_char(first_value(getattr(self.control_model, "output_dtype", "?"), "?"))

        self.visual_tflite_to_keras = [None] * self.visual_num_inputs
        for entry in self.split["visual_inputs"]:
            self.visual_tflite_to_keras[int(entry["tflite_input_index"])] = int(entry["keras_input_index"])
        self.control_roles = [None] * self.control_num_inputs
        for entry in self.split["control_inputs"]:
            role = entry.get("role")
            self.control_roles[int(entry["tflite_input_index"])] = role
        self.embedding_tflite_idx = self.control_roles.index("embedding")
        self.state_tflite_idx = self.control_roles.index("state")
        self.embedding_dim = self.control_numels[self.embedding_tflite_idx]
        self.state_dim = self.control_numels[self.state_tflite_idx]
        self.embedding_scale = self.control_scales[self.embedding_tflite_idx]
        self.embedding_zp = self.control_zps[self.embedding_tflite_idx]
        self.state_scale = self.control_scales[self.state_tflite_idx]
        self.state_zp = self.control_zps[self.state_tflite_idx]
        boundary = self.split.get("embedding_boundary", {})
        self.embedding_raw_copy = bool(boundary.get("raw_int8_copy", False))
        if self.embedding_raw_copy:
            if (abs(self.visual_out_scale - self.embedding_scale) > 1e-9 or
                    int(self.visual_out_zp) != int(self.embedding_zp)):
                raise RuntimeError("embedding boundary metadata says raw copy, but model scales differ")
        self.state_input_path = self.input_path_by_keras.get(self.split["state_full_input_index"])
        if self.state_input_path is None:
            raise RuntimeError("could not locate dense state companion input path")

        image_scale = self.visual_scales[0]
        image_zp = self.visual_zps[0]
        for i in range(1, self.visual_num_inputs):
            if abs(self.visual_scales[i] - image_scale) > 1e-9 or abs(self.visual_zps[i] - image_zp) > 1e-6:
                raise RuntimeError("visual input %d scale/zp differs from input 0" % i)
        self.image_scale = image_scale
        self.image_zp = image_zp
        self.image_direct_s8 = (
            abs(self.image_scale - IMAGE_U8_SCALE) < 1e-6
            and int(self.image_zp) == -128
        )

        self.dense_components, self.action_history_length, self.accel_history_length = parse_dense_layout(
            self.camera["dense_observation"], self.state_dim
        )

        self.visual_ref_path = self.split["visual_output"]["path"]
        self.split_output_path = self.split["split_int8_output"]["path"]
        self.output_numel = shape_numel(normalize_shapes(self.control_model.output_shape)[0])
        if self.output_numel != ACTION_DIM:
            raise RuntimeError("control output dim %d != 4 actions" % self.output_numel)

        print("camera: %dx%d stack=%d stride=%d target=%d inputs=%d" %
              (self.camera["width"], self.camera["height"], self.camera["stack"],
               self.camera["stride"], 1 if self.camera["has_target"] else 0,
               self.camera["num_image_inputs"]))
        print("state_dim:", self.state_dim,
              "action_history:", self.action_history_length,
              "accel_history:", self.accel_history_length)
        print("visual tflite->keras:", self.visual_tflite_to_keras)
        print("control roles:", self.control_roles)
        print("embedding raw-copy:", self.embedding_raw_copy)
        print("image direct signed int8:", self.image_direct_s8)

        self.visual_predict_raw = False
        self.control_predict_raw = False

    def visual_output_values(self, y_flat):
        if not self.visual_predict_raw:
            return y_flat
        out = [0.0] * len(y_flat)
        for i in range(len(y_flat)):
            out[i] = (float(y_flat[i]) - self.visual_out_zp) * self.visual_out_scale
        return out

    def control_output_values(self, y_flat):
        if not self.control_predict_raw:
            return y_flat
        out = [0.0] * len(y_flat)
        for i in range(len(y_flat)):
            out[i] = (float(y_flat[i]) - self.control_out_zp) * self.control_out_scale
        return out

    def quantize_embedding_into(self, embedding_q, y_flat):
        values = self.visual_output_values(y_flat)
        for i in range(self.embedding_dim):
            embedding_q[i] = q_byte(values[i], self.embedding_scale, self.embedding_zp)

    def copy_raw_embedding_into(self, embedding_q, y_flat):
        for i in range(self.embedding_dim):
            embedding_q[i] = int(y_flat[i]) & 0xFF

    def set_embedding_from_visual(self, embedding_q, y_flat):
        if self.embedding_raw_copy and self.visual_predict_raw:
            self.copy_raw_embedding_into(embedding_q, y_flat)
        else:
            self.quantize_embedding_into(embedding_q, y_flat)

    def self_check(self, visual_sources_by_keras, visual_feeders, embedding_q,
                   state_q, control_feeders):
        output_size = os.stat(self.split_output_path)[6]
        if output_size == 0 or output_size % (self.output_numel * 4) != 0:
            raise RuntimeError("split output bin size %d invalid" % output_size)
        n_check = output_size // (self.output_numel * 4)
        visual_ref_numel = self.embedding_dim
        visual_err_max = 0.0
        split_err_max = 0.0
        for sample_idx in range(n_check):
            for tflite_i in range(self.visual_num_inputs):
                keras_i = self.visual_tflite_to_keras[tflite_i]
                path = self.split["visual_inputs"][tflite_i]["source_input_path"]
                visual_sources_by_keras[keras_i] = q_bytes_from_float_sample(
                    path, sample_idx, self.visual_numels[tflite_i],
                    self.visual_scales[tflite_i], self.visual_zps[tflite_i]
                )

            y_visual_raw = self.visual_model.predict(visual_feeders)[0].flatten()
            visual_ref = load_sample_float32(self.visual_ref_path, sample_idx, visual_ref_numel)
            if sample_idx == 0 and self.visual_out_dtype in ("b", "B"):
                err_float = max_abs_diff(y_visual_raw, visual_ref)
                y_deq = [(float(y_visual_raw[i]) - self.visual_out_zp) * self.visual_out_scale
                         for i in range(len(y_visual_raw))]
                err_raw = max_abs_diff(y_deq, visual_ref)
                self.visual_predict_raw = err_raw < err_float
                if self.embedding_raw_copy:
                    if self.visual_predict_raw:
                        print("embedding boundary: raw int8 copy")
                    else:
                        print("embedding boundary: API returned float, quantize fallback")
            visual_values = self.visual_output_values(y_visual_raw)
            visual_err = max_abs_diff(visual_values, visual_ref)
            if visual_err > visual_err_max:
                visual_err_max = visual_err
            self.set_embedding_from_visual(embedding_q, y_visual_raw)

            state_sample = load_sample_float32(self.state_input_path, sample_idx, self.state_dim)
            for i in range(self.state_dim):
                state_q[i] = q_byte(state_sample[i], self.state_scale, self.state_zp)

            y_control_raw = self.control_model.predict(control_feeders)[0].flatten()
            split_ref = load_sample_float32(self.split_output_path, sample_idx, self.output_numel)
            if sample_idx == 0 and self.control_out_dtype in ("b", "B"):
                err_float = max_abs_diff(y_control_raw, split_ref)
                y_deq = [(float(y_control_raw[i]) - self.control_out_zp) * self.control_out_scale
                         for i in range(len(y_control_raw))]
                err_raw = max_abs_diff(y_deq, split_ref)
                self.control_predict_raw = err_raw < err_float
            control_values = self.control_output_values(y_control_raw)
            split_err = max_abs_diff(control_values, split_ref)
            if split_err > split_err_max:
                split_err_max = split_err
            print("[%3d] visual_err=%.6g split_err=%.6g" %
                  (sample_idx, visual_err, split_err))

        print("self-check: visual max_abs_err=%.6g split max_abs_err=%.6g over %d samples" %
              (visual_err_max, split_err_max, n_check))
        if visual_err_max > SELF_CHECK_FAIL_TOL:
            raise RuntimeError("visual self-check err %.6g > %.6g" %
                               (visual_err_max, SELF_CHECK_FAIL_TOL))
        if split_err_max > SELF_CHECK_FAIL_TOL:
            raise RuntimeError("split self-check err %.6g > %.6g" %
                               (split_err_max, SELF_CHECK_FAIL_TOL))


def run():
    print_mem("boot")
    print("cwd:", os.getcwd())
    print("listdir(cwd):", os.listdir("."))

    configure_imu()
    runtime = SplitRuntime()

    img_h = runtime.camera["height"]
    img_w = runtime.camera["width"]
    frame_stack_n = runtime.camera["stack"]
    frame_stride = runtime.camera["stride"]
    has_target = runtime.camera["has_target"]
    target_keras_index = frame_stack_n if has_target else -1
    frame_history_length = frame_stride * (frame_stack_n - 1) + 1

    csi0 = csi.CSI()
    csi0.reset()
    csi0.pixformat(csi.RGB565)
    csi0.framesize(csi.QVGA)
    sensor_w, sensor_h = csi0.width(), csi0.height()
    csi0.framerate(VISION_HZ)
    for _ in range(10):
        csi0.snapshot()
    if sensor_w * img_h != sensor_h * img_w:
        raise RuntimeError("sensor aspect %dx%d does not match model %dx%d" %
                           (sensor_w, sensor_h, img_w, img_h))

    frame_draw_hint = image.BILINEAR | image.SCALE_ASPECT_IGNORE
    scaled_frame_rgb565 = image.Image(img_w, img_h, image.RGB565)
    frame_bytes = img_h * img_w * 3
    frame_history_q = [bytearray(frame_bytes) for _ in range(frame_history_length)]
    target_q = bytearray(frame_bytes)
    scaled_frame_u8 = None if runtime.image_direct_s8 else bytearray(frame_bytes)

    visual_sources_by_keras = [None] * runtime.camera["num_image_inputs"]
    for i in range(len(visual_sources_by_keras)):
        visual_sources_by_keras[i] = frame_history_q[0]
    if has_target:
        visual_sources_by_keras[target_keras_index] = target_q

    def make_visual_feeder(tflite_i):
        keras_i = runtime.visual_tflite_to_keras[tflite_i]
        def feeder(buf, shape, dtype):
            buf[:] = visual_sources_by_keras[keras_i]
        return feeder

    visual_feeders = [make_visual_feeder(i) for i in range(runtime.visual_num_inputs)]

    embedding_q = bytearray(runtime.embedding_dim)
    state_q = bytearray(runtime.state_dim)
    zero_embedding = q_byte(0.0, runtime.embedding_scale, runtime.embedding_zp)
    for i in range(runtime.embedding_dim):
        embedding_q[i] = zero_embedding

    action_ring_q = bytearray(runtime.action_history_length * ACTION_DIM)
    accel_ring_q = bytearray(runtime.accel_history_length * ACCEL_DIM)
    zero_state = q_byte(0.0, runtime.state_scale, runtime.state_zp)
    for i in range(len(action_ring_q)):
        action_ring_q[i] = zero_state
    for i in range(len(accel_ring_q)):
        accel_ring_q[i] = zero_state
    action_write_ptr = 0
    accel_write_ptr = 0

    def make_control_feeder(tflite_i):
        role = runtime.control_roles[tflite_i]
        def feeder(buf, shape, dtype):
            if role == "embedding":
                buf[:] = embedding_q
            else:
                buf[:] = state_q
        return feeder

    control_feeders = [make_control_feeder(i) for i in range(runtime.control_num_inputs)]

    def resize_quantize_frame(src_img, dst_q, tflite_i):
        scaled_frame_rgb565.draw_image(src_img, 0, 0, hint=frame_draw_hint)
        if runtime.image_direct_s8:
            scaled_frame_rgb565.to_ndarray(dtype="b", buffer=dst_q)
            return
        scaled_frame_rgb565.to_ndarray(dtype="B", buffer=scaled_frame_u8)
        scale = runtime.visual_scales[tflite_i]
        zp = runtime.visual_zps[tflite_i]
        for i in range(len(dst_q)):
            dst_q[i] = q_byte(scaled_frame_u8[i] / 255.0, scale, zp)

    def update_visual_sources(history_write_ptr):
        visual_sources_by_keras[0] = frame_history_q[history_write_ptr]
        for f in range(1, frame_stack_n):
            slot = (history_write_ptr - f * frame_stride) % frame_history_length
            visual_sources_by_keras[f] = frame_history_q[slot]
        if has_target:
            visual_sources_by_keras[target_keras_index] = target_q

    def build_state(world_z, ang_vel, linear_accel):
        for comp in runtime.dense_components:
            off = comp["offset"]
            kind = comp["kind"]
            if kind == "orientation_world_z":
                pack3(state_q, off, runtime.state_scale, runtime.state_zp,
                      world_z[0], world_z[1], world_z[2])
            elif kind == "angular_velocity":
                pack3(state_q, off, runtime.state_scale, runtime.state_zp,
                      ang_vel[0], ang_vel[1], ang_vel[2])
            elif kind == "linear_acceleration":
                pack3(state_q, off, runtime.state_scale, runtime.state_zp,
                      linear_accel[0], linear_accel[1], linear_accel[2])
            elif kind == "linear_acceleration_history":
                copy_history_3(state_q, off, accel_ring_q,
                               accel_write_ptr, runtime.accel_history_length)
            elif kind == "action_history":
                copy_history_4(state_q, off, action_ring_q,
                               action_write_ptr, runtime.action_history_length)

    runtime.self_check(visual_sources_by_keras, visual_feeders, embedding_q,
                       state_q, control_feeders)

    mahony = MahonyFilter()
    uart_bridge = machine.UART(UART_BRIDGE_PORT, UART_BRIDGE_BAUD, timeout=0, timeout_char=0)
    frame_tx = bytearray(13)
    rx_line_buf = bytearray()
    reset_button = machine.Pin("SW", machine.Pin.IN, machine.Pin.PULL_UP)
    reset_button_last = reset_button.value()

    history_write_ptr = 0
    target_captured = False
    last_t = time.ticks_us()
    next_deadline = time.ticks_add(last_t, TICK_US)
    tick = 0
    substep = 0
    action_sum_0 = 0.0
    action_sum_1 = 0.0
    action_sum_2 = 0.0
    action_sum_3 = 0.0
    accel_sum_x = 0.0
    accel_sum_y = 0.0
    accel_sum_z = 0.0
    a0_raw = 0.0
    a1_raw = 0.0
    a2_raw = 0.0
    a3_raw = 0.0
    visual_age = 0

    print("starting split visual policy loop control=%dHz vision=%dHz substeps=%d" %
          (CONTROL_HZ, VISION_HZ, CONTROL_SUBSTEPS))

    while True:
        t0 = time.ticks_us()

        reset_button_cur = reset_button.value()
        if reset_button_last == 1 and reset_button_cur == 0:
            target_captured = False
            print("target reset (button)")
        reset_button_last = reset_button_cur

        ax_orig, ay_orig, az_orig = imu.acceleration_mg()
        ax = -az_orig
        ay =  ay_orig
        az =  ax_orig
        gx_orig, gy_orig, gz_orig = imu.angular_rate_mdps()
        gx = -gz_orig
        gy =  gy_orig
        gz =  gx_orig
        t_imu = time.ticks_us()

        dt = time.ticks_diff(t0, last_t) * 1e-6
        last_t = t0
        if dt <= 0.0 or dt > 0.5:
            dt = 1.0 / CONTROL_HZ

        linear_accel = (ax * MG_TO_MPS2, ay * MG_TO_MPS2, az * MG_TO_MPS2)
        gx_rad = gx * MDPS_TO_RADPS
        gy_rad = gy * MDPS_TO_RADPS
        gz_rad = gz * MDPS_TO_RADPS
        mahony.update(linear_accel[0], linear_accel[1], linear_accel[2],
                      gx_rad, gy_rad, gz_rad, dt)
        world_z = mahony.orientation_world_z()
        ang_vel = (gx_rad, gy_rad, gz_rad)
        t_mahony = time.ticks_us()

        visual_updated = 0
        t_snapshot = t_mahony
        t_downsample = t_mahony
        t_visual = t_mahony
        if substep == 0:
            img = csi0.snapshot()
            if img is None:
                raise RuntimeError("camera snapshot failed")
            t_snapshot = time.ticks_us()
            resize_quantize_frame(img, frame_history_q[history_write_ptr], 0)
            t_downsample = time.ticks_us()
            if not target_captured:
                target_q[:] = frame_history_q[history_write_ptr]
                for i in range(frame_history_length):
                    if i != history_write_ptr:
                        frame_history_q[i][:] = frame_history_q[history_write_ptr]
                target_captured = True
            update_visual_sources(history_write_ptr)
            y_visual = runtime.visual_model.predict(visual_feeders)[0].flatten()
            runtime.set_embedding_from_visual(embedding_q, y_visual)
            history_write_ptr = (history_write_ptr + 1) % frame_history_length
            visual_age = 0
            visual_updated = 1
            t_visual = time.ticks_us()
        else:
            visual_age += 1

        build_state(world_z, ang_vel, linear_accel)
        t_state = time.ticks_us()

        y_control = runtime.control_model.predict(control_feeders)[0].flatten()
        y_f32 = runtime.control_output_values(y_control)
        a0_raw = float(y_f32[0])
        a1_raw = float(y_f32[1])
        a2_raw = float(y_f32[2])
        a3_raw = float(y_f32[3])
        a0 = clamp(a0_raw, -1.0, 1.0)
        a1 = clamp(a1_raw, -1.0, 1.0)
        a2 = clamp(a2_raw, -1.0, 1.0)
        a3 = clamp(a3_raw, -1.0, 1.0)
        t_control = time.ticks_us()

        build_frame_into(frame_tx, a0, a1, a2, a3)
        uart_bridge.write(frame_tx)
        t_tx = time.ticks_us()

        n_avail = uart_bridge.any()
        if n_avail:
            chunk = uart_bridge.read(n_avail)
            if chunk:
                rx_line_buf.extend(chunk)
                while True:
                    nl = rx_line_buf.find(b"\n")
                    if nl < 0:
                        break
                    line = bytes(rx_line_buf[:nl]).rstrip(b"\r")
                    rx_line_buf = rx_line_buf[nl + 1:]
                    if b"[u1br] arm rising edge" in line:
                        target_captured = False
                        print("target reset (arm rising edge)")
                    try:
                        print("[cf]", line.decode("utf-8"))
                    except UnicodeError:
                        print("[cf-bin]", line)
                if len(rx_line_buf) > UART_RX_LINE_MAX:
                    print("[cf-overflow]", bytes(rx_line_buf))
                    rx_line_buf = bytearray()
        t_rx = time.ticks_us()

        action_sum_0 += a0
        action_sum_1 += a1
        action_sum_2 += a2
        action_sum_3 += a3
        if runtime.accel_history_length > 0:
            accel_sum_x += linear_accel[0]
            accel_sum_y += linear_accel[1]
            accel_sum_z += linear_accel[2]

        substep += 1
        history_published = 0
        substep_print = substep
        if substep >= CONTROL_SUBSTEPS:
            if runtime.action_history_length > 0:
                off = action_write_ptr * ACTION_DIM
                action_ring_q[off] = q_byte(action_sum_0 * INV_CONTROL_SUBSTEPS,
                                            runtime.state_scale, runtime.state_zp)
                action_ring_q[off + 1] = q_byte(action_sum_1 * INV_CONTROL_SUBSTEPS,
                                                runtime.state_scale, runtime.state_zp)
                action_ring_q[off + 2] = q_byte(action_sum_2 * INV_CONTROL_SUBSTEPS,
                                                runtime.state_scale, runtime.state_zp)
                action_ring_q[off + 3] = q_byte(action_sum_3 * INV_CONTROL_SUBSTEPS,
                                                runtime.state_scale, runtime.state_zp)
                action_write_ptr = (action_write_ptr + 1) % runtime.action_history_length
            if runtime.accel_history_length > 0:
                off = accel_write_ptr * ACCEL_DIM
                accel_ring_q[off] = q_byte(accel_sum_x * INV_CONTROL_SUBSTEPS,
                                           runtime.state_scale, runtime.state_zp)
                accel_ring_q[off + 1] = q_byte(accel_sum_y * INV_CONTROL_SUBSTEPS,
                                               runtime.state_scale, runtime.state_zp)
                accel_ring_q[off + 2] = q_byte(accel_sum_z * INV_CONTROL_SUBSTEPS,
                                               runtime.state_scale, runtime.state_zp)
                accel_write_ptr = (accel_write_ptr + 1) % runtime.accel_history_length
            action_sum_0 = 0.0
            action_sum_1 = 0.0
            action_sum_2 = 0.0
            action_sum_3 = 0.0
            accel_sum_x = 0.0
            accel_sum_y = 0.0
            accel_sum_z = 0.0
            substep = 0
            history_published = 1
            substep_print = CONTROL_SUBSTEPS
        t_actions = time.ticks_us()

        if tick % DIAG_PRINT_EVERY == 0:
            elapsed_us = time.ticks_diff(t_actions, t0)
            imu_us = time.ticks_diff(t_imu, t0)
            mahony_us = time.ticks_diff(t_mahony, t_imu)
            snapshot_us = time.ticks_diff(t_snapshot, t_mahony)
            downsample_us = time.ticks_diff(t_downsample, t_snapshot)
            visual_us = time.ticks_diff(t_visual, t_downsample)
            state_us = time.ticks_diff(t_state, t_visual)
            control_us = time.ticks_diff(t_control, t_state)
            tx_us = time.ticks_diff(t_tx, t_control)
            rx_us = time.ticks_diff(t_rx, t_tx)
            actions_us = time.ticks_diff(t_actions, t_rx)
            delay_us = time.ticks_diff(next_deadline, time.ticks_us())
            print("us=%5d imu=%4d mah=%4d cam=%5d ds=%4d vis=%4d st=%4d ctl=%4d "
                  "tx=%3d rx=%3d act=%3d delay=%d sub=%d/%d hist=%d vupd=%d vage=%d "
                  "a=%+.2f,%+.2f,%+.2f,%+.2f wz=%+.2f,%+.2f,%+.2f av=%+.2f,%+.2f,%+.2f" %
                  (elapsed_us, imu_us, mahony_us, snapshot_us, downsample_us,
                   visual_us, state_us, control_us, tx_us, rx_us, actions_us,
                   delay_us, substep_print, CONTROL_SUBSTEPS, history_published,
                   visual_updated, visual_age, a0, a1, a2, a3,
                   world_z[0], world_z[1], world_z[2],
                   ang_vel[0], ang_vel[1], ang_vel[2]))

        tick += 1
        delay = time.ticks_diff(next_deadline, time.ticks_us())
        if delay > 0:
            time.sleep_us(delay)
        next_deadline = time.ticks_add(next_deadline, TICK_US)


if __name__ == "__main__":
    run()
