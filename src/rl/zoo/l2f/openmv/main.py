import gc
import json
import math
import os
import time

import imu
import machine
import ml
from ulab import numpy as np
try:
    import ustruct
except ImportError:
    import struct as ustruct


TRAINING_CONTROL_HZ = 100
CONTROL_SUBSTEPS = 5
CONTROL_HZ = TRAINING_CONTROL_HZ * CONTROL_SUBSTEPS
TICK_US = 1_000_000 // CONTROL_HZ
DIAG_PRINT_EVERY = CONTROL_HZ
INV_CONTROL_SUBSTEPS = 1.0 / CONTROL_SUBSTEPS

SETPOINT_ROLL_RAD = 0.0
SETPOINT_PITCH_RAD = 0.0
SETPOINT_YAW_RATE_RAD_S = 0.0
SETPOINT_THRUST_G = 1.0
SETPOINT_MAX_TILT_RAD = 0.5235987755982988
SETPOINT_MAX_YAW_RATE_RAD_S = 2.0
SETPOINT_MIN_THRUST_G = 0.4
SETPOINT_MAX_THRUST_G = 1.4
SETPOINT_TIMEOUT_US = 300_000

# From ENVIRONMENT_ATTITUDE_SETPOINT_FACTORY::dynamics.hovering_throttle_relative.
HOVERING_THROTTLE_RELATIVE = 0.6864
HOVER_ACTION = 2.0 * HOVERING_THROTTLE_RELATIVE - 1.0

UART_BRIDGE_PORT = 4
UART_BRIDGE_BAUD = 115200
UART_RX_LINE_MAX = 256

IMU_CTRL1_XL = 0x10
IMU_CTRL2_G = 0x11
IMU_CTRL4_C = 0x13
IMU_CTRL6_C = 0x15

FRAME_START_MASK = 0x80
FRAME_FLAGS = 0x00
FRAME_START_BYTE = FRAME_START_MASK | FRAME_FLAGS
FRAME_TYPE_MASK = 0x7F

SETPOINT_FRAME_TYPE_ATTITUDE = 0x02
SETPOINT_FRAME_START_BYTE = FRAME_START_MASK | SETPOINT_FRAME_TYPE_ATTITUDE
SETPOINT_RAW_BYTES = 11
SETPOINT_DATA_BYTES = 13
SETPOINT_CRC_PAYLOAD_BYTES = 10
SETPOINT_INV_SCALE = 1.0e-4

MG_TO_MPS2 = 9.80665e-3
MDPS_TO_RADPS = math.pi / (180.0 * 1000.0)

SELF_CHECK_MAX_ERR = 1.0e-3
SELF_CHECK_INT8_MAX_ERR = 0.5
SELF_CHECK_INT8_FAIL_LSBS = 100

FP32_BYTES = 4
INT8_BYTES = 1
BASE_INPUT_FLOATS = 10
ACCEL_DIM = 3
ACTION_DIM = 4
STATE_PREFIX_FMT = "<ffffffffff"
ACCEL_FMT = "<fff"
ACTION_FMT = "<ffff"


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
    int8_vela = [f for f in files if f.endswith("_vela.tflite") and ".int8" in f]
    fp32_vela = [f for f in files if f.endswith("_vela.tflite") and ".int8" not in f]
    int8_plain = [
        f for f in files
        if f.endswith(".int8.tflite") and not f.endswith("_vela.tflite")
    ]
    fp32_plain = [
        f for f in files
        if f.endswith(".tflite") and ".int8" not in f and not f.endswith("_vela.tflite")
    ]
    if int8_vela:
        model = sorted(int8_vela, key=len)[0]
    elif fp32_vela:
        model = sorted(fp32_vela, key=len)[0]
    elif int8_plain:
        model = sorted(int8_plain, key=len)[0]
    else:
        model = pick_first(files, lambda f: f in fp32_plain, ".tflite")
    indexed_inputs = [(input_bin_index(f), f) for f in files]
    indexed_inputs = [(i, f) for i, f in indexed_inputs if i >= 0]
    if not indexed_inputs:
        raise RuntimeError("no .example_input.<i>.bin found; files=%r" % (files,))
    indexed_inputs.sort()
    inputs = [f for _, f in indexed_inputs]
    out = pick_first(files, lambda f: f.endswith(".example_output.bin"), ".example_output.bin")
    meta = pick_first(files, lambda f: f.endswith(".example_meta.json"), ".example_meta.json")
    return model, inputs, out, meta


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


def first_dtype(dtypes):
    if isinstance(dtypes, (list, tuple)):
        dtypes = dtypes[0]
    return chr(dtypes) if isinstance(dtypes, int) else dtypes


def first_value(values, default):
    if values is None:
        return default
    if isinstance(values, (list, tuple)):
        if len(values) == 0:
            return default
        return values[0]
    return values


def first_attr(obj, name, default):
    return first_value(getattr(obj, name, default), default)


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


def make_storage_feeder(storage):
    def feeder(buf, shape, dtype):
        buf[:] = storage
    return feeder


def parse_history_length(observation, name):
    marker = name + "("
    pos = observation.find(marker)
    if pos < 0:
        return None
    start = pos + len(marker)
    end = observation.find(")", start)
    if end < 0:
        return None
    try:
        return int(observation[start:end])
    except ValueError:
        return None


def observation_from_meta(meta):
    actor_meta = meta.get("actor_meta")
    if actor_meta is None:
        return ""
    env = actor_meta.get("environment")
    if env is None:
        return ""
    return env.get("observation", "")


class ModelRuntime:
    def __init__(self):
        self.model_path, self.input_paths, self.output_path, self.meta_path = autodetect_paths()
        print("MODEL_PATH :", repr(self.model_path), "size:", os.stat(self.model_path)[6])
        for i, p in enumerate(self.input_paths):
            print("INPUT_%d    :" % i, repr(p), "size:", os.stat(p)[6])
        print("OUTPUT_PATH:", repr(self.output_path), "size:", os.stat(self.output_path)[6])
        print("META_PATH  :", repr(self.meta_path))
        with open(self.meta_path) as f:
            self.meta = json.load(f)

        self.model = ml.Model(self.model_path)
        print_mem("after ml.Model()")
        print(self.model)

        self.input_shapes = normalize_shapes(self.model.input_shape)
        self.output_shapes = normalize_shapes(self.model.output_shape)
        self.num_inputs = len(self.input_shapes)
        if self.num_inputs != 1 or len(self.input_paths) != 1:
            raise RuntimeError("attitude-setpoint policy expects one input, model=%d bins=%d" %
                               (self.num_inputs, len(self.input_paths)))

        self.input_shape = self.input_shapes[0]
        self.output_shape = self.output_shapes[0]
        self.input_numel = shape_numel(self.input_shape)
        self.output_numel = shape_numel(self.output_shape)
        self.input_dtype = first_dtype(getattr(self.model, "input_dtype", "?"))
        self.output_dtype = first_dtype(getattr(self.model, "output_dtype", "?"))
        self.quantized_input = self.input_dtype in ("b", "B")
        self.quantized_output = self.output_dtype in ("b", "B")
        self.quantized = self.quantized_input or self.quantized_output
        self.input_value_bytes = INT8_BYTES if self.quantized_input else FP32_BYTES
        self.input_scale = float(first_attr(self.model, "input_scale", 1.0))
        self.input_zero_point = float(first_attr(self.model, "input_zero_point", 0.0))
        self.output_scale = float(first_attr(self.model, "output_scale", 1.0))
        self.output_zero_point = float(first_attr(self.model, "output_zero_point", 0.0))
        self.int8_output_path = None
        if "int8_output" in self.meta:
            candidate = self.meta["int8_output"].get("path")
            if candidate:
                try:
                    os.stat(candidate)
                    self.int8_output_path = candidate
                except OSError:
                    print("meta references int8_output %r but file is absent" %
                          candidate)

        print("input shape:", self.input_shape, "dtype:", self.input_dtype)
        print("output shape:", self.output_shape, "dtype:", self.output_dtype)
        if self.quantized:
            print("input quantization: scale=%g zp=%g" %
                  (self.input_scale, self.input_zero_point))
            print("output quantization: scale=%g zp=%g" %
                  (self.output_scale, self.output_zero_point))
            if self.quantized_input and self.input_scale <= 0.0:
                raise RuntimeError("quantized input has invalid scale %g" %
                                   self.input_scale)
        if self.output_numel != 4:
            raise RuntimeError("expected 4 motor actions, got output dim %d" % self.output_numel)

        observation = observation_from_meta(self.meta)
        print("observation:", observation)
        self.accel_history_length = parse_history_length(
            observation, "LinearAccelerationBodyFrameHistory"
        )
        self.action_history_length = parse_history_length(observation, "ActionHistory")
        if self.accel_history_length is None and self.action_history_length is None:
            remainder = self.input_numel - BASE_INPUT_FLOATS
            if remainder > 0 and remainder % (ACCEL_DIM + ACTION_DIM) == 0:
                inferred = remainder // (ACCEL_DIM + ACTION_DIM)
                self.accel_history_length = inferred
                self.action_history_length = inferred
                print("history lengths inferred from input dim:", inferred)
            elif remainder >= 0 and remainder % ACTION_DIM == 0:
                self.accel_history_length = 0
                self.action_history_length = remainder // ACTION_DIM
                print("action-only history length inferred from input dim:",
                      self.action_history_length)
            else:
                raise RuntimeError("could not infer history lengths from input dim %d" %
                                   self.input_numel)
        else:
            if self.accel_history_length is None:
                self.accel_history_length = 0
            if self.action_history_length is None:
                remainder = self.input_numel - BASE_INPUT_FLOATS - ACCEL_DIM * self.accel_history_length
                if remainder >= 0 and remainder % ACTION_DIM == 0:
                    self.action_history_length = remainder // ACTION_DIM
                    print("action history length inferred from input dim:",
                          self.action_history_length)
                else:
                    raise RuntimeError("could not infer action history length from input dim %d" %
                                       self.input_numel)

        expected_dim = (BASE_INPUT_FLOATS +
                        ACCEL_DIM * self.accel_history_length +
                        ACTION_DIM * self.action_history_length)
        if self.input_numel != expected_dim:
            raise RuntimeError("input dim %d != expected %d from observation layout" %
                               (self.input_numel, expected_dim))
        print("accel_history_length:", self.accel_history_length)
        print("action_history_length:", self.action_history_length)

    def q_byte(self, value):
        q = int(round(float(value) / self.input_scale + self.input_zero_point))
        if self.input_dtype == "B":
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

    def pack_values(self, storage, byte_offset, values):
        n = len(values)
        if self.quantized_input:
            for i in range(n):
                storage[byte_offset + i] = self.q_byte(values[i])
            return
        if n == 10:
            ustruct.pack_into(STATE_PREFIX_FMT, storage, byte_offset,
                              values[0], values[1], values[2], values[3],
                              values[4], values[5], values[6],
                              values[7], values[8], values[9])
        elif n == 3:
            ustruct.pack_into(ACCEL_FMT, storage, byte_offset,
                              values[0], values[1], values[2])
        elif n == 4:
            ustruct.pack_into(ACTION_FMT, storage, byte_offset,
                              values[0], values[1], values[2], values[3])
        else:
            for i in range(n):
                ustruct.pack_into("<f", storage, byte_offset + i * FP32_BYTES,
                                  values[i])

    def output_values(self, y_flat):
        if not self.quantized_output or self.output_scale <= 0.0:
            return y_flat
        max_abs = 0.0
        for i in range(len(y_flat)):
            v_abs = abs(float(y_flat[i]))
            if v_abs > max_abs:
                max_abs = v_abs
        if max_abs <= 4.0:
            return y_flat
        out = [0.0] * len(y_flat)
        for i in range(len(y_flat)):
            out[i] = (float(y_flat[i]) - self.output_zero_point) * self.output_scale
        return out

    def predict_from_storage(self, storage):
        y_raw = self.model.predict([make_storage_feeder(storage)])[0]
        return self.output_values(y_raw.flatten())

    def self_check(self):
        output_size = os.stat(self.output_path)[6]
        if output_size == 0 or output_size % (self.output_numel * 4) != 0:
            raise RuntimeError("output bin size %d is not a positive multiple of %d" %
                               (output_size, self.output_numel * 4))
        n_check = output_size // (self.output_numel * 4)
        expected_input = self.input_numel * 4 * n_check
        actual_input = os.stat(self.input_paths[0])[6]
        if actual_input != expected_input:
            raise RuntimeError("input bin size %d != expected %d" %
                               (actual_input, expected_input))

        err_max = 0.0
        int8_err_max = 0.0
        last_storage = None
        for sample_idx in range(n_check):
            x = load_sample_float32(self.input_paths[0], sample_idx, self.input_numel)
            y_ref = load_sample_float32(self.output_path, sample_idx, self.output_numel)
            storage = bytearray(self.input_numel * self.input_value_bytes)
            self.pack_values(storage, 0, x)
            last_storage = storage
            y_f32 = self.predict_from_storage(storage)
            err = max_abs_diff(y_f32, y_ref)
            if err > err_max:
                err_max = err
            if self.int8_output_path is not None:
                y_int8_ref = load_sample_float32(
                    self.int8_output_path, sample_idx, self.output_numel
                )
                int8_err = max_abs_diff(y_f32, y_int8_ref)
                if int8_err > int8_err_max:
                    int8_err_max = int8_err
                print("[%3d] fp32_err=%.6g int8_err=%.6g" %
                      (sample_idx, err, int8_err))
            else:
                print("[%3d] fp32_err=%.6g" % (sample_idx, err))

        print("self-check: fp32 max_abs_err=%.6g over %d samples" %
              (err_max, n_check))
        if self.int8_output_path is not None:
            fail_tol = min(SELF_CHECK_INT8_FAIL_LSBS * self.output_scale,
                           SELF_CHECK_INT8_MAX_ERR)
            if fail_tol <= 0.0:
                fail_tol = SELF_CHECK_INT8_MAX_ERR
            print("self-check: int8 max_abs_err=%.6g fail_tol=%.6g" %
                  (int8_err_max, fail_tol))
            if int8_err_max > fail_tol:
                raise RuntimeError("int8 self-check err %.6g > %.6g" %
                                   (int8_err_max, fail_tol))
        elif self.quantized:
            if err_max > SELF_CHECK_INT8_MAX_ERR:
                raise RuntimeError("quantized self-check err %.6g > %.6g" %
                                   (err_max, SELF_CHECK_INT8_MAX_ERR))
        elif err_max > SELF_CHECK_MAX_ERR:
            raise RuntimeError("fp32 self-check err %.6g > %.6g" %
                               (err_max, SELF_CHECK_MAX_ERR))
        if last_storage is not None:
            self.predict_from_storage(last_storage)


def finite_reasonable(v):
    return v == v and -1.0e6 < v < 1.0e6


def clamp(v, lo, hi):
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


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


def shift_history_bytes(buf, offset, n_slots, entry_bytes):
    total_bytes = n_slots * entry_bytes
    if total_bytes > entry_bytes:
        buf[offset + entry_bytes:offset + total_bytes] = buf[offset:offset + total_bytes - entry_bytes]


def push_accel_history(runtime, buf, offset, n_slots, ax, ay, az):
    if n_slots <= 0:
        return
    entry_bytes = ACCEL_DIM * runtime.input_value_bytes
    shift_history_bytes(buf, offset, n_slots, entry_bytes)
    runtime.pack_values(buf, offset, (ax, ay, az))


def push_action_history(runtime, buf, offset, n_slots, a0, a1, a2, a3):
    if n_slots <= 0:
        return
    entry_bytes = ACTION_DIM * runtime.input_value_bytes
    shift_history_bytes(buf, offset, n_slots, entry_bytes)
    runtime.pack_values(buf, offset, (a0, a1, a2, a3))


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


def unpack7_into(data, data_len, raw, raw_len):
    acc = 0
    nbits = 0
    out_idx = 0
    for i in range(data_len):
        acc = (acc << 7) | (data[i] & 0x7F)
        nbits += 7
        while nbits >= 8 and out_idx < raw_len:
            nbits -= 8
            raw[out_idx] = (acc >> nbits) & 0xFF
            out_idx += 1


raw_payload = bytearray(10)
crc_payload = bytearray(9)


def sanitize_action(a):
    if not finite_reasonable(a):
        return 0.0
    return clamp(a, -1.0, 1.0)


def build_frame_into(frame13, a0, a1, a2, a3):
    start_byte = FRAME_START_BYTE
    frame13[0] = start_byte
    for idx, a in ((0, a0), (2, a1), (4, a2), (6, a3)):
        a = sanitize_action(a)
        pwm = int((a + 1.0) * 32767.5 + 0.5)
        pwm = clamp(pwm, 0, 0xFFFF)
        raw_payload[idx] = (pwm >> 8) & 0xFF
        raw_payload[idx + 1] = pwm & 0xFF
    crc_payload[0] = start_byte
    for i in range(8):
        crc_payload[i + 1] = raw_payload[i]
    crc = crc16_ccitt(crc_payload, 9)
    raw_payload[8] = (crc >> 8) & 0xFF
    raw_payload[9] = crc & 0xFF
    pack7(raw_payload, 10, frame13, 1)


def project_setpoint_tilt(roll, pitch):
    sin_roll = math.sin(roll)
    cos_roll = math.cos(roll)
    sin_pitch = math.sin(pitch)
    cos_pitch = math.cos(pitch)
    world_z_body_x = -sin_pitch
    world_z_body_y = cos_pitch * sin_roll
    world_z_body_z = cos_pitch * cos_roll

    cos_tilt_max = math.cos(SETPOINT_MAX_TILT_RAD)
    if world_z_body_z >= cos_tilt_max:
        return roll, pitch

    horizontal = math.sqrt(world_z_body_x * world_z_body_x +
                           world_z_body_y * world_z_body_y)
    if horizontal < 1.0e-6:
        return 0.0, 0.0

    sin_tilt_max = math.sin(SETPOINT_MAX_TILT_RAD)
    world_z_body_x *= sin_tilt_max / horizontal
    world_z_body_y *= sin_tilt_max / horizontal
    world_z_body_z = cos_tilt_max

    roll = math.atan2(world_z_body_y, world_z_body_z)
    pitch = math.atan2(-world_z_body_x,
                       math.sqrt(world_z_body_y * world_z_body_y +
                                 world_z_body_z * world_z_body_z))
    return roll, pitch


def s16be(buf, off):
    v = (buf[off] << 8) | buf[off + 1]
    if v & 0x8000:
        v -= 0x10000
    return v


def u16be(buf, off):
    return (buf[off] << 8) | buf[off + 1]


def sanitize_setpoint(roll, pitch, yaw_rate, thrust_g):
    if not finite_reasonable(roll):
        roll = SETPOINT_ROLL_RAD
    if not finite_reasonable(pitch):
        pitch = SETPOINT_PITCH_RAD
    if not finite_reasonable(yaw_rate):
        yaw_rate = SETPOINT_YAW_RATE_RAD_S
    if not finite_reasonable(thrust_g):
        thrust_g = SETPOINT_THRUST_G

    roll, pitch = project_setpoint_tilt(roll, pitch)
    yaw_rate = clamp(yaw_rate, -SETPOINT_MAX_YAW_RATE_RAD_S,
                     SETPOINT_MAX_YAW_RATE_RAD_S)
    thrust_g = clamp(thrust_g, SETPOINT_MIN_THRUST_G, SETPOINT_MAX_THRUST_G)
    return roll, pitch, yaw_rate, thrust_g


class SetpointReceiver:
    def __init__(self):
        self.roll = SETPOINT_ROLL_RAD
        self.pitch = SETPOINT_PITCH_RAD
        self.yaw_rate = SETPOINT_YAW_RATE_RAD_S
        self.thrust_g = SETPOINT_THRUST_G
        self.seq = -1
        self.have = False
        self.last_us = 0

        self.frames_ok = 0
        self.frames_bad_crc = 0
        self.frames_unknown = 0
        self.frames_restart = 0

        self.data_buf = bytearray(SETPOINT_DATA_BYTES)
        self.raw_buf = bytearray(SETPOINT_RAW_BYTES)
        self.crc_buf = bytearray(SETPOINT_CRC_PAYLOAD_BYTES)
        self.rx_line_buf = bytearray()
        self.start_byte = 0
        self.data_idx = -1

    def current(self, now_us):
        if self.have and time.ticks_diff(now_us, self.last_us) <= SETPOINT_TIMEOUT_US:
            return self.roll, self.pitch, self.yaw_rate, self.thrust_g
        return (SETPOINT_ROLL_RAD, SETPOINT_PITCH_RAD,
                SETPOINT_YAW_RATE_RAD_S, SETPOINT_THRUST_G)

    def age_ms(self, now_us):
        if not self.have:
            return -1
        age = time.ticks_diff(now_us, self.last_us)
        if age < 0:
            return -1
        return age // 1000

    def poll(self, uart_bridge):
        n_avail = uart_bridge.any()
        if not n_avail:
            return
        chunk = uart_bridge.read(n_avail)
        if not chunk:
            return
        for b in chunk:
            self.feed_byte(b)

    def feed_byte(self, b):
        if b & FRAME_START_MASK:
            frame_type = b & FRAME_TYPE_MASK
            if frame_type == SETPOINT_FRAME_TYPE_ATTITUDE:
                if self.data_idx >= 0:
                    self.frames_restart += 1
                self.start_byte = b
                self.data_idx = 0
            else:
                self.frames_unknown += 1
                self.data_idx = -1
            return

        if self.data_idx >= 0:
            self.data_buf[self.data_idx] = b
            self.data_idx += 1
            if self.data_idx == SETPOINT_DATA_BYTES:
                self.decode_frame()
                self.data_idx = -1
            return

        self.rx_line_buf.append(b)
        while True:
            nl = self.rx_line_buf.find(b"\n")
            if nl < 0:
                break
            line = bytes(self.rx_line_buf[:nl]).rstrip(b"\r")
            self.rx_line_buf = self.rx_line_buf[nl + 1:]
            try:
                print("[cf]", line.decode("utf-8"))
            except UnicodeError:
                print("[cf-bin]", line)
        if len(self.rx_line_buf) > UART_RX_LINE_MAX:
            print("[cf-overflow]", bytes(self.rx_line_buf))
            self.rx_line_buf = bytearray()

    def decode_frame(self):
        unpack7_into(self.data_buf, SETPOINT_DATA_BYTES,
                     self.raw_buf, SETPOINT_RAW_BYTES)

        self.crc_buf[0] = self.start_byte
        for i in range(9):
            self.crc_buf[i + 1] = self.raw_buf[i]
        rx_crc = u16be(self.raw_buf, 9)
        ex_crc = crc16_ccitt(self.crc_buf, SETPOINT_CRC_PAYLOAD_BYTES)
        if rx_crc != ex_crc:
            self.frames_bad_crc += 1
            return

        roll = s16be(self.raw_buf, 1) * SETPOINT_INV_SCALE
        pitch = s16be(self.raw_buf, 3) * SETPOINT_INV_SCALE
        yaw_rate = s16be(self.raw_buf, 5) * SETPOINT_INV_SCALE
        thrust_g = u16be(self.raw_buf, 7) * SETPOINT_INV_SCALE
        self.roll, self.pitch, self.yaw_rate, self.thrust_g = sanitize_setpoint(
            roll, pitch, yaw_rate, thrust_g
        )
        self.seq = self.raw_buf[0]
        self.have = True
        self.last_us = time.ticks_us()
        self.frames_ok += 1


def run():
    print_mem("boot")
    print("cwd:", os.getcwd())
    print("listdir(cwd):", os.listdir("."))

    configure_imu()

    runtime = ModelRuntime()
    runtime.self_check()

    input_storage = bytearray(runtime.input_numel * runtime.input_value_bytes)
    accel_entry_bytes = ACCEL_DIM * runtime.input_value_bytes
    action_entry_bytes = ACTION_DIM * runtime.input_value_bytes
    accel_offset_bytes = BASE_INPUT_FLOATS * runtime.input_value_bytes
    action_offset_bytes = accel_offset_bytes + runtime.accel_history_length * accel_entry_bytes
    action_history_bytes = runtime.action_history_length * action_entry_bytes
    for off in range(action_offset_bytes,
                     action_offset_bytes + action_history_bytes,
                     action_entry_bytes):
        runtime.pack_values(input_storage, off,
                            (HOVER_ACTION, HOVER_ACTION, HOVER_ACTION,
                             HOVER_ACTION))

    input_feeder = make_storage_feeder(input_storage)

    mahony = MahonyFilter()
    uart_bridge = machine.UART(UART_BRIDGE_PORT, UART_BRIDGE_BAUD, timeout=0, timeout_char=0)
    frame_tx = bytearray(13)
    setpoint_rx = SetpointReceiver()

    last_t = time.ticks_us()
    next_deadline = time.ticks_add(last_t, TICK_US)
    tick = 0
    substep = 0
    accel_sum_x = 0.0
    accel_sum_y = 0.0
    accel_sum_z = 0.0
    action_sum_0 = 0.0
    action_sum_1 = 0.0
    action_sum_2 = 0.0
    action_sum_3 = 0.0

    mode = "int8" if runtime.quantized else "fp32"
    print("starting %dHz %s attitude-setpoint policy loop" %
          (CONTROL_HZ, mode))
    print("training control history=%dHz substeps=%d" %
          (TRAINING_CONTROL_HZ, CONTROL_SUBSTEPS))
    print("fallback setpoint roll=%.3f pitch=%.3f yaw_rate=%.3f thrust_g=%.3f" %
          (SETPOINT_ROLL_RAD, SETPOINT_PITCH_RAD,
           SETPOINT_YAW_RATE_RAD_S, SETPOINT_THRUST_G))
    print("setpoint rx type=0x%02x timeout_ms=%d limits tilt=%.3f yaw=%.3f thrust=%.1f..%.1f" %
          (SETPOINT_FRAME_TYPE_ATTITUDE, SETPOINT_TIMEOUT_US // 1000,
           SETPOINT_MAX_TILT_RAD, SETPOINT_MAX_YAW_RATE_RAD_S,
           SETPOINT_MIN_THRUST_G, SETPOINT_MAX_THRUST_G))
    print("uart bridge frame flags=0x%02x hover_action=%.4f" %
          (FRAME_FLAGS, HOVER_ACTION))

    while True:
        t0 = time.ticks_us()
        setpoint_rx.poll(uart_bridge)

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

        ax_mps2 = ax * MG_TO_MPS2
        ay_mps2 = ay * MG_TO_MPS2
        az_mps2 = az * MG_TO_MPS2
        gx_rad = gx * MDPS_TO_RADPS
        gy_rad = gy * MDPS_TO_RADPS
        gz_rad = gz * MDPS_TO_RADPS
        mahony.update(ax_mps2, ay_mps2, az_mps2, gx_rad, gy_rad, gz_rad, dt)
        world_z = mahony.orientation_world_z()
        t_mahony = time.ticks_us()

        if runtime.accel_history_length > 0:
            accel_sum_x += ax_mps2
            accel_sum_y += ay_mps2
            accel_sum_z += az_mps2

        sp_now = time.ticks_us()
        sp_roll, sp_pitch, sp_yaw_rate, sp_thrust_g = setpoint_rx.current(sp_now)
        sp_age_ms = setpoint_rx.age_ms(sp_now)

        runtime.pack_values(
            input_storage, 0,
            (sp_roll, sp_pitch, sp_yaw_rate, sp_thrust_g,
             world_z[0], world_z[1], world_z[2],
             gx_rad, gy_rad, gz_rad)
        )
        t_state = time.ticks_us()

        y_raw = runtime.model.predict([input_feeder])[0]
        y_f32 = runtime.output_values(y_raw.flatten())
        a0_raw = float(y_f32[0])
        a1_raw = float(y_f32[1])
        a2_raw = float(y_f32[2])
        a3_raw = float(y_f32[3])
        a0 = sanitize_action(a0_raw)
        a1 = sanitize_action(a1_raw)
        a2 = sanitize_action(a2_raw)
        a3 = sanitize_action(a3_raw)
        t_predict = time.ticks_us()

        build_frame_into(frame_tx, a0, a1, a2, a3)
        uart_bridge.write(frame_tx)
        t_tx = time.ticks_us()

        setpoint_rx.poll(uart_bridge)
        t_rx = time.ticks_us()

        action_sum_0 += a0
        action_sum_1 += a1
        action_sum_2 += a2
        action_sum_3 += a3
        substep += 1
        history_published = 0
        substep_print = substep
        if substep >= CONTROL_SUBSTEPS:
            if runtime.accel_history_length > 0:
                push_accel_history(
                    runtime, input_storage, accel_offset_bytes,
                    runtime.accel_history_length,
                    accel_sum_x * INV_CONTROL_SUBSTEPS,
                    accel_sum_y * INV_CONTROL_SUBSTEPS,
                    accel_sum_z * INV_CONTROL_SUBSTEPS
                )
            push_action_history(
                runtime, input_storage, action_offset_bytes,
                runtime.action_history_length,
                action_sum_0 * INV_CONTROL_SUBSTEPS,
                action_sum_1 * INV_CONTROL_SUBSTEPS,
                action_sum_2 * INV_CONTROL_SUBSTEPS,
                action_sum_3 * INV_CONTROL_SUBSTEPS
            )
            accel_sum_x = 0.0
            accel_sum_y = 0.0
            accel_sum_z = 0.0
            action_sum_0 = 0.0
            action_sum_1 = 0.0
            action_sum_2 = 0.0
            action_sum_3 = 0.0
            substep = 0
            history_published = 1
            substep_print = CONTROL_SUBSTEPS
        t_actions = time.ticks_us()

        elapsed_us = time.ticks_diff(t_actions, t0)
        imu_us = time.ticks_diff(t_imu, t0)
        mahony_us = time.ticks_diff(t_mahony, t_imu)
        state_us = time.ticks_diff(t_state, t_mahony)
        predict_us = time.ticks_diff(t_predict, t_state)
        tx_us = time.ticks_diff(t_tx, t_predict)
        rx_us = time.ticks_diff(t_rx, t_tx)
        actions_us = time.ticks_diff(t_actions, t_rx)
        deadline_delay_us = time.ticks_diff(next_deadline, time.ticks_us())

        if tick % DIAG_PRINT_EVERY == 0:
            print("us=%5d imu=%4d mah=%4d st=%4d inf=%4d tx=%3d rx=%3d act=%3d "
                  "delay=%d sub=%d/%d hist=%d a=%+.2f,%+.2f,%+.2f,%+.2f wz=%+.2f,%+.2f,%+.2f "
                  "rate=%+.2f,%+.2f,%+.2f "
                  "acc=%+.2f,%+.2f,%+.2f "
                  "sp=%+.3f,%+.3f,%+.3f,%.3f sp_age=%d sp_seq=%d sp_ok=%d sp_crc=%d sp_unk=%d" %
                  (elapsed_us, imu_us, mahony_us, state_us, predict_us,
                   tx_us, rx_us, actions_us, deadline_delay_us,
                   substep_print, CONTROL_SUBSTEPS, history_published,
                   a0, a1, a2, a3,
                   world_z[0], world_z[1], world_z[2],
                   gx_rad, gy_rad, gz_rad,
                   ax_mps2, ay_mps2, az_mps2,
                   sp_roll, sp_pitch, sp_yaw_rate, sp_thrust_g,
                   sp_age_ms, setpoint_rx.seq, setpoint_rx.frames_ok,
                   setpoint_rx.frames_bad_crc, setpoint_rx.frames_unknown))

        tick += 1
        delay = time.ticks_diff(next_deadline, time.ticks_us())
        if delay > 0:
            time.sleep_us(delay)
        next_deadline = time.ticks_add(next_deadline, TICK_US)


if __name__ == "__main__":
    run()
