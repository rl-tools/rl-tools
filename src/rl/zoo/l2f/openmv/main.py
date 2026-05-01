import gc
import json
import math
import os
import socket
import struct
import time

import imu
import machine
import ml
import micropython
from ulab import numpy as np


AP_SSID = "l2f-setpoint"
AP_PASS = "attitude123"
AP_COUNTRY = "US"
AP_CHANNEL = 6
UDP_PORT = 5005

TICK_US = 10_000
DIAG_PRINT_EVERY = 25
FAILSAFE_TIMEOUT_US = 250_000

MAX_TILT_RAD = 0.5235987755982988
MAX_YAW_RATE = 2.0
THRUST_MIN_G = 0.4
THRUST_MAX_G = 1.4
FAILSAFE_THRUST_G = 0.4

UART_BRIDGE_PORT = 4
UART_BRIDGE_BAUD = 115200
FRAME_START_MASK = 0x80
FRAME_FLAGS = 0x00
FRAME_START_BYTE = FRAME_START_MASK | FRAME_FLAGS
RX_LINE_MAX = 256

MG_TO_MPS2 = 9.80665e-3
MDPS_TO_RADPS = math.pi / (180.0 * 1000.0)

ASP_FORMAT = "<4sIIffff"
ASP_SIZE = 28
ASP_MAGIC = b"ASP2"


def print_mem(label):
    gc.collect()
    print("%-24s free=%d alloc=%d" % (label, gc.mem_free(), gc.mem_alloc()))


def wifi_ap():
    import network

    try:
        network.country(AP_COUNTRY)
    except Exception as e:
        print("country() not supported:", e)
    try:
        network.WLAN(network.STA_IF).active(False)
    except Exception:
        pass

    ap = network.WLAN(network.AP_IF)

    def try_cfg(**aliases):
        for k, v in aliases.items():
            try:
                ap.config(**{k: v})
                return True
            except (ValueError, OSError, TypeError):
                continue
        return False

    try_cfg(ssid=AP_SSID, essid=AP_SSID)
    if AP_PASS:
        try_cfg(key=AP_PASS, password=AP_PASS)
        sec = getattr(ap, "WPA_WPA2",
                      getattr(ap, "SEC_WPA_WPA2",
                              getattr(network, "AUTH_WPA2_PSK", 3)))
        try_cfg(security=sec, authmode=sec)
    else:
        sec = getattr(ap, "OPEN",
                      getattr(ap, "SEC_OPEN",
                              getattr(network, "AUTH_OPEN", 0)))
        try_cfg(security=sec, authmode=sec)
    try_cfg(channel=AP_CHANNEL)

    ap.active(True)
    for _ in range(50):
        if ap.active():
            break
        time.sleep_ms(100)
    ip = ap.ifconfig()[0]
    print("AP up: ssid=%s ip=%s port=%d ifconfig=%s" %
          (AP_SSID, ip, UDP_PORT, ap.ifconfig()))
    return ap, ip


def udp_socket():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    except Exception:
        pass
    s.bind(("0.0.0.0", UDP_PORT))
    s.setblocking(False)
    print("UDP listening on 0.0.0.0:%d" % UDP_PORT)
    return s


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
    files = os.listdir(".")
    vela = [f for f in files if f.endswith(".int8_vela.tflite") or f.endswith("_vela.tflite")]
    if vela:
        model = sorted(vela, key=len)[0]
    else:
        model = pick_first(
            files,
            lambda f: f.endswith(".tflite") and not f.endswith(".int8.tflite"),
            ".tflite (prefer Vela-compiled)",
        )
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
        if d < 0:
            d = -d
        if d > err:
            err = d
    return err


def dtype_char(dtype):
    return chr(dtype) if isinstance(dtype, int) else dtype


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

        self.num_inputs = len(self.model.input_shape)
        if self.num_inputs != 1 or len(self.input_paths) != 1:
            raise RuntimeError("attitude-setpoint policy expects one input, model=%d bins=%d" %
                               (self.num_inputs, len(self.input_paths)))

        self.input_shape = self.model.input_shape[0]
        self.input_numel = shape_numel(self.input_shape)
        self.input_dtype = dtype_char(self.model.input_dtype[0])
        self.input_scale = float(self.model.input_scale[0])
        self.input_zp = float(self.model.input_zero_point[0])
        self.output_shape = self.model.output_shape[0]
        self.output_numel = shape_numel(self.output_shape)
        self.output_scale = float(self.model.output_scale[0])
        self.output_zp = float(self.model.output_zero_point[0])

        print("input shape:", self.input_shape, "dtype:", self.input_dtype,
              "scale:", self.input_scale, "zp:", self.input_zp)
        print("output shape:", self.output_shape, "scale:", self.output_scale,
              "zp:", self.output_zp)

        if self.input_numel < 13 or (self.input_numel - 13) % 4 != 0:
            raise RuntimeError("input dim %d is incompatible with AttitudeSetpoint obs" %
                               self.input_numel)
        self.action_history_length = (self.input_numel - 13) // 4
        print("action_history_length:", self.action_history_length)
        if self.input_dtype not in ("b", "B"):
            raise RuntimeError("live callable path expects int8/uint8 input, got %r" %
                               (self.input_dtype,))

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

        int8_out_path = None
        if "int8_output" in self.meta and self.meta["int8_output"].get("path"):
            candidate = self.meta["int8_output"]["path"]
            try:
                os.stat(candidate)
                int8_out_path = candidate
            except OSError:
                print("int8 reference %r absent; strict check skipped" % candidate)

        fail_tol = min(100 * self.output_scale, 0.5)
        err_float_max = 0.0
        err_int8_max = 0.0
        x_in = None
        for sample_idx in range(n_check):
            x = load_sample_float32(self.input_paths[0], sample_idx, self.input_numel)
            y_ref = load_sample_float32(self.output_path, sample_idx, self.output_numel)
            x_in = x.reshape(self.input_shape)
            y_raw = self.model.predict([x_in])[0]
            y_f32 = y_raw.flatten()
            err = max_abs_diff(y_f32, y_ref)
            if err > err_float_max:
                err_float_max = err
            if int8_out_path is not None:
                y_ref_int8 = load_sample_float32(int8_out_path, sample_idx, self.output_numel)
                err_int8 = max_abs_diff(y_f32, y_ref_int8)
                if err_int8 > err_int8_max:
                    err_int8_max = err_int8
                if err_int8 > fail_tol:
                    raise RuntimeError("sample %d int8 wiring err %g > %g" %
                                       (sample_idx, err_int8, fail_tol))
                print("[%3d] float_err=%.6g int8_err=%.6g" %
                      (sample_idx, err, err_int8))
            else:
                print("[%3d] float_err=%.6g" % (sample_idx, err))

        print("self-check: float max_abs_err=%.6g over %d samples" %
              (err_float_max, n_check))
        if int8_out_path is not None:
            print("self-check: int8 max_abs_err=%.6g fail_tol=%g" %
                  (err_int8_max, fail_tol))
        self.model.predict([x_in])


def finite_reasonable(v):
    return v == v and -1.0e6 < v < 1.0e6


def clamp(v, lo, hi):
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def parse_attitude_setpoint_packet(data, setpoint):
    if len(data) != ASP_SIZE:
        return None
    try:
        magic, seq, armed, roll, pitch, yaw_rate, thrust_g = struct.unpack(ASP_FORMAT, data)
    except Exception:
        return None
    if magic != ASP_MAGIC:
        return None
    if not (finite_reasonable(roll) and finite_reasonable(pitch) and
            finite_reasonable(yaw_rate) and finite_reasonable(thrust_g)):
        return None
    setpoint[0] = clamp(roll, -MAX_TILT_RAD, MAX_TILT_RAD)
    setpoint[1] = clamp(pitch, -MAX_TILT_RAD, MAX_TILT_RAD)
    setpoint[2] = clamp(yaw_rate, -MAX_YAW_RATE, MAX_YAW_RATE)
    setpoint[3] = clamp(thrust_g, THRUST_MIN_G, THRUST_MAX_G)
    return seq, armed != 0


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

            self.bx += self.ki * ex * dt
            self.by += self.ki * ey * dt
            self.bz += self.ki * ez * dt
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

    def angular_velocity_corrected(self, gx, gy, gz):
        return gx - self.bx, gy - self.by, gz - self.bz


def q_byte(v, scale, zp, dtype):
    q = int(round(v / scale + zp))
    if dtype == "B":
        q = clamp(q, 0, 255)
        return q
    q = clamp(q, -128, 127)
    return q & 0xFF


@micropython.viper
def copy_action_history_newest_first(
    dst: ptr8, dst_offset: int, src: ptr8, write_ptr: int, n_slots: int, n_dim: int
):
    cur = write_ptr - 1
    if cur < 0:
        cur += n_slots
    i = 0
    while i < n_slots:
        s = cur * n_dim
        d = dst_offset + i * n_dim
        dst[d]     = src[s]
        dst[d + 1] = src[s + 1]
        dst[d + 2] = src[s + 2]
        dst[d + 3] = src[s + 3]
        cur -= 1
        if cur < 0:
            cur += n_slots
        i += 1


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


def build_frame_into(frame13, a0, a1, a2, a3):
    start_byte = FRAME_START_BYTE
    frame13[0] = start_byte
    for idx, a in ((0, a0), (2, a1), (4, a2), (6, a3)):
        a = clamp(a, -1.0, 1.0)
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


def run():
    print_mem("boot")
    print("cwd:", os.getcwd())
    print("listdir(cwd):", os.listdir("."))
    ap, ip = wifi_ap()
    udp = udp_socket()
    runtime = ModelRuntime()
    runtime.self_check()

    input_q = bytearray(runtime.input_numel)
    action_history_q = bytearray(runtime.action_history_length * 4)
    zero_action = q_byte(0.0, runtime.input_scale, runtime.input_zp, runtime.input_dtype)
    for i in range(len(action_history_q)):
        action_history_q[i] = zero_action
    action_write_ptr = 0

    def feed_input(buf, shape, dtype):
        buf[:] = input_q

    feeders = [feed_input]

    setpoint = [0.0, 0.0, 0.0, FAILSAFE_THRUST_G]
    active_setpoint = [0.0, 0.0, 0.0, FAILSAFE_THRUST_G]
    last_packet_us = time.ticks_add(time.ticks_us(), -FAILSAFE_TIMEOUT_US - 1)
    last_seq = -1
    udp_armed = False
    packet_count = 0
    bad_packet_count = 0

    mahony = MahonyFilter()
    uart_bridge = machine.UART(UART_BRIDGE_PORT, UART_BRIDGE_BAUD, timeout=0, timeout_char=0)
    frame_tx = bytearray(13)
    rx_line_buf = bytearray()

    last_t = time.ticks_us()
    next_deadline = time.ticks_add(last_t, TICK_US)
    tick = 0
    print("starting 100Hz attitude-setpoint policy loop")
    print("connect host to SSID=%s, send ASP2 UDP to %s:%d" % (AP_SSID, ip, UDP_PORT))

    while True:
        t0 = time.ticks_us()

        while True:
            try:
                data, _addr = udp.recvfrom(64)
            except OSError:
                break
            parsed = parse_attitude_setpoint_packet(data, setpoint)
            if parsed is None:
                bad_packet_count += 1
            else:
                seq, armed = parsed
                last_seq = seq
                udp_armed = armed
                packet_count += 1
                last_packet_us = t0
        t_udp = time.ticks_us()

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
            dt = 0.01

        ax_mps2 = ax * MG_TO_MPS2
        ay_mps2 = ay * MG_TO_MPS2
        az_mps2 = az * MG_TO_MPS2
        gx_rad = gx * MDPS_TO_RADPS
        gy_rad = gy * MDPS_TO_RADPS
        gz_rad = gz * MDPS_TO_RADPS
        mahony.update(ax_mps2, ay_mps2, az_mps2, gx_rad, gy_rad, gz_rad, dt)
        world_z = mahony.orientation_world_z()
        ang_vel = mahony.angular_velocity_corrected(gx_rad, gy_rad, gz_rad)
        t_mahony = time.ticks_us()

        packet_age_us = time.ticks_diff(t0, last_packet_us)
        stale = packet_age_us > FAILSAFE_TIMEOUT_US
        tx_enabled = (not stale) and udp_armed
        if runtime.action_history_length > 0 and not tx_enabled:
            for i in range(len(action_history_q)):
                action_history_q[i] = zero_action
            action_write_ptr = 0
        a0_raw = 0.0
        a1_raw = 0.0
        a2_raw = 0.0
        a3_raw = 0.0
        if tx_enabled:
            active_setpoint[0] = setpoint[0]
            active_setpoint[1] = setpoint[1]
            active_setpoint[2] = setpoint[2]
            active_setpoint[3] = setpoint[3]

            values = (
                active_setpoint[0], active_setpoint[1], active_setpoint[2], active_setpoint[3],
                world_z[0], world_z[1], world_z[2],
                ang_vel[0], ang_vel[1], ang_vel[2],
                ax_mps2, ay_mps2, az_mps2,
            )
            for i in range(13):
                input_q[i] = q_byte(values[i], runtime.input_scale, runtime.input_zp, runtime.input_dtype)
            if runtime.action_history_length > 0:
                copy_action_history_newest_first(
                    input_q, 13, action_history_q, action_write_ptr, runtime.action_history_length, 4
                )
            t_state = time.ticks_us()

            y_raw = runtime.model.predict(feeders)[0]
            y_f32 = y_raw.flatten()
            a0_raw = float(y_f32[0])
            a1_raw = float(y_f32[1])
            a2_raw = float(y_f32[2])
            a3_raw = float(y_f32[3])
            t_predict = time.ticks_us()

            build_frame_into(frame_tx, a0_raw, a1_raw, a2_raw, a3_raw)
            uart_bridge.write(frame_tx)
        else:
            t_state = time.ticks_us()
            t_predict = t_state
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
                    try:
                        print("[cf]", line.decode("utf-8"))
                    except UnicodeError:
                        print("[cf-bin]", line)
                if len(rx_line_buf) > RX_LINE_MAX:
                    print("[cf-overflow]", bytes(rx_line_buf))
                    rx_line_buf = bytearray()
        t_rx = time.ticks_us()

        if tx_enabled and runtime.action_history_length > 0:
            slot_off = action_write_ptr * 4
            action_history_q[slot_off]     = q_byte(clamp(a0_raw, -1.0, 1.0), runtime.input_scale, runtime.input_zp, runtime.input_dtype)
            action_history_q[slot_off + 1] = q_byte(clamp(a1_raw, -1.0, 1.0), runtime.input_scale, runtime.input_zp, runtime.input_dtype)
            action_history_q[slot_off + 2] = q_byte(clamp(a2_raw, -1.0, 1.0), runtime.input_scale, runtime.input_zp, runtime.input_dtype)
            action_history_q[slot_off + 3] = q_byte(clamp(a3_raw, -1.0, 1.0), runtime.input_scale, runtime.input_zp, runtime.input_dtype)
            action_write_ptr = (action_write_ptr + 1) % runtime.action_history_length
        t_actions = time.ticks_us()

        if tick % DIAG_PRINT_EVERY == 0:
            elapsed_us = time.ticks_diff(t_actions, t0)
            udp_us = time.ticks_diff(t_udp, t0)
            imu_us = time.ticks_diff(t_imu, t_udp)
            mahony_us = time.ticks_diff(t_mahony, t_imu)
            state_us = time.ticks_diff(t_state, t_mahony)
            predict_us = time.ticks_diff(t_predict, t_state)
            tx_us = time.ticks_diff(t_tx, t_predict)
            rx_us = time.ticks_diff(t_rx, t_tx)
            actions_us = time.ticks_diff(t_actions, t_rx)
            print("us=%5d udp=%3d imu=%4d mah=%4d st=%4d inf=%4d tx=%3d rx=%3d act=%3d "
                  "seq=%d age=%d stale=%d armed=%d sent=%d bad=%d sp=%+.2f,%+.2f,%+.2f,%.2f "
                  "a=%+.2f,%+.2f,%+.2f,%+.2f wz=%+.2f,%+.2f,%+.2f" %
                  (elapsed_us, udp_us, imu_us, mahony_us, state_us, predict_us,
                   tx_us, rx_us, actions_us, last_seq, packet_age_us, 1 if stale else 0,
                   1 if udp_armed else 0, 1 if tx_enabled else 0, bad_packet_count,
                   active_setpoint[0], active_setpoint[1],
                   active_setpoint[2], active_setpoint[3],
                   clamp(a0_raw, -1.0, 1.0), clamp(a1_raw, -1.0, 1.0),
                   clamp(a2_raw, -1.0, 1.0), clamp(a3_raw, -1.0, 1.0),
                   world_z[0], world_z[1], world_z[2]))

        tick += 1
        delay = time.ticks_diff(next_deadline, time.ticks_us())
        if delay > 0:
            time.sleep_us(delay)
        next_deadline = time.ticks_add(next_deadline, TICK_US)


if __name__ == "__main__":
    run()
