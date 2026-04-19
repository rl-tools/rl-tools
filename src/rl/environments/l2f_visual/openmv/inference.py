import gc
import os
import time
import ml
from ulab import numpy as np

MODEL_PATH = None  # auto-detected below (prefers Vela-compiled, i.e. non-".int8.tflite")
INPUT_PATHS = None
OUTPUT_PATH = None
SAMPLE_INDEX = 0
TIMING_ITERS = 100


def pick_first(files, predicate, label):
    matches = [f for f in files if predicate(f)]
    if not matches:
        raise RuntimeError("no %s found; files=%r" % (label, files))
    return sorted(matches, key=len)[0]


def input_bin_index(fname):
    # Accepts <base>.example_input.<i>.bin — returns i, or -1 if no match.
    tail = ".bin"
    if not fname.endswith(tail):
        return -1
    stem = fname[: -len(tail)]
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
    model = pick_first(
        files,
        lambda f: f.endswith(".tflite") and not f.endswith(".int8.tflite"),
        ".tflite (non-int8, i.e. Vela-compiled)",
    )
    indexed_inputs = [(input_bin_index(f), f) for f in files]
    indexed_inputs = [(i, f) for i, f in indexed_inputs if i >= 0]
    if not indexed_inputs:
        raise RuntimeError("no .example_input.<i>.bin found; files=%r" % (files,))
    indexed_inputs.sort()
    inputs = [f for _, f in indexed_inputs]
    out = pick_first(files, lambda f: f.endswith(".example_output.bin"), ".example_output.bin")
    return model, inputs, out


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


def print_mem(label):
    gc.collect()
    print("%-24s free=%d alloc=%d" % (label, gc.mem_free(), gc.mem_alloc()))


print_mem("boot")
print("cwd:", os.getcwd())
print("listdir('/'):", os.listdir("/"))
try:
    print("listdir(cwd):", os.listdir("."))
except OSError as e:
    print("listdir(cwd) failed:", e)

MODEL_PATH, INPUT_PATHS, OUTPUT_PATH = autodetect_paths()
print("MODEL_PATH :", repr(MODEL_PATH), "size:", os.stat(MODEL_PATH)[6])
for i, p in enumerate(INPUT_PATHS):
    print("INPUT_%d    :" % i, repr(p), "size:", os.stat(p)[6])
print("OUTPUT_PATH:", repr(OUTPUT_PATH), "size:", os.stat(OUTPUT_PATH)[6])

model = ml.Model(MODEL_PATH)
print_mem("after ml.Model()")
print(model)

num_inputs = len(model.input_shape)
if num_inputs != len(INPUT_PATHS):
    raise RuntimeError("model has %d inputs but found %d input bins" % (num_inputs, len(INPUT_PATHS)))

in_shapes = [model.input_shape[i] for i in range(num_inputs)]
in_numels = [shape_numel(s) for s in in_shapes]
in_dtypes = [model.input_dtype[i] for i in range(num_inputs)]
in_scales = [float(model.input_scale[i]) for i in range(num_inputs)]
in_zps = [float(model.input_zero_point[i]) for i in range(num_inputs)]
out_shape = model.output_shape[0]
out_numel = shape_numel(out_shape)
out_dtype = model.output_dtype[0]
out_scale = float(model.output_scale[0])
out_zp = float(model.output_zero_point[0])

for i in range(num_inputs):
    print("input_%d shape:" % i, in_shapes[i], "dtype:", in_dtypes[i], "scale:", in_scales[i], "zp:", in_zps[i])
print("output_shape:", out_shape, "dtype:", out_dtype, "scale:", out_scale, "zp:", out_zp)

x_f32_list = [
    load_sample_float32(INPUT_PATHS[i], SAMPLE_INDEX, in_numels[i]) for i in range(num_inputs)
]
y_ref = load_sample_float32(OUTPUT_PATH, SAMPLE_INDEX, out_numel)

x_in_list = [x_f32_list[i].reshape(in_shapes[i]) for i in range(num_inputs)]

y_raw = model.predict(x_in_list)[0]  # warm-up (first call is slower — arena touches, caches)
y_f32 = y_raw.flatten()

diff = y_f32 - y_ref
err = max(float(np.max(diff)), -float(np.min(diff)))
print("batch_size:", int(in_shapes[0][0]))
print("output   :", y_f32)
print("reference:", y_ref)
print("max_abs_err:", err)


def time_n(fn, n):
    ts = [0] * n
    for i in range(n):
        t0 = time.ticks_us()
        fn()
        t1 = time.ticks_us()
        ts[i] = time.ticks_diff(t1, t0)
    srt = sorted(ts)
    return srt[0], srt[n // 2], sum(ts) / n, srt[-1]


print("--- timing %d iters: ndarray input (includes per-call float->int8 quant) ---" % TIMING_ITERS)
mn, med, mean, mx = time_n(lambda: model.predict(x_in_list), TIMING_ITERS)
print("min/median/mean/max us: %d / %d / %.1f / %d   fps(mean): %.1f" % (mn, med, mean, mx, 1e6 / mean))

# Pre-quantize once per input; feed via callables so predict() just memcpys the bytes
# into each tensor buffer (skips py_ml_process_input's per-element float->int8 loop).
x_q_bytes_list = [
    bytes(np.array(np.clip((x_f32_list[i] / in_scales[i]) + in_zps[i], -128, 127), dtype=np.int8))
    for i in range(num_inputs)
]
for i in range(num_inputs):
    assert len(x_q_bytes_list[i]) == in_numels[i], (len(x_q_bytes_list[i]), in_numels[i])


def make_feeder(bytes_for_input):
    def feeder(buf, shape, dtype):
        buf[:] = bytes_for_input
    return feeder


feeders = [make_feeder(x_q_bytes_list[i]) for i in range(num_inputs)]

model.predict(feeders)  # warm-up with callable path
print("--- timing %d iters: pre-quantized via callable (pure NPU + in-graph CPU ops) ---" % TIMING_ITERS)
mn, med, mean, mx = time_n(lambda: model.predict(feeders), TIMING_ITERS)
print("min/median/mean/max us: %d / %d / %.1f / %d   fps(mean): %.1f" % (mn, med, mean, mx, 1e6 / mean))





import imu
import time
import math


class MahonyFilter:
    # State: quaternion q (body -> world) and gyro bias b.
    # v = R(q)^T * [0, 0, 1] is the estimated direction of gravity-reaction in the body
    # frame (the direction the accelerometer should read when stationary). The PI controller
    # drives the error e = a_hat x v toward zero while tracking gyro bias.
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
            if self.bx > self.max_bias:  self.bx = self.max_bias
            elif self.bx < -self.max_bias: self.bx = -self.max_bias
            if self.by > self.max_bias:  self.by = self.max_bias
            elif self.by < -self.max_bias: self.by = -self.max_bias
            if self.bz > self.max_bias:  self.bz = self.max_bias
            elif self.bz < -self.max_bias: self.bz = -self.max_bias

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

    def gravity_body(self):
        # World "up" direction expressed in body frame (= accel reading when stationary)
        vx = 2.0 * (self.q1 * self.q3 - self.q0 * self.q2)
        vy = 2.0 * (self.q2 * self.q3 + self.q0 * self.q1)
        vz = self.q0 * self.q0 - self.q1 * self.q1 - self.q2 * self.q2 + self.q3 * self.q3
        # Gravity vector (points down) = -g * "up"
        return -self.g_ref * vx, -self.g_ref * vy, -self.g_ref * vz

    def orientation_body_z(self):
        # Third column of R(q): body +Z axis expressed in world frame.
        # Matches l2f obs::OrientationBodyZ — unit vector, no g scaling.
        x = 2.0 * (self.q1 * self.q3 + self.q0 * self.q2)
        y = 2.0 * (self.q2 * self.q3 - self.q0 * self.q1)
        z = self.q0 * self.q0 - self.q1 * self.q1 - self.q2 * self.q2 + self.q3 * self.q3
        return x, y, z

    def angular_velocity_corrected(self, gx, gy, gz):
        return gx - self.bx, gy - self.by, gz - self.bz


MG_TO_MPS2 = 9.80665e-3
MDPS_TO_RADPS = math.pi / (180.0 * 1000.0)

import sensor

# ---- Constants mirrored from imitation_cuda.cu ----
FRAME_STACK_N = 5
FRAME_STACK_STRIDE = 20           # 100 Hz / 20 = 5 Hz frame capture
ACTION_HISTORY_LENGTH = 64
ACTION_DIM = 4
IMG_H, IMG_W, IMG_C = 64, 64, 3
LOGICAL_IMG_C = FRAME_STACK_N * IMG_C + IMG_C  # 18
COMBINED_IMG_C = 24                            # cuDNN-aligned pad
STATE_DIM = 3 + 3 + ACTION_HISTORY_LENGTH * ACTION_DIM  # 262
TICK_US = 10_000                  # 100 Hz period
IMAGE_INPUT_IDX = 0
STATE_INPUT_IDX = 1
RGB565_BIG_ENDIAN = True          # flip if colors look wrong

assert num_inputs == 2, num_inputs
assert in_numels[IMAGE_INPUT_IDX] == IMG_H * IMG_W * COMBINED_IMG_C, (in_numels[IMAGE_INPUT_IDX], IMG_H * IMG_W * COMBINED_IMG_C)
assert in_numels[STATE_INPUT_IDX] == STATE_DIM, (in_numels[STATE_INPUT_IDX], STATE_DIM)

# ---- Camera ----
sensor.reset()
sensor.set_pixformat(sensor.RGB565)
try:
    sensor.set_framesize(sensor.B64X64)
except Exception:
    sensor.set_framesize(sensor.QQVGA)
    try:
        sensor.set_windowing(((sensor.width() - IMG_W) // 2,
                              (sensor.height() - IMG_H) // 2, IMG_W, IMG_H))
    except Exception:
        pass
sensor.skip_frames(time=500)

# ---- Quantization ----
_img_scale   = in_scales[IMAGE_INPUT_IDX]
_img_zp      = in_zps[IMAGE_INPUT_IDX]
_state_scale = in_scales[STATE_INPUT_IDX]
_state_zp    = in_zps[STATE_INPUT_IDX]

def _q_byte(v, scale, zp):
    q = int(round(v / scale + zp))
    if q < -128: q = -128
    elif q > 127: q = 127
    return q & 0xFF

_image_lut = bytes(_q_byte(p / 255.0, _img_scale, _img_zp) for p in range(256))
PAD_BYTE = _q_byte(0.0, _img_scale, _img_zp)

# ---- Buffers ----
# LUT-quantized frames (int8 bit pattern stored as uint8)
frame_ring_q = [bytearray(IMG_H * IMG_W * IMG_C) for _ in range(FRAME_STACK_N)]
target_q     = bytearray(IMG_H * IMG_W * IMG_C)

# Combined NHWC tensor as (H*W, 24). Pad channels preset once.
combined_arr = np.zeros((IMG_H * IMG_W, COMBINED_IMG_C), dtype=np.uint8)
combined_arr[:, LOGICAL_IMG_C:COMBINED_IMG_C] = PAD_BYTE
cached_image_bytes = bytes(combined_arr)

# State vector (int8 bit pattern stored as uint8)
state_int8 = bytearray(STATE_DIM)

# Action history ring: raw policy outputs (not clipped on store)
action_ring = [[0.0, 0.0, 0.0, 0.0] for _ in range(ACTION_HISTORY_LENGTH)]

# ---- Helpers ----
def decode_rgb565_to_rgb888(raw_bytes):
    # Vectorized RGB565 -> RGB888 decode via ulab.
    raw = np.frombuffer(raw_bytes, dtype=np.uint8).reshape((IMG_H * IMG_W, 2))
    if RGB565_BIG_ENDIAN:
        hi = raw[:, 0]; lo = raw[:, 1]
    else:
        hi = raw[:, 1]; lo = raw[:, 0]
    r5 = hi // 8
    g_hi = hi - r5 * 8
    g_lo = lo // 32
    g6 = g_hi * 8 + g_lo
    b5 = lo - g_lo * 32
    rgb = np.empty((IMG_H * IMG_W, 3), dtype=np.uint8)
    rgb[:, 0] = r5 * 8
    rgb[:, 1] = g6 * 4
    rgb[:, 2] = b5 * 8
    return bytes(rgb)

def recompose_image():
    # Interleave the 5 frames (newest-first) + target into channels 0..17.
    # Pad channels 18..23 are set once at init.
    global cached_image_bytes
    for f in range(FRAME_STACK_N):
        src_idx = (frame_write_ptr - 1 - f) % FRAME_STACK_N
        src = np.frombuffer(frame_ring_q[src_idx], dtype=np.uint8).reshape((IMG_H * IMG_W, IMG_C))
        combined_arr[:, f*IMG_C:(f+1)*IMG_C] = src
    tgt = np.frombuffer(target_q, dtype=np.uint8).reshape((IMG_H * IMG_W, IMG_C))
    combined_arr[:, FRAME_STACK_N*IMG_C:FRAME_STACK_N*IMG_C + IMG_C] = tgt
    cached_image_bytes = bytes(combined_arr)

def build_state_int8(body_z, ang_vel):
    # Layout: [orientation_body_z(3), angular_velocity(3), action_history newest-first(256)]
    state_int8[0] = _q_byte(body_z[0], _state_scale, _state_zp)
    state_int8[1] = _q_byte(body_z[1], _state_scale, _state_zp)
    state_int8[2] = _q_byte(body_z[2], _state_scale, _state_zp)
    state_int8[3] = _q_byte(ang_vel[0], _state_scale, _state_zp)
    state_int8[4] = _q_byte(ang_vel[1], _state_scale, _state_zp)
    state_int8[5] = _q_byte(ang_vel[2], _state_scale, _state_zp)
    idx = 6
    cur = (action_write_ptr - 1) % ACTION_HISTORY_LENGTH
    for _ in range(ACTION_HISTORY_LENGTH):
        slot = action_ring[cur]
        for ai in range(ACTION_DIM):
            v = slot[ai]
            if v < -1.0: v = -1.0
            elif v > 1.0: v = 1.0
            state_int8[idx] = _q_byte(v, _state_scale, _state_zp)
            idx += 1
        cur = (cur - 1) % ACTION_HISTORY_LENGTH

def image_feeder(buf, shape, dtype):
    buf[:] = cached_image_bytes

def state_feeder(buf, shape, dtype):
    buf[:] = state_int8

feeders_live = [image_feeder, state_feeder]

# ---- Main loop ----
mahony = MahonyFilter()
last_t = time.ticks_us()
next_deadline = time.ticks_add(last_t, TICK_US)
target_captured = False
frame_write_ptr = 0
action_write_ptr = 0
tick = 0
print("starting 100Hz policy inference loop")

while True:
    t0 = time.ticks_us()

    ax_orig, ay_orig, az_orig = imu.acceleration_mg()
    ax = -az_orig
    ay =  ay_orig
    az =  ax_orig
    gx_orig, gy_orig, gz_orig = imu.angular_rate_mdps()
    gx = -gz_orig
    gy =  gy_orig
    gz =  gx_orig

    dt = time.ticks_diff(t0, last_t) * 1e-6
    last_t = t0
    if dt <= 0.0 or dt > 0.5:
        dt = 0.01

    gx_rad = gx * MDPS_TO_RADPS
    gy_rad = gy * MDPS_TO_RADPS
    gz_rad = gz * MDPS_TO_RADPS
    mahony.update(
        ax * MG_TO_MPS2, ay * MG_TO_MPS2, az * MG_TO_MPS2,
        gx_rad, gy_rad, gz_rad, dt,
    )

    if tick % FRAME_STACK_STRIDE == 0:
        img = sensor.snapshot()
        rgb888 = decode_rgb565_to_rgb888(img.bytearray())
        frame_ring_q[frame_write_ptr][:] = rgb888.translate(_image_lut)
        if not target_captured:
            target_q[:] = frame_ring_q[frame_write_ptr]
            target_captured = True
            for i in range(FRAME_STACK_N):
                if i != frame_write_ptr:
                    frame_ring_q[i][:] = frame_ring_q[frame_write_ptr]
        frame_write_ptr = (frame_write_ptr + 1) % FRAME_STACK_N
        recompose_image()

    if not target_captured:
        tick += 1
        delay = time.ticks_diff(next_deadline, time.ticks_us())
        if delay > 0:
            time.sleep_us(delay)
        next_deadline = time.ticks_add(next_deadline, TICK_US)
        continue

    body_z = mahony.orientation_body_z()
    ang_vel = mahony.angular_velocity_corrected(gx_rad, gy_rad, gz_rad)
    build_state_int8(body_z, ang_vel)

    y_raw = model.predict(feeders_live)[0]
    yq = y_raw.flatten()
    a0_raw = (float(yq[0]) - out_zp) * out_scale
    a1_raw = (float(yq[1]) - out_zp) * out_scale
    a2_raw = (float(yq[2]) - out_zp) * out_scale
    a3_raw = (float(yq[3]) - out_zp) * out_scale

    slot = action_ring[action_write_ptr]
    slot[0] = a0_raw; slot[1] = a1_raw; slot[2] = a2_raw; slot[3] = a3_raw
    action_write_ptr = (action_write_ptr + 1) % ACTION_HISTORY_LENGTH

    a0 = -1.0 if a0_raw < -1.0 else (1.0 if a0_raw > 1.0 else a0_raw)
    a1 = -1.0 if a1_raw < -1.0 else (1.0 if a1_raw > 1.0 else a1_raw)
    a2 = -1.0 if a2_raw < -1.0 else (1.0 if a2_raw > 1.0 else a2_raw)
    a3 = -1.0 if a3_raw < -1.0 else (1.0 if a3_raw > 1.0 else a3_raw)

    elapsed_us = time.ticks_diff(time.ticks_us(), t0)
    print("us=%5d a=%+5.2f,%+5.2f,%+5.2f,%+5.2f bz=%+5.2f,%+5.2f,%+5.2f av=%+6.2f,%+6.2f,%+6.2f" %
          (elapsed_us, a0, a1, a2, a3, body_z[0], body_z[1], body_z[2],
           ang_vel[0], ang_vel[1], ang_vel[2]))

    tick += 1
    delay = time.ticks_diff(next_deadline, time.ticks_us())
    if delay > 0:
        time.sleep_us(delay)
    next_deadline = time.ticks_add(next_deadline, TICK_US)
