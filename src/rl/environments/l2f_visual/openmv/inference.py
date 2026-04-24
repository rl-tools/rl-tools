import gc
import os
import time
import json
import ml
import machine
import micropython
from ulab import numpy as np

USE_NEURAL_POLICY = False  # False -> classical PD attitude controller (skips NN/camera/state pipeline)

MODEL_PATH = None  # auto-detected below (prefers Vela-compiled, i.e. non-".int8.tflite")
INPUT_PATHS = None
OUTPUT_PATH = None
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

if USE_NEURAL_POLICY:
    MODEL_PATH, INPUT_PATHS, OUTPUT_PATH, META_PATH = autodetect_paths()
    print("MODEL_PATH :", repr(MODEL_PATH), "size:", os.stat(MODEL_PATH)[6])
    for i, p in enumerate(INPUT_PATHS):
        print("INPUT_%d    :" % i, repr(p), "size:", os.stat(p)[6])
    print("OUTPUT_PATH:", repr(OUTPUT_PATH), "size:", os.stat(OUTPUT_PATH)[6])
    print("META_PATH  :", repr(META_PATH))
    with open(META_PATH) as _f:
        META = json.load(_f)

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

    # Auto-detect how many samples the companion bins hold (all bins must agree).
    _output_size = os.stat(OUTPUT_PATH)[6]
    if _output_size == 0 or _output_size % (out_numel * 4) != 0:
        raise RuntimeError("output bin size %d is not a positive multiple of %d bytes/sample"
                           % (_output_size, out_numel * 4))
    N_CHECK = _output_size // (out_numel * 4)
    for _i, _p in enumerate(INPUT_PATHS):
        _expected = in_numels[_i] * 4 * N_CHECK
        _actual = os.stat(_p)[6]
        if _actual != _expected:
            raise RuntimeError("input %d bin size %d != expected %d (= %d samples of %d floats)"
                               % (_i, _actual, _expected, N_CHECK, in_numels[_i]))

    # Strict wiring check setup: compare against the int8 model's expected output saved by
    # the converter (when present in the meta JSON). Any meaningful delta here
    # indicates input ordering / dtype / packing mismatch — quantization noise has
    # already been subtracted on both sides since both produce identical int8 ops.
    # Two thresholds:
    #   TIGHT_TOL ≈ 2 LSBs — Vela should ideally hit this (pure NPU graph). Won't
    #     hit it if any op falls back to CPU (e.g. dynamic-shape Flatten); the
    #     CPU↔NPU re-quantization handoffs inject ~10-30 LSBs of drift.
    #   FAIL_TOL  ≈ 100 LSBs (bounded by 0.5 absolute) — a real wiring bug
    #     (wrong feeder slot, byte misorder, dtype mismatch) blows past this
    #     by a wide margin. Anything below FAIL_TOL but above TIGHT_TOL is most
    #     likely Vela compilation drift, not a deployment bug.
    INT8_OUT_PATH = None
    if "int8_output" in META and META["int8_output"].get("path"):
        candidate = META["int8_output"]["path"]
        try:
            os.stat(candidate)
            INT8_OUT_PATH = candidate
        except OSError:
            print("meta references int8_output %r but file is absent — skipping strict check"
                  % candidate)
    TIGHT_TOL = max(2 * out_scale, 1e-5)
    FAIL_TOL = min(100 * out_scale, 0.5)

    print("batch_size:", int(in_shapes[0][0]))
    print("checking %d samples from companion bins" % N_CHECK)

    err_float_max = 0.0
    err_int8_max = 0.0
    worst_float_idx = -1
    worst_int8_idx = -1
    x_f32_list = None
    x_in_list = None
    for sample_idx in range(N_CHECK):
        x_f32_list = [
            load_sample_float32(INPUT_PATHS[i], sample_idx, in_numels[i]) for i in range(num_inputs)
        ]
        y_ref = load_sample_float32(OUTPUT_PATH, sample_idx, out_numel)
        x_in_list = [x_f32_list[i].reshape(in_shapes[i]) for i in range(num_inputs)]

        y_raw = model.predict(x_in_list)[0]  # warm-up on first iter (arena touches, caches)
        y_f32 = y_raw.flatten()

        diff = y_f32 - y_ref
        err = max(float(np.max(diff)), -float(np.min(diff)))
        if err > err_float_max:
            err_float_max = err
            worst_float_idx = sample_idx

        if INT8_OUT_PATH is not None:
            y_ref_int8 = load_sample_float32(INT8_OUT_PATH, sample_idx, out_numel)
            diff_int8 = y_f32 - y_ref_int8
            err_int8 = max(float(np.max(diff_int8)), -float(np.min(diff_int8)))
            if err_int8 > err_int8_max:
                err_int8_max = err_int8
                worst_int8_idx = sample_idx
            if err_int8 > FAIL_TOL:
                raise RuntimeError(
                    "sample %d: deployed int8 output differs from converter's int8 reference by %g "
                    "(> fail-threshold %g). Almost certainly a wiring bug: input ordering, "
                    "dtype, or byte packing." % (sample_idx, err_int8, FAIL_TOL))
            print("[%3d] float_err=%.6g  int8_err=%.6g" % (sample_idx, err, err_int8))
        else:
            print("[%3d] float_err=%.6g" % (sample_idx, err))

    print("over %d samples: float max_abs_err=%.6g (sample %d, incl. quantization noise)" %
          (N_CHECK, err_float_max, worst_float_idx))
    if INT8_OUT_PATH is not None:
        print("over %d samples: int8 max_abs_err=%.6g (sample %d, tight=%g, fail=%g)" %
              (N_CHECK, err_int8_max, worst_int8_idx, TIGHT_TOL, FAIL_TOL))
        if err_int8_max > TIGHT_TOL:
            print("WIRING OK (Vela drift): %g (in %g..%g) — likely from Vela's "
                  "CPU↔NPU op-placement boundaries, not a wiring issue."
                  % (err_int8_max, TIGHT_TOL, FAIL_TOL))
        else:
            print("WIRING OK (bit-tight): %g <= %g" % (err_int8_max, TIGHT_TOL))
    else:
        print("no int8_output in meta JSON — strict wiring check skipped")


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

TICK_US = 10_000                  # 100 Hz period
DIAG_PRINT_EVERY = 10

UART_BRIDGE_PORT = 4
UART_BRIDGE_BAUD = 115200
FRAME_SYNC_BYTE = 0x80
RX_LINE_MAX = 256

# Classical PD attitude controller (target: identity orientation, zero angular velocity).
# Gains are in normalized action units; one unit of tau_* translates to a +/- per-motor
# offset around hover and therefore to ~arm*dF/da N*m of physical torque. With
# J ~ 1e-5 kg*m^2 and per-motor thrust authority ~0.05 N/unit, Kp=0.3 lands near
# omega_n ~ 10 rad/s and Kd=0.05 near zeta ~ 0.7 -- a gentle starting point. Tune on bench.
CLASSIC_KP_ATT = 0.30
CLASSIC_KD_RATE = 0.05
CLASSIC_HOVER_THROTTLE = 0.45    # BENCH TEST: center actions around 0; keep CF disarmed and
                                # watch `a=...` in the diag print while tilting by hand to
                                # verify axis signs. Raise to ~0.45 once the signs look right.


def classic_attitude_action(mahony, ang_vel):
    # e_R driven by world +Z expressed in body frame (3rd row of R(q));
    # cross with desired body_up (0,0,1) gives the body-frame error axis (vy, -vx, 0).
    # e_R_z = 0 -> no yaw target, only yaw-rate damping.
    # Mix matches l2f crazyflie.h rotor positions and torque signs:
    #   M0(+x,-y, tz=-1)  M1(-x,-y, tz=+1)  M2(-x,+y, tz=-1)  M3(+x,+y, tz=+1)
    q0, q1, q2, q3 = mahony.q0, mahony.q1, mahony.q2, mahony.q3
    vx = 2.0 * (q1 * q3 - q0 * q2)
    vy = 2.0 * (q2 * q3 + q0 * q1)
    tau_x = -CLASSIC_KP_ATT * vy - CLASSIC_KD_RATE * ang_vel[0]
    tau_y =  CLASSIC_KP_ATT * vx - CLASSIC_KD_RATE * ang_vel[1]
    tau_z =                       -CLASSIC_KD_RATE * ang_vel[2]
    T = CLASSIC_HOVER_THROTTLE
    a0 = T - tau_x - tau_y - tau_z
    a1 = T - tau_x + tau_y + tau_z
    a2 = T + tau_x + tau_y - tau_z
    a3 = T + tau_x - tau_y + tau_z
    return a0, a1, a2, a3


if USE_NEURAL_POLICY:
    import image
    import csi

    # ---- Constants mirrored from imitation_cuda.cu ----
    FRAME_STACK_N = 10
    FRAME_STACK_STRIDE = 10           # 100 Hz / 20 = 200 ms spacing between stacked frames
    FRAME_STACK_HISTORY_LENGTH = FRAME_STACK_STRIDE * (FRAME_STACK_N - 1) + 1  # 81
    ACTION_HISTORY_LENGTH = 64
    ACTION_DIM = 4
    IMG_H, IMG_W, IMG_C = 50, 80, 3
    N_IMAGE_INPUTS = FRAME_STACK_N + 1
    N_INPUTS = N_IMAGE_INPUTS + 1
    TARGET_KERAS_INDEX = FRAME_STACK_N
    STATE_KERAS_INDEX = N_IMAGE_INPUTS
    STATE_DIM = 3 + 3 + ACTION_HISTORY_LENGTH * ACTION_DIM  # 262

    assert num_inputs == N_INPUTS, (num_inputs, N_INPUTS)
    # Build tflite_input_index -> keras_input_index mapping from the meta file.
    # keras_input_index 0..(FRAME_STACK_N-1) = frame slots (newest first),
    # keras_input_index FRAME_STACK_N = target frame, keras_input_index N_IMAGE_INPUTS = state.
    tflite_to_keras = [None] * num_inputs
    for _entry in META["inputs"]:
        tflite_to_keras[int(_entry["tflite_input_index"])] = int(_entry["keras_input_index"])
    for _i, _k in enumerate(tflite_to_keras):
        if _k is None:
            raise RuntimeError("meta JSON missing tflite input %d" % _i)
    print("tflite -> keras input mapping:", tflite_to_keras)

    # Per-input size sanity checks.
    for _ti in range(num_inputs):
        _ki = tflite_to_keras[_ti]
        if _ki == STATE_KERAS_INDEX:
            assert in_numels[_ti] == STATE_DIM, (in_numels[_ti], STATE_DIM, _ti, _ki)
        else:
            assert in_numels[_ti] == IMG_H * IMG_W * IMG_C, (in_numels[_ti], IMG_H * IMG_W * IMG_C, _ti, _ki)

    # ---- Camera ----
    # Use the sensor's QVGA 320x200 mode. Aspect 1.6:1 matches IMG_W:IMG_H (80:50)
    # exactly, so the GPU bilinear scaler can resize into the model input without crop.
    csi0 = csi.CSI()
    csi0.reset()
    csi0.pixformat(csi.RGB565)
    csi0.framesize(csi.QVGA)
    _sensor_w, _sensor_h = csi0.width(), csi0.height()
    csi0.framerate(100)
    # csi0.auto_exposure(False, exposure_us=4000)
    for _ in range(10):
        csi0.snapshot()
    assert _sensor_w * IMG_H == _sensor_h * IMG_W, (
        "sensor aspect %dx%d does not match target %dx%d" % (_sensor_w, _sensor_h, IMG_W, IMG_H)
    )
    FRAME_DRAW_HINT = image.BILINEAR | image.SCALE_ASPECT_IGNORE

    # ---- Quantization ----
    # Locate the state's tflite slot and a representative image slot. All 6 image
    # inputs were quantized from the same source distribution, so they share scale/zp.
    _state_tflite_idx = next(i for i in range(num_inputs) if tflite_to_keras[i] == STATE_KERAS_INDEX)
    _img_tflite_idx   = next(i for i in range(num_inputs) if tflite_to_keras[i] != STATE_KERAS_INDEX)
    _img_scale   = in_scales[_img_tflite_idx]
    _img_zp      = in_zps[_img_tflite_idx]
    _state_scale = in_scales[_state_tflite_idx]
    _state_zp    = in_zps[_state_tflite_idx]
    # Verify all image inputs really do share scale/zp (sanity check).
    for _ti in range(num_inputs):
        if tflite_to_keras[_ti] != STATE_KERAS_INDEX:
            if abs(in_scales[_ti] - _img_scale) > 1e-9 or abs(in_zps[_ti] - _img_zp) > 1e-6:
                print("WARN: image input %d has scale/zp (%g/%g) differing from %g/%g"
                      % (_ti, in_scales[_ti], in_zps[_ti], _img_scale, _img_zp))

    def _q_byte(v, scale, zp):
        q = int(round(v / scale + zp))
        if q < -128: q = -128
        elif q > 127: q = 127
        return q & 0xFF

    def _dtype_char(dtype):
        return chr(dtype) if isinstance(dtype, int) else dtype

    _image_lut = bytearray(256)
    for p in range(256):
        _image_lut[p] = _q_byte(p / 255.0, _img_scale, _img_zp)
    _img_dtype_char = _dtype_char(in_dtypes[_img_tflite_idx])

    def _probe_direct_image_export(img_dtype_char):
        if img_dtype_char not in ("b", "B"):
            return False
        probe_colors = (
            (0, 0, 0),
            (255, 255, 255),
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (17, 93, 211),
            (123, 45, 67),
            (250, 130, 10),
        )
        probe = image.Image(len(probe_colors), 1, image.RGB565)
        for x, rgb in enumerate(probe_colors):
            probe.set_pixel(x, 0, rgb)
        probe_u8 = bytearray(len(probe_colors) * 3)
        probe_q = bytearray(len(probe_colors) * 3)
        probe.to_ndarray(dtype="B", buffer=probe_u8)
        probe.to_ndarray(dtype=img_dtype_char, buffer=probe_q)
        for i in range(len(probe_u8)):
            if probe_q[i] != _image_lut[probe_u8[i]]:
                return False
        return True

    # ---- Buffers ----
    # Circular history of quantized frames, one per loop tick. At each tick the frame
    # stack is gathered from stride-spaced offsets from the write pointer — the same
    # sliding-window scheme used at training time (imitation_cuda.cu:1810 write, :1821 gather).
    frame_history_q = [bytearray(IMG_H * IMG_W * IMG_C) for _ in range(FRAME_STACK_HISTORY_LENGTH)]
    target_q        = bytearray(IMG_H * IMG_W * IMG_C)
    scaled_frame_rgb565 = image.Image(IMG_W, IMG_H, image.RGB565)
    _img_direct_dtype = _img_dtype_char if _probe_direct_image_export(_img_dtype_char) else None
    scaled_frame_u8 = None if _img_direct_dtype is not None else bytearray(IMG_H * IMG_W * IMG_C)
    if _img_direct_dtype is not None:
        print("image path: draw_image(BILINEAR) + to_ndarray(%r) direct" % _img_direct_dtype)
    else:
        print("image path: draw_image(BILINEAR) + to_ndarray('B') + LUT quantize")

    # State vector (int8 bit pattern stored as uint8)
    state_int8 = bytearray(STATE_DIM)

    # Action history ring: pre-quantized int8 bytes (clipped to [-1,1] before quantize on store).
    action_ring_q = bytearray(ACTION_HISTORY_LENGTH * ACTION_DIM)
    _zero_action_byte = _q_byte(0.0, _state_scale, _state_zp)
    for _i in range(len(action_ring_q)):
        action_ring_q[_i] = _zero_action_byte

    # ---- Helpers ----
    # Per-keras-input source bytearray references. The model's leading Concat lets us
    # feed each frame slot as its own contiguous tensor — no NHWC interleave needed.
    # Indices [0:FRAME_STACK_N] are dynamic frame-history slots (refreshed each tick),
    # index TARGET_KERAS_INDEX is the target frame, index STATE_KERAS_INDEX is the state.
    sources_by_keras = [None] * N_INPUTS
    sources_by_keras[TARGET_KERAS_INDEX] = target_q
    sources_by_keras[STATE_KERAS_INDEX] = state_int8
    for _k in range(FRAME_STACK_N):
        sources_by_keras[_k] = frame_history_q[0]  # placeholder until first tick

    def resize_quantize_frame(src_img, dst_q):
        scaled_frame_rgb565.draw_image(src_img, 0, 0, hint=FRAME_DRAW_HINT)
        if _img_direct_dtype is not None:
            scaled_frame_rgb565.to_ndarray(dtype=_img_direct_dtype, buffer=dst_q)
            return
        scaled_frame_rgb565.to_ndarray(dtype="B", buffer=scaled_frame_u8)
        lut = _image_lut
        src = scaled_frame_u8
        for i in range(len(dst_q)):
            dst_q[i] = lut[src[i]]

    @micropython.viper
    def _viper_copy_action_history_newest_first(
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

    def build_state_int8(body_z, ang_vel):
        # Layout: [orientation_body_z(3), angular_velocity(3), action_history newest-first(256)]
        state_int8[0] = _q_byte(body_z[0], _state_scale, _state_zp)
        state_int8[1] = _q_byte(body_z[1], _state_scale, _state_zp)
        state_int8[2] = _q_byte(body_z[2], _state_scale, _state_zp)
        state_int8[3] = _q_byte(ang_vel[0], _state_scale, _state_zp)
        state_int8[4] = _q_byte(ang_vel[1], _state_scale, _state_zp)
        state_int8[5] = _q_byte(ang_vel[2], _state_scale, _state_zp)
        _viper_copy_action_history_newest_first(
            state_int8, 6, action_ring_q, action_write_ptr, ACTION_HISTORY_LENGTH, ACTION_DIM
        )

    def _quantize_action_clip(v):
        if v < -1.0: v = -1.0
        elif v > 1.0: v = 1.0
        return _q_byte(v, _state_scale, _state_zp)

    def make_live_feeder(keras_idx):
        def feeder(buf, shape, dtype):
            buf[:] = sources_by_keras[keras_idx]
        return feeder

    feeders_live = [make_live_feeder(tflite_to_keras[_ti]) for _ti in range(num_inputs)]


# ---- Offboard UART frame (mirror of crazyflie-firmware uart1_bridge.c) ----
# Frame: 0x80 sync byte (MSB=1) + 12 data bytes (MSB=0) encoding a 10-byte payload
# MSB-first at 7 bits per byte. Payload = 4x u16 BE motor PWM + CRC-16-CCITT BE
# over those 8 bytes (init 0xFFFF, poly 0x1021).
def _crc16_ccitt(buf, n):
    crc = 0xFFFF
    for i in range(n):
        crc ^= buf[i] << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc

def _pack7(raw, raw_len, out, out_offset):
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
        w += 1

_raw_payload = bytearray(10)

def _build_frame_into(frame13, a0, a1, a2, a3):
    for idx, a in ((0, a0), (2, a1), (4, a2), (6, a3)):
        if a < -1.0: a = -1.0
        elif a > 1.0: a = 1.0
        pwm = int((a + 1.0) * 32767.5 + 0.5)
        if pwm < 0: pwm = 0
        elif pwm > 0xFFFF: pwm = 0xFFFF
        _raw_payload[idx] = (pwm >> 8) & 0xFF
        _raw_payload[idx + 1] = pwm & 0xFF
    crc = _crc16_ccitt(_raw_payload, 8)
    _raw_payload[8] = (crc >> 8) & 0xFF
    _raw_payload[9] = crc & 0xFF
    _pack7(_raw_payload, 10, frame13, 1)

# ---- Main loop ----
mahony = MahonyFilter()
last_t = time.ticks_us()
next_deadline = time.ticks_add(last_t, TICK_US)
tick = 0
reset_button = machine.Pin('SW', machine.Pin.IN, machine.Pin.PULL_UP)
reset_button_last = reset_button.value()

if USE_NEURAL_POLICY:
    target_captured = False
    history_write_ptr = 0
    action_write_ptr = 0

uart_bridge = machine.UART(UART_BRIDGE_PORT, UART_BRIDGE_BAUD, timeout=0, timeout_char=0)
frame_tx = bytearray(13)
frame_tx[0] = FRAME_SYNC_BYTE
rx_line_buf = bytearray()

if USE_NEURAL_POLICY:
    print("starting 100Hz policy inference loop (NN)")
else:
    print("starting 100Hz policy inference loop (classical PD)")

while True:
    t0 = time.ticks_us()

    reset_button_cur = reset_button.value()
    if reset_button_last == 1 and reset_button_cur == 0:
        if USE_NEURAL_POLICY:
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
        dt = 0.01

    gx_rad = gx * MDPS_TO_RADPS
    gy_rad = gy * MDPS_TO_RADPS
    gz_rad = gz * MDPS_TO_RADPS
    mahony.update(
        ax * MG_TO_MPS2, ay * MG_TO_MPS2, az * MG_TO_MPS2,
        gx_rad, gy_rad, gz_rad, dt,
    )
    t_mahony = time.ticks_us()

    body_z = mahony.orientation_body_z()
    ang_vel = mahony.angular_velocity_corrected(gx_rad, gy_rad, gz_rad)

    if USE_NEURAL_POLICY:
        img = csi0.snapshot()
        assert img is not None
        assert img.width() == _sensor_w
        assert img.height() == _sensor_h
        t_snapshot = time.ticks_us()
        resize_quantize_frame(img, frame_history_q[history_write_ptr])
        t_downsample = time.ticks_us()
        if not target_captured:
            target_q[:] = frame_history_q[history_write_ptr]
            for i in range(FRAME_STACK_HISTORY_LENGTH):
                if i != history_write_ptr:
                    frame_history_q[i][:] = frame_history_q[history_write_ptr]
            target_captured = True
        sources_by_keras[0] = frame_history_q[history_write_ptr]
        for _f in range(1, FRAME_STACK_N):
            _slot = (history_write_ptr - _f * FRAME_STACK_STRIDE) % FRAME_STACK_HISTORY_LENGTH
            sources_by_keras[_f] = frame_history_q[_slot]
        t_frames = time.ticks_us()

        build_state_int8(body_z, ang_vel)
        t_state = time.ticks_us()

        y_raw = model.predict(feeders_live)[0]
        y_f32 = y_raw.flatten()
        a0_raw = float(y_f32[0])
        a1_raw = float(y_f32[1])
        a2_raw = float(y_f32[2])
        a3_raw = float(y_f32[3])
        t_predict = time.ticks_us()
    else:
        a0_raw, a1_raw, a2_raw, a3_raw = classic_attitude_action(mahony, ang_vel)
        t_snapshot = t_mahony
        t_downsample = t_mahony
        t_frames = t_mahony
        t_state = t_mahony
        t_predict = t_mahony

    _build_frame_into(frame_tx, a0_raw, a1_raw, a2_raw, a3_raw)
    uart_bridge.write(frame_tx)
    t_tx = time.ticks_us()

    n_avail = uart_bridge.any()
    if n_avail:
        chunk = uart_bridge.read(n_avail)
        if chunk:
            rx_line_buf.extend(chunk)
            while True:
                nl = rx_line_buf.find(b'\n')
                if nl < 0:
                    break
                line = bytes(rx_line_buf[:nl]).rstrip(b'\r')
                rx_line_buf = rx_line_buf[nl + 1:]
                if b"[u1br] arm rising edge" in line:
                    if USE_NEURAL_POLICY:
                        target_captured = False
                        print("target reset (arm rising edge)")
                try:
                    print("[cf]", line.decode('utf-8'))
                except UnicodeError:
                    print("[cf-bin]", line)
            if len(rx_line_buf) > RX_LINE_MAX:
                print("[cf-overflow]", bytes(rx_line_buf))
                rx_line_buf = bytearray()
    t_rx = time.ticks_us()

    if USE_NEURAL_POLICY:
        slot_off = action_write_ptr * ACTION_DIM
        action_ring_q[slot_off]     = _quantize_action_clip(a0_raw)
        action_ring_q[slot_off + 1] = _quantize_action_clip(a1_raw)
        action_ring_q[slot_off + 2] = _quantize_action_clip(a2_raw)
        action_ring_q[slot_off + 3] = _quantize_action_clip(a3_raw)
        action_write_ptr = (action_write_ptr + 1) % ACTION_HISTORY_LENGTH
    t_actions = time.ticks_us()

    elapsed_us = time.ticks_diff(t_actions, t0)
    if tick % DIAG_PRINT_EVERY == 0:
        imu_us = time.ticks_diff(t_imu, t0)
        mahony_us = time.ticks_diff(t_mahony, t_imu)
        snapshot_us = time.ticks_diff(t_snapshot, t_mahony)
        downsample_us = time.ticks_diff(t_downsample, t_snapshot)
        frames_us = time.ticks_diff(t_frames, t_downsample)
        state_us = time.ticks_diff(t_state, t_frames)
        predict_us = time.ticks_diff(t_predict, t_state)
        tx_us = time.ticks_diff(t_tx, t_predict)
        rx_us = time.ticks_diff(t_rx, t_tx)
        actions_us = time.ticks_diff(t_actions, t_rx)
        a0 = -1.0 if a0_raw < -1.0 else (1.0 if a0_raw > 1.0 else a0_raw)
        a1 = -1.0 if a1_raw < -1.0 else (1.0 if a1_raw > 1.0 else a1_raw)
        a2 = -1.0 if a2_raw < -1.0 else (1.0 if a2_raw > 1.0 else a2_raw)
        a3 = -1.0 if a3_raw < -1.0 else (1.0 if a3_raw > 1.0 else a3_raw)
        print("us=%5d imu=%4d mah=%4d cam=%5d ds=%4d frm=%4d st=%4d inf=%4d tx=%4d rx=%4d act=%4d "
              "a=%+5.2f,%+5.2f,%+5.2f,%+5.2f bz=%+5.2f,%+5.2f,%+5.2f av=%+6.2f,%+6.2f,%+6.2f" %
              (elapsed_us, imu_us, mahony_us, snapshot_us, downsample_us, frames_us, state_us, predict_us,
               tx_us, rx_us, actions_us,
               a0, a1, a2, a3, body_z[0], body_z[1], body_z[2],
               ang_vel[0], ang_vel[1], ang_vel[2]))

    if USE_NEURAL_POLICY:
        history_write_ptr = (history_write_ptr + 1) % FRAME_STACK_HISTORY_LENGTH
    tick += 1
    delay = time.ticks_diff(next_deadline, time.ticks_us())
    if delay > 0:
        time.sleep_us(delay)
    next_deadline = time.ticks_add(next_deadline, TICK_US)
