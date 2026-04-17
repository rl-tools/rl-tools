import gc
import os
import time
import ml
from ulab import numpy as np

MODEL_PATH = None  # auto-detected below (prefers Vela-compiled, i.e. non-".int8.tflite")
INPUT_PATH = None
OUTPUT_PATH = None
SAMPLE_INDEX = 0
TIMING_ITERS = 100


def pick_first(files, predicate, label):
    matches = [f for f in files if predicate(f)]
    if not matches:
        raise RuntimeError("no %s found; files=%r" % (label, files))
    return sorted(matches, key=len)[0]


def autodetect_paths():
    files = os.listdir(".")
    model = pick_first(
        files,
        lambda f: f.endswith(".tflite") and not f.endswith(".int8.tflite"),
        ".tflite (non-int8, i.e. Vela-compiled)",
    )
    inp = pick_first(files, lambda f: f.endswith(".example_input.bin"), ".example_input.bin")
    out = pick_first(files, lambda f: f.endswith(".example_output.bin"), ".example_output.bin")
    return model, inp, out


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

MODEL_PATH, INPUT_PATH, OUTPUT_PATH = autodetect_paths()
print("MODEL_PATH :", repr(MODEL_PATH), "size:", os.stat(MODEL_PATH)[6])
print("INPUT_PATH :", repr(INPUT_PATH), "size:", os.stat(INPUT_PATH)[6])
print("OUTPUT_PATH:", repr(OUTPUT_PATH), "size:", os.stat(OUTPUT_PATH)[6])

model = ml.Model(MODEL_PATH)
print_mem("after ml.Model()")
print(model)

in_shape = model.input_shape[0]
out_shape = model.output_shape[0]
in_numel = shape_numel(in_shape)
out_numel = shape_numel(out_shape)
in_dtype = model.input_dtype[0]
out_dtype = model.output_dtype[0]
in_scale = float(model.input_scale[0])
in_zp = float(model.input_zero_point[0])
out_scale = float(model.output_scale[0])
out_zp = float(model.output_zero_point[0])

print("input_shape :", in_shape, "dtype:", in_dtype, "scale:", in_scale, "zp:", in_zp)
print("output_shape:", out_shape, "dtype:", out_dtype, "scale:", out_scale, "zp:", out_zp)

x_f32 = load_sample_float32(INPUT_PATH, SAMPLE_INDEX, in_numel)
y_ref = load_sample_float32(OUTPUT_PATH, SAMPLE_INDEX, out_numel)

x_in = x_f32.reshape(in_shape)

y_raw = model.predict([x_in])[0]  # warm-up (first call is slower — arena touches, caches)
y_f32 = y_raw.flatten()

diff = y_f32 - y_ref
err = max(float(np.max(diff)), -float(np.min(diff)))
print("batch_size:", int(in_shape[0]))
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
mn, med, mean, mx = time_n(lambda: model.predict([x_in]), TIMING_ITERS)
print("min/median/mean/max us: %d / %d / %.1f / %d   fps(mean): %.1f" % (mn, med, mean, mx, 1e6 / mean))

# Pre-quantize once; feed via callable so predict() just memcpys the bytes into the
# tensor buffer (skips py_ml_process_input's per-element float->int8 loop).
x_q_bytes = bytes(np.array(np.clip((x_f32 / in_scale) + in_zp, -128, 127), dtype=np.int8))
assert len(x_q_bytes) == in_numel, (len(x_q_bytes), in_numel)


def feed_prequantized(buf, shape, dtype):
    buf[:] = x_q_bytes


model.predict([feed_prequantized])  # warm-up with callable path
print("--- timing %d iters: pre-quantized via callable (pure NPU + in-graph CPU ops) ---" % TIMING_ITERS)
mn, med, mean, mx = time_n(lambda: model.predict([feed_prequantized]), TIMING_ITERS)
print("min/median/mean/max us: %d / %d / %.1f / %d   fps(mean): %.1f" % (mn, med, mean, mx, 1e6 / mean))
