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
