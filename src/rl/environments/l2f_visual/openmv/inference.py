import gc
import os
import time
import ml
from ulab import numpy as np

MODEL_PATH = None  # auto-detected below (prefers Vela-compiled, i.e. non-".int8.tflite")
INPUT_PATH = None
OUTPUT_PATH = None
SAMPLE_INDEX = 0


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

if in_dtype == 'f':
    x_in = x_f32.reshape(in_shape)
else:
    x_q = (x_f32 / in_scale) + in_zp
    x_q = np.array(np.clip(x_q, -128, 127), dtype=np.int8)
    x_in = x_q.reshape(in_shape)

t0 = time.ticks_us()
y_raw = model.predict([x_in])[0]
t1 = time.ticks_us()
print("inference_us:", time.ticks_diff(t1, t0))

y_flat = np.array(y_raw.flatten(), dtype=np.float)
if out_dtype == 'f':
    y_f32 = y_flat
else:
    y_f32 = (y_flat - out_zp) * out_scale

diff = y_f32 - y_ref
err = max(float(np.max(diff)), -float(np.min(diff)))
print("output   :", y_f32)
print("reference:", y_ref)
print("max_abs_err:", err)
