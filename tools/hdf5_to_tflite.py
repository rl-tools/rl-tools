"""Convert an RLtools HDF5 checkpoint to a TFLite flatbuffer.

Parses the layer tree in the HDF5 file, rebuilds the architecture as a Keras
functional model with the original weights copied in, converts that to TFLite,
and verifies both models against the `example/input`/`example/output` pair
stored in the checkpoint.

Supported layer types: parallel, sequential, conv2d, dense, flatten, standardize.
Supported activations: RELU, IDENTITY, FAST_TANH.

With `--quantize int8`, additionally performs full-integer post-training
quantization using a representative dataset and writes:
  - `<base>.int8.tflite`  — int8 weights/activations, int8 IO (NPU deployment).
  - `<base>.quantized.h5` — source HDF5 cloned, with every conv/dense weight and
                            bias dataset overwritten by its per-channel
                            dequantized float32 values. Drop-in replacement for
                            RLtools eval to measure the weight-quantization
                            impact on reward.

Note: activation quantization error is NOT simulated in `.quantized.h5`; that
error only manifests when running `.int8.tflite` directly.

Requires the virtualenv at `.venv` with tensorflow installed.
"""
import argparse
import os
import shutil
import sys

import h5py
import numpy as np
import tensorflow as tf


def fast_tanh(x):
    x = tf.clip_by_value(x, -3.0, 3.0)
    x2 = x * x
    return x * (27.0 + x2) / (27.0 + 9.0 * x2)


ACTIVATIONS = {
    "RELU": tf.nn.relu,
    "IDENTITY": lambda x: x,
    "FAST_TANH": fast_tanh,
}


def attr(group, name, default=None):
    if name in group.attrs:
        v = group.attrs[name]
        return v.decode() if isinstance(v, bytes) else v
    return default


def ordered_layer_indices(group):
    return sorted(int(k) for k in group["layers"].keys())


def sanitize_name(h5_path):
    return h5_path.lstrip("/").replace("/", "_")


def build_layer(h5_layer, tensor):
    kind = attr(h5_layer, "type")
    if kind == "conv2d":
        out_c = int(attr(h5_layer, "output_channels"))
        in_c = int(attr(h5_layer, "input_channels"))
        kh = int(attr(h5_layer, "kernel_height"))
        kw = int(attr(h5_layer, "kernel_width"))
        sh = int(attr(h5_layer, "stride_h"))
        sw = int(attr(h5_layer, "stride_w"))
        ph = int(attr(h5_layer, "padding_h"))
        pw = int(attr(h5_layer, "padding_w"))
        norm = attr(h5_layer, "normalization", "NONE")
        if norm != "NONE":
            raise NotImplementedError(f"conv2d normalization={norm} not supported")
        act_name = attr(h5_layer, "activation_function")
        if act_name not in ACTIVATIONS:
            raise NotImplementedError(f"activation {act_name}")

        w = h5_layer["weights/parameters"][:]
        b = h5_layer["biases/parameters"][:]
        assert w.shape == (out_c, kh, kw, in_c), (
            f"unexpected conv weight shape {w.shape}, expected {(out_c, kh, kw, in_c)}"
        )
        w = np.transpose(w, (1, 2, 3, 0))

        if ph or pw:
            tensor = tf.keras.layers.ZeroPadding2D(padding=(ph, pw))(tensor)
        conv = tf.keras.layers.Conv2D(
            filters=out_c,
            kernel_size=(kh, kw),
            strides=(sh, sw),
            padding="valid",
            activation=None,
            use_bias=True,
            name=sanitize_name(h5_layer.name),
        )
        tensor = conv(tensor)
        conv.set_weights([w, b])
        tensor = tf.keras.layers.Lambda(ACTIVATIONS[act_name])(tensor)
        return tensor

    if kind == "dense":
        act_name = attr(h5_layer, "activation_function")
        if act_name not in ACTIVATIONS:
            raise NotImplementedError(f"activation {act_name}")
        w = h5_layer["weights/parameters"][:]
        b = h5_layer["biases/parameters"][:]
        out_dim, in_dim = w.shape
        w_tf = w.T.astype(np.float32)

        dense = tf.keras.layers.Dense(
            units=out_dim,
            activation=None,
            use_bias=True,
            name=sanitize_name(h5_layer.name),
        )
        tensor = dense(tensor)
        dense.set_weights([w_tf, b.astype(np.float32)])
        tensor = tf.keras.layers.Lambda(ACTIVATIONS[act_name])(tensor)
        return tensor

    if kind == "flatten":
        return tf.keras.layers.Flatten()(tensor)

    if kind == "standardize":
        mean = h5_layer["mean/parameters"][:].astype(np.float32)
        precision = h5_layer["precision/parameters"][:].astype(np.float32)
        return tf.keras.layers.Lambda(
            lambda x, m=mean, p=precision: (x - tf.constant(m)) * tf.constant(p)
        )(tensor)

    raise NotImplementedError(f"unsupported layer type: {kind}")


def build_sequential(h5_group, tensor):
    for i in ordered_layer_indices(h5_group):
        tensor = build_layer(h5_group["layers"][str(i)], tensor)
    return tensor


def parallel_input_dims(h5_parallel):
    dims = []
    i = 0
    while f"input_dim_{i}" in h5_parallel.attrs:
        dims.append(int(h5_parallel.attrs[f"input_dim_{i}"]))
        i += 1
    return dims


def maybe_reshape_for_conv(branch_group, flat_dim, tensor):
    btype = attr(branch_group, "type")
    if btype != "sequential":
        return tensor
    first_idx = ordered_layer_indices(branch_group)[0]
    first = branch_group["layers"][str(first_idx)]
    if attr(first, "type") != "conv2d":
        return tensor
    in_c = int(attr(first, "input_channels"))
    spatial = flat_dim // in_c
    side = int(round(spatial ** 0.5))
    if side * side != spatial or spatial * in_c != flat_dim:
        raise ValueError(
            f"cannot infer square HxW for conv input: flat={flat_dim}, channels={in_c}"
        )
    return tf.keras.layers.Reshape((side, side, in_c))(tensor)


def build_parallel(h5_parallel, tensor):
    dims = parallel_input_dims(h5_parallel)
    total = sum(dims)
    assert tensor.shape[-1] == total, (
        f"parallel input dim {tensor.shape[-1]} != sum of branch dims {total}"
    )

    branch_outputs = []
    offset = 0
    for i, d in enumerate(dims):
        branch_name = f"branch_{i}"
        branch_group = h5_parallel[branch_name]
        start, end = offset, offset + d
        offset = end
        branch_input = tf.keras.layers.Lambda(
            lambda x, s=start, e=end: x[..., s:e]
        )(tensor)

        branch_input = maybe_reshape_for_conv(branch_group, d, branch_input)

        btype = attr(branch_group, "type")
        if btype == "sequential":
            branch_out = build_sequential(branch_group, branch_input)
        elif btype == "parallel":
            branch_out = build_parallel(branch_group, branch_input)
        else:
            raise NotImplementedError(f"parallel branch type {btype}")
        branch_outputs.append(branch_out)

    concat = tf.keras.layers.Concatenate(axis=-1)(branch_outputs)

    head_group = h5_parallel["head"]
    htype = attr(head_group, "type")
    if htype == "sequential":
        return build_sequential(head_group, concat)
    if htype == "parallel":
        return build_parallel(head_group, concat)
    raise NotImplementedError(f"parallel head type {htype}")


def find_model_group(h5_root):
    if "actor" in h5_root:
        return h5_root["actor"]
    top = [k for k in h5_root.keys() if k != "example" and "type" in h5_root[k].attrs]
    assert len(top) == 1, f"cannot locate top-level model group, candidates={top}"
    return h5_root[top[0]]


def build_model(h5_root):
    model_group = find_model_group(h5_root)
    mtype = attr(model_group, "type")
    if mtype == "parallel":
        dims = parallel_input_dims(model_group)
        total_input = sum(dims)
    elif mtype == "sequential":
        example_in = h5_root["example/input"]
        total_input = int(example_in.shape[-1])
    else:
        raise NotImplementedError(f"top-level type {mtype}")

    inputs = tf.keras.Input(shape=(total_input,), dtype=tf.float32)
    if mtype == "parallel":
        outputs = build_parallel(model_group, inputs)
    else:
        outputs = build_sequential(model_group, inputs)

    return tf.keras.Model(inputs=inputs, outputs=outputs)


def collect_quant_layers(group):
    kind = attr(group, "type")
    if kind == "sequential":
        out = []
        for i in ordered_layer_indices(group):
            out.extend(collect_quant_layers(group["layers"][str(i)]))
        return out
    if kind == "parallel":
        out = []
        dims = parallel_input_dims(group)
        for i in range(len(dims)):
            out.extend(collect_quant_layers(group[f"branch_{i}"]))
        out.extend(collect_quant_layers(group["head"]))
        return out
    if kind in ("dense", "conv2d"):
        w = group["weights/parameters"]
        b = group["biases/parameters"]
        return [(group.name, kind, tuple(w.shape), tuple(b.shape))]
    return []


def convert_float(keras_model):
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    return converter.convert()


def load_representative_samples(path, expected_dim, max_samples):
    if path.endswith(".npy"):
        arr = np.load(path)
    else:
        raw = np.fromfile(path, dtype=np.float32)
        if expected_dim <= 0 or raw.size % expected_dim != 0:
            raise ValueError(
                f"raw file size {raw.size} not divisible by expected_dim {expected_dim}"
            )
        arr = raw.reshape(-1, expected_dim)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim > 2:
        arr = arr.reshape(-1, arr.shape[-1])
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[-1] != expected_dim:
        raise ValueError(
            f"representative data last-dim {arr.shape[-1]} != model input dim {expected_dim}"
        )
    if max_samples > 0 and arr.shape[0] > max_samples:
        arr = arr[:max_samples]
    return arr


def convert_int8(keras_model, rep_samples, io_dtype):
    def rep_gen():
        for sample in rep_samples:
            yield [sample[np.newaxis].astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    if io_dtype == "int8":
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    return converter.convert()


def extract_quantized_weights(tflite_bytes, ordered_layers):
    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()

    ops = interp._get_ops_details()
    td_by_idx = {td["index"]: td for td in interp.get_tensor_details()}

    quant_op_kinds = ("FULLY_CONNECTED", "CONV_2D")
    quant_ops = [o for o in ops if o["op_name"] in quant_op_kinds]

    if len(quant_ops) != len(ordered_layers):
        raise RuntimeError(
            f"tflite has {len(quant_ops)} {quant_op_kinds} ops, "
            f"expected {len(ordered_layers)} from HDF5"
        )

    result = {}
    for op, layer_info in zip(quant_ops, ordered_layers):
        h5_path, kind, exp_w_shape, exp_b_shape = layer_info
        inputs = list(op["inputs"])
        w_idx = int(inputs[1])
        b_idx = int(inputs[2]) if len(inputs) > 2 and inputs[2] >= 0 else None

        w_int = interp.get_tensor(w_idx)
        actual_w_shape = tuple(int(x) for x in w_int.shape)
        if actual_w_shape != exp_w_shape:
            raise RuntimeError(
                f"weight shape mismatch at {h5_path}: tflite {actual_w_shape} "
                f"vs h5 {exp_w_shape} (op #{op['index']}, kind={op['op_name']})"
            )

        qp = td_by_idx[w_idx]["quantization_parameters"]
        scales = np.asarray(qp["scales"], dtype=np.float32)
        zps = np.asarray(qp["zero_points"], dtype=np.int32)
        qdim = int(qp["quantized_dimension"])
        shape_b = [1] * w_int.ndim
        shape_b[qdim] = -1
        scales_r = scales.reshape(shape_b)
        zps_r = zps.reshape(shape_b).astype(np.float32)
        w_deq = (w_int.astype(np.float32) - zps_r) * scales_r

        b_deq = None
        if b_idx is not None:
            b_int = interp.get_tensor(b_idx)
            b_qp = td_by_idx[b_idx]["quantization_parameters"]
            b_scales = np.asarray(b_qp["scales"], dtype=np.float64)
            b_zps = np.asarray(b_qp["zero_points"], dtype=np.int64)
            if b_scales.size > 0:
                if b_scales.size == 1:
                    b_deq = ((b_int.astype(np.float64) - b_zps) * b_scales).astype(np.float32)
                else:
                    b_deq = ((b_int.astype(np.float64) - b_zps.reshape(-1)) * b_scales.reshape(-1)).astype(np.float32)
            else:
                b_deq = b_int.astype(np.float32)
            actual_b_shape = tuple(int(x) for x in b_deq.shape)
            if actual_b_shape != exp_b_shape:
                raise RuntimeError(
                    f"bias shape mismatch at {h5_path}: tflite {actual_b_shape} "
                    f"vs h5 {exp_b_shape}"
                )

        result[h5_path] = (w_deq, b_deq)

    return result


def write_quantized_h5(src_path, dst_path, dequant_by_path):
    shutil.copyfile(src_path, dst_path)
    with h5py.File(dst_path, "r+") as f:
        for h5_path, (w_deq, b_deq) in dequant_by_path.items():
            w_ds = f[h5_path + "/weights/parameters"]
            w_ds[...] = w_deq
            if b_deq is not None:
                b_ds = f[h5_path + "/biases/parameters"]
                b_ds[...] = b_deq


def run_tflite(tflite_bytes, x):
    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    inp = interp.get_input_details()[0]
    interp.resize_tensor_input(inp["index"], list(x.shape))
    interp.allocate_tensors()
    out = interp.get_output_details()[0]
    interp.set_tensor(inp["index"], x.astype(np.float32))
    interp.invoke()
    return interp.get_tensor(out["index"])


def run_tflite_int8(tflite_bytes, x):
    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    in_detail = interp.get_input_details()[0]
    out_detail = interp.get_output_details()[0]

    in_dim = int(in_detail["shape"][-1])
    flat = np.ascontiguousarray(x, dtype=np.float32).reshape(-1, in_dim)

    in_scale, in_zp = in_detail["quantization"]
    out_scale, out_zp = out_detail["quantization"]

    outputs = []
    for sample in flat:
        s = sample[np.newaxis]
        if in_detail["dtype"] == np.int8:
            q = np.round(s / in_scale + in_zp).clip(-128, 127).astype(np.int8)
            interp.set_tensor(in_detail["index"], q)
        else:
            interp.set_tensor(in_detail["index"], s.astype(np.float32))
        interp.invoke()
        raw = interp.get_tensor(out_detail["index"])
        if out_detail["dtype"] == np.int8:
            y = (raw.astype(np.float32) - out_zp) * out_scale
        else:
            y = raw.astype(np.float32)
        outputs.append(y)
    return np.concatenate(outputs, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="path to .h5 checkpoint")
    ap.add_argument("-o", "--output", help="output .tflite path (default: input with .tflite suffix)")
    ap.add_argument("--tolerance", type=float, default=1e-4, help="max absolute error for float tflite")
    ap.add_argument("--quantize", choices=["none", "int8"], default="none",
                    help="also emit an int8-quantized tflite and fake-quant .quantized.h5")
    ap.add_argument("--quantize-io", choices=["int8", "float32"], default="int8",
                    help="I/O dtype of the int8 tflite (default int8 for NPU deployment)")
    ap.add_argument("--representative-data", default=None,
                    help="path to .npy or raw-float32 .bin of (N, input_dim) for PTQ calibration; "
                         "falls back to example/input with a warning")
    ap.add_argument("--representative-samples", type=int, default=256,
                    help="cap on calibration samples; 0 means use all")
    ap.add_argument("--quantize-tolerance", type=float, default=0.2,
                    help="max absolute error tolerance for int8 tflite vs reference")
    ap.add_argument("--fast-tanh-substitute", choices=["keep", "tanh"], default="tanh",
                    help="for int8 mode only: replace the polynomial FAST_TANH with tf.nn.tanh "
                         "(tanh uses a lookup table under int8 and is well-supported on NPUs; "
                         "the polynomial form triggers a division by zero in the int8 DIV kernel)")
    args = ap.parse_args()

    out_path = args.output or os.path.splitext(args.input)[0] + ".tflite"

    with h5py.File(args.input, "r") as f:
        model = build_model(f)
        x = f["example/input"][:].astype(np.float32)
        y_ref = f["example/output"][:].astype(np.float32)
        ordered_layers = collect_quant_layers(find_model_group(f))

    model.summary(line_length=120)

    y_keras = model.predict(x, verbose=0)
    keras_err = np.max(np.abs(y_keras - y_ref))
    print(f"Keras vs reference: max_abs_err={keras_err:.6g}")
    print(f"  ref:   {y_ref.reshape(-1)}")
    print(f"  keras: {y_keras.reshape(-1)}")

    tflite_bytes = convert_float(model)
    with open(out_path, "wb") as f:
        f.write(tflite_bytes)
    print(f"wrote {len(tflite_bytes)} bytes to {out_path}")

    base, _ = os.path.splitext(out_path)
    in_path = base + ".example_input.bin"
    out_example_path = base + ".example_output.bin"
    x_flat = np.ascontiguousarray(x, dtype=np.float32)
    y_flat = np.ascontiguousarray(y_ref, dtype=np.float32)
    with open(in_path, "wb") as f:
        f.write(x_flat.tobytes())
    with open(out_example_path, "wb") as f:
        f.write(y_flat.tobytes())
    print(f"wrote {x_flat.nbytes} bytes to {in_path}  (shape={x_flat.shape}, dtype=float32)")
    print(f"wrote {y_flat.nbytes} bytes to {out_example_path}  (shape={y_flat.shape}, dtype=float32)")

    with open(in_path, "rb") as f:
        x_reloaded = np.frombuffer(f.read(), dtype=np.float32).reshape(x_flat.shape)
    with open(out_example_path, "rb") as f:
        y_reloaded = np.frombuffer(f.read(), dtype=np.float32).reshape(y_flat.shape)
    assert np.array_equal(x_reloaded, x_flat), "companion input diverges from source"
    assert np.array_equal(y_reloaded, y_flat), "companion output diverges from source"

    with open(out_path, "rb") as f:
        reloaded = f.read()
    y_tflite = run_tflite(reloaded, x_reloaded)
    tflite_err = np.max(np.abs(y_tflite - y_reloaded))
    print(f"TFLite (via companion files) vs reference: max_abs_err={tflite_err:.6g}")
    print(f"  tflite: {y_tflite.reshape(-1)}")

    worst = max(keras_err, tflite_err)
    if worst > args.tolerance:
        print(f"FAIL: max error {worst:.6g} > tolerance {args.tolerance:.6g}", file=sys.stderr)
        return 1
    print(f"OK: within tolerance {args.tolerance:.6g}")

    if args.quantize == "none":
        return 0

    print()
    print("=== int8 quantization ===")

    if not ordered_layers:
        print("no quantizable layers (conv2d/dense) in HDF5 — nothing to do",
              file=sys.stderr)
        return 1

    input_dim = int(model.input.shape[-1])
    if args.representative_data is None:
        print("WARNING: --representative-data not given, using example/input "
              "(usually too small/non-diverse for good PTQ calibration)",
              file=sys.stderr)
        rep_samples = x_flat.reshape(-1, input_dim)
    else:
        rep_samples = load_representative_samples(
            args.representative_data, input_dim, args.representative_samples
        )
    print(f"representative-dataset: {rep_samples.shape[0]} samples of dim {rep_samples.shape[-1]}")

    if args.fast_tanh_substitute == "tanh":
        print("substituting FAST_TANH → tf.nn.tanh for int8 conversion "
              "(weights in .quantized.h5 will reflect this substitution)")
        saved = ACTIVATIONS["FAST_TANH"]
        ACTIVATIONS["FAST_TANH"] = tf.nn.tanh
        try:
            with h5py.File(args.input, "r") as f:
                int8_src_model = build_model(f)
        finally:
            ACTIVATIONS["FAST_TANH"] = saved
    else:
        int8_src_model = model

    int8_bytes = convert_int8(int8_src_model, rep_samples, args.quantize_io)
    int8_path = base + ".int8.tflite"
    with open(int8_path, "wb") as f:
        f.write(int8_bytes)
    print(f"wrote {len(int8_bytes)} bytes to {int8_path}")

    y_int8 = run_tflite_int8(int8_bytes, x_flat)
    y_int8 = y_int8.reshape(y_flat.shape)
    int8_err = np.max(np.abs(y_int8 - y_flat))
    print(f"TFLite (int8) vs reference: max_abs_err={int8_err:.6g}")
    print(f"  int8:  {y_int8.reshape(-1)}")

    dequant_by_path = extract_quantized_weights(int8_bytes, ordered_layers)
    quant_h5 = base + ".quantized.h5"
    write_quantized_h5(args.input, quant_h5, dequant_by_path)
    print(f"wrote {quant_h5} (weight fake-quant, activations still float)")

    with h5py.File(quant_h5, "r") as f:
        fq_model = build_model(f)
    y_fq = fq_model.predict(x, verbose=0)
    fq_err = np.max(np.abs(y_fq - y_flat))
    print(f"HDF5 (weight-fake-quant) vs reference: max_abs_err={fq_err:.6g}")
    print(f"  fq:    {y_fq.reshape(-1)}")

    if int8_err > args.quantize_tolerance:
        print(f"FAIL: int8 max error {int8_err:.6g} > tolerance "
              f"{args.quantize_tolerance:.6g}", file=sys.stderr)
        return 1
    print(f"OK (int8): within tolerance {args.quantize_tolerance:.6g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
