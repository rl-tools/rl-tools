"""Convert an RLtools HDF5 checkpoint to a TFLite flatbuffer.

Parses the layer tree in the HDF5 file, rebuilds the architecture as a Keras
functional model with the original weights copied in, converts that to TFLite,
and verifies both models against the `example/inputs/*`/`example/outputs/0` pair
stored in the checkpoint.

For parallel top-level models, the Keras model is multi-input — one
`tf.keras.Input` per branch with that branch's feature shape, matching the
struct-of-arrays layout of `example/inputs/{0,1,...}`. This gives the TFLite
int8 converter a separate quantization scale per input tensor (critical when
one branch carries image pixels in [0,1] and another carries state values
with a wider dynamic range).

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


def branch_input_shape(branch_group):
    csv = attr(branch_group, "input_shape")
    if csv is None:
        raise ValueError(f"branch {branch_group.name} missing input_shape attribute")
    return tuple(int(x) for x in csv.split(","))


def num_branches(h5_parallel):
    return int(attr(h5_parallel, "num_branches"))


def branch_flat_dim(branch_group):
    shape = branch_input_shape(branch_group)
    flat = 1
    for d in shape[2:]:  # drop leading STEPS, BATCH dims
        flat *= d
    return flat


def branch_feature_shape(branch_group):
    shape = branch_input_shape(branch_group)
    return tuple(shape[2:])  # drop leading STEPS, BATCH dims


def build_parallel(h5_parallel, tensor):
    n = num_branches(h5_parallel)
    dims = [branch_flat_dim(h5_parallel[f"branch_{i}"]) for i in range(n)]
    total = sum(dims)
    assert tensor.shape[-1] == total, (
        f"parallel input dim {tensor.shape[-1]} != sum of branch dims {total}"
    )

    branch_outputs = []
    offset = 0
    for i in range(n):
        branch_group = h5_parallel[f"branch_{i}"]
        d = dims[i]
        start, end = offset, offset + d
        offset = end
        branch_input = tf.keras.layers.Lambda(
            lambda x, s=start, e=end: x[..., s:e]
        )(tensor)

        feature_shape = branch_feature_shape(branch_group)
        if len(feature_shape) > 1:
            branch_input = tf.keras.layers.Reshape(feature_shape)(branch_input)

        btype = attr(branch_group, "type")
        if btype == "sequential":
            branch_out = build_sequential(branch_group, branch_input)
        elif btype == "parallel":
            branch_out = build_parallel(branch_group, branch_input)
        else:
            raise NotImplementedError(f"parallel branch type {btype}")
        branch_outputs.append(branch_out)

    concat = tf.keras.layers.Concatenate(axis=-1)(branch_outputs)

    has_head = int(attr(h5_parallel, "has_head")) != 0
    if not has_head:
        return concat
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


def example_input_tensors(h5_root):
    inputs_group = h5_root["example/inputs"]
    keys = sorted(inputs_group.keys(), key=int)
    return [inputs_group[k][:].astype(np.float32) for k in keys]


def example_output_tensor(h5_root):
    return h5_root["example/outputs/0"][:].astype(np.float32)


def build_branch_output(branch_group, branch_input):
    btype = attr(branch_group, "type")
    if btype == "sequential":
        return build_sequential(branch_group, branch_input)
    if btype == "parallel":
        feature_shape = tuple(int(d) for d in branch_input.shape[1:])
        flat = (
            tf.keras.layers.Flatten()(branch_input)
            if len(feature_shape) > 1
            else branch_input
        )
        return build_parallel(branch_group, flat)
    raise NotImplementedError(f"branch type {btype}")


def build_head_output(head_group, tensor):
    htype = attr(head_group, "type")
    if htype == "sequential":
        return build_sequential(head_group, tensor)
    if htype == "parallel":
        return build_parallel(head_group, tensor)
    raise NotImplementedError(f"head type {htype}")


def build_model(h5_root):
    model_group = find_model_group(h5_root)
    mtype = attr(model_group, "type")
    if mtype == "parallel":
        n = num_branches(model_group)
        branch_inputs = []
        branch_outputs = []
        for i in range(n):
            branch_group = model_group[f"branch_{i}"]
            feature_shape = branch_feature_shape(branch_group)
            # Pad with a leading zero so SavedModel's alphabetical signature
            # sort preserves declaration order (in_00 < in_01 < ... < in_09 <
            # in_10 < ...). Without this, a run with ≥10 inputs would see
            # in_10 ordered before in_2.
            inp = tf.keras.Input(shape=feature_shape, dtype=tf.float32, name=f"in_{i:02d}")
            branch_inputs.append(inp)
            branch_outputs.append(build_branch_output(branch_group, inp))

        concat = (
            tf.keras.layers.Concatenate(axis=-1)(branch_outputs)
            if n > 1
            else branch_outputs[0]
        )

        if int(attr(model_group, "has_head")) != 0:
            outputs = build_head_output(model_group["head"], concat)
        else:
            outputs = concat
        return tf.keras.Model(inputs=branch_inputs, outputs=outputs)

    if mtype == "sequential":
        example_in_0 = h5_root["example/inputs/0"]
        feature_shape = tuple(int(d) for d in example_in_0.shape[2:])
        inp = tf.keras.Input(shape=feature_shape, dtype=tf.float32, name="in_00")
        outputs = build_sequential(model_group, inp)
        return tf.keras.Model(inputs=inp, outputs=outputs)

    raise NotImplementedError(f"top-level type {mtype}")


def model_input_shapes(model):
    raw = model.input if isinstance(model.input, list) else [model.input]
    return [tuple(int(d) for d in t.shape[1:]) for t in raw]


def wrap_split_image_input(model, n_split, channels_per):
    """Replace the model's first input (assumed shape (H, W, C_total)) with
    `n_split` separate inputs of shape (H, W, channels_per). They are joined
    by a Concat along the channel axis, then zero-padded to C_total channels
    (so the original first conv still sees the expected input dimensionality).

    The rest of the inputs are kept as-is. This is intended for OpenMV /
    Ethos-U deployment where avoiding a CPU-side NHWC interleave matters
    more than runtime simplicity.
    """
    original_inputs = list(model.inputs) if isinstance(model.inputs, (list, tuple)) else [model.inputs]
    if not original_inputs:
        raise RuntimeError("model has no inputs")
    img_input = original_inputs[0]
    other_inputs = original_inputs[1:]
    img_shape = tuple(int(d) for d in img_input.shape[1:])
    if len(img_shape) != 3:
        raise ValueError(f"--split-image-input expects first input rank 3 (H,W,C), got {img_shape}")
    H, W, C_total = img_shape
    n_logical = n_split * channels_per
    n_pad = C_total - n_logical
    if n_pad < 0:
        raise ValueError(
            f"--split-image-input N={n_split} × channels-per={channels_per} = {n_logical} "
            f"exceeds first-input channel count C_total={C_total}"
        )

    new_img_inputs = [
        tf.keras.Input(shape=(H, W, channels_per), dtype=img_input.dtype, name=f"in_{i:02d}")
        for i in range(n_split)
    ]
    new_other_inputs = [
        tf.keras.Input(
            shape=tuple(int(d) for d in t.shape[1:]),
            dtype=t.dtype,
            name=f"in_{n_split + j:02d}",
        )
        for j, t in enumerate(other_inputs)
    ]

    if n_split > 1:
        concat = tf.keras.layers.Concatenate(axis=-1, name="split_image_concat")(new_img_inputs)
    else:
        concat = new_img_inputs[0]
    if n_pad > 0:
        padded = tf.keras.layers.Lambda(
            lambda x, p=n_pad: tf.pad(x, [[0, 0], [0, 0], [0, 0], [0, p]]),
            name="split_image_pad",
        )(concat)
    else:
        padded = concat

    output = model([padded] + new_other_inputs)
    return tf.keras.Model(inputs=new_img_inputs + new_other_inputs, outputs=output)


def split_first_branch_image(per_branch, n_split, channels_per):
    """Split per_branch[0] (..., C_total) into n_split arrays along the last
    axis, dropping any remaining padding channels. The other branches are
    returned unchanged.
    """
    img = per_branch[0]
    if img.ndim < 1:
        raise ValueError("first per-branch array has no axes to split")
    if img.shape[-1] < n_split * channels_per:
        raise ValueError(
            f"first per-branch array has {img.shape[-1]} channels, needs "
            f"≥ {n_split * channels_per}"
        )
    parts = [
        np.ascontiguousarray(img[..., i * channels_per:(i + 1) * channels_per])
        for i in range(n_split)
    ]
    return parts + per_branch[1:]


def collect_quant_layers(group):
    kind = attr(group, "type")
    if kind == "sequential":
        out = []
        for i in ordered_layer_indices(group):
            out.extend(collect_quant_layers(group["layers"][str(i)]))
        return out
    if kind == "parallel":
        out = []
        n = num_branches(group)
        for i in range(n):
            out.extend(collect_quant_layers(group[f"branch_{i}"]))
        has_head = int(attr(group, "has_head")) != 0
        if has_head:
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


def convert_int8(keras_model, per_branch_inputs, io_dtype, float_tflite_bytes):
    order = _tflite_input_order(float_tflite_bytes)
    if sorted(order) != list(range(len(per_branch_inputs))):
        raise RuntimeError(
            f"tflite input order {order} does not map cleanly onto "
            f"{len(per_branch_inputs)} keras inputs"
        )
    reordered = [per_branch_inputs[i] for i in order]
    n_samples = reordered[0].shape[0]

    def rep_gen():
        for i in range(n_samples):
            yield [arr[i : i + 1].astype(np.float32) for arr in reordered]

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


import re as _re


def _keras_index_from_tflite_name(name):
    # TFLite input tensor names look like "serving_default_input_0:0".
    m = _re.search(r"in_(\d+)", name)
    if not m:
        raise RuntimeError(f"cannot recover keras input index from tflite name {name!r}")
    return int(m.group(1))


def _sort_details_by_name(details, expected_count):
    if len(details) != expected_count:
        raise RuntimeError(
            f"tflite has {len(details)} tensors but {expected_count} were expected"
        )
    return sorted(details, key=lambda d: _keras_index_from_tflite_name(d["name"]))


def _tflite_input_order(tflite_bytes):
    # Returns a list such that tflite_input[i] corresponds to keras input
    # index `order[i]`. Used to align a representative dataset with the
    # tflite converter's internal input ordering (which may differ from the
    # Keras-declared order after SavedModel round-tripping).
    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    details = interp.get_input_details()
    return [_keras_index_from_tflite_name(d["name"]) for d in details]


def run_tflite(tflite_bytes, inputs):
    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    in_details = _sort_details_by_name(interp.get_input_details(), len(inputs))
    for detail, arr in zip(in_details, inputs):
        interp.resize_tensor_input(detail["index"], list(arr.shape))
    interp.allocate_tensors()
    out_detail = interp.get_output_details()[0]
    for detail, arr in zip(in_details, inputs):
        interp.set_tensor(detail["index"], arr.astype(np.float32))
    interp.invoke()
    return interp.get_tensor(out_detail["index"])


def run_tflite_int8(tflite_bytes, inputs, return_raw=False):
    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    in_details = _sort_details_by_name(interp.get_input_details(), len(inputs))
    out_detail = interp.get_output_details()[0]

    n_samples = inputs[0].shape[0]
    for arr in inputs:
        if arr.shape[0] != n_samples:
            raise RuntimeError(
                f"inconsistent sample count across inputs: {[a.shape[0] for a in inputs]}"
            )

    out_scale, out_zp = out_detail["quantization"]

    outputs = []
    raw_outputs = []
    for i in range(n_samples):
        for detail, arr in zip(in_details, inputs):
            s = arr[i : i + 1]
            if detail["dtype"] == np.int8:
                in_scale, in_zp = detail["quantization"]
                q = np.round(s / in_scale + in_zp).clip(-128, 127).astype(np.int8)
                interp.set_tensor(detail["index"], q)
            else:
                interp.set_tensor(detail["index"], s.astype(np.float32))
        interp.invoke()
        raw = interp.get_tensor(out_detail["index"])
        if out_detail["dtype"] == np.int8:
            y = (raw.astype(np.float32) - out_zp) * out_scale
            raw_outputs.append(raw.astype(np.int8))
        else:
            y = raw.astype(np.float32)
            raw_outputs.append(None)
        outputs.append(y)
    dequant = np.concatenate(outputs, axis=0)
    if return_raw:
        if any(r is None for r in raw_outputs):
            raw_arr = None
        else:
            raw_arr = np.concatenate(raw_outputs, axis=0)
        return dequant, raw_arr, float(out_scale), int(out_zp), str(out_detail["dtype"].__name__)
    return dequant


def print_summary(lines):
    print()
    print("=" * 72)
    print("SUMMARY (re-printed without intermediate tflite/keras logs)")
    print("=" * 72)
    for line in lines:
        print(line)
    print("=" * 72)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="path to .h5 checkpoint")
    ap.add_argument("-o", "--output", help="output .tflite path (default: input with .tflite suffix)")
    ap.add_argument("--tolerance", type=float, default=1e-4, help="max absolute error for float tflite")
    ap.add_argument("--quantize", choices=["none", "int8"], default="none",
                    help="also emit an int8-quantized tflite and fake-quant .quantized.h5")
    ap.add_argument("--quantize-io", choices=["int8", "float32"], default="int8",
                    help="I/O dtype of the int8 tflite (default int8 for NPU deployment)")
    ap.add_argument("--quantize-tolerance", type=float, default=0.2,
                    help="max absolute error tolerance for int8 tflite vs reference")
    ap.add_argument("--fast-tanh-substitute", choices=["keep", "tanh"], default="tanh",
                    help="for int8 mode only: replace the polynomial FAST_TANH with tf.nn.tanh "
                         "(tanh uses a lookup table under int8 and is well-supported on NPUs; "
                         "the polynomial form triggers a division by zero in the int8 DIV kernel)")
    ap.add_argument("--split-image-input", type=int, default=None, metavar="N",
                    help="experimental: replace the first input (assumed (H,W,C_total)) with "
                         "N inputs of (H,W,channels-per) joined by a leading Concat (and zero-pad "
                         "if N*channels-per < C_total). Used to avoid the CPU-side NHWC interleave "
                         "on OpenMV / Ethos-U deployment.")
    ap.add_argument("--split-image-channels-per", type=int, default=3,
                    help="channels per split input when --split-image-input is set (default 3)")
    args = ap.parse_args()

    out_path = args.output or os.path.splitext(args.input)[0] + ".tflite"

    with h5py.File(args.input, "r") as f:
        model = build_model(f)
        example_inputs = example_input_tensors(f)
        y_ref = example_output_tensor(f)
        ordered_layers = collect_quant_layers(find_model_group(f))
    # Canonical example shape is [T, B, ...features] (T=1 for non-recurrent).
    # Fold T into the batch axis so each per-branch array is (T*B, ...features).
    per_branch = [
        t.reshape((t.shape[0] * t.shape[1],) + t.shape[2:]).astype(np.float32)
        for t in example_inputs
    ]
    n_examples = per_branch[0].shape[0]
    y_ref = y_ref.reshape((n_examples,) + y_ref.shape[2:]).astype(np.float32)

    if args.split_image_input is not None:
        n_split = args.split_image_input
        c_per = args.split_image_channels_per
        print(f"--split-image-input: wrapping model with {n_split} image inputs "
              f"of {c_per} channels each (leading Concat + zero-pad if needed)")
        model = wrap_split_image_input(model, n_split, c_per)
        per_branch = split_first_branch_image(per_branch, n_split, c_per)

    input_shapes = model_input_shapes(model)
    if len(per_branch) != len(input_shapes):
        raise RuntimeError(
            f"model has {len(input_shapes)} inputs but example/inputs has "
            f"{len(per_branch)} tensors"
        )
    for i, (arr, shape) in enumerate(zip(per_branch, input_shapes)):
        if tuple(arr.shape[1:]) != shape:
            raise RuntimeError(
                f"example/inputs/{i} feature shape {arr.shape[1:]} != model "
                f"input_{i} shape {shape}"
            )

    model.summary(line_length=120)

    summary_lines = []
    def report(line):
        print(line)
        summary_lines.append(line)

    def err_stats(y_pred, y_true):
        diff = np.abs(y_pred - y_true)
        return float(np.max(diff)), float(np.mean(diff))

    y_keras = model.predict(per_branch, verbose=0)
    keras_err, keras_mean = err_stats(y_keras, y_ref)
    report(f"Keras vs reference: max_abs_err={keras_err:.6g}  mean_abs_err={keras_mean:.6g}")
    report(f"  ref:   {y_ref.reshape(-1)}")
    report(f"  keras: {y_keras.reshape(-1)}")

    tflite_bytes = convert_float(model)
    with open(out_path, "wb") as f:
        f.write(tflite_bytes)
    print(f"wrote {len(tflite_bytes)} bytes to {out_path}")

    base, _ = os.path.splitext(out_path)
    out_example_path = base + ".example_output.bin"
    meta_path = base + ".example_meta.json"
    per_branch_contig = [np.ascontiguousarray(a, dtype=np.float32) for a in per_branch]
    y_flat = np.ascontiguousarray(y_ref, dtype=np.float32)

    # The tflite converter alphabetizes inputs through SavedModel, so
    # tflite's input index i does not in general equal keras input index i.
    # On-device consumers (OpenMV / Vela-compiled models) read inputs in
    # tflite index order, so name the bin files accordingly:
    #   example_input.<tflite_i>.bin  == data for tflite input tflite_i
    # tflite_order[tflite_i] = keras_i.
    tflite_order = _tflite_input_order(tflite_bytes)
    in_paths = [base + f".example_input.{i}.bin" for i in range(len(per_branch_contig))]
    for tflite_i, path in enumerate(in_paths):
        arr = per_branch_contig[tflite_order[tflite_i]]
        with open(path, "wb") as f:
            f.write(arr.tobytes())
        print(f"wrote {arr.nbytes} bytes to {path}  "
              f"(tflite_input={tflite_i}, keras_input={tflite_order[tflite_i]}, "
              f"shape={arr.shape}, dtype=float32)")
    with open(out_example_path, "wb") as f:
        f.write(y_flat.tobytes())
    print(f"wrote {y_flat.nbytes} bytes to {out_example_path}  (shape={y_flat.shape}, dtype=float32)")

    import json
    meta = {
        "inputs": [
            {
                "path": os.path.basename(in_paths[tflite_i]),
                "tflite_input_index": tflite_i,
                "keras_input_index": tflite_order[tflite_i],
                "shape": list(per_branch_contig[tflite_order[tflite_i]].shape),
                "dtype": "float32",
            }
            for tflite_i in range(len(per_branch_contig))
        ],
        "output": {
            "path": os.path.basename(out_example_path),
            "shape": list(y_flat.shape),
            "dtype": "float32",
        },
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {meta_path}")

    # Reload bins (tflite order) and reshuffle back into keras order for the
    # Keras model validation and the name-matched run_tflite path.
    per_branch_reloaded = [None] * len(per_branch_contig)
    for tflite_i, path in enumerate(in_paths):
        keras_i = tflite_order[tflite_i]
        src = per_branch_contig[keras_i]
        with open(path, "rb") as f:
            reloaded_arr = np.frombuffer(f.read(), dtype=np.float32).reshape(src.shape)
        assert np.array_equal(reloaded_arr, src), f"companion input {path} diverges from source"
        per_branch_reloaded[keras_i] = reloaded_arr
    with open(out_example_path, "rb") as f:
        y_reloaded = np.frombuffer(f.read(), dtype=np.float32).reshape(y_flat.shape)
    assert np.array_equal(y_reloaded, y_flat), "companion output diverges from source"

    with open(out_path, "rb") as f:
        reloaded = f.read()
    y_tflite = run_tflite(reloaded, per_branch_reloaded)
    tflite_err, tflite_mean = err_stats(y_tflite, y_reloaded)
    report(
        f"TFLite (via companion files) vs reference: "
        f"max_abs_err={tflite_err:.6g}  mean_abs_err={tflite_mean:.6g}"
    )
    report(f"  tflite: {y_tflite.reshape(-1)}")

    worst = max(keras_err, tflite_err)
    if worst > args.tolerance:
        report(f"FAIL: max error {worst:.6g} > tolerance {args.tolerance:.6g}")
        print_summary(summary_lines)
        print(f"FAIL: max error {worst:.6g} > tolerance {args.tolerance:.6g}", file=sys.stderr)
        return 1
    report(f"OK: within tolerance {args.tolerance:.6g}")

    if args.quantize == "none":
        print_summary(summary_lines)
        return 0

    print()
    print("=== int8 quantization ===")

    if not ordered_layers:
        print("no quantizable layers (conv2d/dense) in HDF5 — nothing to do",
              file=sys.stderr)
        return 1

    print(f"representative-dataset: {n_examples} samples across {len(per_branch_contig)} inputs")

    if args.fast_tanh_substitute == "tanh":
        print("substituting FAST_TANH → tf.nn.tanh for int8 conversion "
              "(weights in .quantized.h5 will reflect this substitution)")
        saved = ACTIVATIONS["FAST_TANH"]
        ACTIVATIONS["FAST_TANH"] = tf.nn.tanh
        try:
            with h5py.File(args.input, "r") as f:
                int8_src_model = build_model(f)
            if args.split_image_input is not None:
                int8_src_model = wrap_split_image_input(
                    int8_src_model, args.split_image_input, args.split_image_channels_per
                )
        finally:
            ACTIVATIONS["FAST_TANH"] = saved
    else:
        int8_src_model = model

    int8_bytes = convert_int8(int8_src_model, per_branch_contig, args.quantize_io, tflite_bytes)
    int8_path = base + ".int8.tflite"
    with open(int8_path, "wb") as f:
        f.write(int8_bytes)
    print(f"wrote {len(int8_bytes)} bytes to {int8_path}")

    with open(int8_path, "rb") as f:
        int8_reloaded = f.read()
    y_int8, y_int8_raw, out_scale, out_zp, out_dtype_name = run_tflite_int8(
        int8_reloaded, per_branch_reloaded, return_raw=True
    )
    y_int8 = y_int8.reshape(y_flat.shape)
    int8_err, int8_mean = err_stats(y_int8, y_flat)
    report(
        f"TFLite (int8, via companion files) vs reference: "
        f"max_abs_err={int8_err:.6g}  mean_abs_err={int8_mean:.6g}"
    )

    # Save the int8 model's expected output as a companion bin so on-device
    # validation can do a strict wiring check (compare deployed output against
    # what THIS converter produced from the same int8 tflite).
    int8_out_path = base + ".example_int8_output.bin"
    int8_out_flat = np.ascontiguousarray(y_int8, dtype=np.float32)
    with open(int8_out_path, "wb") as f:
        f.write(int8_out_flat.tobytes())
    print(f"wrote {int8_out_flat.nbytes} bytes to {int8_out_path}  "
          f"(shape={int8_out_flat.shape}, dtype=float32)")
    int8_out_raw_path = None
    if y_int8_raw is not None:
        y_int8_raw = y_int8_raw.reshape(y_flat.shape).astype(np.int8)
        int8_out_raw_path = base + ".example_int8_output_raw.bin"
        with open(int8_out_raw_path, "wb") as f:
            f.write(np.ascontiguousarray(y_int8_raw).tobytes())
        print(f"wrote {y_int8_raw.nbytes} bytes to {int8_out_raw_path}  "
              f"(shape={y_int8_raw.shape}, dtype=int8)")

    # Re-write meta JSON with the int8_output block so on-device consumers can
    # discover the strict-comparison artifacts. Existing keys preserved.
    meta["int8_output"] = {
        "path": os.path.basename(int8_out_path),
        "raw_path": (os.path.basename(int8_out_raw_path) if int8_out_raw_path else None),
        "shape": list(int8_out_flat.shape),
        "dtype": "float32",
        "raw_dtype": ("int8" if int8_out_raw_path else None),
        "scale": out_scale,
        "zero_point": out_zp,
        "tflite_output_dtype": out_dtype_name,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"updated {meta_path} with int8_output block")
    report(f"  int8:  {y_int8.reshape(-1)}")

    dequant_by_path = extract_quantized_weights(int8_bytes, ordered_layers)
    quant_h5 = base + ".quantized.h5"
    write_quantized_h5(args.input, quant_h5, dequant_by_path)
    print(f"wrote {quant_h5} (weight fake-quant, activations still float)")

    with h5py.File(quant_h5, "r") as f:
        fq_model = build_model(f)
    if args.split_image_input is not None:
        fq_model = wrap_split_image_input(
            fq_model, args.split_image_input, args.split_image_channels_per
        )
    y_fq = fq_model.predict(per_branch, verbose=0)
    fq_err, fq_mean = err_stats(y_fq, y_flat)
    report(
        f"HDF5 (weight-fake-quant) vs reference: "
        f"max_abs_err={fq_err:.6g}  mean_abs_err={fq_mean:.6g}"
    )
    report(f"  fq:    {y_fq.reshape(-1)}")

    if int8_err > args.quantize_tolerance:
        report(f"FAIL: int8 max error {int8_err:.6g} > tolerance "
               f"{args.quantize_tolerance:.6g}")
        print_summary(summary_lines)
        print(f"FAIL: int8 max error {int8_err:.6g} > tolerance "
              f"{args.quantize_tolerance:.6g}", file=sys.stderr)
        return 0
    report(f"OK (int8): within tolerance {args.quantize_tolerance:.6g}")
    print_summary(summary_lines)
    return 0


if __name__ == "__main__":
    sys.exit(main())
