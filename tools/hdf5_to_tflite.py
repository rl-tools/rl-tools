"""Convert an RLtools HDF5 checkpoint to a TFLite flatbuffer.

Parses the layer tree in the HDF5 file, rebuilds the architecture as a Keras
functional model with the original weights copied in, converts that to TFLite,
and verifies both models against the `example/input`/`example/output` pair
stored in the checkpoint.

Supported layer types: parallel, sequential, conv2d, dense, flatten, standardize.
Supported activations: RELU, IDENTITY, FAST_TANH.

Requires the virtualenv at `.venv` with tensorflow installed.
"""
import argparse
import os
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

        dense = tf.keras.layers.Dense(units=out_dim, activation=None, use_bias=True)
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


def build_model(h5_root):
    if "actor" in h5_root:
        model_group = h5_root["actor"]
    else:
        top = [k for k in h5_root.keys() if k != "example" and "type" in h5_root[k].attrs]
        assert len(top) == 1, f"cannot locate top-level model group, candidates={top}"
        model_group = h5_root[top[0]]

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


def convert(keras_model):
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    return converter.convert()


def run_tflite(tflite_bytes, x):
    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    interp.set_tensor(inp["index"], x.astype(np.float32))
    interp.invoke()
    return interp.get_tensor(out["index"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="path to .h5 checkpoint")
    ap.add_argument("-o", "--output", help="output .tflite path (default: input with .tflite suffix)")
    ap.add_argument("--tolerance", type=float, default=1e-4, help="max absolute error")
    args = ap.parse_args()

    out_path = args.output or os.path.splitext(args.input)[0] + ".tflite"

    with h5py.File(args.input, "r") as f:
        model = build_model(f)
        x = f["example/input"][:].astype(np.float32)
        y_ref = f["example/output"][:].astype(np.float32)

    model.summary(line_length=120)

    y_keras = model.predict(x, verbose=0)
    keras_err = np.max(np.abs(y_keras - y_ref))
    print(f"Keras vs reference: max_abs_err={keras_err:.6g}")
    print(f"  ref:   {y_ref.reshape(-1)}")
    print(f"  keras: {y_keras.reshape(-1)}")

    tflite_bytes = convert(model)
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
