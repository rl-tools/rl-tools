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


class FakeQuantConvBlock(tf.keras.layers.Layer):
    """Conv2D → bias → activation → per-tensor output fake-quant, with the
    kernel wrapped in a per-channel weight fake-quant. Used for QAT fine-tuning.

    The inner Conv2D layer owns the trainable kernel / bias; this wrapper adds
    STE-fake-quant around them using scales pre-harvested from an int8 tflite.
    """
    def __init__(self, inner_conv, w_scales, out_min, out_max, activation_fn,
                 pad_h=0, pad_w=0, stride_h=1, stride_w=1, **kwargs):
        super().__init__(**kwargs)
        self.inner_conv = inner_conv
        self._w_scales = tf.constant(np.asarray(w_scales, dtype=np.float32))
        self._out_min = tf.constant(float(out_min), dtype=tf.float32)
        self._out_max = tf.constant(float(out_max), dtype=tf.float32)
        self._activation_fn = activation_fn
        self._pad = (int(pad_h), int(pad_w))
        self._strides = (int(stride_h), int(stride_w))

    def call(self, x):
        if self._pad[0] or self._pad[1]:
            x = tf.pad(x, [[0, 0], [self._pad[0], self._pad[0]],
                           [self._pad[1], self._pad[1]], [0, 0]])
        max_w = 127.0 * self._w_scales
        w_fq = tf.quantization.fake_quant_with_min_max_vars_per_channel(
            self.inner_conv.kernel, min=-max_w, max=max_w, num_bits=8)
        y = tf.nn.conv2d(x, w_fq,
                         strides=(1, self._strides[0], self._strides[1], 1),
                         padding="VALID")
        if self.inner_conv.use_bias:
            y = tf.nn.bias_add(y, self.inner_conv.bias)
        y = self._activation_fn(y)
        y = tf.quantization.fake_quant_with_min_max_vars(
            y, min=self._out_min, max=self._out_max, num_bits=8)
        return y


class FakeQuantDenseBlock(tf.keras.layers.Layer):
    """Dense → bias → activation → per-tensor output fake-quant, kernel
    wrapped in per-channel weight fake-quant."""
    def __init__(self, inner_dense, w_scales, out_min, out_max, activation_fn, **kwargs):
        super().__init__(**kwargs)
        self.inner_dense = inner_dense
        self._w_scales = tf.constant(np.asarray(w_scales, dtype=np.float32))
        self._out_min = tf.constant(float(out_min), dtype=tf.float32)
        self._out_max = tf.constant(float(out_max), dtype=tf.float32)
        self._activation_fn = activation_fn

    def call(self, x):
        max_w = 127.0 * self._w_scales
        w_fq = tf.quantization.fake_quant_with_min_max_vars_per_channel(
            self.inner_dense.kernel, min=-max_w, max=max_w, num_bits=8)
        y = tf.matmul(x, w_fq)
        if self.inner_dense.use_bias:
            y = y + self.inner_dense.bias
        y = self._activation_fn(y)
        y = tf.quantization.fake_quant_with_min_max_vars(
            y, min=self._out_min, max=self._out_max, num_bits=8)
        return y


def build_layer(h5_layer, tensor, fq_scales=None):
    kind = attr(h5_layer, "type")
    name = sanitize_name(h5_layer.name)
    fq = fq_scales.get(name) if fq_scales is not None else None

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

        conv = tf.keras.layers.Conv2D(
            filters=out_c,
            kernel_size=(kh, kw),
            strides=(sh, sw),
            padding="valid",
            activation=None,
            use_bias=True,
            name=name,
        )

        if fq is None:
            if ph or pw:
                tensor = tf.keras.layers.ZeroPadding2D(padding=(ph, pw))(tensor)
            tensor = conv(tensor)
            conv.set_weights([w, b])
            tensor = tf.keras.layers.Lambda(ACTIVATIONS[act_name])(tensor)
        else:
            conv.build((None, None, None, in_c))
            conv.set_weights([w, b])
            block = FakeQuantConvBlock(
                conv,
                w_scales=fq["w_scales"],
                out_min=fq["out_min"],
                out_max=fq["out_max"],
                activation_fn=ACTIVATIONS[act_name],
                pad_h=ph, pad_w=pw, stride_h=sh, stride_w=sw,
                name=f"{name}_fq",
            )
            tensor = block(tensor)
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
            name=name,
        )
        if fq is None:
            tensor = dense(tensor)
            dense.set_weights([w_tf, b.astype(np.float32)])
            tensor = tf.keras.layers.Lambda(ACTIVATIONS[act_name])(tensor)
        else:
            dense.build((None, in_dim))
            dense.set_weights([w_tf, b.astype(np.float32)])
            block = FakeQuantDenseBlock(
                dense,
                w_scales=fq["w_scales"],
                out_min=fq["out_min"],
                out_max=fq["out_max"],
                activation_fn=ACTIVATIONS[act_name],
                name=f"{name}_fq",
            )
            tensor = block(tensor)
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


def build_sequential(h5_group, tensor, fq_scales=None):
    for i in ordered_layer_indices(h5_group):
        tensor = build_layer(h5_group["layers"][str(i)], tensor, fq_scales=fq_scales)
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


def build_parallel(h5_parallel, tensor, fq_scales=None):
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
            branch_out = build_sequential(branch_group, branch_input, fq_scales=fq_scales)
        elif btype == "parallel":
            branch_out = build_parallel(branch_group, branch_input, fq_scales=fq_scales)
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
        return build_sequential(head_group, concat, fq_scales=fq_scales)
    if htype == "parallel":
        return build_parallel(head_group, concat, fq_scales=fq_scales)
    raise NotImplementedError(f"parallel head type {htype}")


def find_model_group(h5_root):
    if "actor" in h5_root:
        return h5_root["actor"]
    top = [k for k in h5_root.keys() if k != "example" and "type" in h5_root[k].attrs]
    assert len(top) == 1, f"cannot locate top-level model group, candidates={top}"
    return h5_root[top[0]]


def parse_camera_slug(meta_string):
    """Extract the Camera observation slug and return (n_img_inputs, channels_per).

    Supported slug forms:
      CameraRGB(fov, H, W)                                 → (1, 3)
      CameraRGBWithTarget(fov, H, W)                       → (2, 3)
      CameraRGBStacked(fov, H, W, stride, N)               → (N, 3)
      CameraRGBStackedWithTarget(fov, H, W, stride, N)     → (N+1, 3)
    Returns None if no Camera slug can be parsed.
    """
    if not meta_string:
        return None
    import json, re as _r
    try:
        meta = json.loads(meta_string)
    except (ValueError, TypeError):
        return None
    obs = meta.get("environment", {}).get("observation", "")
    if not obs:
        return None
    # First top-level token (image obs) ends at the first ", " at nesting depth 0.
    # The state slug internally uses dots + parentheses (e.g.
    # "OrientationBodyZ.ActionHistory(64)") so depth-tracked splitting is robust.
    depth = 0
    image_tok = None
    for i, ch in enumerate(obs):
        if ch == "(": depth += 1
        elif ch == ")": depth -= 1
        elif ch == "," and depth == 0:
            image_tok = obs[:i].strip()
            break
    if image_tok is None:
        image_tok = obs.strip()
    m = _r.match(r"^(Camera\w+)\s*\((.*)\)\s*$", image_tok)
    if not m:
        return None
    name, inner = m.group(1), m.group(2)
    args = [a.strip() for a in inner.split(",")]
    if name == "CameraRGB":
        return (1, 3)
    if name == "CameraRGBWithTarget":
        return (2, 3)
    if name == "CameraRGBStacked":
        if len(args) < 5: return None
        try: return (int(args[4]), 3)
        except ValueError: return None
    if name == "CameraRGBStackedWithTarget":
        if len(args) < 5: return None
        try: return (int(args[4]) + 1, 3)
        except ValueError: return None
    return None


def first_conv_input_channels(h5_root):
    """Walk the model tree to the first conv2d layer and return its
    input_channels attribute, or None if the model has no conv2d layers.
    """
    model_group = find_model_group(h5_root)
    def walk(g):
        kind = attr(g, "type")
        if kind == "sequential":
            for i in ordered_layer_indices(g):
                l = g["layers"][str(i)]
                lk = attr(l, "type")
                if lk == "conv2d":
                    return int(attr(l, "input_channels"))
                found = walk(l) if lk in ("sequential", "parallel") else None
                if found is not None:
                    return found
            return None
        if kind == "parallel":
            n = int(attr(g, "num_branches"))
            for i in range(n):
                found = walk(g[f"branch_{i}"])
                if found is not None:
                    return found
            return None
        return None
    return walk(model_group)


def example_input_tensors(h5_root):
    inputs_group = h5_root["example/inputs"]
    keys = sorted(inputs_group.keys(), key=int)
    return [inputs_group[k][:].astype(np.float32) for k in keys]


def example_output_tensor(h5_root):
    return h5_root["example/outputs/0"][:].astype(np.float32)


def build_branch_output(branch_group, branch_input, fq_scales=None):
    btype = attr(branch_group, "type")
    if btype == "sequential":
        return build_sequential(branch_group, branch_input, fq_scales=fq_scales)
    if btype == "parallel":
        feature_shape = tuple(int(d) for d in branch_input.shape[1:])
        flat = (
            tf.keras.layers.Flatten()(branch_input)
            if len(feature_shape) > 1
            else branch_input
        )
        return build_parallel(branch_group, flat, fq_scales=fq_scales)
    raise NotImplementedError(f"branch type {btype}")


def build_head_output(head_group, tensor, fq_scales=None):
    htype = attr(head_group, "type")
    if htype == "sequential":
        return build_sequential(head_group, tensor, fq_scales=fq_scales)
    if htype == "parallel":
        return build_parallel(head_group, tensor, fq_scales=fq_scales)
    raise NotImplementedError(f"head type {htype}")


def build_model(h5_root, fq_scales=None):
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
            branch_outputs.append(build_branch_output(branch_group, inp, fq_scales=fq_scales))

        concat = (
            tf.keras.layers.Concatenate(axis=-1)(branch_outputs)
            if n > 1
            else branch_outputs[0]
        )

        if int(attr(model_group, "has_head")) != 0:
            outputs = build_head_output(model_group["head"], concat, fq_scales=fq_scales)
        else:
            outputs = concat
        return tf.keras.Model(inputs=branch_inputs, outputs=outputs)

    if mtype == "sequential":
        example_in_0 = h5_root["example/inputs/0"]
        feature_shape = tuple(int(d) for d in example_in_0.shape[2:])
        inp = tf.keras.Input(shape=feature_shape, dtype=tf.float32, name="in_00")
        outputs = build_sequential(model_group, inp, fq_scales=fq_scales)
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


def harvest_int8_scales(tflite_bytes, ordered_layers):
    """Map each Conv2D/Dense Keras layer name to the int8 tflite's fake-quant
    parameters: per-channel weight scales (symmetric, zp=0) and per-tensor
    output (post-activation-fusion) min/max + scale/zp.
    """
    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    ops = interp._get_ops_details()
    td_by_idx = {td["index"]: td for td in interp.get_tensor_details()}

    quant_op_kinds = ("FULLY_CONNECTED", "CONV_2D")
    quant_ops = [o for o in ops if o["op_name"] in quant_op_kinds]
    if len(quant_ops) != len(ordered_layers):
        raise RuntimeError(
            f"tflite has {len(quant_ops)} quant ops, expected "
            f"{len(ordered_layers)} from HDF5"
        )

    result = {}
    for op, layer_info in zip(quant_ops, ordered_layers):
        h5_path, kind, _exp_w, _exp_b = layer_info
        inputs = list(op["inputs"])
        outputs = list(op["outputs"])
        w_idx = int(inputs[1])
        y_idx = int(outputs[0])
        w_qp = td_by_idx[w_idx]["quantization_parameters"]
        y_qp = td_by_idx[y_idx]["quantization_parameters"]
        w_scales = np.asarray(w_qp["scales"], dtype=np.float32)
        w_zps = np.asarray(w_qp["zero_points"], dtype=np.int32)
        y_scales = np.asarray(y_qp["scales"], dtype=np.float32)
        y_zps = np.asarray(y_qp["zero_points"], dtype=np.int32)
        if y_scales.size != 1:
            raise RuntimeError(
                f"output of {h5_path} is per-channel (n={y_scales.size}); expected per-tensor"
            )
        out_scale = float(y_scales[0])
        out_zp = int(y_zps[0])
        result[sanitize_name(h5_path)] = {
            "w_scales": w_scales,
            "w_zps": w_zps,
            "out_scale": out_scale,
            "out_zp": out_zp,
            "out_min": (-128 - out_zp) * out_scale,
            "out_max": (127 - out_zp) * out_scale,
        }
    return result


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


def write_quantized_h5(src_path, dst_path, dequant_by_path, example_limit=None):
    shutil.copyfile(src_path, dst_path)
    with h5py.File(dst_path, "r+") as f:
        for h5_path, (w_deq, b_deq) in dequant_by_path.items():
            w_ds = f[h5_path + "/weights/parameters"]
            w_ds[...] = w_deq
            if b_deq is not None:
                b_ds = f[h5_path + "/biases/parameters"]
                b_ds[...] = b_deq
        if example_limit is not None and "example" in f:
            # Example tensors have canonical shape [T, B, ...features]; truncate
            # along the batch axis (1). h5py doesn't support in-place resize for
            # datasets created without maxshape, so delete + recreate.
            def _truncate(group):
                for key in list(group.keys()):
                    ds = group[key]
                    if isinstance(ds, h5py.Dataset) and ds.ndim >= 2:
                        limit = min(int(example_limit), ds.shape[1])
                        if limit < ds.shape[1]:
                            data = ds[:, :limit, ...]
                            attrs = dict(ds.attrs)
                            del group[key]
                            new_ds = group.create_dataset(key, data=data)
                            for k, v in attrs.items():
                                new_ds.attrs[k] = v
            if "inputs" in f["example"]:
                _truncate(f["example/inputs"])
            if "outputs" in f["example"]:
                _truncate(f["example/outputs"])


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


def _int8_train_test_err(int8_bytes, per_branch_contig, y_flat, train_idx, test_idx):
    """Returns (train_max, train_mean, test_max, test_mean)."""
    y_all = run_tflite_int8(int8_bytes, per_branch_contig).reshape(y_flat.shape)
    diff = np.abs(y_all - y_flat)
    diff_flat = diff.reshape(diff.shape[0], -1)
    train_d = diff_flat[train_idx]
    test_d  = diff_flat[test_idx]
    return (float(train_d.max()), float(train_d.mean()),
            float(test_d.max()),  float(test_d.mean()))


def qat_finetune(*, args, h5_path, teacher_model, int8_src_model, int8_src_model_inner,
                 int8_bytes_v0, ordered_layers, per_branch_contig, y_flat, n_examples,
                 report, tflite_bytes):
    """Fine-tune the int8_src_model's Conv/Dense weights with fake-quant-in-the-loop
    (STE) distillation from the teacher on a shuffled train split. Returns the
    int8_bytes produced by a second convert_int8 call on the fine-tuned weights."""
    print()
    print("=== qat fine-tune ===")

    # 1. Harvest per-layer scales from the first int8 conversion.
    fq_scales = harvest_int8_scales(int8_bytes_v0, ordered_layers)
    print(f"harvested int8 scales for {len(fq_scales)} Conv/Dense layers")

    # 2. Build the fake-quant mirror model (same tanh substitution as int8_src_model).
    substitute = (args.fast_tanh_substitute == "tanh")
    if substitute:
        saved = ACTIVATIONS["FAST_TANH"]
        ACTIVATIONS["FAST_TANH"] = tf.nn.tanh
    try:
        with h5py.File(h5_path, "r") as f:
            fq_inner = build_model(f, fq_scales=fq_scales)
    finally:
        if substitute:
            ACTIVATIONS["FAST_TANH"] = saved
    if args.split_image_input is not None:
        fq_model = wrap_split_image_input(
            fq_inner, args.split_image_input, args.split_image_channels_per
        )
    else:
        fq_model = fq_inner

    # 3. Copy int8_src_model weights → fq_inner (redundant since both built from
    # the same h5, but defensive — ensures identical starting point).
    for layer in fq_inner.layers:
        if isinstance(layer, (FakeQuantConvBlock, FakeQuantDenseBlock)):
            inner = layer.inner_conv if isinstance(layer, FakeQuantConvBlock) else layer.inner_dense
            src_layer = int8_src_model_inner.get_layer(inner.name)
            inner.set_weights(src_layer.get_weights())

    # 4. Shuffle + train/test split.
    rng_split = np.random.default_rng(args.qat_seed)
    perm = rng_split.permutation(n_examples)
    n_test = max(1, int(round(n_examples * args.qat_test_fraction)))
    if n_test >= n_examples:
        raise RuntimeError(f"--qat-test-fraction={args.qat_test_fraction} leaves no training samples")
    test_idx = np.sort(perm[:n_test])
    train_idx = np.sort(perm[n_test:])
    train_inputs = [np.ascontiguousarray(a[train_idx]) for a in per_branch_contig]
    test_inputs  = [np.ascontiguousarray(a[test_idx])  for a in per_branch_contig]
    print(f"shuffled split: {len(train_idx)} train / {len(test_idx)} test (seed={args.qat_seed})")

    # 5. Pre-finetune baseline.
    pre = _int8_train_test_err(int8_bytes_v0, per_branch_contig, y_flat, train_idx, test_idx)
    report(f"QAT baseline (pre-finetune int8):  "
           f"train max={pre[0]:.4g} mean={pre[1]:.4g}  |  "
           f"test  max={pre[2]:.4g} mean={pre[3]:.4g}")

    # 6. Precompute teacher targets (frozen; saves forward passes in the loop).
    teacher_train_y = teacher_model(
        [tf.constant(a) for a in train_inputs], training=False
    ).numpy()
    teacher_test_y = teacher_model(
        [tf.constant(a) for a in test_inputs], training=False
    ).numpy()

    # Trainable variables: the inner Conv/Dense kernels + biases.
    train_vars = []
    for layer in fq_inner.layers:
        if isinstance(layer, (FakeQuantConvBlock, FakeQuantDenseBlock)):
            inner = layer.inner_conv if isinstance(layer, FakeQuantConvBlock) else layer.inner_dense
            train_vars.extend(inner.trainable_variables)
    optim = tf.keras.optimizers.Adam(args.qat_lr)

    @tf.function
    def step(xs, y_target):
        with tf.GradientTape() as tape:
            y_pred = fq_model(xs, training=True)
            loss = tf.reduce_mean((y_pred - y_target) ** 2)
        grads = tape.gradient(loss, train_vars)
        optim.apply_gradients(zip(grads, train_vars))
        return loss

    rng_batch = np.random.default_rng(args.qat_seed + 1)
    n_train = len(train_idx)
    bs = min(args.qat_batch_size, n_train)
    train_inputs_tf = [tf.constant(a) for a in train_inputs]
    test_inputs_tf = [tf.constant(a) for a in test_inputs]
    teacher_test_y_tf = tf.constant(teacher_test_y)
    print(f"fine-tuning {args.qat_steps} steps, batch={bs}, lr={args.qat_lr}")
    for step_i in range(args.qat_steps):
        batch = rng_batch.choice(n_train, size=bs, replace=False)
        batch_idx_tf = tf.constant(batch, dtype=tf.int32)
        batch_xs = [tf.gather(a, batch_idx_tf) for a in train_inputs_tf]
        batch_y = tf.gather(tf.constant(teacher_train_y), batch_idx_tf)
        loss = step(batch_xs, batch_y)
        if step_i % 50 == 0 or step_i == args.qat_steps - 1:
            test_pred = fq_model(test_inputs_tf, training=False)
            test_loss = float(tf.reduce_mean((test_pred - teacher_test_y_tf) ** 2).numpy())
            print(f"  [qat {step_i:4d}] train_mse={float(loss.numpy()):.6g} "
                  f"test_mse={test_loss:.6g}")

    # 7. Copy fine-tuned weights from fq_inner back to int8_src_model_inner.
    for layer in fq_inner.layers:
        if isinstance(layer, (FakeQuantConvBlock, FakeQuantDenseBlock)):
            inner = layer.inner_conv if isinstance(layer, FakeQuantConvBlock) else layer.inner_dense
            int8_src_model_inner.get_layer(inner.name).set_weights(inner.get_weights())

    # 8. Re-run int8 conversion on the updated weights.
    int8_bytes_v1 = convert_int8(int8_src_model, per_branch_contig, args.quantize_io, tflite_bytes)

    # 9. Post-finetune metrics.
    post = _int8_train_test_err(int8_bytes_v1, per_branch_contig, y_flat, train_idx, test_idx)
    def delta(a, b):
        return (b - a) / max(abs(a), 1e-12) * 100.0
    report(f"QAT final  (post-finetune int8):    "
           f"train max={post[0]:.4g} mean={post[1]:.4g}  |  "
           f"test  max={post[2]:.4g} mean={post[3]:.4g}")
    report(f"QAT delta (post-pre): "
           f"train max Δ={delta(pre[0], post[0]):+.1f}% mean Δ={delta(pre[1], post[1]):+.1f}%  |  "
           f"test  max Δ={delta(pre[2], post[2]):+.1f}% mean Δ={delta(pre[3], post[3]):+.1f}%")

    # Heuristic guidance on the outcome.
    train_improved = post[1] < pre[1]
    test_improved  = post[3] < pre[3]
    if train_improved and test_improved:
        gap = (pre[1] - post[1]) - (pre[3] - post[3])  # train delta - test delta
        if gap > 0.5 * (pre[1] - post[1]):
            report("QAT note: train improves notably more than test — mild overfit to calibration set")
        else:
            report("QAT note: train and test both improved — looks like a genuine gain")
    elif train_improved and not test_improved:
        report("QAT WARNING: test error got worse while train improved — overfitting, "
               "consider fewer --qat-steps or more calibration samples")
    elif not train_improved:
        report("QAT WARNING: no improvement on train — fine-tune not useful; "
               "consider --qat-lr / --qat-steps changes or skipping --qat-finetune")

    return int8_bytes_v1


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
                    help="override the auto-derived image-input split: replace the first input "
                         "(assumed (H,W,C_total)) with N inputs of (H,W,channels-per) joined by a "
                         "leading Concat (and zero-pad if N*channels-per < C_total). If omitted, N "
                         "is auto-derived from the checkpoint's Camera observation slug. Used to "
                         "avoid the CPU-side NHWC interleave on OpenMV / Ethos-U deployment.")
    ap.add_argument("--split-image-channels-per", type=int, default=None,
                    help="override channels per split input (default: 3 when --split-image-input is "
                         "set explicitly, or whatever the Camera slug implies when auto-deriving)")
    ap.add_argument("--no-split-image-input", action="store_true",
                    help="disable the image-input split entirely (overrides both auto-derive and "
                         "--split-image-input)")
    ap.add_argument("--example-bin-limit", type=int, default=None, metavar="N",
                    help="limit the number of samples written to the companion .bin files (inputs, "
                         "output, int8 output) to the first N (default: all samples from the h5 "
                         "example set). Int8 calibration still uses the full example set; this "
                         "only affects the deployed artifacts, e.g. to fit them in OpenMV flash.")
    ap.add_argument("--qat-finetune", action="store_true",
                    help="after the initial int8 conversion, fine-tune the float model's Conv/Dense "
                         "weights with fake-quant ops (STE) in the forward pass, using distillation "
                         "from the original float model on a shuffled train split of example/inputs. "
                         "Then re-run the int8 conversion. Requires --quantize=int8; no-op otherwise.")
    ap.add_argument("--qat-steps", type=int, default=500,
                    help="number of gradient steps for --qat-finetune (default 500)")
    ap.add_argument("--qat-lr", type=float, default=1e-4,
                    help="learning rate for --qat-finetune (default 1e-4)")
    ap.add_argument("--qat-batch-size", type=int, default=16,
                    help="batch size for --qat-finetune (default 16)")
    ap.add_argument("--qat-test-fraction", type=float, default=0.2,
                    help="fraction of example/inputs held out for --qat-finetune overfitting check "
                         "(default 0.2); the remainder is used for gradient updates")
    ap.add_argument("--qat-seed", type=int, default=0,
                    help="RNG seed controlling the shuffle before the --qat-finetune train/test split")
    args = ap.parse_args()

    out_path = args.output or os.path.splitext(args.input)[0] + ".tflite"

    with h5py.File(args.input, "r") as f:
        model = build_model(f)
        example_inputs = example_input_tensors(f)
        y_ref = example_output_tensor(f)
        ordered_layers = collect_quant_layers(find_model_group(f))
    # Keep a pre-(split-image-wrap) reference so get_layer() can reach the
    # inner Conv/Dense weights — wrap_split_image_input replaces `model` with
    # a Keras Model whose top-level children are the wrapper inputs, not the
    # inner Conv/Dense layers. QAT fine-tuning writes weights through this.
    model_inner = model
    # Canonical example shape is [T, B, ...features] (T=1 for non-recurrent).
    # Fold T into the batch axis so each per-branch array is (T*B, ...features).
    per_branch = [
        t.reshape((t.shape[0] * t.shape[1],) + t.shape[2:]).astype(np.float32)
        for t in example_inputs
    ]
    n_examples = per_branch[0].shape[0]
    y_ref = y_ref.reshape((n_examples,) + y_ref.shape[2:]).astype(np.float32)

    # Decide whether and how to split the first image input.
    #   --no-split-image-input   → disabled (explicit)
    #   --split-image-input N    → explicit override (user-supplied values)
    #   neither                  → auto-derive from the checkpoint's Camera slug
    split_mode = None  # one of "disabled", "override", "auto", "none"
    n_split = None
    c_per = None
    if args.no_split_image_input:
        split_mode = "disabled"
    elif args.split_image_input is not None:
        split_mode = "override"
        n_split = int(args.split_image_input)
        c_per = int(args.split_image_channels_per) if args.split_image_channels_per is not None else 3
    else:
        with h5py.File(args.input, "r") as f:
            meta_bytes = f["actor"].attrs.get("meta") if "actor" in f else None
            meta_str = meta_bytes.decode() if isinstance(meta_bytes, bytes) else meta_bytes
            first_conv_c = first_conv_input_channels(f)
        slug = parse_camera_slug(meta_str) if meta_str else None
        if slug is None:
            split_mode = "none"
        else:
            split_mode = "auto"
            n_split = slug[0]
            c_per = int(args.split_image_channels_per) if args.split_image_channels_per is not None else slug[1]

    args.split_image_input = n_split
    args.split_image_channels_per = c_per

    if split_mode == "disabled":
        print("image-input split: disabled via --no-split-image-input")
    elif split_mode == "none":
        print("image-input split: no Camera slug in meta; skipping (pass --split-image-input N "
              "to force)")
    else:
        c_total = None
        try:
            c_total = int(model.inputs[0].shape[-1])
        except Exception:
            pass
        pad = (c_total - n_split * c_per) if c_total is not None else None
        if split_mode == "auto":
            print(f"image-input split: auto from Camera slug → N={n_split}, channels_per={c_per}"
                  + (f", pad={pad} (first-conv channels={c_total})" if c_total is not None else ""))
        else:
            print(f"image-input split: user override → N={n_split}, channels_per={c_per}"
                  + (f", pad={pad} (first-conv channels={c_total})" if c_total is not None else ""))
        if pad is not None and pad < 0:
            raise RuntimeError(
                f"image-input split: N*channels_per = {n_split * c_per} exceeds the first conv's "
                f"{c_total} input channels. Check the Camera slug / network layout or pass "
                f"--no-split-image-input."
            )
        if split_mode == "auto" and pad is not None and pad > 0:
            print(f"  note: non-zero pad — the network expects {c_total} channels but the slug "
                  f"accounts for only {n_split * c_per}; the extra {pad} channels will be zero-filled")
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

    if args.example_bin_limit is not None:
        bin_limit = min(int(args.example_bin_limit), n_examples)
    else:
        bin_limit = n_examples
    per_branch_bin = [np.ascontiguousarray(a[:bin_limit]) for a in per_branch_contig]
    y_bin = np.ascontiguousarray(y_flat[:bin_limit])
    if bin_limit != n_examples:
        print(f"--example-bin-limit: writing first {bin_limit} of {n_examples} samples "
              f"to companion bin files")

    # The tflite converter alphabetizes inputs through SavedModel, so
    # tflite's input index i does not in general equal keras input index i.
    # On-device consumers (OpenMV / Vela-compiled models) read inputs in
    # tflite index order, so name the bin files accordingly:
    #   example_input.<tflite_i>.bin  == data for tflite input tflite_i
    # tflite_order[tflite_i] = keras_i.
    tflite_order = _tflite_input_order(tflite_bytes)
    in_paths = [base + f".example_input.{i}.bin" for i in range(len(per_branch_bin))]
    for tflite_i, path in enumerate(in_paths):
        arr = per_branch_bin[tflite_order[tflite_i]]
        with open(path, "wb") as f:
            f.write(arr.tobytes())
        print(f"wrote {arr.nbytes} bytes to {path}  "
              f"(tflite_input={tflite_i}, keras_input={tflite_order[tflite_i]}, "
              f"shape={arr.shape}, dtype=float32)")
    with open(out_example_path, "wb") as f:
        f.write(y_bin.tobytes())
    print(f"wrote {y_bin.nbytes} bytes to {out_example_path}  (shape={y_bin.shape}, dtype=float32)")

    import json
    meta = {
        "inputs": [
            {
                "path": os.path.basename(in_paths[tflite_i]),
                "tflite_input_index": tflite_i,
                "keras_input_index": tflite_order[tflite_i],
                "shape": list(per_branch_bin[tflite_order[tflite_i]].shape),
                "dtype": "float32",
            }
            for tflite_i in range(len(per_branch_bin))
        ],
        "output": {
            "path": os.path.basename(out_example_path),
            "shape": list(y_bin.shape),
            "dtype": "float32",
        },
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {meta_path}")

    # Reload bins (tflite order) and reshuffle back into keras order for the
    # Keras model validation and the name-matched run_tflite path.
    per_branch_reloaded = [None] * len(per_branch_bin)
    for tflite_i, path in enumerate(in_paths):
        keras_i = tflite_order[tflite_i]
        src = per_branch_bin[keras_i]
        with open(path, "rb") as f:
            reloaded_arr = np.frombuffer(f.read(), dtype=np.float32).reshape(src.shape)
        assert np.array_equal(reloaded_arr, src), f"companion input {path} diverges from source"
        per_branch_reloaded[keras_i] = reloaded_arr
    with open(out_example_path, "rb") as f:
        y_reloaded = np.frombuffer(f.read(), dtype=np.float32).reshape(y_bin.shape)
    assert np.array_equal(y_reloaded, y_bin), "companion output diverges from source"

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
                int8_src_model_inner = build_model(f)
            if args.split_image_input is not None:
                int8_src_model = wrap_split_image_input(
                    int8_src_model_inner, args.split_image_input, args.split_image_channels_per
                )
            else:
                int8_src_model = int8_src_model_inner
        finally:
            ACTIVATIONS["FAST_TANH"] = saved
    else:
        int8_src_model_inner = model_inner
        int8_src_model = model

    int8_bytes = convert_int8(int8_src_model, per_branch_contig, args.quantize_io, tflite_bytes)

    if args.qat_finetune:
        int8_bytes = qat_finetune(
            args=args,
            h5_path=args.input,
            teacher_model=model,
            int8_src_model=int8_src_model,
            int8_src_model_inner=int8_src_model_inner,
            int8_bytes_v0=int8_bytes,
            ordered_layers=ordered_layers,
            per_branch_contig=per_branch_contig,
            y_flat=y_flat,
            n_examples=n_examples,
            report=report,
            tflite_bytes=tflite_bytes,
        )

    int8_path = base + ".int8.tflite"
    with open(int8_path, "wb") as f:
        f.write(int8_bytes)
    print(f"wrote {len(int8_bytes)} bytes to {int8_path}")

    with open(int8_path, "rb") as f:
        int8_reloaded = f.read()
    y_int8, y_int8_raw, out_scale, out_zp, out_dtype_name = run_tflite_int8(
        int8_reloaded, per_branch_reloaded, return_raw=True
    )
    y_int8 = y_int8.reshape(y_bin.shape)
    int8_err, int8_mean = err_stats(y_int8, y_bin)
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
        y_int8_raw = y_int8_raw.reshape(y_bin.shape).astype(np.int8)
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
    write_quantized_h5(args.input, quant_h5, dequant_by_path, example_limit=bin_limit)
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
