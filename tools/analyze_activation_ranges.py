"""Per-layer activation analysis for int8 quantization diagnosis.

Reads an RLtools HDF5 checkpoint + the paired `.int8.tflite` produced by
`hdf5_to_tflite.py`, runs every example/inputs sample through the Keras model
with per-layer introspection outputs, and prints a table comparing the actual
float activation range with the int8 scale the converter chose.

What to look for:
  * `sat%`: fraction of POST-activation values outside the int8 representable
    range at that layer. A few percent is fine; >5% means calibration chose a
    range too tight for the outlier tail.
  * `headroom`: int8_range / (2 * absmax). Well-calibrated lies around 1.0
    — values < 0.5 mean a long-tailed distribution where the converter threw
    resolution at the outliers and is under-resolving the bulk.
  * weight-scale dynamic range: if per-channel scales span >10× within a
    single layer, quantization error on the low-scale channels becomes
    non-negligible and often points to a training-side fix (weight decay).

"Pre-act" lines are the raw conv/dense output (useful for weight-scale
context); "post-act" lines are what the int8 tflite actually quantizes
(activation fused into the preceding op).
"""
import argparse
import os
import sys

import h5py
import numpy as np
import tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hdf5_to_tflite as h2t


def tflite_tensor_info(path):
    interp = tf.lite.Interpreter(model_content=open(path, "rb").read())
    interp.allocate_tensors()
    details = interp.get_tensor_details()
    info = {}
    for d in details:
        q = d["quantization_parameters"]
        scales = np.asarray(q["scales"], dtype=np.float32)
        zps = np.asarray(q["zero_points"], dtype=np.int32)
        info[d["index"]] = {
            "name": d["name"],
            "dtype": d["dtype"].__name__ if hasattr(d["dtype"], "__name__") else str(d["dtype"]),
            "shape": tuple(int(x) for x in d["shape"]),
            "scales": scales,
            "zps": zps,
        }
    return info


def stats(arr):
    flat = arr.reshape(-1).astype(np.float64)
    mn = float(flat.min()); mx = float(flat.max())
    p01, p99 = float(np.quantile(flat, 0.01)), float(np.quantile(flat, 0.99))
    absmax = max(abs(mn), abs(mx))
    std = float(flat.std())
    return mn, mx, p01, p99, absmax, std


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("h5", help="path to the RLtools .h5 checkpoint")
    ap.add_argument("--tflite", default=None, help="path to paired .int8.tflite (default: <h5>.int8.tflite)")
    ap.add_argument("--max-samples", type=int, default=None)
    args = ap.parse_args()

    tflite_path = args.tflite or (os.path.splitext(args.h5)[0] + ".int8.tflite")

    with h5py.File(args.h5, "r") as f:
        keras_model = h2t.build_model(f)
        example_inputs = h2t.example_input_tensors(f)
        y_ref = h2t.example_output_tensor(f)

    per_branch = [
        t.reshape((t.shape[0] * t.shape[1],) + t.shape[2:]).astype(np.float32)
        for t in example_inputs
    ]
    n_examples = per_branch[0].shape[0]
    y_ref = y_ref.reshape((n_examples,) + y_ref.shape[2:]).astype(np.float32)
    if args.max_samples is not None:
        per_branch = [a[: args.max_samples] for a in per_branch]
        y_ref = y_ref[: args.max_samples]
        n_examples = per_branch[0].shape[0]

    print(f"samples: {n_examples}  output mean_abs_err (ref): {float(np.abs(y_ref).mean()):.4g}")
    print()

    # Identify (conv/dense, activation-lambda) pairs via graph connectivity:
    # for each Conv/Dense, find the Lambda whose input tensor is produced by
    # that layer. (Walking keras_model.layers in list order mispairs when two
    # Dense layers from parallel branches sit adjacent in the list.)
    def producer(tensor):
        kh = getattr(tensor, "_keras_history", None)
        if kh is None: return None
        return kh.layer if hasattr(kh, "layer") else kh[0]

    pairs = []
    layers = keras_model.layers
    for layer in layers:
        if not isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.Dense)):
            continue
        act_layer = None
        for other in layers:
            if not isinstance(other, tf.keras.layers.Lambda):
                continue
            try:
                inp = other.input
            except Exception:
                continue
            if producer(inp) is layer:
                act_layer = other
                break
        pairs.append((layer, act_layer))

    # Build a multi-output probe model — emit BOTH pre-act and post-act outputs.
    probe_names = []
    probe_outputs = []
    for main, act in pairs:
        probe_names.append(main.name + " (pre)")
        probe_outputs.append(main.output)
        if act is not None:
            probe_names.append(main.name + " (post)")
            probe_outputs.append(act.output)
    probe = tf.keras.Model(inputs=keras_model.inputs, outputs=probe_outputs)
    acts = probe.predict(per_branch, verbose=0)
    if not isinstance(acts, (list, tuple)):
        acts = [acts]
    name_to_stats = dict(zip(probe_names, [stats(a) for a in acts]))
    name_to_act = dict(zip(probe_names, acts))

    info = tflite_tensor_info(tflite_path)

    def find_activation_int8(layer_name):
        best = None
        for idx, t in info.items():
            if t["dtype"] != "int8" or t["scales"].size != 1:
                continue
            if layer_name in t["name"]:
                if best is None or len(t["name"]) < len(best["name"]):
                    best = t
        return best

    print(f"{'layer':<36} {'kind':<9}  "
          f"{'min':>8} {'max':>8} {'p01':>8} {'p99':>8} {'absmax':>8} {'std':>8}   "
          f"{'int8 scale':>11} {'int8 range':>13} {'headroom':>9} {'sat%':>6}")
    print("-" * 155)
    for main, act in pairs:
        pre_stats = name_to_stats.get(main.name + " (pre)")
        post_stats = name_to_stats.get(main.name + " (post)")
        int8 = find_activation_int8(main.name)

        if pre_stats is not None:
            mn, mx, p01, p99, absmax, std = pre_stats
            print(f"{main.name:<36} {'pre-act':<9}  "
                  f"{mn:>8.3g} {mx:>8.3g} {p01:>8.3g} {p99:>8.3g} {absmax:>8.3g} {std:>8.3g}   "
                  f"{'-':>11} {'-':>13} {'-':>9} {'-':>6}")
        if post_stats is not None and act is not None:
            mn, mx, p01, p99, absmax, std = post_stats
            if int8 is not None:
                scale = float(int8["scales"][0]); zp = int(int8["zps"][0])
                lo = (-128 - zp) * scale; hi = (127 - zp) * scale
                headroom = (hi - lo) / max(2 * absmax, 1e-12)
                flat = name_to_act[main.name + " (post)"].reshape(-1).astype(np.float64)
                sat = float(((flat < lo) | (flat > hi)).mean()) * 100
                print(f"{main.name:<36} {'post-act':<9}  "
                      f"{mn:>8.3g} {mx:>8.3g} {p01:>8.3g} {p99:>8.3g} {absmax:>8.3g} {std:>8.3g}   "
                      f"{scale:>11.4g} {lo:>6.2g}..{hi:<5.2g} {headroom:>9.2f} {sat:>5.1f}")
            else:
                print(f"{main.name:<36} {'post-act':<9}  "
                      f"{mn:>8.3g} {mx:>8.3g} {p01:>8.3g} {p99:>8.3g} {absmax:>8.3g} {std:>8.3g}   "
                      f"{'-':>11} {'-':>13} {'-':>9} {'-':>6}")

    # ---- Weight scale analysis ----
    print()
    print("=== weight-scale dynamic range per layer (per-channel int8) ===")
    print(f"{'layer':<36}  {'w absmax':>9} {'w std':>9}  {'scale min':>12} {'scale max':>12} "
          f"{'max/min':>9} {'n_ch':>6}")
    for main, _ in pairs:
        ws = main.get_weights()
        if not ws:
            continue
        w = ws[0]
        w_abs = float(np.abs(w).max()); w_std = float(w.std())

        matched = None
        for idx, t in info.items():
            if t["dtype"] != "int8":
                continue
            if t["shape"] and (tuple(t["shape"]) == tuple(w.shape)):
                matched = t; break
            if len(t["shape"]) == len(w.shape) == 4:
                if tuple(t["shape"]) == (w.shape[3], w.shape[0], w.shape[1], w.shape[2]):
                    matched = t; break
        if matched is None or matched["scales"].size == 1:
            s_min = s_max = ratio = None
            n_ch = 0 if matched is None else int(matched["scales"].size)
        else:
            scales = matched["scales"].astype(np.float64)
            s_min, s_max = float(scales.min()), float(scales.max())
            ratio = s_max / max(s_min, 1e-20)
            n_ch = int(scales.size)

        def fmt(x, w=12):
            return f"{x:>{w}.4g}" if x is not None else f"{'-':>{w}}"
        def fmt_ratio(x, w=9):
            return f"{x:>{w}.2f}" if x is not None else f"{'-':>{w}}"
        print(f"{main.name:<36}  {w_abs:>9.3g} {w_std:>9.3g}  {fmt(s_min)} {fmt(s_max)} "
              f"{fmt_ratio(ratio)} {n_ch:>6}")


if __name__ == "__main__":
    main()
