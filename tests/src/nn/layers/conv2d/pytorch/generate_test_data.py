"""
Generate test data for Conv2d layer verification.
Produces an HDF5 file with multiple test cases covering edge cases:
  - basic_3x3: 3x3 kernel, no padding, stride 1
  - padded_3x3: 3x3 kernel, padding 1, stride 1 ("same" padding)
  - strided_5x5: 5x5 kernel, stride 2, padding 2
  - pointwise_1x1: 1x1 kernel, no padding, stride 1
  - relu_3x3: 3x3 kernel, padding 1, stride 1, ReLU activation
  - nonsquare_3x5: 3x5 kernel, padding (1,2), stride 1

All data is stored in NHWC (batch, height, width, channels) format.
Weights are stored in PyTorch's [OUT_C, IN_C, KH, KW] format.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
import numpy as np
import os

torch.manual_seed(42)

test_cases = [
    {
        "name": "basic_3x3",
        "in_channels": 3, "out_channels": 8,
        "kernel_size": (3, 3), "stride": (1, 1), "padding": (0, 0),
        "height": 8, "width": 8, "batch_size": 2,
        "activation": "identity",
    },
    {
        "name": "padded_3x3",
        "in_channels": 3, "out_channels": 16,
        "kernel_size": (3, 3), "stride": (1, 1), "padding": (1, 1),
        "height": 8, "width": 8, "batch_size": 2,
        "activation": "identity",
    },
    {
        "name": "strided_5x5",
        "in_channels": 8, "out_channels": 16,
        "kernel_size": (5, 5), "stride": (2, 2), "padding": (2, 2),
        "height": 16, "width": 16, "batch_size": 2,
        "activation": "identity",
    },
    {
        "name": "pointwise_1x1",
        "in_channels": 16, "out_channels": 32,
        "kernel_size": (1, 1), "stride": (1, 1), "padding": (0, 0),
        "height": 4, "width": 4, "batch_size": 2,
        "activation": "identity",
    },
    {
        "name": "relu_3x3",
        "in_channels": 3, "out_channels": 8,
        "kernel_size": (3, 3), "stride": (1, 1), "padding": (1, 1),
        "height": 8, "width": 8, "batch_size": 2,
        "activation": "relu",
    },
    {
        "name": "nonsquare_3x5",
        "in_channels": 4, "out_channels": 8,
        "kernel_size": (3, 5), "stride": (1, 1), "padding": (1, 2),
        "height": 6, "width": 10, "batch_size": 2,
        "activation": "identity",
    },
]

script_dir = os.path.dirname(os.path.abspath(__file__))
# Navigate up to the repo root: pytorch/ -> conv2d/ -> layers/ -> nn/ -> src/ -> tests/ -> (repo root)
repo_root = os.path.normpath(os.path.join(script_dir, "..", "..", "..", "..", "..", ".."))
output_dir = os.path.join(repo_root, "tests", "data")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "conv2d_test_data.h5")

print(f"Saving to: {output_path}")

with h5py.File(output_path, "w") as f:
    for tc in test_cases:
        print(f"Generating test case: {tc['name']}")
        g = f.create_group(tc["name"])

        conv = nn.Conv2d(
            tc["in_channels"], tc["out_channels"],
            tc["kernel_size"],
            stride=tc["stride"],
            padding=tc["padding"],
            bias=True,
        ).double()

        # Random input in NCHW format
        input_nchw = torch.randn(
            tc["batch_size"], tc["in_channels"], tc["height"], tc["width"],
            dtype=torch.float64, requires_grad=True,
        )

        # Forward pass
        pre_act_nchw = conv(input_nchw)

        if tc["activation"] == "relu":
            output_nchw = F.relu(pre_act_nchw)
        else:
            output_nchw = pre_act_nchw

        # Random upstream gradient
        d_output_nchw = torch.randn_like(output_nchw)

        # Backward pass
        output_nchw.backward(d_output_nchw)

        # Convert NCHW -> NHWC for storage
        input_nhwc = input_nchw.detach().permute(0, 2, 3, 1).contiguous().numpy()
        output_nhwc = output_nchw.detach().permute(0, 2, 3, 1).contiguous().numpy()
        d_output_nhwc = d_output_nchw.detach().permute(0, 2, 3, 1).contiguous().numpy()
        d_input_nhwc = input_nchw.grad.detach().permute(0, 2, 3, 1).contiguous().numpy()

        # Weights: [OUT_C, IN_C, KH, KW]
        weights = conv.weight.detach().numpy()
        biases = conv.bias.detach().numpy()
        d_weights = conv.weight.grad.detach().numpy()
        d_biases = conv.bias.grad.detach().numpy()

        g.create_dataset("input", data=input_nhwc)
        g.create_dataset("output", data=output_nhwc)
        g.create_dataset("weights", data=weights)
        g.create_dataset("biases", data=biases)
        g.create_dataset("d_output", data=d_output_nhwc)
        g.create_dataset("d_input", data=d_input_nhwc)
        g.create_dataset("d_weights", data=d_weights)
        g.create_dataset("d_biases", data=d_biases)

        # Store config as attributes for reference
        g.attrs["in_channels"] = tc["in_channels"]
        g.attrs["out_channels"] = tc["out_channels"]
        g.attrs["kernel_h"] = tc["kernel_size"][0]
        g.attrs["kernel_w"] = tc["kernel_size"][1]
        g.attrs["stride_h"] = tc["stride"][0]
        g.attrs["stride_w"] = tc["stride"][1]
        g.attrs["padding_h"] = tc["padding"][0]
        g.attrs["padding_w"] = tc["padding"][1]
        g.attrs["height"] = tc["height"]
        g.attrs["width"] = tc["width"]
        g.attrs["batch_size"] = tc["batch_size"]
        g.attrs["activation"] = tc["activation"]

        print(f"  Input:  {input_nhwc.shape}")
        print(f"  Output: {output_nhwc.shape}")
        print(f"  Weights: {weights.shape}, Biases: {biases.shape}")

print("Done!")
