"""
Generate test data for Conv2d layer verification.
Produces an HDF5 file with multiple test cases covering edge cases:
  - basic_3x3: 3x3 kernel, no padding, stride 1
  - padded_3x3: 3x3 kernel, padding 1, stride 1 ("same" padding)
  - strided_5x5: 5x5 kernel, stride 2, padding 2
  - pointwise_1x1: 1x1 kernel, no padding, stride 1
  - relu_3x3: 3x3 kernel, padding 1, stride 1, ReLU activation
  - nonsquare_3x5: 3x5 kernel, padding (1,2), stride 1
  - batchnorm_3x3: 3x3 kernel, padding 1, BatchNorm, identity activation
  - layernorm_3x3: 3x3 kernel, padding 1, LayerNorm (GroupNorm(1)), identity activation
  - batchnorm_relu: 3x3 kernel, padding 1, BatchNorm + ReLU
  - layernorm_relu: 3x3 kernel, padding 1, LayerNorm + ReLU

All data is stored in NHWC (batch, height, width, channels) format.
Weights are stored in NHWC filter layout [OUT_C, KH, KW, IN_C].
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
        "normalization": "none",
    },
    {
        "name": "padded_3x3",
        "in_channels": 3, "out_channels": 16,
        "kernel_size": (3, 3), "stride": (1, 1), "padding": (1, 1),
        "height": 8, "width": 8, "batch_size": 2,
        "activation": "identity",
        "normalization": "none",
    },
    {
        "name": "strided_5x5",
        "in_channels": 8, "out_channels": 16,
        "kernel_size": (5, 5), "stride": (2, 2), "padding": (2, 2),
        "height": 16, "width": 16, "batch_size": 2,
        "activation": "identity",
        "normalization": "none",
    },
    {
        "name": "pointwise_1x1",
        "in_channels": 16, "out_channels": 32,
        "kernel_size": (1, 1), "stride": (1, 1), "padding": (0, 0),
        "height": 4, "width": 4, "batch_size": 2,
        "activation": "identity",
        "normalization": "none",
    },
    {
        "name": "relu_3x3",
        "in_channels": 3, "out_channels": 8,
        "kernel_size": (3, 3), "stride": (1, 1), "padding": (1, 1),
        "height": 8, "width": 8, "batch_size": 2,
        "activation": "relu",
        "normalization": "none",
    },
    {
        "name": "nonsquare_3x5",
        "in_channels": 4, "out_channels": 8,
        "kernel_size": (3, 5), "stride": (1, 1), "padding": (1, 2),
        "height": 6, "width": 10, "batch_size": 2,
        "activation": "identity",
        "normalization": "none",
    },
    # ======== Normalization test cases ========
    {
        "name": "batchnorm_3x3",
        "in_channels": 3, "out_channels": 16,
        "kernel_size": (3, 3), "stride": (1, 1), "padding": (1, 1),
        "height": 8, "width": 8, "batch_size": 4,
        "activation": "identity",
        "normalization": "batch_norm",
    },
    {
        "name": "layernorm_3x3",
        "in_channels": 3, "out_channels": 16,
        "kernel_size": (3, 3), "stride": (1, 1), "padding": (1, 1),
        "height": 8, "width": 8, "batch_size": 2,
        "activation": "identity",
        "normalization": "layer_norm",
    },
    {
        "name": "batchnorm_relu",
        "in_channels": 3, "out_channels": 8,
        "kernel_size": (3, 3), "stride": (1, 1), "padding": (1, 1),
        "height": 8, "width": 8, "batch_size": 4,
        "activation": "relu",
        "normalization": "batch_norm",
    },
    {
        "name": "layernorm_relu",
        "in_channels": 3, "out_channels": 8,
        "kernel_size": (3, 3), "stride": (1, 1), "padding": (1, 1),
        "height": 8, "width": 8, "batch_size": 2,
        "activation": "relu",
        "normalization": "layer_norm",
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

        # Create normalization layer if needed
        norm_layer = None
        norm_type = tc.get("normalization", "none")
        if norm_type == "batch_norm":
            norm_layer = nn.BatchNorm2d(tc["out_channels"], eps=1e-5, momentum=0.1).double()
            # Initialize gamma and beta randomly for a non-trivial test
            nn.init.uniform_(norm_layer.weight, 0.5, 1.5)  # gamma
            nn.init.uniform_(norm_layer.bias, -0.5, 0.5)   # beta
            norm_layer.train()  # training mode: use batch stats
        elif norm_type == "layer_norm":
            # GroupNorm(1, C) normalizes over (C, H, W) per sample with per-channel gamma/beta
            norm_layer = nn.GroupNorm(1, tc["out_channels"], eps=1e-5).double()
            # Initialize gamma and beta randomly
            nn.init.uniform_(norm_layer.weight, 0.5, 1.5)  # gamma
            nn.init.uniform_(norm_layer.bias, -0.5, 0.5)   # beta

        # Random input in NCHW format
        input_nchw = torch.randn(
            tc["batch_size"], tc["in_channels"], tc["height"], tc["width"],
            dtype=torch.float64, requires_grad=True,
        )

        # Forward pass: conv -> [norm] -> [activation]
        conv_out_nchw = conv(input_nchw)

        if norm_layer is not None:
            normed_nchw = norm_layer(conv_out_nchw)
        else:
            normed_nchw = conv_out_nchw

        if tc["activation"] == "relu":
            output_nchw = F.relu(normed_nchw)
        else:
            output_nchw = normed_nchw

        # Random upstream gradient
        d_output_nchw = torch.randn_like(output_nchw)

        # Backward pass
        output_nchw.backward(d_output_nchw)

        # Convert NCHW -> NHWC for storage
        input_nhwc = input_nchw.detach().permute(0, 2, 3, 1).contiguous().numpy()
        output_nhwc = output_nchw.detach().permute(0, 2, 3, 1).contiguous().numpy()
        d_output_nhwc = d_output_nchw.detach().permute(0, 2, 3, 1).contiguous().numpy()
        d_input_nhwc = input_nchw.grad.detach().permute(0, 2, 3, 1).contiguous().numpy()

        # Weights: [OUT_C, KH, KW, IN_C] (NHWC filter layout)
        weights = conv.weight.detach().permute(0, 2, 3, 1).contiguous().numpy()
        biases = conv.bias.detach().numpy()
        d_weights = conv.weight.grad.detach().permute(0, 2, 3, 1).contiguous().numpy()
        d_biases = conv.bias.grad.detach().numpy()

        g.create_dataset("input", data=input_nhwc)
        g.create_dataset("output", data=output_nhwc)
        g.create_dataset("weights", data=weights)
        g.create_dataset("biases", data=biases)
        g.create_dataset("d_output", data=d_output_nhwc)
        g.create_dataset("d_input", data=d_input_nhwc)
        g.create_dataset("d_weights", data=d_weights)
        g.create_dataset("d_biases", data=d_biases)

        # Save normalization parameters and gradients
        if norm_layer is not None:
            gamma = norm_layer.weight.detach().numpy()  # [OUT_C]
            beta = norm_layer.bias.detach().numpy()      # [OUT_C]
            d_gamma = norm_layer.weight.grad.detach().numpy()
            d_beta = norm_layer.bias.grad.detach().numpy()
            g.create_dataset("gamma", data=gamma)
            g.create_dataset("beta", data=beta)
            g.create_dataset("d_gamma", data=d_gamma)
            g.create_dataset("d_beta", data=d_beta)

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
        g.attrs["normalization"] = norm_type

        print(f"  Input:  {input_nhwc.shape}")
        print(f"  Output: {output_nhwc.shape}")
        print(f"  Weights: {weights.shape}, Biases: {biases.shape}")
        if norm_layer is not None:
            print(f"  Gamma: {gamma.shape}, Beta: {beta.shape}")
            print(f"  d_gamma: {d_gamma.shape}, d_beta: {d_beta.shape}")

print("Done!")
