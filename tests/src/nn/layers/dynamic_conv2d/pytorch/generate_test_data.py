"""
Generate test data for DynamicConv2d layer verification.
Per-sample depthwise or full convolution with runtime-provided kernel weights.

Data layout:
  - data: NHWC [batch_size, height, width, channels_in]
  - kernel_weights (depthwise): [batch_size, channels, kernel_h, kernel_w]
  - kernel_weights (full conv): [batch_size, channels_out, channels_in, kernel_h, kernel_w]
  - output: NHWC [batch_size, output_h, output_w, channels_out]
"""

import torch
import torch.nn.functional as F
import h5py
import os

torch.manual_seed(42)

test_cases = [
    {
        "name": "basic_3x3",
        "channels": 8, "kernel_size": (3, 3),
        "stride": (1, 1), "padding": (0, 0),
        "height": 8, "width": 8, "batch_size": 2,
        "activation": "identity",
    },
    {
        "name": "padded_3x3",
        "channels": 16, "kernel_size": (3, 3),
        "stride": (1, 1), "padding": (1, 1),
        "height": 8, "width": 8, "batch_size": 2,
        "activation": "identity",
    },
    {
        "name": "strided_3x3",
        "channels": 32, "kernel_size": (3, 3),
        "stride": (2, 2), "padding": (1, 1),
        "height": 16, "width": 16, "batch_size": 2,
        "activation": "identity",
    },
    {
        "name": "pointwise_1x1",
        "channels": 16, "kernel_size": (1, 1),
        "stride": (1, 1), "padding": (0, 0),
        "height": 4, "width": 4, "batch_size": 2,
        "activation": "identity",
    },
    {
        "name": "relu_3x3",
        "channels": 8, "kernel_size": (3, 3),
        "stride": (1, 1), "padding": (1, 1),
        "height": 8, "width": 8, "batch_size": 2,
        "activation": "relu",
    },
    {
        "name": "strided_relu",
        "channels": 16, "kernel_size": (3, 3),
        "stride": (2, 2), "padding": (1, 1),
        "height": 16, "width": 16, "batch_size": 2,
        "activation": "relu",
    },
    {
        "name": "large_batch",
        "channels": 64, "kernel_size": (3, 3),
        "stride": (2, 2), "padding": (1, 1),
        "height": 16, "width": 16, "batch_size": 8,
        "activation": "identity",
    },
    {
        "name": "nonsquare_3x5",
        "channels": 8, "kernel_size": (3, 5),
        "stride": (1, 1), "padding": (1, 2),
        "height": 6, "width": 10, "batch_size": 2,
        "activation": "identity",
    },
    {
        "name": "single_channel",
        "channels": 1, "kernel_size": (3, 3),
        "stride": (1, 1), "padding": (1, 1),
        "height": 8, "width": 8, "batch_size": 2,
        "activation": "identity",
    },
    # Full (non-depthwise) convolution test cases
    {
        "name": "full_basic_3x3",
        "channels_in": 8, "channels_out": 16, "kernel_size": (3, 3),
        "stride": (1, 1), "padding": (0, 0),
        "height": 8, "width": 8, "batch_size": 2,
        "activation": "identity", "full_conv": True,
    },
    {
        "name": "full_padded_3x3",
        "channels_in": 8, "channels_out": 4, "kernel_size": (3, 3),
        "stride": (1, 1), "padding": (1, 1),
        "height": 8, "width": 8, "batch_size": 2,
        "activation": "identity", "full_conv": True,
    },
    {
        "name": "full_strided_3x3",
        "channels_in": 16, "channels_out": 32, "kernel_size": (3, 3),
        "stride": (2, 2), "padding": (1, 1),
        "height": 16, "width": 16, "batch_size": 2,
        "activation": "identity", "full_conv": True,
    },
    {
        "name": "full_relu_3x3",
        "channels_in": 8, "channels_out": 16, "kernel_size": (3, 3),
        "stride": (1, 1), "padding": (1, 1),
        "height": 8, "width": 8, "batch_size": 2,
        "activation": "relu", "full_conv": True,
    },
    {
        "name": "full_strided_relu",
        "channels_in": 16, "channels_out": 8, "kernel_size": (3, 3),
        "stride": (2, 2), "padding": (1, 1),
        "height": 16, "width": 16, "batch_size": 2,
        "activation": "relu", "full_conv": True,
    },
]

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.normpath(os.path.join(script_dir, "..", "..", "..", "..", "..", ".."))
output_dir = os.path.join(repo_root, "tests", "data")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "dynamic_conv2d_test_data.h5")

print(f"Saving to: {output_path}")

with h5py.File(output_path, "w") as f:
    for tc in test_cases:
        print(f"Generating test case: {tc['name']}")
        g = f.create_group(tc["name"])

        BS = tc["batch_size"]
        H = tc["height"]
        W = tc["width"]
        KH, KW = tc["kernel_size"]
        stride = tc["stride"]
        padding = tc["padding"]
        is_full_conv = tc.get("full_conv", False)

        if is_full_conv:
            C_IN = tc["channels_in"]
            C_OUT = tc["channels_out"]
        else:
            C_IN = tc["channels"]
            C_OUT = C_IN

        # Input data in NCHW (for PyTorch conv2d)
        data_nchw = torch.randn(BS, C_IN, H, W, dtype=torch.float64, requires_grad=True)

        if is_full_conv:
            # Per-sample full convolution kernel: [BS, C_OUT, C_IN, KH, KW]
            kernel_weights = torch.randn(BS, C_OUT, C_IN, KH, KW, dtype=torch.float64, requires_grad=True)

            output_parts = []
            for b in range(BS):
                # input: [1, C_IN, H, W], weight: [C_OUT, C_IN, KH, KW]
                out = F.conv2d(data_nchw[b:b+1], kernel_weights[b],
                               stride=stride, padding=padding)
                output_parts.append(out)
            output_nchw = torch.cat(output_parts, dim=0)  # [BS, C_OUT, OH, OW]
        else:
            # Per-sample depthwise kernel: [BS, C_IN, KH, KW]
            kernel_weights = torch.randn(BS, C_IN, KH, KW, dtype=torch.float64, requires_grad=True)

            output_parts = []
            for b in range(BS):
                # input: [1, C, H, W], weight: [C, 1, KH, KW] for depthwise
                out = F.conv2d(data_nchw[b:b+1], kernel_weights[b].unsqueeze(1),
                               stride=stride, padding=padding, groups=C_IN)
                output_parts.append(out)
            output_nchw = torch.cat(output_parts, dim=0)  # [BS, C_IN, OH, OW]

        # Apply activation
        if tc["activation"] == "relu":
            output_nchw = F.relu(output_nchw)

        # Random upstream gradient
        d_output_nchw = torch.randn_like(output_nchw)

        # Backward
        output_nchw.backward(d_output_nchw)

        d_data_nchw = data_nchw.grad
        d_kernel_weights = kernel_weights.grad

        # Convert NCHW -> NHWC for storage
        data_nhwc = data_nchw.detach().permute(0, 2, 3, 1).contiguous().numpy()
        output_nhwc = output_nchw.detach().permute(0, 2, 3, 1).contiguous().numpy()
        d_output_nhwc = d_output_nchw.detach().permute(0, 2, 3, 1).contiguous().numpy()
        d_data_nhwc = d_data_nchw.detach().permute(0, 2, 3, 1).contiguous().numpy()

        # kernel_weights and d_kernel_weights: [BS, C, KH, KW] or [BS, C_OUT, C_IN, KH, KW]
        kw_np = kernel_weights.detach().numpy()
        d_kw_np = d_kernel_weights.detach().numpy()

        g.create_dataset("data", data=data_nhwc)
        g.create_dataset("kernel_weights", data=kw_np)
        g.create_dataset("output", data=output_nhwc)
        g.create_dataset("d_output", data=d_output_nhwc)
        g.create_dataset("d_data", data=d_data_nhwc)
        g.create_dataset("d_kernel_weights", data=d_kw_np)

        # Store config as attributes
        g.attrs["channels_in"] = C_IN
        g.attrs["channels_out"] = C_OUT
        g.attrs["kernel_h"] = KH
        g.attrs["kernel_w"] = KW
        g.attrs["stride_h"] = stride[0]
        g.attrs["stride_w"] = stride[1]
        g.attrs["padding_h"] = padding[0]
        g.attrs["padding_w"] = padding[1]
        g.attrs["height"] = H
        g.attrs["width"] = W
        g.attrs["batch_size"] = BS
        g.attrs["activation"] = tc["activation"]
        g.attrs["full_conv"] = is_full_conv

        print(f"  Data:  {data_nhwc.shape}")
        print(f"  KernelWeights: {kw_np.shape}")
        print(f"  Output: {output_nhwc.shape}")

print("Done!")
