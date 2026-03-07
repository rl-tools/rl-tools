"""
Generate test data for Upsample2d (bilinear) layer verification.
Produces an HDF5 file with multiple test cases covering:
  - scale2_small: 2x upscale, small spatial dims
  - scale2_rect: 2x upscale, non-square input
  - scale3: 3x upscale
  - scale4: 4x upscale
  - scale2_multichannel: 2x upscale, many channels

All data is stored in NHWC (batch, height, width, channels) format.
Uses PyTorch's F.interpolate with mode='bilinear', align_corners=False.
"""

import torch
import torch.nn.functional as F
import h5py
import os

torch.manual_seed(42)

test_cases = [
    {
        "name": "scale2_small",
        "channels": 3,
        "height": 4, "width": 4, "batch_size": 2,
        "scale_h": 2, "scale_w": 2,
    },
    {
        "name": "scale2_rect",
        "channels": 8,
        "height": 3, "width": 5, "batch_size": 2,
        "scale_h": 2, "scale_w": 2,
    },
    {
        "name": "scale3",
        "channels": 4,
        "height": 4, "width": 4, "batch_size": 2,
        "scale_h": 3, "scale_w": 3,
    },
    {
        "name": "scale4",
        "channels": 3,
        "height": 3, "width": 3, "batch_size": 2,
        "scale_h": 4, "scale_w": 4,
    },
    {
        "name": "scale2_multichannel",
        "channels": 64,
        "height": 7, "width": 7, "batch_size": 2,
        "scale_h": 2, "scale_w": 2,
    },
]

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.normpath(os.path.join(script_dir, "..", "..", "..", "..", "..", ".."))
output_dir = os.path.join(repo_root, "tests", "data")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "upsample2d_test_data.h5")

print(f"Saving to: {output_path}")

with h5py.File(output_path, "w") as f:
    for tc in test_cases:
        print(f"Generating test case: {tc['name']}")
        g = f.create_group(tc["name"])

        B = tc["batch_size"]
        C = tc["channels"]
        H = tc["height"]
        W = tc["width"]
        SH = tc["scale_h"]
        SW = tc["scale_w"]
        OH = H * SH
        OW = W * SW

        # Random input in NCHW format (PyTorch convention)
        input_nchw = torch.randn(B, C, H, W, dtype=torch.float64, requires_grad=True)

        # Forward: bilinear upsample (align_corners=False)
        output_nchw = F.interpolate(input_nchw, size=(OH, OW), mode='bilinear', align_corners=False)

        # Random upstream gradient
        d_output_nchw = torch.randn_like(output_nchw)

        # Backward pass
        output_nchw.backward(d_output_nchw)

        # Convert NCHW -> NHWC for storage
        input_nhwc = input_nchw.detach().permute(0, 2, 3, 1).contiguous().numpy()
        output_nhwc = output_nchw.detach().permute(0, 2, 3, 1).contiguous().numpy()
        d_output_nhwc = d_output_nchw.detach().permute(0, 2, 3, 1).contiguous().numpy()
        d_input_nhwc = input_nchw.grad.detach().permute(0, 2, 3, 1).contiguous().numpy()

        g.create_dataset("input", data=input_nhwc)
        g.create_dataset("output", data=output_nhwc)
        g.create_dataset("d_output", data=d_output_nhwc)
        g.create_dataset("d_input", data=d_input_nhwc)

        g.attrs["channels"] = C
        g.attrs["height"] = H
        g.attrs["width"] = W
        g.attrs["batch_size"] = B
        g.attrs["scale_h"] = SH
        g.attrs["scale_w"] = SW

        print(f"  Input:  {input_nhwc.shape}")
        print(f"  Output: {output_nhwc.shape}")

print("Done!")
