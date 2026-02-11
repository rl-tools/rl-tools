#!/usr/bin/env python3
"""
Export timm/resnet18.a1_in1k to RLtools HDF5 format with test data.
Generates resnet18_test_data.h5 in tests/data/.
"""
import torch
import torch.nn as nn
import numpy as np
import h5py
import os
import sys

# ============================================================
# Load model from safetensors
# ============================================================
try:
    import timm
    from safetensors.torch import load_file
except ImportError:
    print("Install: pip install timm safetensors h5py")
    sys.exit(1)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
SAFETENSORS_PATH = os.path.join(REPO_ROOT, "tests", "data", "model.safetensors")
OUTPUT_PATH = os.path.join(REPO_ROOT, "tests", "data", "resnet18_test_data.h5")

print(f"Loading model from: {SAFETENSORS_PATH}")
state_dict = load_file(SAFETENSORS_PATH)
model = timm.create_model('resnet18', pretrained=False, num_classes=1000)
model.load_state_dict(state_dict)
model = model.eval().double()  # Use double for numerical precision

# ============================================================
# Helper functions
# ============================================================
def to_nhwc(tensor):
    """Convert NCHW tensor to NHWC numpy array (double)."""
    return tensor.detach().double().permute(0, 2, 3, 1).contiguous().numpy()

def to_numpy(tensor):
    """Convert tensor to numpy array (double)."""
    return tensor.detach().double().numpy()

def save_conv2d_layer(group, conv_weight, conv_bias, bn_weight, bn_bias, bn_running_mean, bn_running_var):
    """Save a Conv2d + BN layer in RLtools format."""
    group.attrs["type"] = "conv2d"
    # Weights: [OC, IC, KH, KW] - same layout as PyTorch
    wg = group.create_group("weights")
    wg.create_dataset("parameters", data=to_numpy(conv_weight))
    # Biases: always zero (BN absorbs bias)
    bg = group.create_group("biases")
    bias_data = np.zeros(conv_weight.shape[0], dtype=np.float64)
    if conv_bias is not None:
        bias_data = to_numpy(conv_bias)
    bg.create_dataset("parameters", data=bias_data)
    # BN gamma, beta, running stats
    gg = group.create_group("gamma")
    gg.create_dataset("parameters", data=to_numpy(bn_weight))
    btg = group.create_group("beta")
    btg.create_dataset("parameters", data=to_numpy(bn_bias))
    group.create_dataset("running_mean", data=to_numpy(bn_running_mean))
    group.create_dataset("running_var", data=to_numpy(bn_running_var))

def save_resnet_block(group, block):
    """Save a BasicBlock in RLtools resnet_block format."""
    group.attrs["type"] = "resnet_block"
    # Conv1 + BN1
    conv1_group = group.create_group("conv1")
    save_conv2d_layer(conv1_group,
        block.conv1.weight, None,
        block.bn1.weight, block.bn1.bias,
        block.bn1.running_mean, block.bn1.running_var)
    # Conv2 + BN2
    conv2_group = group.create_group("conv2")
    save_conv2d_layer(conv2_group,
        block.conv2.weight, None,
        block.bn2.weight, block.bn2.bias,
        block.bn2.running_mean, block.bn2.running_var)
    # Downsample (if exists)
    if block.downsample is not None:
        ds_group = group.create_group("downsample")
        ds_conv = block.downsample[0]  # Conv2d
        ds_bn = block.downsample[1]    # BatchNorm2d
        save_conv2d_layer(ds_group,
            ds_conv.weight, None,
            ds_bn.weight, ds_bn.bias,
            ds_bn.running_mean, ds_bn.running_var)

# ============================================================
# Generate test data with intermediate activations
# ============================================================
torch.manual_seed(42)
np.random.seed(42)

# Random input in the expected range (ImageNet-normalized)
input_tensor = torch.randn(1, 3, 224, 224, dtype=torch.float64)
x = input_tensor.clone()

# Collect intermediate activations
intermediates = {}

with torch.no_grad():
    # Stem: conv1 + bn1 + relu
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.act1(x)  # ReLU
    intermediates["after_stem"] = to_nhwc(x)

    # MaxPool
    x = model.maxpool(x)
    intermediates["after_maxpool"] = to_nhwc(x)

    # Layer 1 (2 blocks, 64 channels, no downsample)
    for i, block in enumerate(model.layer1):
        x = block(x)
        intermediates[f"after_layer1_block{i}"] = to_nhwc(x)

    # Layer 2 (2 blocks, 128 channels, first has downsample)
    for i, block in enumerate(model.layer2):
        x = block(x)
        intermediates[f"after_layer2_block{i}"] = to_nhwc(x)

    # Layer 3 (2 blocks, 256 channels, first has downsample)
    for i, block in enumerate(model.layer3):
        x = block(x)
        intermediates[f"after_layer3_block{i}"] = to_nhwc(x)

    # Layer 4 (2 blocks, 512 channels, first has downsample)
    for i, block in enumerate(model.layer4):
        x = block(x)
        intermediates[f"after_layer4_block{i}"] = to_nhwc(x)

    # Global average pool
    x = model.global_pool(x)
    intermediates["after_avgpool"] = to_numpy(x)

    # FC
    x = model.fc(x)
    intermediates["output"] = to_numpy(x)

print(f"Input shape: {input_tensor.shape}")
for name, arr in intermediates.items():
    print(f"  {name}: {arr.shape}")

# ============================================================
# Write HDF5 file
# ============================================================
print(f"\nWriting to: {OUTPUT_PATH}")
with h5py.File(OUTPUT_PATH, "w") as f:
    # ---- Model weights in Sequential format ----
    model_group = f.create_group("model")
    model_group.attrs["type"] = "sequential"
    layers_group = model_group.create_group("layers")

    # Layer 0: Stem Conv2d + BN + ReLU
    stem_group = layers_group.create_group("0")
    save_conv2d_layer(stem_group,
        model.conv1.weight, None,
        model.bn1.weight, model.bn1.bias,
        model.bn1.running_mean, model.bn1.running_var)

    # Layer 1: MaxPool2d (no parameters)
    mp_group = layers_group.create_group("1")
    mp_group.attrs["type"] = "max_pool2d"

    # Layers 2-9: ResNet blocks
    block_idx = 2
    for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
        layer_module = getattr(model, layer_name)
        for block in layer_module:
            block_group = layers_group.create_group(str(block_idx))
            save_resnet_block(block_group, block)
            block_idx += 1

    # Layer 10: AvgPool2d (no parameters)
    avg_group = layers_group.create_group("10")
    avg_group.attrs["type"] = "avg_pool2d"

    # Layer 11: Dense (FC)
    fc_group = layers_group.create_group("11")
    fc_group.attrs["type"] = "dense"
    wg = fc_group.create_group("weights")
    wg.create_dataset("parameters", data=to_numpy(model.fc.weight))
    bg = fc_group.create_group("biases")
    bg.create_dataset("parameters", data=to_numpy(model.fc.bias))

    # ---- Test data ----
    test_group = f.create_group("test_data")
    test_group.create_dataset("input", data=to_nhwc(input_tensor))
    for name, arr in intermediates.items():
        test_group.create_dataset(name, data=arr)

print("Done! HDF5 file written successfully.")
print(f"\nSequential layer mapping:")
print(f"  0: stem_conv (7x7, BN+ReLU)")
print(f"  1: maxpool (3x3, stride=2, pad=1)")
print(f"  2: layer1.0 (BasicBlock, 64ch, stride=1)")
print(f"  3: layer1.1 (BasicBlock, 64ch, stride=1)")
print(f"  4: layer2.0 (BasicBlock, 128ch, stride=2, downsample)")
print(f"  5: layer2.1 (BasicBlock, 128ch, stride=1)")
print(f"  6: layer3.0 (BasicBlock, 256ch, stride=2, downsample)")
print(f"  7: layer3.1 (BasicBlock, 256ch, stride=1)")
print(f"  8: layer4.0 (BasicBlock, 512ch, stride=2, downsample)")
print(f"  9: layer4.1 (BasicBlock, 512ch, stride=1)")
print(f" 10: global_avgpool")
print(f" 11: fc (Dense, 512->1000)")
