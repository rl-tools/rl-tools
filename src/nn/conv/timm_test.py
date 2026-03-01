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
SAFETENSORS_PATH = os.path.join(REPO_ROOT, "tests", "data", "timm_resnet18.a1_in1k")
OUTPUT_PATH = os.path.join(REPO_ROOT, "tests", "data", "resnet18_test_data.h5")
OUTPUT_PATH_BF16 = os.path.join(REPO_ROOT, "tests", "data", "resnet18_test_data_bf16.h5")

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

def conv_weight_to_nhwc(weight):
    """Convert PyTorch conv weight [OC, IC, KH, KW] to NHWC filter layout [OC, KH, KW, IC]."""
    return weight.detach().double().permute(0, 2, 3, 1).contiguous().numpy()

def save_conv2d_layer(group, conv_weight, conv_bias, bn_weight, bn_bias, bn_running_mean, bn_running_var):
    """Save a Conv2d + BN layer in RLtools format."""
    group.attrs["type"] = "conv2d"
    # Weights: [OC, KH, KW, IC] - NHWC filter layout
    wg = group.create_group("weights")
    wg.create_dataset("parameters", data=conv_weight_to_nhwc(conv_weight))
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
# Generate test data with intermediate activations (forward)
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
# Generate backward pass test data
# ============================================================
print("\nComputing backward pass...")
model.zero_grad()
input_grad = input_tensor.clone().requires_grad_(True)
output = model(input_grad)
# Use sum as the loss so d_output = ones
loss = output.sum()
loss.backward()

gradient_data = {}
# d_input
gradient_data["d_input"] = to_nhwc(input_grad.grad)
# d_output (ones, same shape as output)
gradient_data["d_output"] = to_numpy(torch.ones_like(output))

def save_conv2d_gradients(grad_dict, conv, bn, prefix):
    """Save gradients for a Conv2d+BN layer into a dict."""
    # d_weights: [OC, KH, KW, IC] (NHWC filter layout)
    grad_dict[f"{prefix}_d_weights"] = conv_weight_to_nhwc(conv.weight.grad)
    # d_biases: Conv has no bias when followed by BN, so zeros
    grad_dict[f"{prefix}_d_biases"] = np.zeros(conv.weight.shape[0], dtype=np.float64)
    # d_gamma, d_beta
    grad_dict[f"{prefix}_d_gamma"] = to_numpy(bn.weight.grad)
    grad_dict[f"{prefix}_d_beta"] = to_numpy(bn.bias.grad)

# Stem gradients
save_conv2d_gradients(gradient_data, model.conv1, model.bn1, "stem")

# ResNet block gradients
block_map = [
    ("layer1", 0), ("layer1", 1),
    ("layer2", 0), ("layer2", 1),
    ("layer3", 0), ("layer3", 1),
    ("layer4", 0), ("layer4", 1),
]
for layer_name, block_idx in block_map:
    layer_module = getattr(model, layer_name)
    block = layer_module[block_idx]
    prefix = f"{layer_name}_block{block_idx}"
    save_conv2d_gradients(gradient_data, block.conv1, block.bn1, f"{prefix}_conv1")
    save_conv2d_gradients(gradient_data, block.conv2, block.bn2, f"{prefix}_conv2")
    if block.downsample is not None:
        save_conv2d_gradients(gradient_data, block.downsample[0], block.downsample[1], f"{prefix}_downsample")

# FC gradients
gradient_data["fc_d_weights"] = to_numpy(model.fc.weight.grad)
gradient_data["fc_d_biases"] = to_numpy(model.fc.bias.grad)

print("Gradient data collected:")
for name, arr in gradient_data.items():
    if isinstance(arr, np.ndarray):
        print(f"  {name}: {arr.shape}")
    else:
        print(f"  {name}: dict")

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

    # ---- Test data (forward) ----
    test_group = f.create_group("test_data")
    test_group.create_dataset("input", data=to_nhwc(input_tensor))
    for name, arr in intermediates.items():
        test_group.create_dataset(name, data=arr)

    # ---- Gradient data (backward) ----
    grad_group = f.create_group("gradient_data")
    for name, data in gradient_data.items():
        if isinstance(data, np.ndarray):
            grad_group.create_dataset(name, data=data)

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

# ============================================================
# BF16 checkpoint: mixed precision matching RLtools TYPE_POLICY_BF16 (on CUDA)
# ============================================================
# Matches RLtools behavior exactly:
#   - Conv weights in bf16 (Parameter category)
#   - BN gamma/beta in fp32 (NormParameter category)
#   - BN running stats in fp32 (NormStatistics category)
#   - BN forward computes in fp32, output materialized as bf16 (Activation category)
#   - Activations/intermediates in bf16

if not torch.cuda.is_available():
    print("\nSkipping BF16 checkpoint: CUDA not available (bf16 backward requires GPU)")
    sys.exit(0)

print(f"\n{'='*60}")
print("Generating BF16 checkpoint (RLtools-matching, on CUDA)...")
print(f"{'='*60}")

device_cuda = torch.device("cuda")

def bf16_round_f32_numpy(tensor):
    """Round tensor through bf16 then to float32 numpy."""
    return tensor.detach().cpu().float().to(torch.bfloat16).float().numpy()

def bf16_to_f32_nhwc(tensor):
    """BF16 NCHW tensor to float32 NHWC numpy."""
    return tensor.detach().cpu().float().permute(0, 2, 3, 1).contiguous().numpy()

def conv_weight_to_nhwc_bf16(weight):
    """Conv weight [OC, IC, KH, KW] through bf16 to NHWC float32."""
    return weight.detach().cpu().float().to(torch.bfloat16).float().permute(0, 2, 3, 1).contiguous().numpy()

def f32_numpy(tensor):
    """Tensor to float32 numpy (no bf16 rounding)."""
    return tensor.detach().cpu().float().numpy()

# Reload model fresh in fp32, move to CUDA
# BN params (gamma, beta, running_mean, running_var) stay fp32.
# Conv weights are cast to bf16 at each forward call.
model_bf16 = timm.create_model('resnet18', pretrained=False, num_classes=1000)
model_bf16.load_state_dict(state_dict)
model_bf16 = model_bf16.eval().to(device_cuda)

def conv_bn_bf16(x, conv, bn, relu=True):
    """Conv(bf16) + BN(fp32) + optional ReLU, materializing bf16 after BN.
    Matches RLtools: conv in bf16, BN accumulates in fp32, output stored as bf16."""
    w = conv.weight.to(torch.bfloat16)
    x = torch.nn.functional.conv2d(x, w, bias=None, stride=conv.stride, padding=conv.padding)
    # Add zero bias in bf16 (RLtools always has a bias tensor, initialized to 0)
    # BN in fp32: promote input, compute with fp32 gamma/beta/stats
    x = bn(x.float()).to(torch.bfloat16)
    if relu:
        x = torch.nn.functional.relu(x)
    return x

def resnet_block_bf16(x, block):
    """BasicBlock forward matching RLtools: bf16 intermediates, fp32 BN."""
    identity = x
    out = conv_bn_bf16(x, block.conv1, block.bn1, relu=True)
    out = conv_bn_bf16(out, block.conv2, block.bn2, relu=False)
    if block.downsample is not None:
        identity = conv_bn_bf16(x, block.downsample[0], block.downsample[1], relu=False)
    out = torch.nn.functional.relu(out + identity)
    return out

def save_conv2d_layer_bf16(group, conv_weight, conv_bias, bn_weight, bn_bias, bn_running_mean, bn_running_var):
    """Save Conv2d + BN layer: conv weights bf16-rounded, BN params fp32."""
    group.attrs["type"] = "conv2d"
    wg = group.create_group("weights")
    wg.create_dataset("parameters", data=conv_weight_to_nhwc_bf16(conv_weight))
    bg = group.create_group("biases")
    bias_data = np.zeros(conv_weight.shape[0], dtype=np.float32)
    if conv_bias is not None:
        bias_data = bf16_round_f32_numpy(conv_bias)
    bg.create_dataset("parameters", data=bias_data)
    gg = group.create_group("gamma")
    gg.create_dataset("parameters", data=f32_numpy(bn_weight))
    btg = group.create_group("beta")
    btg.create_dataset("parameters", data=f32_numpy(bn_bias))
    group.create_dataset("running_mean", data=f32_numpy(bn_running_mean))
    group.create_dataset("running_var", data=f32_numpy(bn_running_var))

def save_resnet_block_bf16(group, block):
    """Save a BasicBlock in RLtools resnet_block format."""
    group.attrs["type"] = "resnet_block"
    conv1_group = group.create_group("conv1")
    save_conv2d_layer_bf16(conv1_group,
        block.conv1.weight, None,
        block.bn1.weight, block.bn1.bias,
        block.bn1.running_mean, block.bn1.running_var)
    conv2_group = group.create_group("conv2")
    save_conv2d_layer_bf16(conv2_group,
        block.conv2.weight, None,
        block.bn2.weight, block.bn2.bias,
        block.bn2.running_mean, block.bn2.running_var)
    if block.downsample is not None:
        ds_group = group.create_group("downsample")
        ds_conv = block.downsample[0]
        ds_bn = block.downsample[1]
        save_conv2d_layer_bf16(ds_group,
            ds_conv.weight, None,
            ds_bn.weight, ds_bn.bias,
            ds_bn.running_mean, ds_bn.running_var)

# Forward pass: manual layer-by-layer with bf16 materialization
torch.manual_seed(42)
np.random.seed(42)
input_bf16 = torch.randn(1, 3, 224, 224, dtype=torch.float32).to(torch.bfloat16).to(device_cuda)
x = input_bf16.clone()

intermediates_bf16 = {}
with torch.no_grad():
    x = conv_bn_bf16(x, model_bf16.conv1, model_bf16.bn1, relu=True)
    intermediates_bf16["after_stem"] = bf16_to_f32_nhwc(x)

    x = torch.nn.functional.max_pool2d(x, kernel_size=3, stride=2, padding=1)
    intermediates_bf16["after_maxpool"] = bf16_to_f32_nhwc(x)

    for i, block in enumerate(model_bf16.layer1):
        x = resnet_block_bf16(x, block)
        intermediates_bf16[f"after_layer1_block{i}"] = bf16_to_f32_nhwc(x)
    for i, block in enumerate(model_bf16.layer2):
        x = resnet_block_bf16(x, block)
        intermediates_bf16[f"after_layer2_block{i}"] = bf16_to_f32_nhwc(x)
    for i, block in enumerate(model_bf16.layer3):
        x = resnet_block_bf16(x, block)
        intermediates_bf16[f"after_layer3_block{i}"] = bf16_to_f32_nhwc(x)
    for i, block in enumerate(model_bf16.layer4):
        x = resnet_block_bf16(x, block)
        intermediates_bf16[f"after_layer4_block{i}"] = bf16_to_f32_nhwc(x)

    x = model_bf16.global_pool(x)
    intermediates_bf16["after_avgpool"] = f32_numpy(x.float())
    w_fc = model_bf16.fc.weight.to(torch.bfloat16)
    b_fc = model_bf16.fc.bias.to(torch.bfloat16)
    x = torch.nn.functional.linear(x, w_fc, b_fc)
    intermediates_bf16["output"] = bf16_round_f32_numpy(x)

print(f"BF16 input shape: {input_bf16.shape}")
for name, arr in intermediates_bf16.items():
    print(f"  {name}: {arr.shape}")

# Backward pass: same manual forward with grad tracking
print("\nComputing BF16 backward pass...")
model_bf16.zero_grad()
x = input_bf16.clone().requires_grad_(True)
input_grad_bf16 = x

x = conv_bn_bf16(x, model_bf16.conv1, model_bf16.bn1, relu=True)
x = torch.nn.functional.max_pool2d(x, kernel_size=3, stride=2, padding=1)
for block in model_bf16.layer1:
    x = resnet_block_bf16(x, block)
for block in model_bf16.layer2:
    x = resnet_block_bf16(x, block)
for block in model_bf16.layer3:
    x = resnet_block_bf16(x, block)
for block in model_bf16.layer4:
    x = resnet_block_bf16(x, block)
x = model_bf16.global_pool(x)
w_fc = model_bf16.fc.weight.to(torch.bfloat16)
b_fc = model_bf16.fc.bias.to(torch.bfloat16)
output_bf16 = torch.nn.functional.linear(x, w_fc, b_fc)
loss_bf16 = output_bf16.sum()
loss_bf16.backward()

gradient_data_bf16 = {}
gradient_data_bf16["d_input"] = bf16_to_f32_nhwc(input_grad_bf16.grad)
gradient_data_bf16["d_output"] = np.ones_like(bf16_round_f32_numpy(output_bf16))

def save_conv2d_gradients_bf16(grad_dict, conv, bn, prefix):
    grad_dict[f"{prefix}_d_weights"] = conv_weight_to_nhwc_bf16(conv.weight.grad)
    grad_dict[f"{prefix}_d_biases"] = np.zeros(conv.weight.shape[0], dtype=np.float32)
    grad_dict[f"{prefix}_d_gamma"] = f32_numpy(bn.weight.grad)
    grad_dict[f"{prefix}_d_beta"] = f32_numpy(bn.bias.grad)

save_conv2d_gradients_bf16(gradient_data_bf16, model_bf16.conv1, model_bf16.bn1, "stem")
for layer_name, block_idx in block_map:
    layer_module = getattr(model_bf16, layer_name)
    block = layer_module[block_idx]
    prefix = f"{layer_name}_block{block_idx}"
    save_conv2d_gradients_bf16(gradient_data_bf16, block.conv1, block.bn1, f"{prefix}_conv1")
    save_conv2d_gradients_bf16(gradient_data_bf16, block.conv2, block.bn2, f"{prefix}_conv2")
    if block.downsample is not None:
        save_conv2d_gradients_bf16(gradient_data_bf16, block.downsample[0], block.downsample[1], f"{prefix}_downsample")
gradient_data_bf16["fc_d_weights"] = bf16_round_f32_numpy(model_bf16.fc.weight.grad)
gradient_data_bf16["fc_d_biases"] = bf16_round_f32_numpy(model_bf16.fc.bias.grad)

print("BF16 gradient data collected:")
for name, arr in gradient_data_bf16.items():
    print(f"  {name}: {arr.shape}")

# Write BF16 HDF5 file
print(f"\nWriting BF16 checkpoint to: {OUTPUT_PATH_BF16}")
with h5py.File(OUTPUT_PATH_BF16, "w") as f:
    model_group = f.create_group("model")
    model_group.attrs["type"] = "sequential"
    layers_group = model_group.create_group("layers")

    stem_group = layers_group.create_group("0")
    save_conv2d_layer_bf16(stem_group,
        model_bf16.conv1.weight, None,
        model_bf16.bn1.weight, model_bf16.bn1.bias,
        model_bf16.bn1.running_mean, model_bf16.bn1.running_var)

    mp_group = layers_group.create_group("1")
    mp_group.attrs["type"] = "max_pool2d"

    block_idx = 2
    for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
        layer_module = getattr(model_bf16, layer_name)
        for block in layer_module:
            block_group = layers_group.create_group(str(block_idx))
            save_resnet_block_bf16(block_group, block)
            block_idx += 1

    avg_group = layers_group.create_group("10")
    avg_group.attrs["type"] = "avg_pool2d"

    fc_group = layers_group.create_group("11")
    fc_group.attrs["type"] = "dense"
    wg = fc_group.create_group("weights")
    wg.create_dataset("parameters", data=bf16_round_f32_numpy(model_bf16.fc.weight))
    bg = fc_group.create_group("biases")
    bg.create_dataset("parameters", data=bf16_round_f32_numpy(model_bf16.fc.bias))

    # Test data (AMP-style forward)
    test_group = f.create_group("test_data")
    test_group.create_dataset("input", data=bf16_to_f32_nhwc(input_bf16))
    for name, arr in intermediates_bf16.items():
        test_group.create_dataset(name, data=arr)

    # Gradient data (AMP-style backward)
    grad_group = f.create_group("gradient_data")
    for name, data in gradient_data_bf16.items():
        if isinstance(data, np.ndarray):
            grad_group.create_dataset(name, data=data)

print("Done! BF16 HDF5 file written successfully.")
