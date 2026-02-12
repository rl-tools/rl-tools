#!/usr/bin/env python3
"""
Minimal ResNet-18 inference using timm/resnet18.a1_in1k on a JPEG image.
Usage: python resnet_inference.py <image.jpg>
"""
import sys
import os
import torch
import numpy as np
from PIL import Image

try:
    import timm
except ImportError:
    print("Install: pip install timm pillow")
    sys.exit(1)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
IMAGE_PATH = os.path.join(REPO_ROOT, "tests", "data", "IMG_8734.jpg")

if len(sys.argv) > 1:
    IMAGE_PATH = sys.argv[1]

# Load model (downloads weights automatically if not cached)
SAFETENSORS_PATH = os.path.join(REPO_ROOT, "tests", "data", "model.safetensors")
if os.path.exists(SAFETENSORS_PATH):
    from safetensors.torch import load_file
    print(f"Loading model from local: {SAFETENSORS_PATH}")
    model = timm.create_model('resnet18', pretrained=False, num_classes=1000)
    model.load_state_dict(load_file(SAFETENSORS_PATH))
else:
    print("Loading pretrained resnet18.a1_in1k from timm hub...")
    model = timm.create_model('resnet18.a1_in1k', pretrained=True, num_classes=1000)
model = model.eval().double()

# Load and preprocess image (matching the C++ pipeline: resize to 224x224, ImageNet normalize)
print(f"Loading image: {IMAGE_PATH}")
img = Image.open(IMAGE_PATH).convert("RGB")
print(f"  Original size: {img.size}")
img = img.resize((224, 224), Image.BILINEAR)
print(f"  Resized to: {img.size}")

img_np = np.array(img, dtype=np.float64) / 255.0  # (H, W, 3) in [0, 1]
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
img_np = (img_np - mean) / std

# Convert to NCHW tensor
input_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).double()

# Run inference
with torch.no_grad():
    logits = model(input_tensor)

logits_np = logits.squeeze().numpy()

# Top-10
indices = np.argsort(logits_np)[::-1][:10]
exp_logits = np.exp(logits_np - logits_np.max())
probs = exp_logits / exp_logits.sum()

print("\nTop-10 predictions:")
print(f"  {'Rank':<6}{'Class':<7}{'Logit':<12}{'Probability'}")
print(f"  {'----':<6}{'-----':<7}{'---------':<12}{'-----------'}")
for k, idx in enumerate(indices):
    print(f"  {k+1:<6}{idx:<7}{logits_np[idx]:<12.5f}{probs[idx]*100:.4f}%")

# All logits
print("\nAll logits:")
for i in range(1000):
    print(f"  class {i}: {logits_np[i]}")
