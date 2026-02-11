from urllib.request import urlopen
from PIL import Image
import torch
import timm

img = Image.open(urlopen(
    'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
))

# model = timm.create_model('mobilenetv3_small_100.lamb_in1k', pretrained=True)
model = timm.create_model('timm/resnet18.a1_in1k', pretrained=True)
model = model.eval()

data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)

output = model(transforms(img).unsqueeze(0))

top5_probabilities, top5_class_indices = torch.topk(output.softmax(dim=1) * 100, k=5)

import json
from urllib.request import urlretrieve

labels_url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
labels_path, _ = urlretrieve(labels_url)
with open(labels_path) as f:
    labels = json.load(f)

for prob, idx in zip(top5_probabilities[0], top5_class_indices[0]):
    print(f"{labels[idx.item()]:>30s}: {prob.item():.2f}%")

print("\n--- Model operations ---")
for name, module in model.named_modules():
    if name:
        print(f"{name:60s} {module.__class__.__name__}")

print("\n--- Model parameters ---")
total_params = 0
for name, param in model.named_parameters():
    print(f"{name:60s} {str(list(param.shape)):20s} {param.numel():>10,}")
    total_params += param.numel()
print(f"{'TOTAL':60s} {'':20s} {total_params:>10,}")

print("\n--- Unique layer instantiations ---")
from collections import OrderedDict
unique = OrderedDict()
for name, module in model.named_modules():
    if not name:
        continue
    cls = module.__class__.__name__
    if cls in ('Sequential', 'BasicBlock', 'Bottleneck', 'SelectAdaptivePool2d', 'Identity'):
        continue
    if cls == 'Conv2d':
        key = f"Conv2d(in={module.in_channels}, out={module.out_channels}, k={module.kernel_size}, s={module.stride}, p={module.padding}, bias={module.bias is not None})"
    elif cls == 'BatchNorm2d':
        key = f"BatchNorm2d(features={module.num_features})"
    elif cls == 'Linear':
        key = f"Linear(in={module.in_features}, out={module.out_features}, bias={module.bias is not None})"
    elif cls == 'MaxPool2d':
        key = f"MaxPool2d(k={module.kernel_size}, s={module.stride}, p={module.padding})"
    elif cls == 'AdaptiveAvgPool2d':
        key = f"AdaptiveAvgPool2d(output_size={module.output_size})"
    elif cls == 'Flatten':
        key = f"Flatten(start_dim={module.start_dim}, end_dim={module.end_dim})"
    elif cls == 'ReLU':
        key = f"ReLU(inplace={module.inplace})"
    else:
        key = repr(module).split('\n')[0]
    if key not in unique:
        unique[key] = []
    unique[key].append(name)

for key, names in unique.items():
    print(f"\n  {key}")
    print(f"    count: {len(names)}")
    print(f"    instances: {', '.join(names)}")