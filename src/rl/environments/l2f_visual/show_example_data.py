import sys
import re
import json
import h5py
import matplotlib.pyplot as plt

MAX_ROWS = int(sys.argv[2]) if len(sys.argv) > 2 else 8

def split_top(s):
    parts, depth, start = [], 0, 0
    for i, c in enumerate(s):
        if c == '(': depth += 1
        elif c == ')': depth -= 1
        elif c == ',' and depth == 0:
            parts.append(s[start:i].strip()); start = i + 1
    parts.append(s[start:].strip())
    return parts

with h5py.File(sys.argv[1], 'r') as f:
    x = f['example/inputs/0'][:]
    meta = json.loads(f['actor'].attrs['meta'])

obs_0 = split_top(meta['environment']['observation'])[0]
is_target_input = obs_0.startswith('TargetImage(')
inner = obs_0[len('TargetImage('):-1] if is_target_input else obs_0
m = re.match(r'CameraRGB(Stacked)?(WithTarget)?\(([^)]*)\)', inner)
assert m, f'unrecognized camera observation: {inner!r}'
args = [a.strip() for a in m.group(3).split(',')]
n_frames = int(args[4]) if m.group(1) else 1
channel_target = m.group(2) is not None
if is_target_input:
    titles = [f'target t-{i}' for i in range(n_frames)]
elif channel_target:
    titles = [f't-{i}' for i in range(n_frames)] + ['target']
else:
    titles = [f't-{i}' for i in range(n_frames)]
n_slots = len(titles)

assert x.ndim == 5, f'expected [TIME_STEP, BATCH, H, W, C], got shape {x.shape}'
assert x.shape[0] == 1, f'expected TIME_STEP=1, got {x.shape[0]}'
img = x[0]
n = min(img.shape[0], MAX_ROWS)
img = img[:n]

fig, ax = plt.subplots(n, n_slots, figsize=(1.5 * n_slots, n * 1.5), squeeze=False)
for i in range(n):
    for j in range(n_slots):
        ax[i, j].imshow(img[i, :, :, j * 3:j * 3 + 3])
        ax[i, j].axis('off')
        if i == 0:
            ax[i, j].set_title(titles[j], fontsize=8)
plt.tight_layout()
plt.show()
