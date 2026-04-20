import sys
import h5py
import matplotlib.pyplot as plt

H, W, C = 50, 80, 24
FRAMES = 5
TITLES = [f't-{i}' for i in range(FRAMES)] + ['target']

with h5py.File(sys.argv[1], 'r') as f:
    x = f['example/inputs/0'][:]

n = x.shape[0]
img = x.reshape(n, H, W, C)

fig, ax = plt.subplots(n, FRAMES + 1, figsize=(9, n * 1.5))
for i in range(n):
    for j in range(FRAMES + 1):
        ax[i, j].imshow(img[i, :, :, j * 3:j * 3 + 3])
        ax[i, j].axis('off')
        if i == 0:
            ax[i, j].set_title(TITLES[j], fontsize=8)
plt.tight_layout()
plt.show()
