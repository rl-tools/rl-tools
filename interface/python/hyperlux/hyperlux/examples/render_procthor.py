"""Render ProcTHOR-Train-1 from the Train200 catalog at an eye-height position inside the house
(four yaw angles) and show the RGB frames and depth maps with matplotlib."""
import math

import matplotlib.pyplot as plt
import numpy as np

import hyperlux
from hyperlux.camera import CameraSpec, Depth, RGB
dataset = hyperlux.datasets.procthor.Train200()
scene_reference = dataset["ProcTHOR-Train-1"].reference

POSITION = np.array([-0.97, -4.42, 1.5])
YAWS_DEGREES = (150.0, 195.0, 240.0, 285.0)

scene = hyperlux.load_scene(scene_reference, fidelity="high")
renderer = hyperlux.HyperLux(width=320, height=240, batch_size=len(YAWS_DEGREES), cameras=(CameraSpec("front", (RGB("rgba8"), Depth())),), fidelity="high")
renderer.init(scene)

cameras = []
for yaw_degrees in YAWS_DEGREES:
    yaw = math.radians(yaw_degrees)
    look_at = POSITION + np.array([math.cos(yaw), math.sin(yaw), 0.0])
    cameras.append(renderer.camera(position=POSITION, look_at=look_at, fov=80.0))
renderer.set_cameras(np.stack(cameras))
renderer.render()

frame = renderer.outputs[RGB("rgba8")].read()
depth = renderer.outputs[Depth()].read()

figure, axes = plt.subplots(2, len(YAWS_DEGREES), figsize=(4 * len(YAWS_DEGREES), 6))
for camera, yaw_degrees in enumerate(YAWS_DEGREES):
    axes[0, camera].imshow(frame[camera, :, :, :3])
    axes[0, camera].set_title(f"yaw {yaw_degrees:.0f}°")
    depth_image = axes[1, camera].imshow(depth[camera], cmap="viridis")
    figure.colorbar(depth_image, ax=axes[1, camera], fraction=0.046)
for axis in axes.flat:
    axis.axis("off")
figure.suptitle(f"ProcTHOR-Train-1 — hyperlux.render ({renderer.backend} backend)")
figure.tight_layout()
plt.show()
