"""Render the ProcTHOR-Train-1 test scene from an eye-height position inside the house
(four yaw angles) and show the RGB frames and depth maps with matplotlib."""
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from hyperdrone import jit, render

ROOT = Path(os.environ.get("HYPERDRONE_RLTOOLS_ROOT", jit.source_root()))
GLB = ROOT / "tests" / "data" / "ProcTHOR-Train-1.glb"

POSITION = np.array([-0.97, -4.42, 1.5])
YAWS_DEGREES = (150.0, 195.0, 240.0, 285.0)

scene = render.load_scene(GLB, shading="high")
renderer = render.Renderer(width=320, height=240, num_cameras=len(YAWS_DEGREES), output="rgbd", shading="high")
renderer.init(scene)

cameras = []
for yaw_degrees in YAWS_DEGREES:
    yaw = math.radians(yaw_degrees)
    look_at = POSITION + np.array([math.cos(yaw), math.sin(yaw), 0.0])
    cameras.append(renderer.camera(position=POSITION, look_at=look_at, fov=math.radians(80)))
renderer.set_cameras(np.stack(cameras))
renderer.render()

frame = renderer.frame()
depth = renderer.depth()

figure, axes = plt.subplots(2, len(YAWS_DEGREES), figsize=(4 * len(YAWS_DEGREES), 6))
for camera, yaw_degrees in enumerate(YAWS_DEGREES):
    axes[0, camera].imshow(frame[camera, :, :, :3])
    axes[0, camera].set_title(f"yaw {yaw_degrees:.0f}°")
    depth_image = axes[1, camera].imshow(depth[camera], cmap="viridis")
    figure.colorbar(depth_image, ax=axes[1, camera], fraction=0.046)
for axis in axes.flat:
    axis.axis("off")
figure.suptitle(f"ProcTHOR-Train-1 — hyperdrone.render ({renderer.backend} backend)")
figure.tight_layout()
plt.show()
