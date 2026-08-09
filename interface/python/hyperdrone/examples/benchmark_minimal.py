"""Minimal form of benchmark.py's default run — the configuration of
`rendering_raytracing_sim_benchmark_medium --scene procthor` (render_only, rgb cell):
4096 static cameras @ 64x64, medium shading, fov 80 degrees, all at one eye position
inside the ProcTHOR house, uniform-SO3 orientations (seed 0), 2s untimed warmup, then a
timed async render_launch loop with a sync every 10 iterations for ~10 seconds."""
import math
import time
from pathlib import Path

import numpy as np

from hyperdrone import render

ROOT = Path(__file__).resolve().parents[4]
NUM_CAMERAS = 4096
WIDTH, HEIGHT = 64, 64
FOV = 1.3962634015954636  # 80 degrees, the renderer config default
EYE = np.array([-3.92, -5.67, 1.0])
WARMUP_SECONDS, SECONDS, SYNC_INTERVAL = 2.0, 10.0, 10

scene = render.load_scene(ROOT / "tests" / "data" / "ProcTHOR-Train-1.glb", shading="medium")
renderer = render.Renderer(width=WIDTH, height=HEIGHT, num_cameras=NUM_CAMERAS, num_probes=1,
                           output="rgb", shading="medium")
renderer.init(scene)


def normalize(vectors):
    return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)


rng = np.random.default_rng(0)
u1, u2, u3 = rng.uniform(size=(3, NUM_CAMERAS))  # Shoemake uniform quaternions
qx = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)
qy = np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
qz = np.sqrt(u1) * np.sin(2 * np.pi * u3)
qw = np.sqrt(u1) * np.cos(2 * np.pi * u3)
forwards = np.stack([1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy + qw * qz), 2 * (qx * qz - qw * qy)], axis=-1)
ups = np.stack([2 * (qx * qz + qw * qy), 2 * (qy * qz - qw * qx), 1 - 2 * (qx * qx + qy * qy)], axis=-1)

scale = 2.0 * math.tan(FOV / 2.0)
du = normalize(np.cross(forwards, ups)) * scale
dv = normalize(np.cross(du, forwards)) * scale
dir_00 = forwards - 0.5 * du + 0.5 * dv
positions = np.tile(EYE, (NUM_CAMERAS, 1))
renderer.set_cameras(np.concatenate([positions, dir_00, du, -dv], axis=-1).astype(np.float32))
renderer.generate_probe_directions()

warmup_start = time.perf_counter()
while time.perf_counter() - warmup_start < WARMUP_SECONDS:
    renderer.render()
renderer.synchronize()

wall_start = time.perf_counter()
num_iterations = 0
while True:
    renderer.render_launch()
    num_iterations += 1
    if num_iterations % SYNC_INTERVAL == 0:
        renderer.render_sync()
        if time.perf_counter() - wall_start >= SECONDS:
            break
renderer.synchronize()
wall_seconds = time.perf_counter() - wall_start

total_frames = num_iterations * NUM_CAMERAS
print(f"Backend:         {renderer.backend}")
print(f"Iterations:      {num_iterations}")
print(f"Wall-clock time: {wall_seconds * 1000:.2f} ms")
print(f"Avg per iter:    {wall_seconds * 1000 / num_iterations:.2f} ms")
print(f"Throughput:      {total_frames / wall_seconds:.1f} frames/sec")
print(f"Total MRays/sec: {total_frames * WIDTH * HEIGHT / 1e6 / wall_seconds:.1f}")
