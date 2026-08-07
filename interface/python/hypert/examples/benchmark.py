"""Python-API counterpart of src/rendering/raytracing/benchmark/benchmark.cpp: same
renderer constants (128x128, 4096 cameras, 64 probes, RGB, medium shading), same
generated cameras, and the same async render_launch loop with a sync every 10
iterations for ~10 seconds."""
import argparse
import math
import os
import time
from pathlib import Path

import hypert

ROOT = Path(os.environ.get("HYPERT_RLTOOLS_ROOT", Path(__file__).resolve().parents[4]))

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", default=ROOT / "tests" / "data" / "ProcTHOR-Train-1.glb")
parser.add_argument("--width", type=int, default=128)
parser.add_argument("--height", type=int, default=128)
arguments = parser.parse_args()

GLB = arguments.model
WIDTH, HEIGHT = arguments.width, arguments.height
NUM_CAMERAS = 4096
NUM_PROBES = 64
FOV = math.radians(80.0)
BENCHMARK_SECONDS = 10.0

scene = hypert.load_scene(GLB, shading="medium")
renderer = hypert.Renderer(width=WIDTH, height=HEIGHT, num_cameras=NUM_CAMERAS, num_probes=NUM_PROBES, output="rgb", shading="medium")
renderer.init(scene)
renderer.generate_cameras(fov=FOV)
renderer.generate_probe_directions()

renderer.render()
renderer.render_sync()

wall_start = time.perf_counter()
num_iterations = 0
while True:
    renderer.render_launch()
    num_iterations += 1
    if num_iterations % 10 == 0:
        renderer.render_sync()
        if time.perf_counter() - wall_start >= BENCHMARK_SECONDS:
            break
renderer.render_sync()
wall_seconds = time.perf_counter() - wall_start

total_frames = num_iterations * NUM_CAMERAS
total_rgb_rays = total_frames * WIDTH * HEIGHT
total_probe_rays = total_frames * NUM_PROBES
total_mrays = (total_rgb_rays + total_probe_rays) / 1e6

print("=== BENCHMARK RESULTS (async, python) ===")
print(f"  Backend:             {renderer.backend}")
print(f"  Cameras per batch:   {NUM_CAMERAS}")
print(f"  Resolution per cam:  {WIDTH}x{HEIGHT}")
print(f"  Probes per camera:   {NUM_PROBES}")
print(f"  Batch iterations:    {num_iterations}")
print(f"  Total frames:        {total_frames}")
print(f"  Wall-clock time:     {wall_seconds * 1000:.2f} ms")
print(f"  Avg per iter:        {wall_seconds * 1000 / num_iterations:.2f} ms")
print(f"  Avg per frame:       {wall_seconds * 1000 / total_frames:.4f} ms")
print(f"  Throughput:          {total_frames / wall_seconds:.1f} frames/sec")
print(f"  Total MRays/sec:     {total_mrays / wall_seconds:.1f}")
print("=========================================")
