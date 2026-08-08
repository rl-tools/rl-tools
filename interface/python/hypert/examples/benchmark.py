"""Python-API counterpart of src/rendering/raytracing/benchmark/benchmark.cpp: same
renderer constants (128x128, 4096 cameras, 64 probes, RGB, medium shading) and the same
async render_launch loop with a sync every 10 iterations for ~10 seconds.

--camera-input selects how per-iteration camera orientations reach the renderer:
  static  no per-iteration update (the original benchmark behavior)
  host    cycle pre-sampled orientation sets from host numpy arrays (H2D upload per iter)
  dlpack  cycle sets pre-uploaded to the GPU, handed over zero-copy via DLPack
  direct  same GPU memory, addressed by set index (no per-iteration DLPack traffic)
The device modes enqueue a device-to-device copy on the render stream: fully async, no
host synchronization, ordered against the launches.
"""
import argparse
import math
import os
import time
from pathlib import Path

import numpy as np

import hypert

ROOT = Path(os.environ.get("HYPERT_RLTOOLS_ROOT", Path(__file__).resolve().parents[4]))

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", default=ROOT / "tests" / "data" / "ProcTHOR-Train-1.glb")
parser.add_argument("--width", type=int, default=128)
parser.add_argument("--height", type=int, default=128)
parser.add_argument("--cameras", type=int, default=4096)
parser.add_argument("--probes", type=int, default=64)
parser.add_argument("--seconds", type=float, default=10.0)
parser.add_argument("--camera-input", choices=("static", "host", "dlpack", "direct"), default="direct")
parser.add_argument("--camera-sets", type=int, default=100)
arguments = parser.parse_args()

WIDTH, HEIGHT = arguments.width, arguments.height
NUM_CAMERAS = arguments.cameras
NUM_PROBES = arguments.probes
FOV = math.radians(80.0)
ASPECT = WIDTH / HEIGHT


def build_procedural_scene(rng, num_pillars=64, half=10.0, height=4.0):
    scene = hypert.Scene()
    room = hypert.Object(name="room")

    def quad(v0, v1, v2, v3, color):
        vertices = np.array([v0, v1, v2, v3], dtype=np.float32)
        indices = np.array([[0, 1, 2], [0, 2, 3], [0, 2, 1], [0, 3, 2]], dtype=np.int32)
        room.add_mesh(hypert.Mesh(vertices, indices, color=color))

    quad((-half, -half, 0), (half, -half, 0), (half, half, 0), (-half, half, 0), (0.5, 0.5, 0.5))
    quad((-half, -half, height), (half, -half, height), (half, half, height), (-half, half, height), (0.9, 0.9, 0.9))
    quad((-half, -half, 0), (half, -half, 0), (half, -half, height), (-half, -half, height), (0.8, 0.3, 0.3))
    quad((-half, half, 0), (half, half, 0), (half, half, height), (-half, half, height), (0.3, 0.8, 0.3))
    quad((-half, -half, 0), (-half, half, 0), (-half, half, height), (-half, -half, height), (0.3, 0.3, 0.8))
    quad((half, -half, 0), (half, half, 0), (half, half, height), (half, -half, height), (0.8, 0.8, 0.3))
    for _ in range(num_pillars):
        x, y = rng.uniform(-half * 0.9, half * 0.9, size=2)
        r = 0.2
        color = tuple(rng.uniform(0.2, 1.0, size=3))
        quad((x - r, y - r, 0), (x + r, y - r, 0), (x + r, y - r, height), (x - r, y - r, height), color)
        quad((x - r, y + r, 0), (x + r, y + r, 0), (x + r, y + r, height), (x - r, y + r, height), color)
        quad((x - r, y - r, 0), (x - r, y + r, 0), (x - r, y + r, height), (x - r, y - r, height), color)
        quad((x + r, y - r, 0), (x + r, y + r, 0), (x + r, y + r, height), (x + r, y - r, height), color)
    scene.add_object(room)
    scene.add_light(hypert.SceneLight.directional(direction=(0.3, 0.2, -1.0), color=(1.0, 1.0, 1.0)))
    scene.add_light(hypert.SceneLight.directional(direction=(-0.5, 0.4, -0.3), color=(0.4, 0.4, 0.5)))
    return scene


def camera_bases(positions, yaws, pitches, fov, aspect):
    """Vectorized equivalent of make_camera_data for forward directions given by yaw/pitch."""
    forward = np.stack(
        [np.cos(yaws) * np.cos(pitches), np.sin(yaws) * np.cos(pitches), np.sin(pitches)],
        axis=-1,
    )
    up = np.array([0.0, 0.0, 1.0])
    scale = 2.0 * math.tan(fov / 2.0)

    def normalize(v):
        return v / np.linalg.norm(v, axis=-1, keepdims=True)

    du = normalize(np.cross(forward, up)) * scale
    dv = normalize(np.cross(du, forward)) * (scale / aspect)
    dir_00 = forward - 0.5 * du + 0.5 * dv
    return np.concatenate([positions, dir_00, du, -dv], axis=-1).astype(np.float32)


scene_source = str(arguments.model)
if Path(arguments.model).exists():
    scene = hypert.load_scene(arguments.model, shading="medium")
else:
    scene_source = "procedural room (GLB not found)"
    scene = build_procedural_scene(np.random.default_rng(0))

renderer = hypert.Renderer(width=WIDTH, height=HEIGHT, num_cameras=NUM_CAMERAS, num_probes=NUM_PROBES, output="rgb", shading="medium")
renderer.init(scene)
renderer.generate_cameras(fov=FOV)
renderer.generate_probe_directions()

set_cameras_for_iteration = lambda iteration: None
if arguments.camera_input != "static":
    rng = np.random.default_rng(1)
    bounds = renderer.scene_bounds
    center, half_extent = np.asarray(bounds["center"]), np.asarray(bounds["half_extent"])
    positions = center + rng.uniform(-0.6, 0.6, size=(NUM_CAMERAS, 3)) * half_extent
    yaws = rng.uniform(0.0, 2.0 * math.pi, size=(arguments.camera_sets, NUM_CAMERAS))
    pitches = rng.uniform(-0.2, 0.2, size=(arguments.camera_sets, NUM_CAMERAS))
    all_sets = np.stack([
        camera_bases(positions, yaws[s], pitches[s], FOV, ASPECT) for s in range(arguments.camera_sets)
    ])  # (sets, cameras, 12)
    num_sets = arguments.camera_sets

    if arguments.camera_input == "host":
        host_sets = [np.ascontiguousarray(all_sets[s]) for s in range(num_sets)]
        set_host = renderer.set_cameras
        set_cameras_for_iteration = lambda iteration: set_host(host_sets[iteration % num_sets])
    else:
        if renderer.backend != "optix":
            raise SystemExit(f"--camera-input {arguments.camera_input} requires the OptiX backend (got {renderer.backend})")
        tensor_set = hypert.cuda_upload(all_sets)
        if arguments.camera_input == "dlpack":
            views = [tensor_set.view(s) for s in range(num_sets)]
            set_device = renderer._renderer.set_cameras_device
            set_cameras_for_iteration = lambda iteration: set_device(views[iteration % num_sets], 0)
        else:
            raw_buffer = tensor_set.raw
            set_from_buffer = renderer._renderer.set_cameras_from_cuda_buffer
            set_cameras_for_iteration = lambda iteration: set_from_buffer(raw_buffer, iteration % num_sets, 0)

        # the device path must be pixel-identical to the host path
        renderer.set_cameras(np.ascontiguousarray(all_sets[0]))
        renderer.render()
        renderer.render_sync()
        host_frame = renderer.frame_raw()
        set_cameras_for_iteration(0)
        renderer.render()
        renderer.render_sync()
        assert np.array_equal(host_frame, renderer.frame_raw()), "device camera path diverged from host path"

renderer.render()
renderer.render_sync()

wall_start = time.perf_counter()
num_iterations = 0
while True:
    set_cameras_for_iteration(num_iterations)
    renderer.render_launch()
    num_iterations += 1
    if num_iterations % 10 == 0:
        renderer.render_sync()
        if time.perf_counter() - wall_start >= arguments.seconds:
            break
renderer.render_sync()
wall_seconds = time.perf_counter() - wall_start

total_frames = num_iterations * NUM_CAMERAS
total_rgb_rays = total_frames * WIDTH * HEIGHT
total_probe_rays = total_frames * NUM_PROBES
total_mrays = (total_rgb_rays + total_probe_rays) / 1e6

print("=== BENCHMARK RESULTS (async, python) ===")
print(f"  Backend:             {renderer.backend}")
print(f"  Scene:               {scene_source}")
print(f"  Camera input:        {arguments.camera_input}" + ("" if arguments.camera_input == "static" else f" ({arguments.camera_sets} orientation sets)"))
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
