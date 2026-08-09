"""Python-API counterpart of the simulator-matrix raytracing benchmark
(src/rendering/raytracing/benchmark/simulator_matrix.cpp), matching the configuration of
`rendering_raytracing_sim_benchmark_medium --scene procthor` (render_only cell):

  4096 cameras @ 64x64, medium shading, RGB output, 1 probe, AA/MB off, fov 80 degrees,
  all cameras at the fixed FLU eye position [-3.92, -5.67, 1.0] inside the ProcTHOR
  house, per-camera orientations sampled once from a seeded distribution
  (uniform_so3 default), 2s untimed warmup, then a timed async render_launch loop with a
  sync every 10 iterations for ~10 seconds.

Orientation sampling uses the same distributions as the C++ benchmark (Shoemake uniform
quaternions / yaw-pitch / look-at jitter) with numpy's RNG, so the workload is
statistically identical but not orientation-for-orientation bit-equal to the C++ run.

--camera-input additionally exercises hyperdrone's per-iteration camera-input paths
(host upload, DLPack device hand-over, direct device indexing); the default (static)
matches the C++ render_only protocol.
"""
import argparse
import math
import os
import time
from pathlib import Path

import numpy as np

import hyperdrone
from hyperdrone import render

ROOT = Path(os.environ.get("HYPERDRONE_RLTOOLS_ROOT", Path(__file__).resolve().parents[4]))

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", default=ROOT / "tests" / "data" / "ProcTHOR-Train-1.glb")
parser.add_argument("--width", type=int, default=64)
parser.add_argument("--height", type=int, default=64)
parser.add_argument("--cameras", type=int, default=4096)
parser.add_argument("--output", choices=("rgb", "depth"), default="rgb")
parser.add_argument("--shading", default="medium")
parser.add_argument("--orientation-mode", choices=("uniform_so3", "random_yaw_pitch", "look_at_scene_jitter"), default="uniform_so3")
parser.add_argument("--seconds", type=float, default=10.0)
parser.add_argument("--iterations", type=int, default=0, help="fixed timed iterations; overrides --seconds when > 0")
parser.add_argument("--warmup-seconds", type=float, default=2.0)
parser.add_argument("--sync-interval", type=int, default=10)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--camera-input", choices=("static", "host", "dlpack", "direct"), default="static")
parser.add_argument("--camera-sets", type=int, default=100)
arguments = parser.parse_args()

WIDTH, HEIGHT = arguments.width, arguments.height
NUM_CAMERAS = arguments.cameras
FOV = 1.3962634015954636  # matches the renderer config default (80 degrees)
ASPECT = WIDTH / HEIGHT
PROCTHOR_EYE = np.array([-3.92, -5.67, 1.0])  # camera_offset(procthor) in the C++ benchmark


def build_procedural_scene(rng, num_pillars=64, half=10.0, height=4.0):
    scene = render.Scene()
    room = render.Object(name="room")

    def quad(v0, v1, v2, v3, color):
        vertices = np.array([v0, v1, v2, v3], dtype=np.float32)
        indices = np.array([[0, 1, 2], [0, 2, 3], [0, 2, 1], [0, 3, 2]], dtype=np.int32)
        room.add_mesh(render.Mesh(vertices, indices, color=color))

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
    scene.add_light(render.SceneLight.directional(direction=(0.3, 0.2, -1.0), color=(1.0, 1.0, 1.0)))
    scene.add_light(render.SceneLight.directional(direction=(-0.5, 0.4, -0.3), color=(0.4, 0.4, 0.5)))
    return scene


def normalize(vectors):
    return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)


def sample_orientations(rng, count, mode, look_at_forward):
    """Per-camera (forward, up) pairs; same distributions as make_camera_poses in the
    C++ simulator-matrix benchmark."""
    world_up = np.array([0.0, 0.0, 1.0])
    if mode == "uniform_so3":
        u1, u2, u3 = rng.uniform(size=(3, count))
        qx = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)
        qy = np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
        qz = np.sqrt(u1) * np.sin(2 * np.pi * u3)
        qw = np.sqrt(u1) * np.cos(2 * np.pi * u3)
        rotation = np.empty((count, 3, 3))
        rotation[:, 0, 0] = 1 - 2 * (qy * qy + qz * qz)
        rotation[:, 0, 1] = 2 * (qx * qy - qw * qz)
        rotation[:, 0, 2] = 2 * (qx * qz + qw * qy)
        rotation[:, 1, 0] = 2 * (qx * qy + qw * qz)
        rotation[:, 1, 1] = 1 - 2 * (qx * qx + qz * qz)
        rotation[:, 1, 2] = 2 * (qy * qz - qw * qx)
        rotation[:, 2, 0] = 2 * (qx * qz - qw * qy)
        rotation[:, 2, 1] = 2 * (qy * qz + qw * qx)
        rotation[:, 2, 2] = 1 - 2 * (qx * qx + qy * qy)
        forwards = rotation[:, :, 0]
        ups = rotation[:, :, 2]
    elif mode == "look_at_scene_jitter":
        yaw_offset = rng.uniform(-0.08, 0.08, size=(count, 1))
        pitch_offset = rng.uniform(-0.06, 0.06, size=(count, 1))
        right = normalize(np.cross(look_at_forward, world_up))
        forwards = normalize(look_at_forward + yaw_offset * right + pitch_offset * world_up)
        ups = np.tile(world_up, (count, 1))
    else:  # random_yaw_pitch
        yaw = rng.uniform(-np.pi, np.pi, size=count)
        pitch = rng.uniform(-0.35, 0.35, size=count)
        forwards = np.stack([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)], axis=-1)
        ups = np.stack([-np.sin(pitch) * np.cos(yaw), -np.sin(pitch) * np.sin(yaw), np.cos(pitch)], axis=-1)
    return forwards, ups


def camera_bases(positions, forwards, ups, fov, aspect):
    """Vectorized make_camera_data: packed (N, 12) bases from per-camera forward/up."""
    scale = 2.0 * math.tan(fov / 2.0)
    du = normalize(np.cross(forwards, ups)) * scale
    dv = normalize(np.cross(du, forwards)) * (scale / aspect)
    dir_00 = forwards - 0.5 * du + 0.5 * dv
    return np.concatenate([positions, dir_00, du, -dv], axis=-1).astype(np.float32)


scene_source = str(arguments.model)
if Path(arguments.model).exists():
    scene = render.load_scene(arguments.model, shading=arguments.shading)
    eye = PROCTHOR_EYE
else:
    scene_source = "procedural room (GLB not found)"
    scene = build_procedural_scene(np.random.default_rng(0))
    eye = None

renderer = render.Renderer(width=WIDTH, height=HEIGHT, num_cameras=NUM_CAMERAS, num_probes=1,
                           output=arguments.output, shading=arguments.shading)
renderer.init(scene)
if eye is None:
    eye = np.asarray(renderer.scene_bounds["center"], dtype=np.float64)

rng = np.random.default_rng(arguments.seed)
positions = np.tile(eye, (NUM_CAMERAS, 1))
look_at_forward = normalize(np.asarray(renderer.scene_bounds["center"], dtype=np.float64) - eye) \
    if not np.allclose(np.asarray(renderer.scene_bounds["center"], dtype=np.float64), eye) else np.array([1.0, 0.0, 0.0])
forwards, ups = sample_orientations(rng, NUM_CAMERAS, arguments.orientation_mode, look_at_forward)
renderer.set_cameras(camera_bases(positions, forwards, ups, FOV, ASPECT))
renderer.generate_probe_directions()

set_cameras_for_iteration = lambda iteration: None
if arguments.camera_input != "static":
    all_sets = np.stack([
        camera_bases(positions, *sample_orientations(rng, NUM_CAMERAS, arguments.orientation_mode, look_at_forward), FOV, ASPECT)
        for _ in range(arguments.camera_sets)
    ])  # (sets, cameras, 12)
    num_sets = arguments.camera_sets

    if arguments.camera_input == "host":
        host_sets = [np.ascontiguousarray(all_sets[s]) for s in range(num_sets)]
        set_host = renderer.set_cameras
        set_cameras_for_iteration = lambda iteration: set_host(host_sets[iteration % num_sets])
    else:
        if renderer.backend != "optix":
            raise SystemExit(f"--camera-input {arguments.camera_input} requires the OptiX backend (got {renderer.backend})")
        tensor_set = hyperdrone.cuda.upload(all_sets)
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
        host_frame = renderer.frame_raw() if arguments.output == "rgb" else renderer.depth()
        set_cameras_for_iteration(0)
        renderer.render()
        renderer.render_sync()
        check_frame = renderer.frame_raw() if arguments.output == "rgb" else renderer.depth()
        assert np.array_equal(host_frame, check_frame), "device camera path diverged from host path"

# untimed warmup: full synchronous renders, matching the C++ protocol
warmup_start = time.perf_counter()
warmup_iterations = 0
while time.perf_counter() - warmup_start < arguments.warmup_seconds or warmup_iterations == 0:
    set_cameras_for_iteration(warmup_iterations)
    renderer.render()
    warmup_iterations += 1
renderer.synchronize()

wall_start = time.perf_counter()
num_iterations = 0
if arguments.iterations > 0:
    for num_iterations in range(1, arguments.iterations + 1):
        set_cameras_for_iteration(num_iterations - 1)
        renderer.render_launch()
        if num_iterations % arguments.sync_interval == 0:
            renderer.render_sync()
else:
    while True:
        set_cameras_for_iteration(num_iterations)
        renderer.render_launch()
        num_iterations += 1
        if num_iterations % arguments.sync_interval == 0:
            renderer.render_sync()
            if time.perf_counter() - wall_start >= arguments.seconds:
                break
if num_iterations % arguments.sync_interval != 0:
    renderer.render_sync()
renderer.synchronize()
wall_seconds = time.perf_counter() - wall_start

total_frames = num_iterations * NUM_CAMERAS
total_pixels = total_frames * WIDTH * HEIGHT
total_mrays = total_pixels / 1e6  # 1 sample/pixel: AA and motion blur are off

print("=== BENCHMARK RESULTS (simulator-matrix config, python) ===")
print(f"  Backend:             {renderer.backend}")
print(f"  Scene:               {scene_source}")
print(f"  Output:              {arguments.output}")
print(f"  Shading:             {arguments.shading}")
print(f"  Cameras per batch:   {NUM_CAMERAS}")
print(f"  Resolution per cam:  {WIDTH}x{HEIGHT}")
print(f"  FOV:                 {math.degrees(FOV):.1f} deg")
print(f"  Camera eye (FLU):    [{eye[0]:.2f}, {eye[1]:.2f}, {eye[2]:.2f}]")
print(f"  Orientation mode:    {arguments.orientation_mode} (seed {arguments.seed})")
print(f"  Camera input:        {arguments.camera_input}" + ("" if arguments.camera_input == "static" else f" ({arguments.camera_sets} orientation sets)"))
print(f"  Warmup:              {warmup_iterations} iterations ({arguments.warmup_seconds:.1f}s untimed)")
print(f"  Batch iterations:    {num_iterations}")
print(f"  Total frames:        {total_frames}")
print(f"  Wall-clock time:     {wall_seconds * 1000:.2f} ms")
print(f"  Avg per iter:        {wall_seconds * 1000 / num_iterations:.2f} ms")
print(f"  Avg per frame:       {wall_seconds * 1000 / total_frames:.4f} ms")
print(f"  Throughput:          {total_frames / wall_seconds:.1f} frames/sec")
print(f"  Pixels/sec:          {total_pixels / wall_seconds:.3e}")
print(f"  Total MRays/sec:     {total_mrays / wall_seconds:.1f}")
print("===========================================================")
