"""Host cost of posing B articulated placements per frame from CUDA tensors (OptiX): set_pose on a placement batch -> update_launch -> render_launch, synchronizing every SYNC_EVERY frames."""

import json
import os
import sys
import time

import numpy as np

import hyperdrone
from hyperdrone import render, set_verbosity
from hyperdrone import _jit as jit
from hyperdrone.camera import CameraSpec, RGB

sys.path.insert(0, str(jit.source_root() / "tools" / "metra"))
import metra


BATCHES = (64, 1024, 4096)
PROPS = 4
FRAMES = 200
WARMUP = 20
SYNC_EVERY = 10
INPUT_SETS = 4


def quad(size, color):
    vertices = np.array([[0, -size, -size], [0, size, -size], [0, size, size], [0, -size, size]], dtype=np.float32)
    return render.Mesh(vertices, np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32), color=color)


def drone():
    assembly = render.ObjectAssembly()
    body = render.Object(name="body")
    body.add_mesh(quad(0.05, (0.3, 0.3, 0.3)))
    assembly.add_part(body, np.eye(4, dtype=np.float32)[:3])
    for prop_i in range(PROPS):
        prop = render.Object(name=f"prop_{prop_i}")
        prop.add_mesh(quad(0.02, (0.8, 0.2, 0.2)))
        rest = np.eye(4, dtype=np.float32)[:3].copy()
        rest[1:3, 3] = (0.05 if prop_i % 2 else -0.05), (0.05 if prop_i < 2 else -0.05)
        assembly.add_part(prop, rest, parent=0)
        assembly.set_joint(prop_i + 1, "revolute", axis=(1, 0, 0), coordinate=prop_i)
    return assembly


def scene():
    result = render.Scene()
    wall = render.Object(name="wall")
    wall.add_mesh(render.Mesh(np.array([[8, -8, -8], [8, 8, -8], [8, 8, 8], [8, -8, 8]], dtype=np.float32), np.array([[0, 2, 1], [0, 3, 2]], dtype=np.int32), color=(0.5, 0.5, 0.5)))
    result.add_object(wall)
    result.add_light(render.SceneLight.directional(direction=(1.0, 0.0, 0.0), color=(1.0, 1.0, 1.0)))
    return result


def inputs(count, rng):
    bases = np.tile(np.eye(4, dtype=np.float32)[:3].reshape(1, 1, 12), (INPUT_SETS, count, 1))
    bases[..., 3] = 4 + rng.uniform(0, 2, (INPUT_SETS, count))
    bases[..., 7] = rng.uniform(-2, 2, (INPUT_SETS, count))
    bases[..., 11] = rng.uniform(-2, 2, (INPUT_SETS, count))
    coordinates = rng.uniform(0, 2 * np.pi, (INPUT_SETS, count, PROPS)).astype(np.float32)
    return hyperdrone._cuda.upload(bases), hyperdrone._cuda.upload(coordinates)


def measure(renderer, placements, rng):
    batch = renderer.placement_batch(placements)
    bases, coordinates = inputs(len(placements), rng)
    set_pose_seconds = 0.0
    update_seconds = 0.0
    started = None
    for frame in range(WARMUP + FRAMES):
        if frame == WARMUP:
            renderer.synchronize()
            set_pose_seconds = update_seconds = 0.0
            started = time.perf_counter()
        index = frame % INPUT_SETS
        before = time.perf_counter()
        renderer.set_pose(batch, render.Pose(bases.view(index)), coordinates=coordinates.view(index))
        after = time.perf_counter()
        renderer.update_launch()
        set_pose_seconds += after - before
        update_seconds += time.perf_counter() - after
        renderer.render_launch()
        if (frame + 1) % SYNC_EVERY == 0:
            renderer.render_sync()
    renderer.synchronize()
    frame_seconds = (time.perf_counter() - started) / FRAMES
    return {"set_pose_host_us": 1e6 * set_pose_seconds / FRAMES, "update_host_us": 1e6 * update_seconds / FRAMES, "frame_ms": 1e3 * frame_seconds}


def main():
    set_verbosity("warning")
    renderer = render.Renderer(width=32, height=32, batch_size=1, num_probes=0, cameras=(CameraSpec("camera", outputs=(RGB(format="rgba8"),)),), fidelity="low", num_overlays=1, max_overlay_instances=(PROPS + 1) * max(BATCHES), max_overlays_per_camera=1)
    if renderer.backend.lower() != "optix":
        raise SystemExit("benchmark_bulk_pose: device pose input needs the OptiX backend")
    pool = render.AssetPool()
    asset = pool.add_assembly(drone())
    renderer.init(scene(), asset_pool=pool)
    renderer.attach(0, 0)
    camera = renderer.camera((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), fov=90)
    renderer.set_cameras(camera[None])
    rng = np.random.default_rng(0)
    report = {"backend": renderer.backend, "frames": FRAMES, "sync_every": SYNC_EVERY}
    for count in BATCHES:
        placements = [renderer.spawn(0, asset) for _ in range(count)]
        renderer.update()
        report[count] = measure(renderer, placements, rng)
        for placement in placements:
            renderer.despawn(placement)
        renderer.update()
    print(json.dumps(report, indent=2))
    metra_url = os.environ.get("METRA_URL")
    if metra_url:
        if "://" not in metra_url:
            metra_url = "http://" + metra_url
        for count in BATCHES:
            for name, value in report[count].items():
                metra.log(f"hyperdrone/bulk_pose/{renderer.backend.lower()}/{count}/{name}", float(value), url=metra_url)


if __name__ == "__main__":
    main()
