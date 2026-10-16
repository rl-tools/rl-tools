"""Compare full and compact camera profiles on a fixed procedural scene; Python allocation metrics exclude native allocations."""

import json
import os
import sys
import time
import tracemalloc
from dataclasses import replace

import numpy as np

from hyperdrone import render, set_verbosity
from hyperdrone import _jit as jit
from hyperdrone.camera import CameraSpec, Depth, RGB
from hyperdrone.render._component import component

sys.path.insert(0, str(jit.source_root() / "tools" / "metra"))
import metra


WIDTH = HEIGHT = 32
BATCH_SIZE = 64
RENDER_ITERATIONS = 100
READ_ITERATIONS = 1000


def measure(scene, profiles):
    configuration = render.RendererConfig(width=WIDTH, height=HEIGHT, cameras=profiles, batch_size=BATCH_SIZE, fidelity="low")
    artifact = jit.build_dir(component()) / "jit" / f"render_{configuration.key()}.so"
    cache_hit = artifact.exists()
    started = time.perf_counter()
    renderer = render.Renderer(WIDTH, HEIGHT, cameras=profiles, batch_size=BATCH_SIZE, fidelity="low")
    jit_seconds = time.perf_counter() - started
    artifact_time = artifact.stat().st_mtime_ns
    renamed_profiles = tuple(replace(camera, name=f"renamed{index}") for index, camera in enumerate(profiles))
    started = time.perf_counter()
    renamed = render.Renderer(WIDTH, HEIGHT, cameras=renamed_profiles, batch_size=BATCH_SIZE, fidelity="low")
    renamed_seconds = time.perf_counter() - started
    assert renamed.config.key() == renderer.config.key()
    assert type(renamed._renderer) is type(renderer._renderer)
    assert artifact.stat().st_mtime_ns == artifact_time
    started = time.perf_counter()
    renderer.init(scene)
    init_seconds = time.perf_counter() - started
    camera = renderer.camera((0, 0, 0), (1, 0, 0), fov=80)
    cameras = np.repeat(camera.reshape(1, 12), 2 * BATCH_SIZE, axis=0)
    cameras[1::2, 0] = 0.25
    renderer.set_cameras(cameras)
    for _ in range(20):
        renderer.render()
    started = time.perf_counter()
    for _ in range(RENDER_ITERATIONS):
        renderer.render_launch()
    renderer.synchronize()
    render_seconds = (time.perf_counter() - started) / RENDER_ITERATIONS
    depth = renderer.outputs[Depth()]
    destination = np.empty(depth.layout.shape, dtype=depth.layout.dtype)
    assert depth.read(out=destination) is destination
    started = time.perf_counter()
    for _ in range(READ_ITERATIONS):
        depth.read(out=destination)
    read_seconds = (time.perf_counter() - started) / READ_ITERATIONS
    tracemalloc.start()
    for _ in range(READ_ITERATIONS):
        depth.read(out=destination)
    python_current, python_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    output_bytes = sum(int(np.prod(output.layout.shape)) * np.dtype(output.layout.dtype).itemsize for output in renderer.outputs.values())
    metrics = {"jit_cache_hit": cache_hit, "jit_request_seconds": jit_seconds, "renamed_request_seconds": renamed_seconds, "init_seconds": init_seconds, "output_bytes": output_bytes, "render_seconds": render_seconds, "read_seconds": read_seconds, "read_bytes": destination.nbytes, "read_bytes_per_second": destination.nbytes / read_seconds, "python_read_retained_bytes": python_current, "python_read_peak_bytes": python_peak}
    return renderer, metrics


def main():
    set_verbosity("warning")
    scene = render.Scene()
    wall = render.Object()
    wall.add_mesh(render.Mesh(np.array([[2, -4, -4], [2, 4, -4], [2, 4, 4], [2, -4, 4]], dtype=np.float32), np.array([[0, 2, 1], [0, 3, 2]], dtype=np.int32), color=(0.3, 0.6, 0.9)))
    scene.add_object(wall)
    full, full_metrics = measure(scene, (CameraSpec("front", (RGB("float32"), Depth())), CameraSpec("rear", (RGB("float32"), Depth()))))
    sparse, sparse_metrics = measure(scene, (CameraSpec("front", (RGB("float32"), Depth())), CameraSpec("rear", (RGB("rgba8"),))))
    for descriptor in (RGB("float32"), Depth()):
        np.testing.assert_array_equal(full.outputs[descriptor].read()[::2], sparse.outputs[descriptor].read())
    report = {"backend": full.backend, "width": WIDTH, "height": HEIGHT, "rigs": BATCH_SIZE, "destination": "cpu", "full": full_metrics, "sparse": sparse_metrics}
    print(json.dumps(report, indent=2))
    metra_url = os.environ.get("METRA_URL")
    if metra_url:
        if "://" not in metra_url:
            metra_url = "http://" + metra_url
        for profile, metrics in (("full", full_metrics), ("sparse", sparse_metrics)):
            for name, value in metrics.items():
                metra.log(f"hyperdrone/camera_profiles/{full.backend.lower()}/{profile}/{name}", float(value), url=metra_url)


if __name__ == "__main__":
    main()
