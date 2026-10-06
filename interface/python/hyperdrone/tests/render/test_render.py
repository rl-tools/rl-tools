import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import hyperdrone
from hyperdrone import jit, render


def test_backend_announcement_reaches_python_for_each_renderer():
    script = """
from hyperdrone import render

first = render.Renderer(width=1, height=1, output="depth", fidelity="low")
second = render.Renderer(width=1, height=1, output="depth", fidelity="low")
print(f"HYPERDRONE_TEST_BACKEND={first.backend}")
print(f"HYPERDRONE_TEST_BACKEND={second.backend}")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    expected_backend = render.backend().lower()
    announcement = f"#rl_tools::rendering::raytracing: backend={expected_backend}"
    announcements = [
        line
        for line in result.stderr.splitlines()
        if line.startswith("#rl_tools::rendering::raytracing: backend=")
    ]
    assert announcements == [announcement, announcement]
    backend_markers = [
        line.removeprefix("HYPERDRONE_TEST_BACKEND=")
        for line in result.stdout.splitlines()
        if line.startswith("HYPERDRONE_TEST_BACKEND=")
    ]
    assert backend_markers == [expected_backend, expected_backend]


def make_quad(center_x, half_size=1.0, color=(1.0, 0.0, 0.0)):
    # a quad in the Y-Z plane at x = center_x, facing -X (towards a camera at the origin
    # looking along +X, the FLU forward axis)
    vertices = np.array(
        [
            [center_x, -half_size, -half_size],
            [center_x, half_size, -half_size],
            [center_x, half_size, half_size],
            [center_x, -half_size, half_size],
        ],
        dtype=np.float32,
    )
    indices = np.array([[0, 2, 1], [0, 3, 2]], dtype=np.int32)
    return render.Mesh(vertices, indices, color=color)


def make_scene(distance=2.0):
    scene = render.Scene()
    wall = render.Object(name="wall")
    wall.add_mesh(make_quad(distance))
    scene.add_object(wall)
    scene.add_light(render.SceneLight.directional(direction=(1.0, 0.0, 0.0), color=(1.0, 1.0, 1.0)))
    return scene


def look_forward(renderer):
    camera = renderer.camera(position=(0.0, 0.0, 0.0), look_at=(1.0, 0.0, 0.0), fov=60.0)
    renderer.set_cameras(np.repeat(camera[None, :, :], renderer.num_cameras, axis=0))


def linear_to_rgb8(value):
    value = np.clip(value, 0, 1)
    return 255 * np.where(value <= 0.0031308, 12.92 * value, 1.055 * value ** (1 / 2.4) - 0.055)


@pytest.mark.parametrize("fidelity", ["high", "veryhigh"])
@pytest.mark.parametrize("ambient", [None, "default", (0, 0, 0), (0.6, 0.3, 0.2), (2, 1.5, 0.5)])
def test_environment_ambient(fidelity, ambient):
    color = np.array([0.2, 0.4, 0.6])
    scene = render.Scene()
    scene.add_mesh(make_quad(2, half_size=10, color=color))
    if ambient == "default":
        scene.set_environment()
    elif ambient is not None:
        scene.set_environment(ambient=ambient)
    renderer = render.Renderer(width=8, height=8, num_probes=0, output="rgb", fidelity=fidelity)
    renderer.init(scene)
    look_forward(renderer)
    renderer.render()
    expected_ambient = (0.1, 0.1, 0.1) if ambient in (None, "default") else ambient
    np.testing.assert_allclose(renderer.frame()[0, 4, 4, :3], linear_to_rgb8(color * expected_ambient), atol=1.1)


@pytest.mark.parametrize("fidelity", ["low", "medium", "high", "veryhigh"])
def test_environment_solid_background(fidelity):
    background = np.array([0.15, 0.4, 0.8])
    scene = render.Scene()
    scene.add_mesh(make_quad(-2))
    scene.set_environment(background=background)
    renderer = render.Renderer(width=8, height=8, num_probes=0, output="rgb", fidelity=fidelity)
    renderer.init(scene)
    look_forward(renderer)
    renderer.render()
    expected = 255 * background if fidelity == "low" else linear_to_rgb8(background)
    frame = renderer.frame()[0, :, :, :3]
    np.testing.assert_allclose(frame, np.broadcast_to(expected, frame.shape), atol=1.1)


@pytest.mark.parametrize("field", ["ambient", "background"])
@pytest.mark.parametrize("invalid", [-0.1, float("nan"), float("inf"), -float("inf"), 1e100])
def test_environment_rejects_invalid_colors(field, invalid):
    scene = render.Scene()
    with pytest.raises(ValueError, match="finite and nonnegative"):
        scene.set_environment(**{field: (0, invalid, 1)})


def test_depth():
    distance = 2.0
    renderer = render.Renderer(width=64, height=64, num_cameras=1, output="depth", fidelity="low")
    renderer.init(make_scene(distance))
    look_forward(renderer)
    renderer.render()
    depth = renderer.depth()
    assert depth.shape == (1, 64, 64)
    center = depth[0, 32, 32]
    assert abs(center - distance) < 1e-2


def test_rgb():
    renderer = render.Renderer(width=32, height=32, num_cameras=2, output="rgb", fidelity="low")
    renderer.init(make_scene())
    look_forward(renderer)
    renderer.render()
    frame = renderer.frame()
    assert frame.shape == (2, 32, 32, 4)
    center = frame[0, 16, 16]
    assert center[0] > 100  # red quad
    assert center[3] == 255
    assert np.array_equal(frame[0], frame[1])  # identical cameras render identically


def make_quad_span(x, y_low, y_high, z_half=5.0):
    vertices = np.array(
        [
            [x, y_low, -z_half],
            [x, y_high, -z_half],
            [x, y_high, z_half],
            [x, y_low, z_half],
        ],
        dtype=np.float32,
    )
    indices = np.array([[0, 2, 1], [0, 3, 2]], dtype=np.int32)
    return render.Mesh(vertices, indices)


def test_segmentation():
    # two disjoint half-planes: instance 0 on -Y (image right), instance 1 on +Y (image left)
    scene = render.Scene()
    for index, (y_low, y_high) in enumerate(((-5.0, -0.05), (0.05, 5.0))):
        wall = render.Object(name=f"wall_{index}")
        wall.add_mesh(make_quad_span(2.0, y_low, y_high))
        scene.add_object(wall)
    renderer = render.Renderer(width=32, height=32, num_cameras=1, output="segmentation", fidelity="low")
    renderer.init(scene)
    look_forward(renderer)
    renderer.render()
    segmentation = renderer.segmentation()
    left = segmentation[0, 16, 8]    # image x grows toward -Y, so +Y (instance 1) is on the left
    right = segmentation[0, 16, 24]
    assert (left, right) == (1, 0)


def test_collision_probes():
    renderer = render.Renderer(width=16, height=16, num_cameras=1, num_probes=4, output="depth", fidelity="low")
    renderer.init(make_scene(3.0))
    look_forward(renderer)
    renderer.generate_probe_directions()
    renderer.probe()
    distances, hits = renderer.collisions()
    assert distances.shape == (1, 4)
    # probe 0 is the camera forward direction and must hit the wall at ~3m
    assert hits[0, 0] == 1
    assert abs(distances[0, 0] - 3.0) < 1e-2


@pytest.mark.parametrize("asynchronous", [False, True])
def test_render_and_probe_are_independent(asynchronous):
    renderer = render.Renderer(width=16, height=16, output="rgbd_segmentation", fidelity="low")
    renderer.init(make_scene(3.0))
    look_forward(renderer)
    renderer.generate_probe_directions()

    def run(operation):
        if asynchronous:
            getattr(renderer, operation + "_launch")()
            getattr(renderer, operation + "_sync")()
        else:
            getattr(renderer, operation)()

    run("render")
    run("probe")
    initial_distances, initial_hits = renderer.collisions()
    assert abs(initial_distances[0, 0] - 3.0) < 1e-2

    renderer.set_cameras(renderer.camera(position=(1, 0, 0), look_at=(2, 0, 0)))
    run("render")
    assert abs(renderer.depth()[0, 8, 8] - 2.0) < 1e-2
    assert renderer.frame()[0, 8, 8, 0] > 100
    assert renderer.segmentation()[0, 8, 8] == 0
    distances, hits = renderer.collisions()
    np.testing.assert_array_equal(distances, initial_distances)
    np.testing.assert_array_equal(hits, initial_hits)
    depth = renderer.depth()
    frame = renderer.frame()
    segmentation = renderer.segmentation()

    renderer.set_cameras(renderer.camera(position=(2, 0, 0), look_at=(3, 0, 0)))
    run("probe")
    distances, hits = renderer.collisions()
    assert hits[0, 0] == 1
    assert abs(distances[0, 0] - 1.0) < 1e-2
    np.testing.assert_array_equal(renderer.depth(), depth)
    np.testing.assert_array_equal(renderer.frame(), frame)
    np.testing.assert_array_equal(renderer.segmentation(), segmentation)


def test_no_probes_compiles_the_probe_pass_out():
    with pytest.raises(ValueError):
        render.Renderer(width=16, height=16, num_cameras=1, num_probes=-1, output="depth", fidelity="low")
    renderer = render.Renderer(width=16, height=16, num_cameras=1, num_probes=0, output="depth", fidelity="low")
    assert renderer.num_probes == 0
    assert not renderer.has_probes
    assert renderer.config.key() != replace(renderer.config, num_probes=1).key()
    renderer.init(make_scene(3.0))
    look_forward(renderer)
    renderer.render()
    assert abs(renderer.depth()[0, 8, 8] - 3.0) < 1e-2
    renderer.render_launch()
    renderer.render_sync()
    renderer.synchronize()
    for call in (
        renderer.generate_probe_directions,
        renderer.collisions,
        renderer.probe,
        renderer.probe_launch,
        renderer.probe_sync,
    ):
        with pytest.raises(RuntimeError, match="num_probes=0"):
            call()


def test_motion_blur_camera_buffers_and_split_render():
    renderer = render.Renderer(
        width=16,
        height=16,
        num_cameras=1,
        output="depth",
        fidelity="low",
        motion_blur_samples=2,
        anti_aliasing_grid=2,
    )
    renderer.init(make_scene(2.0))
    camera_open = renderer.camera(position=(0.0, 0.0, 0.0), look_at=(1.0, 0.0, 0.0))
    camera_close = renderer.camera(position=(-1.0, 0.0, 0.0), look_at=(1.0, 0.0, 0.0))
    renderer.set_motion_blur_cameras(camera_open[None], camera_close[None])
    renderer.render_launch()
    renderer.render_sync()
    blurred_depth = renderer.depth()[0, 8, 8]
    assert 2.0 < blurred_depth < 3.0

    renderer.set_cameras(camera_open[None])
    renderer.render()
    assert abs(renderer.depth()[0, 8, 8] - 2.0) < 1e-2


def test_dynamic_motion_blur_object():
    # static camera (shutter open == close), overlay translating from x=2 to x=4 across the
    # shutter: 2 samples at midpoint times hit x=2.5 and x=3.5, so the blurred depth is 3.0
    scene = make_scene(8.0)
    asset_pool = render.AssetPool()
    dynamic = render.Object(name="dynamic")
    dynamic.add_mesh(make_quad(0.0, half_size=1.0, color=(0.0, 1.0, 0.0)))
    asset = asset_pool.add_object(dynamic)
    renderer = render.Renderer(
        width=1,
        height=1,
        output="depth",
        fidelity="low",
        motion_blur_samples=2,
        num_overlays=1,
        max_overlay_instances=2,
        max_overlays_per_camera=1,
        dynamic_motion_blur=True,
    )
    renderer.init(scene, asset_pool)
    camera = renderer.camera(position=(0.0, 0.0, 0.0), look_at=(1.0, 0.0, 0.0), fov=60.0)
    renderer.set_motion_blur_cameras(camera[None], camera[None])
    renderer.attach(0, 0)
    placement = renderer.spawn(0, asset, render.make_transform(position=(4.0, 0.0, 0.0)))
    renderer.set_transform_pair(0, placement, render.make_transform(position=(2.0, 0.0, 0.0)), render.make_transform(position=(4.0, 0.0, 0.0)))
    renderer.update()
    renderer.render()
    assert abs(renderer.depth()[0, 0, 0] - 3.0) < 1e-2

    # producer path: the same shutter pair through the transforms_pair tensor + on-device
    # expansion must reproduce the host-verb result
    num_slots = 1 * 2  # num_overlays * max_overlay_instances
    pairs = np.zeros((2, num_slots, 12), dtype=np.float32)
    pairs[0, placement[0]] = np.asarray(render.make_transform(position=(2.0, 0.0, 0.0)), dtype=np.float32).reshape(12)
    pairs[1, placement[0]] = np.asarray(render.make_transform(position=(4.0, 0.0, 0.0)), dtype=np.float32).reshape(12)
    renderer.set_transforms_pair(pairs)
    renderer.expand_motion_transforms()
    renderer.update()
    renderer.render()
    assert abs(renderer.depth()[0, 0, 0] - 3.0) < 1e-2


def test_overlay_pipeline(tmp_path):
    scene = make_scene(8.0)
    asset_pool = render.AssetPool()
    dynamic = render.Object(name="dynamic", segmentation_class=23)
    dynamic.add_mesh(make_quad(0.0, half_size=0.35, color=(1.0, 0.0, 0.0)))
    asset = asset_pool.add_object(dynamic)
    assembly_path = tmp_path / "overlay_assembly.glb"
    write_minimal_glb(assembly_path)
    assembly_asset = asset_pool.add_assembly(render.load_assembly(assembly_path, fidelity="low"))
    assert asset_pool.num_assets == 2

    renderer = render.Renderer(
        width=48,
        height=48,
        num_cameras=2,
        num_probes=2,
        output="rgbd_segmentation",
        fidelity="low",
        num_overlays=2,
        max_overlay_instances=2,
        max_overlays_per_camera=2,
    )
    renderer.init(scene, asset_pool)
    look_forward(renderer)
    assert renderer.can_attach(0, 0)
    assert renderer.can_spawn(0, asset)
    renderer.attach(0, 0)
    placement = renderer.spawn(0, asset, render.make_transform(position=(4.0, 0.75, 0.0)))
    assert placement == (0, 1, 0)
    renderer.update()
    renderer.render()

    instance_id = scene.num_instances + placement[0]
    segmentation_live = renderer.segmentation(copy=False)
    first_mask = segmentation_live[0] == instance_id
    assert first_mask.sum() > 4
    assert not np.any(segmentation_live[1] == instance_id)
    assert np.mean(renderer.frame()[0][first_mask, 0]) > 100
    assert np.mean(renderer.depth()[0][first_mask]) < 5.0

    renderer.set_transform(0, placement, render.make_transform(position=(4.0, -0.75, 0.0)))
    renderer.update()
    renderer.render_launch()
    renderer.render_sync()
    second_view = renderer.segmentation(copy=False)
    second_mask = second_view[0] == instance_id
    assert np.shares_memory(segmentation_live, second_view)
    assert second_mask.sum() > 4
    assert not np.array_equal(first_mask, second_mask)

    renderer.detach(0, 0)
    renderer.attach(1, 0)
    renderer.update()
    renderer.render()
    segmentation = renderer.segmentation()
    assert not np.any(segmentation[0] == instance_id)
    assert np.any(segmentation[1] == instance_id)

    renderer.despawn(0, placement)
    renderer.update()
    renderer.render()
    assert not np.any(renderer.segmentation() == instance_id)
    assert renderer.can_spawn(0, asset)
    replacement = renderer.spawn(0, asset, render.make_transform(position=(4.0, -0.75, 0.0)))
    assert replacement == placement
    renderer.update()
    renderer.render()
    assert np.any(renderer.segmentation()[1] == instance_id)

    assert not renderer.can_spawn(0, assembly_asset)
    renderer.despawn(0, replacement)
    assert renderer.can_spawn(0, assembly_asset)
    assembly_placement = renderer.spawn(0, assembly_asset, render.make_transform())
    assert assembly_placement == (0, 2, 1)
    renderer.set_part_transform(
        0,
        assembly_placement,
        1,
        render.make_transform(position=(0.0, 2.0, 0.0)),
    )
    renderer.update()
    renderer.render()
    segmentation = renderer.segmentation()
    assert np.any(segmentation[1] == instance_id)
    assert np.any(segmentation[1] == instance_id + 1)
    assert not np.any(segmentation[0] == instance_id)
    assert not np.any(segmentation[0] == instance_id + 1)


def test_semantic_segmentation_with_overlay():
    scene = make_scene(8.0)
    scene.set_object_segmentation_class(0, 7)
    asset_pool = render.AssetPool()
    dynamic = render.Object(name="semantic", segmentation_class=23)
    dynamic.add_mesh(make_quad(0.0, half_size=0.25))
    asset = asset_pool.add_object(dynamic)
    renderer = render.Renderer(
        width=24,
        height=24,
        output="segmentation",
        fidelity="low",
        num_overlays=1,
        max_overlay_instances=1,
        max_overlays_per_camera=1,
        semantic_segmentation=True,
    )
    renderer.init(scene, asset_pool)
    look_forward(renderer)
    renderer.attach(0, 0)
    renderer.spawn(0, asset, render.make_transform(position=(4.0, 0.0, 0.0)))
    renderer.update()
    renderer.render()
    classes = renderer.segmentation()
    assert np.any(classes == 7)
    assert np.any(classes == 23)


def test_jit_cache_reuse():
    config = render.RendererConfig(
        width=16, height=16, num_cameras=1, num_probes=1, fidelity=0, output_mode=2,
        motion_blur_samples=1, anti_aliasing_grid=1, num_overlays=0,
        max_overlay_instances=0, max_overlays_per_camera=0, semantic_segmentation=False,
    )
    from hyperdrone.render import _component
    library = jit.build_dir(_component.component()) / "jit" / f"render_{config.key()}.so"
    renderer = render.Renderer(width=16, height=16, num_cameras=1, output="depth", fidelity="low")
    assert library.exists()
    modification_time = library.stat().st_mtime
    renderer_again = render.Renderer(width=16, height=16, num_cameras=1, output="depth", fidelity="low")
    assert library.stat().st_mtime == modification_time


def test_zero_copy_host_view():
    renderer = render.Renderer(width=16, height=16, num_cameras=1, output="depth", fidelity="low")
    renderer.init(make_scene(2.0))
    look_forward(renderer)
    renderer.render()
    first_view = renderer.depth(copy=False)
    assert abs(first_view[0, 8, 8] - 2.0) < 1e-2
    # stepping the camera back must update the earlier view in place (it aliases the
    # renderer's staging buffer)
    camera = renderer.camera(position=(-1.0, 0.0, 0.0), look_at=(1.0, 0.0, 0.0), fov=60.0)
    renderer.set_cameras(camera[None])
    renderer.render()
    second_view = renderer.depth(copy=False)
    assert np.shares_memory(first_view, second_view)
    assert abs(first_view[0, 8, 8] - 3.0) < 1e-2


def test_dlpack_export():
    renderer = render.Renderer(width=16, height=16, num_cameras=1, output="rgbd", fidelity="low")
    renderer.init(make_scene(2.0))
    look_forward(renderer)
    renderer.render()
    frame_live = renderer.frame_dlpack()
    frame_raw_live = renderer.frame_raw_dlpack()
    depth_live = renderer.depth_dlpack()
    device = frame_live.__dlpack_device__()
    assert frame_raw_live.__dlpack_device__() == device
    if renderer.backend == "optix":
        assert device == (2, 0)  # kDLCUDA
    else:
        assert device[0] == 1  # kDLCPU: live buffer is host-visible, numpy can alias it
        frame_array = np.from_dlpack(frame_live)
        frame_raw_array = np.from_dlpack(frame_raw_live)
        depth_array = np.from_dlpack(depth_live)
        assert frame_array.shape == (1, 16, 16, 4)
        assert frame_array.dtype == np.uint8
        assert np.array_equal(frame_array, renderer.frame())
        assert frame_raw_array.shape == (1, 16, 16)
        assert frame_raw_array.dtype == np.uint32
        assert np.array_equal(frame_raw_array, renderer.frame_raw())
        assert np.shares_memory(frame_array, frame_raw_array)
        assert np.array_equal(depth_array, renderer.depth())


def test_dlpack_camera_input():
    # any DLPack producer works as camera input; numpy's own arrays go through the same path
    renderer = render.Renderer(width=16, height=16, num_cameras=1, output="depth", fidelity="low")
    renderer.init(make_scene(2.0))
    camera = renderer.camera(position=(0.0, 0.0, 0.0), look_at=(1.0, 0.0, 0.0), fov=60.0)

    class DLPackOnly:
        def __init__(self, array):
            self._array = array

        def __dlpack__(self, **kwargs):
            return self._array.__dlpack__(**kwargs)

        def __dlpack_device__(self):
            return self._array.__dlpack_device__()

    renderer.set_cameras(DLPackOnly(np.ascontiguousarray(camera[None])))
    renderer.render()
    assert abs(renderer.depth()[0, 8, 8] - 2.0) < 1e-2


@pytest.mark.skipif(render.backend() != "OPTIX", reason="device camera input requires the OptiX backend")
def test_device_camera_input():
    renderer = render.Renderer(width=32, height=32, num_cameras=2, output="depth", fidelity="low")
    renderer.init(make_scene(2.0))
    cameras = np.stack([
        np.asarray(renderer.camera(position=(0.0, 0.0, 0.0), look_at=(1.0, 0.0, 0.0), fov=60.0)).reshape(12),
        np.asarray(renderer.camera(position=(-1.0, 0.0, 0.0), look_at=(1.0, 0.0, 0.0), fov=60.0)).reshape(12),
    ])
    renderer.set_cameras(cameras)
    renderer.render()
    host_frame = renderer.depth()

    tensor_set = hyperdrone.cuda.upload(cameras[None])  # one set of two cameras
    renderer.set_cameras(tensor_set.view(0))            # dispatches on __dlpack_device__ == kDLCUDA
    renderer.render()
    device_frame = renderer.depth()

    assert np.array_equal(host_frame, device_frame)
    assert abs(device_frame[0, 16, 16] - 2.0) < 1e-2
    assert abs(device_frame[1, 16, 16] - 3.0) < 1e-2


def test_renderer_lifecycle():
    # two full create/render/destroy cycles in one process (regression: OWL's PinnedHostMem
    # cudaFree-on-pinned-memory bug killed the second lifecycle on the OptiX backend)
    import gc

    for _ in range(2):
        renderer = render.Renderer(width=16, height=16, num_cameras=1, output="depth", fidelity="low")
        renderer.init(make_scene())
        look_forward(renderer)
        renderer.render()
        assert renderer.depth().shape == (1, 16, 16)
        del renderer
        gc.collect()


def test_error_on_wrong_output():
    renderer = render.Renderer(width=16, height=16, num_cameras=1, output="depth", fidelity="low")
    renderer.init(make_scene())
    with pytest.raises(RuntimeError):
        renderer.frame()


def write_minimal_glb(path, color=(0.0, 1.0, 0.0)):
    """Two glTF scene-root nodes sharing one quad mesh in the glTF x=2 plane. The FLU
    swizzle (x, -z, y) keeps x, so the first quad sits straight ahead of a camera at the
    origin looking along +X at distance 2; the second sits behind it at x=5. Two roots
    also make Assimp keep a wrapper root, which the assembly loader requires."""
    import json
    import struct

    positions = np.array(
        [
            [2.0, -1.0, -1.0],
            [2.0, 1.0, -1.0],
            [2.0, 1.0, 1.0],
            [2.0, -1.0, 1.0],
        ],
        dtype=np.float32,
    )
    indices = np.array([0, 2, 1, 0, 3, 2], dtype=np.uint16)
    position_bytes = positions.tobytes()
    index_bytes = indices.tobytes()  # 12 bytes, already 4-byte aligned
    binary = position_bytes + index_bytes
    gltf = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0, 1]}],
        "nodes": [
            {"mesh": 0, "name": "quad"},
            {"mesh": 0, "name": "quad_far", "translation": [3.0, 0.0, 0.0]},
        ],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "material": 0}]}],
        "materials": [{"pbrMetallicRoughness": {"baseColorFactor": [*color, 1.0], "metallicFactor": 0.0, "roughnessFactor": 1.0}}],
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(position_bytes), "target": 34962},
            {"buffer": 0, "byteOffset": len(position_bytes), "byteLength": len(index_bytes), "target": 34963},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 4, "type": "VEC3", "min": [2.0, -1.0, -1.0], "max": [2.0, 1.0, 1.0]},
            {"bufferView": 1, "componentType": 5123, "count": 6, "type": "SCALAR"},
        ],
    }
    json_bytes = json.dumps(gltf).encode()
    json_bytes += b" " * ((4 - len(json_bytes) % 4) % 4)
    length = 12 + 8 + len(json_bytes) + 8 + len(binary)
    with open(path, "wb") as handle:
        handle.write(struct.pack("<III", 0x46546C67, 2, length))
        handle.write(struct.pack("<II", len(json_bytes), 0x4E4F534A))
        handle.write(json_bytes)
        handle.write(struct.pack("<II", len(binary), 0x004E4942))
        handle.write(binary)


def test_glb_roundtrip(tmp_path):
    glb = tmp_path / "quad.glb"
    write_minimal_glb(glb)
    scene = render.load_scene(glb, fidelity="medium")
    assert scene.num_objects == 1
    assert scene.num_instances == 1
    assert scene.num_lights == 3  # neutral fill lights added for light-less files
    renderer = render.Renderer(width=32, height=32, num_cameras=1, output="rgbd", fidelity="medium")
    renderer.init(scene)
    look_forward(renderer)
    renderer.render()
    frame = renderer.frame()
    depth = renderer.depth()
    center = frame[0, 16, 16]
    assert center[1] > center[0]  # green quad
    assert abs(depth[0, 16, 16] - 2.0) < 1e-2  # swizzled to x=+2 in FLU


def test_glb_assembly_names(tmp_path):
    glb = tmp_path / "quad.glb"
    write_minimal_glb(glb)
    assembly = render.load_assembly(glb, fidelity="low")
    assert assembly.num_objects == 2
    assert assembly.object_names() == ["quad", "quad_far"]
    scene = render.Scene()
    first, count = scene.add_assembly(assembly)
    assert (first, count) == (0, 2)


GLB = Path(os.environ.get("HYPERDRONE_RLTOOLS_ROOT", Path(__file__).resolve().parents[5])) / "tests" / "data" / "ProcTHOR-Train-1.glb"


@pytest.mark.skipif(not GLB.exists(), reason="ProcTHOR test scene not available")
def test_glb_scene():
    scene = render.load_scene(GLB, fidelity="high")
    assert scene.num_objects >= 1
    assert scene.num_instances >= 1
    renderer = render.Renderer(width=64, height=48, num_cameras=1, output="rgbd", fidelity="high")
    renderer.init(scene)
    bounds = renderer.scene_bounds
    assert bounds["camera_radius"] > 0
    renderer.generate_cameras()
    renderer.render()
    frame = renderer.frame()
    depth = renderer.depth()
    assert frame.shape == (1, 48, 64, 4)
    assert depth.shape == (1, 48, 64)
    assert frame[..., :3].max() > 0  # something visible
    assert (depth > 0).all()


@pytest.mark.skipif(not GLB.exists(), reason="ProcTHOR test scene not available")
def test_glb_assembly_segmentation_names():
    assembly = render.load_assembly(GLB, fidelity="low")
    assert assembly.num_objects >= 1
    names = assembly.object_names()
    assert len(names) == assembly.num_objects
    scene = render.Scene()
    first, count = scene.add_assembly(assembly)
    assert count == assembly.num_parts
    assert scene.num_instances == count

@pytest.fixture
def labelled_glb(tmp_path):
    import json
    import struct

    vertices = np.array([[2, -.25, -.25], [2, .25, -.25], [2, .25, .25],
                         [2, -.25, -.25], [2, .25, .25], [2, -.25, .25]], dtype='<f4')
    doc = {
        'asset': {'version': '2.0'}, 'scene': 0,
        'scenes': [{'nodes': [0, 1, 2, 3], 'extras': {'rl_tools': {'segmentation': {
            'taxonomy': 'test', 'classes': {'Unknown': 0, 'Chair': 21, 'Wall': 162}}}}}],
        'nodes': [{'name': f'object-{i}', 'mesh': 0, 'translation': [0, 0, i - 1.5],
                   'extras': {'rl_tools': {'segmentation_class': c}}}
                  for i, c in enumerate([21, 21, 162, 0])],
        'meshes': [{'primitives': [{'attributes': {'POSITION': 0}}]}],
        'buffers': [{'byteLength': vertices.nbytes}],
        'bufferViews': [{'buffer': 0, 'byteLength': vertices.nbytes}],
        'accessors': [{'bufferView': 0, 'componentType': 5126, 'count': 6, 'type': 'VEC3',
                       'min': [2, -.25, -.25], 'max': [2, .25, .25]}],
    }
    encoded = json.dumps(doc).encode()
    encoded += b' ' * (-len(encoded) % 4)
    path = tmp_path / 'labels.glb'
    path.write_bytes(struct.pack('<5I', 0x46546c67, 2, 28 + len(encoded) + vertices.nbytes,
                                 len(encoded), 0x4e4f534a) + encoded +
                     struct.pack('<2I', vertices.nbytes, 0x004e4942) + vertices.tobytes())
    return path


@pytest.mark.parametrize('semantic', [False, True])
def test_loaded_instance_and_semantic_labels(labelled_glb, semantic):
    scene = render.load_scene(labelled_glb, rgb=False, preserve_instances=True)
    assert scene.num_instances == 4
    assert scene.segmentation_taxonomy == 'test'
    assert scene.segmentation_classes == {0: 'Unknown', 21: 'Chair', 162: 'Wall'}
    renderer = render.Renderer(width=24, height=24, num_cameras=4, fidelity='low',
                               output='segmentation', semantic_segmentation=semantic)
    renderer.init(scene)
    cameras = [renderer.camera(position=(0, 1.5-i, 0), look_at=(2, 1.5-i, 0)) for i in range(4)]
    renderer.set_cameras(np.array(cameras, dtype=np.float32))
    renderer.render()
    labels = renderer.segmentation()
    assert renderer.segmentation_mode == ('semantic' if semantic else 'instance')
    assert labels[:, 12, 12].tolist() == ([21, 21, 162, 0] if semantic else [0, 1, 2, 3])
    assert np.any(labels == 0xffffffff)
    if not semantic:
        assert scene.object_name(scene.instance_object(int(labels[1, 12, 12]))) == 'object-1'


def test_loaded_semantics_survive_assembly_and_pool(labelled_glb):
    assembly = render.load_assembly(labelled_glb, rgb=False)
    assert assembly.segmentation_classes[21] == 'Chair'
    scene = render.Scene()
    scene.add_assembly(assembly)
    scene.add_assembly(assembly, render.make_transform(position=(0, 0, 3)))
    assert scene.num_instances == 8
    assert scene.object_segmentation_class(4) == 21
    assert scene.segmentation_taxonomy == 'test'
    pool = render.AssetPool()
    asset = pool.add_assembly(assembly)
    renderer = render.Renderer(width=24, height=24, fidelity='low', output='segmentation',
                               semantic_segmentation=True, num_overlays=1,
                               max_overlay_instances=4, max_overlays_per_camera=1)
    renderer.init(scene, pool)
    renderer.attach(0, 0)
    handle = renderer.spawn(0, asset, render.make_transform(position=(-1, 0, 0)))
    camera = renderer.camera(position=(0, 1.5, 0), look_at=(2, 1.5, 0))
    renderer.set_cameras(camera[None])
    renderer.update()
    renderer.render()
    assert renderer.segmentation()[0, 12, 12] == 21
    renderer.set_transform(0, handle, render.make_transform(position=(-1, 2, 0)))
    renderer.update()
    renderer.render()
    assert renderer.segmentation()[0, 12, 12] == 162


def test_welded_semantics_rejected(labelled_glb):
    scene = render.load_scene(labelled_glb, rgb=False)
    assert scene.num_instances == 1
    renderer = render.Renderer(width=24, height=24, fidelity='low', output='segmentation',
                               semantic_segmentation=True)
    with pytest.raises(ValueError, match='welded classes'):
        renderer.init(scene)


@pytest.mark.parametrize('target', ['object', 'scene', 'assembly', 'pool'])
def test_explicit_class_assignment_repairs_welded_labels(labelled_glb, target):
    obj = render.load_object(labelled_glb, rgb=False)
    assert not obj.segmentation_class_valid
    scene = render.Scene()
    pool = None
    if target == 'object':
        with pytest.raises(ValueError, match='absent from the taxonomy'):
            obj.segmentation_class = 7
        assert not obj.segmentation_class_valid
        obj.segmentation_class = 21
        assert obj.segmentation_class_valid
        assert obj.segmentation_taxonomy == 'test'
        scene.add_object(obj)
    elif target == 'scene':
        scene.add_object(obj)
        scene.set_object_segmentation_class(0, 21)
    elif target == 'assembly':
        assembly = render.load_assembly(labelled_glb, rgb=False)
        assembly.set_segmentation_class(0, 162)
        assert assembly.segmentation_class(0) == 162
        assembly.remap_segmentation_classes({0: 21, 21: 21, 162: 21},
                                             taxonomy='test', classes=assembly.segmentation_classes)
        scene.add_assembly(assembly)
    else:
        pool = render.AssetPool()
        asset = pool.add_object(obj)
        pool.set_object_segmentation_class(asset, 0, 21)
        assert pool.object_segmentation_class(asset, 0) == 21
    renderer = render.Renderer(width=24, height=24, fidelity='low', output='segmentation',
                               semantic_segmentation=True, num_overlays=1,
                               max_overlay_instances=4, max_overlays_per_camera=1)
    renderer.init(scene, pool)
    if pool is not None:
        renderer.attach(0, 0)
        renderer.spawn(0, asset, render.make_transform(position=(0, 0, 0)))
        renderer.update()
    camera = renderer.camera(position=(0, 1.5, 0), look_at=(2, 1.5, 0))
    renderer.set_cameras(camera[None])
    renderer.render()
    assert renderer.segmentation()[0, 12, 12] == 21


def test_failed_semantic_init_can_be_retried(labelled_glb):
    scene = render.load_scene(labelled_glb, rgb=False)
    renderer = render.Renderer(width=24, height=24, fidelity='low', output='segmentation',
                               semantic_segmentation=True)
    with pytest.raises(ValueError, match='welded classes'):
        renderer.init(scene)
    assert renderer._scene is None
    scene.set_object_segmentation_class(0, 21)
    renderer.init(scene)
    camera = renderer.camera(position=(0, 1.5, 0), look_at=(2, 1.5, 0))
    renderer.set_cameras(camera[None])
    renderer.render()
    assert renderer.segmentation()[0, 12, 12] == 21


def test_label_remapping_across_scene_and_pool(labelled_glb):
    assembly = render.load_assembly(labelled_glb, rgb=False)
    source_classes = assembly.segmentation_classes
    scene = render.Scene()
    scene.add_assembly(assembly)
    pool = render.AssetPool()
    asset = pool.add_assembly(assembly)
    mapping = {0: 0, 21: 2147483648, 162: 4294967294}
    classes = {0: 'Unknown', 2147483648: 'Furniture', 4294967294: 'Structure'}
    for target in (scene, assembly, pool):
        for bad_mapping in ({0: 0, 21: 2147483648}, {0: 0, 21: 2147483648, 162: 999}):
            with pytest.raises(ValueError):
                target.remap_segmentation_classes(bad_mapping, taxonomy='custom', classes=classes)
            assert target.segmentation_taxonomy == 'test'
            assert target.segmentation_classes == source_classes
        target.remap_segmentation_classes(mapping, taxonomy='custom', classes=classes)
        assert target.segmentation_taxonomy == 'custom'
        assert target.segmentation_classes == classes
    assert [scene.object_segmentation_class(i) for i in range(4)] == [2147483648, 2147483648, 4294967294, 0]
    assert [assembly.segmentation_class(i) for i in range(4)] == [2147483648, 2147483648, 4294967294, 0]
    assert [pool.object_segmentation_class(asset, i) for i in range(4)] == [2147483648, 2147483648, 4294967294, 0]
    renderer = render.Renderer(width=24, height=24, fidelity='low', output='segmentation',
                               semantic_segmentation=True, num_overlays=1,
                               max_overlay_instances=4, max_overlays_per_camera=1)
    renderer.init(scene, pool)
    renderer.attach(0, 0)
    handle = renderer.spawn(0, asset, render.make_transform(position=(-1, 0, 0)))
    camera = renderer.camera(position=(0, 1.5, 0), look_at=(2, 1.5, 0))
    renderer.set_cameras(camera[None])
    renderer.update()
    renderer.render()
    assert renderer.segmentation()[0, 12, 12] == 2147483648
    renderer.set_transform(0, handle, render.make_transform(position=(-1, 2, 0)))
    renderer.update()
    renderer.render()
    assert renderer.segmentation()[0, 12, 12] == 4294967294


def test_object_and_pool_remapping_is_explicit(labelled_glb):
    obj = render.load_object(labelled_glb, rgb=False)
    with pytest.raises(ValueError, match='welded classes'):
        obj.remap_segmentation_classes({0: 7})
    obj.assign_segmentation_class(7, taxonomy='custom', classes={0: 'Unknown', 7: 'Furniture'})
    assert obj.segmentation_class_valid
    assert obj.segmentation_taxonomy == 'custom'
    with pytest.raises(ValueError):
        obj.assign_segmentation_class(9, taxonomy='broken', classes={9: 'MissingUnknown'})
    assert obj.segmentation_class == 7
    assert obj.segmentation_taxonomy == 'custom'
    pool = render.AssetPool()
    asset = pool.add_object(obj)
    pool.remap_asset_segmentation_classes(asset, {7: 8})
    assert pool.segmentation_taxonomy == ''
    assert pool.object_segmentation_class(asset, 0) == 8
    assert obj.segmentation_class == 7
    obj.remap_segmentation_classes({7: 4294967294})
    assert obj.segmentation_taxonomy == ''
    assert obj.segmentation_classes == {}
    assert obj.segmentation_class == 4294967294
    with pytest.raises(ValueError, match='reserved'):
        obj.segmentation_class = 0xffffffff
    assert obj.segmentation_class == 4294967294
    with pytest.raises(ValueError, match='reserved'):
        render.Object(segmentation_class=0xffffffff)
