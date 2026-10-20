from hyperdrone.camera import CameraSpec, RGB, Depth, Segmentation, Normals, Flow, FlowStatus, Probes
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import hyperdrone
from hyperdrone import _jit as jit, render


def test_backend_announcement_reaches_python_for_each_renderer():
    script = """
from hyperdrone.camera import CameraSpec, RGB, Depth, Segmentation, Normals, Flow, FlowStatus, Probes

from hyperdrone import render

first = render.Renderer(width=1, height=1, cameras=(CameraSpec("camera", outputs=(Depth(), Probes(),)),), fidelity="low")
second = render.Renderer(width=1, height=1, cameras=(CameraSpec("camera", outputs=(Depth(), Probes(),)),), fidelity="low")
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
    expected_backend = render._backend().lower()
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


def test_verbosity_reaches_loaded_and_future_renderers():
    script = """
from hyperdrone.camera import CameraSpec, RGB, Depth, Segmentation, Normals, Flow, FlowStatus, Probes

import hyperdrone
from hyperdrone import render

hyperdrone.set_verbosity("warning")
quiet = render.Renderer(width=1, height=1, cameras=(CameraSpec("camera", outputs=(Depth(), Probes(),)),), fidelity="low")
hyperdrone.set_verbosity("debug")
quiet.generate_probe_directions()
loud = render.Renderer(width=2, height=1, cameras=(CameraSpec("camera", outputs=(Depth(), Probes(),)),), fidelity="low")
hyperdrone.set_verbosity("warning")
quiet.generate_probe_directions()
loud.generate_probe_directions()
hyperdrone.set_verbosity("debug")
quiet.generate_probe_directions()
loud.generate_probe_directions()
hyperdrone.set_verbosity("warning")
del quiet, loud
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    prefix = "#rl_tools::rendering::raytracing: "
    messages = [line.removeprefix(prefix) for line in result.stderr.splitlines()
                if line.startswith((prefix + "Generated", prefix + "backend="))]
    assert messages == ["Generated 1 probe directions per camera", f"backend={render._backend().lower()}",
                        "Generated 1 probe directions per camera", "Generated 1 probe directions per camera"]


def test_verbosity_rejects_unknown_levels():
    previous = hyperdrone.verbosity()
    try:
        hyperdrone.set_verbosity("debug")
        with pytest.raises(ValueError):
            hyperdrone.set_verbosity("verbose")
        assert hyperdrone.verbosity() == "debug"
    finally:
        hyperdrone.set_verbosity(previous)


def test_verbosity_before_jit_import():
    subprocess.run([sys.executable, "-c", """
import sys
import hyperdrone
hyperdrone.set_verbosity("warning")
assert hyperdrone.verbosity() == "warning"
assert "hyperdrone._jit" not in sys.modules
"""], check=True)


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
    renderer = render.Renderer(width=8, height=8, num_probes=0, cameras=(CameraSpec("camera", outputs=(RGB(format="rgba8"),)),), fidelity=fidelity)
    renderer.init(scene)
    look_forward(renderer)
    renderer.render()
    expected_ambient = (0.1, 0.1, 0.1) if ambient in (None, "default") else ambient
    np.testing.assert_allclose(renderer.outputs[RGB(format="rgba8")].read()[0, 4, 4, :3], linear_to_rgb8(color * expected_ambient), atol=1.1)


@pytest.mark.parametrize("fidelity", ["low", "medium", "high", "veryhigh"])
def test_environment_solid_background(fidelity):
    background = np.array([0.15, 0.4, 0.8])
    scene = render.Scene()
    scene.add_mesh(make_quad(-2))
    scene.set_environment(background=background)
    renderer = render.Renderer(width=8, height=8, num_probes=0, cameras=(CameraSpec("camera", outputs=(RGB(format="rgba8"),)),), fidelity=fidelity)
    renderer.init(scene)
    look_forward(renderer)
    renderer.render()
    expected = linear_to_rgb8(background)
    frame = renderer.outputs[RGB(format="rgba8")].read()[0, :, :, :3]
    np.testing.assert_allclose(frame, np.broadcast_to(expected, frame.shape), atol=1.1)


@pytest.mark.parametrize("field", ["ambient", "background"])
@pytest.mark.parametrize("invalid", [-0.1, float("nan"), float("inf"), -float("inf"), 1e100])
def test_environment_rejects_invalid_colors(field, invalid):
    scene = render.Scene()
    with pytest.raises(ValueError, match="finite and nonnegative"):
        scene.set_environment(**{field: (0, invalid, 1)})


def test_depth():
    distance = 2.0
    renderer = render.Renderer(width=64, height=64, batch_size=1, cameras=(CameraSpec("camera", outputs=(Depth(), Probes(),)),), fidelity="low")
    renderer.init(make_scene(distance))
    look_forward(renderer)
    renderer.render()
    depth = renderer.outputs[Depth()].read()
    assert depth.shape == (1, 64, 64)
    center = depth[0, 32, 32]
    assert abs(center - distance) < 1e-2


def test_rgb():
    renderer = render.Renderer(width=32, height=32, batch_size=2, cameras=(CameraSpec("camera", outputs=(RGB(format="rgba8"), Probes(),)),), fidelity="low")
    renderer.init(make_scene())
    look_forward(renderer)
    renderer.render()
    frame = renderer.outputs[RGB(format="rgba8")].read()
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
    renderer = render.Renderer(width=32, height=32, batch_size=1, cameras=(CameraSpec("camera", outputs=(Segmentation(), Probes(),)),), fidelity="low")
    renderer.init(scene)
    look_forward(renderer)
    renderer.render()
    segmentation = renderer.outputs[Segmentation()].read()
    left = segmentation[0, 16, 8]    # image x grows toward -Y, so +Y (instance 1) is on the left
    right = segmentation[0, 16, 24]
    assert (left, right) == (1, 0)


def test_collision_probes():
    renderer = render.Renderer(width=16, height=16, batch_size=1, num_probes=4, cameras=(CameraSpec("camera", outputs=(Depth(), Probes(),)),), fidelity="low")
    renderer.init(make_scene(3.0))
    look_forward(renderer)
    renderer.generate_probe_directions()
    renderer.probe()
    distances, hits = renderer.outputs[Probes()].read().values()
    assert distances.shape == (1, 4)
    assert hits.dtype == np.int32
    assert np.isposinf(distances[hits == 0]).all()
    # probe 0 is the camera forward direction and must hit the wall at ~3m
    assert hits[0, 0] == 1
    assert abs(distances[0, 0] - 3.0) < 1e-2


@pytest.mark.parametrize("asynchronous", [False, True])
def test_render_and_probe_are_independent(asynchronous):
    renderer = render.Renderer(width=16, height=16, cameras=(CameraSpec("camera", outputs=(RGB(format="rgba8"), Depth(), Segmentation(), Probes(),)),), fidelity="low")
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
    initial_distances, initial_hits = renderer.outputs[Probes()].read().values()
    assert abs(initial_distances[0, 0] - 3.0) < 1e-2

    renderer.set_cameras(renderer.camera(position=(1, 0, 0), look_at=(2, 0, 0)))
    run("render")
    assert abs(renderer.outputs[Depth()].read()[0, 8, 8] - 2.0) < 1e-2
    assert renderer.outputs[RGB(format="rgba8")].read()[0, 8, 8, 0] > 100
    assert renderer.outputs[Segmentation()].read()[0, 8, 8] == 0
    distances, hits = renderer.outputs[Probes()].read().values()
    np.testing.assert_array_equal(distances, initial_distances)
    np.testing.assert_array_equal(hits, initial_hits)
    depth = renderer.outputs[Depth()].read()
    frame = renderer.outputs[RGB(format="rgba8")].read()
    segmentation = renderer.outputs[Segmentation()].read()

    renderer.set_cameras(renderer.camera(position=(2, 0, 0), look_at=(3, 0, 0)))
    run("probe")
    distances, hits = renderer.outputs[Probes()].read().values()
    assert hits[0, 0] == 1
    assert abs(distances[0, 0] - 1.0) < 1e-2
    np.testing.assert_array_equal(renderer.outputs[Depth()].read(), depth)
    np.testing.assert_array_equal(renderer.outputs[RGB(format="rgba8")].read(), frame)
    np.testing.assert_array_equal(renderer.outputs[Segmentation()].read(), segmentation)


def test_no_probes_compiles_the_probe_pass_out():
    with pytest.raises(ValueError):
        render.Renderer(width=16, height=16, batch_size=1, num_probes=-1, cameras=(CameraSpec("camera", outputs=(Depth(), Probes(),)),), fidelity="low")
    renderer = render.Renderer(width=16, height=16, batch_size=1, num_probes=0, cameras=(CameraSpec("camera", outputs=(Depth(),)),), fidelity="low")
    assert renderer.num_probes == 0
    assert not renderer.has_probes
    assert renderer.config.key() == replace(renderer.config, num_probes=1).key()
    renderer.init(make_scene(3.0))
    look_forward(renderer)
    renderer.render()
    assert abs(renderer.outputs[Depth()].read()[0, 8, 8] - 3.0) < 1e-2
    renderer.render_launch()
    renderer.render_sync()
    renderer.synchronize()
    for call in (
        renderer.generate_probe_directions,
        renderer.probe,
        renderer.probe_launch,
        renderer.probe_sync,
    ):
        with pytest.raises(RuntimeError, match="Probes output"):
            call()


def test_motion_blur_camera_buffers_and_split_render():
    renderer = render.Renderer(
        width=16,
        height=16,
        batch_size=1,
        cameras=(CameraSpec("camera", outputs=(Depth(), Probes(),)),),
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
    assert abs(renderer.outputs[Depth()].read()[0, 8, 8] - 3.0) < 1e-5

    renderer.set_cameras(camera_open[None])
    renderer.render()
    assert abs(renderer.outputs[Depth()].read()[0, 8, 8] - 2.0) < 1e-2


def test_dynamic_motion_blur_object():
    scene = make_scene(8.0)
    asset_pool = render.AssetPool()
    dynamic = render.Object(name="dynamic")
    dynamic.add_mesh(make_quad(0.0, half_size=1.0, color=(0.0, 1.0, 0.0)))
    asset = asset_pool.add_object(dynamic)
    renderer = render.Renderer(
        width=1,
        height=1,
        cameras=(CameraSpec("camera", outputs=(Depth(), Probes(),)),),
        fidelity="low",
        motion_blur_samples=2,
        num_overlays=1,
        max_overlay_instances=2,
        max_overlays_per_camera=1,
        dynamic_motion_blur=True,
    )
    renderer.init(scene, asset_pool=asset_pool)
    camera = renderer.camera(position=(0.0, 0.0, 0.0), look_at=(1.0, 0.0, 0.0), fov=60.0)
    renderer.set_motion_blur_cameras(camera[None], camera[None])
    renderer.attach(0, 0)
    placement = renderer.spawn(0, asset, render.Pose(render.make_transform(position=(4.0, 0.0, 0.0))))
    renderer.set_pose(placement, render.Pose(render.make_transform(position=(2.0, 0.0, 0.0))), render.Pose(render.make_transform(position=(4.0, 0.0, 0.0))))
    renderer.update()
    renderer.render()
    assert abs(renderer.outputs[Depth()].read()[0, 0, 0] - 4.0) < 1e-5


def test_overlay_pipeline(tmp_path):
    scene = make_scene(8.0)
    asset_pool = render.AssetPool()
    dynamic = render.Object(name="dynamic", segmentation_class=23)
    dynamic.add_mesh(make_quad(0.0, half_size=0.35, color=(1.0, 0.0, 0.0)))
    asset = asset_pool.add_object(dynamic)
    assembly_path = tmp_path / "overlay_assembly.glb"
    write_minimal_glb(assembly_path)
    assembly = render.load_assembly(assembly_path, fidelity="low")
    assembly.set_joint(1, "prismatic", axis=(-3.0, 2.0, 0.0))
    assert assembly.num_coordinates == 1
    assembly_asset = asset_pool.add_assembly(assembly)
    assert asset_pool.num_assets == 2

    renderer = render.Renderer(
        width=48,
        height=48,
        batch_size=2,
        num_probes=2,
        cameras=(CameraSpec("camera", outputs=(RGB(format="rgba8"), Depth(), Segmentation(), Probes(),)),),
        fidelity="low",
        num_overlays=2,
        max_overlay_instances=2,
        max_overlays_per_camera=2,
    )
    renderer.init(scene, asset_pool=asset_pool)
    look_forward(renderer)
    assert renderer.can_attach(0, 0)
    assert renderer.can_spawn(0, asset)
    renderer.attach(0, 0)
    placement = renderer.spawn(0, asset, render.Pose(render.make_transform(position=(4.0, 0.75, 0.0))))
    assert (placement.first_slot, placement.num_parts) == (0, 1)
    renderer.update()
    renderer.render()

    instance_id = scene.num_instances + placement.first_slot
    segmentation_live = renderer.outputs[Segmentation()].read()
    first_mask = segmentation_live[0] == instance_id
    assert first_mask.sum() > 4
    assert not np.any(segmentation_live[1] == instance_id)
    assert np.mean(renderer.outputs[RGB(format="rgba8")].read()[0][first_mask, 0]) > 100
    assert np.mean(renderer.outputs[Depth()].read()[0][first_mask]) < 5.0

    renderer.set_pose(placement, render.Pose(render.make_transform(position=(4.0, -0.75, 0.0))))
    renderer.update()
    renderer.render_launch()
    renderer.render_sync()
    second_view = renderer.outputs[Segmentation()].read(out=segmentation_live)
    second_mask = second_view[0] == instance_id
    assert np.shares_memory(segmentation_live, second_view)
    assert second_mask.sum() > 4
    assert not np.array_equal(first_mask, second_mask)

    renderer.detach(0, 0)
    renderer.attach(1, 0)
    renderer.update()
    renderer.render()
    segmentation = renderer.outputs[Segmentation()].read()
    assert not np.any(segmentation[0] == instance_id)
    assert np.any(segmentation[1] == instance_id)

    renderer.despawn(placement)
    renderer.update()
    renderer.render()
    assert not np.any(renderer.outputs[Segmentation()].read() == instance_id)
    assert renderer.can_spawn(0, asset)
    replacement = renderer.spawn(0, asset, render.Pose(render.make_transform(position=(4.0, -0.75, 0.0))))
    assert replacement.first_slot == placement.first_slot
    assert replacement != placement
    with pytest.raises(ValueError, match="stale"):
        renderer.set_pose(placement, render.Pose(render.make_transform()))
    renderer.update()
    renderer.render()
    assert np.any(renderer.outputs[Segmentation()].read()[1] == instance_id)

    assert not renderer.can_spawn(0, assembly_asset)
    renderer.despawn(replacement)
    assert renderer.can_spawn(0, assembly_asset)
    assembly_placement = renderer.spawn(0, assembly_asset, render.Pose(render.make_transform()))
    assert (assembly_placement.first_slot, assembly_placement.num_parts) == (0, 2)
    # part 0 stays at its rest frame, the prismatic joint slides part 1 from its rest at x = 3
    # into view at (0, 2, 0)
    renderer.set_pose(assembly_placement, render.Pose(), coordinates=[np.sqrt(13.0)])
    renderer.update()
    renderer.render()
    segmentation = renderer.outputs[Segmentation()].read()
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
        cameras=(CameraSpec("camera", outputs=(Segmentation(labels="semantic"), Probes(),)),),
        fidelity="low",
        num_overlays=1,
        max_overlay_instances=1,
        max_overlays_per_camera=1,
    )
    renderer.init(scene, asset_pool=asset_pool)
    look_forward(renderer)
    renderer.attach(0, 0)
    renderer.spawn(0, asset, render.Pose(render.make_transform(position=(4.0, 0.0, 0.0))))
    renderer.update()
    renderer.render()
    classes = renderer.outputs[Segmentation(labels="semantic")].read()
    assert np.any(classes == 7)
    assert np.any(classes == 23)


def test_jit_cache_reuse():
    config = render.RendererConfig(
        width=16, height=16, batch_size=1, num_probes=1, fidelity="low", cameras=(CameraSpec("camera", outputs=(Depth(), Probes(),)),),
        motion_blur_samples=1, anti_aliasing_grid=1, num_overlays=0,
        max_overlay_instances=0, max_overlays_per_camera=0,
    )
    from hyperdrone.render import _component
    library = jit.build_dir(_component.component()) / "jit" / f"render_{config.key()}.so"
    renderer = render.Renderer(width=16, height=16, batch_size=1, cameras=(CameraSpec("camera", outputs=(Depth(), Probes(),)),), fidelity="low")
    assert library.exists()
    modification_time = library.stat().st_mtime
    renderer_again = render.Renderer(**{name: getattr(config, name) for name in config.__dataclass_fields__})
    assert renderer_again.config == config
    assert library.stat().st_mtime == modification_time


def test_different_renderer_configurations_share_scene():
    scene = make_scene(3.0)
    depth_renderer = render.Renderer(width=16, height=16, cameras=(CameraSpec("camera", outputs=(Depth(), Probes(),)),), fidelity="low")
    rgbd_renderer = render.Renderer(width=16, height=16, cameras=(CameraSpec("camera", outputs=(RGB(format="rgba8"), Depth(), Probes(),)),), fidelity="low")
    for renderer in (depth_renderer, rgbd_renderer):
        renderer.init(scene)
        look_forward(renderer)
        renderer.render()
        assert abs(renderer.outputs[Depth()].read()[0, 8, 8] - 3.0) < 1e-2

    frame = rgbd_renderer.outputs[RGB(format="rgba8")].read()
    camera = depth_renderer.camera(position=(1, 0, 0), look_at=(2, 0, 0))
    depth_renderer.set_cameras(camera[None])
    depth_renderer.render()
    assert abs(depth_renderer.outputs[Depth()].read()[0, 8, 8] - 2.0) < 1e-2
    assert abs(rgbd_renderer.outputs[Depth()].read()[0, 8, 8] - 3.0) < 1e-2
    np.testing.assert_array_equal(rgbd_renderer.outputs[RGB(format="rgba8")].read(), frame)
    rgbd_renderer.render()
    assert abs(rgbd_renderer.outputs[Depth()].read()[0, 8, 8] - 3.0) < 1e-2


def test_output_views_keep_renderer_and_scene_alive():
    import gc

    scene = make_scene(2.0)
    renderer = render.Renderer(width=16, height=16, cameras=(CameraSpec("camera", outputs=(Depth(), Probes(),)),), fidelity="low")
    renderer.init(scene)
    look_forward(renderer)
    renderer.render()
    snapshot = renderer.outputs[Depth()].read()
    views = [renderer.outputs[Depth()].read()]
    if renderer.backend != "optix":
        views.append(np.from_dlpack(renderer.outputs[Depth()].blocks[0]))
    del renderer, scene
    gc.collect()
    for view in views:
        np.testing.assert_array_equal(view, snapshot)


def test_reused_host_output():
    renderer = render.Renderer(width=16, height=16, batch_size=1, cameras=(CameraSpec("camera", outputs=(Depth(), Probes(),)),), fidelity="low")
    renderer.init(make_scene(2.0))
    look_forward(renderer)
    renderer.render()
    first_view = renderer.outputs[Depth()].read()
    assert abs(first_view[0, 8, 8] - 2.0) < 1e-2
    camera = renderer.camera(position=(-1.0, 0.0, 0.0), look_at=(1.0, 0.0, 0.0), fov=60.0)
    renderer.set_cameras(camera[None])
    renderer.render()
    second_view = renderer.outputs[Depth()].read(out=first_view)
    assert np.shares_memory(first_view, second_view)
    assert abs(first_view[0, 8, 8] - 3.0) < 1e-2


def test_dlpack_export():
    renderer = render.Renderer(width=16, height=16, batch_size=1, cameras=(CameraSpec("camera", outputs=(RGB(format="rgba8"), Depth(), Probes(),)),), fidelity="low")
    renderer.init(make_scene(2.0))
    look_forward(renderer)
    renderer.render()
    frame_live = renderer.outputs[RGB(format="rgba8")].blocks[0]
    frame_raw_live = renderer.outputs[RGB(format="rgba8")].packed.blocks[0]
    depth_live = renderer.outputs[Depth()].blocks[0]
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
        assert np.array_equal(frame_array, renderer.outputs[RGB(format="rgba8")].read())
        assert frame_raw_array.shape == (1, 16, 16)
        assert frame_raw_array.dtype == np.uint32
        assert np.array_equal(frame_raw_array, renderer.outputs[RGB(format="rgba8")].packed.read())
        assert np.shares_memory(frame_array, frame_raw_array)
        assert np.array_equal(depth_array, renderer.outputs[Depth()].read())


def test_dlpack_camera_input():
    # any DLPack producer works as camera input; numpy's own arrays go through the same path
    renderer = render.Renderer(width=16, height=16, batch_size=1, cameras=(CameraSpec("camera", outputs=(Depth(), Probes(),)),), fidelity="low")
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
    assert abs(renderer.outputs[Depth()].read()[0, 8, 8] - 2.0) < 1e-2


@pytest.mark.skipif(render._backend() != "optix", reason="device camera input requires the OptiX backend")
def test_device_camera_input():
    renderer = render.Renderer(width=32, height=32, batch_size=2, cameras=(CameraSpec("camera", outputs=(Depth(), Probes(),)),), fidelity="low")
    renderer.init(make_scene(2.0))
    cameras = np.stack([
        np.asarray(renderer.camera(position=(0.0, 0.0, 0.0), look_at=(1.0, 0.0, 0.0), fov=60.0)).reshape(12),
        np.asarray(renderer.camera(position=(-1.0, 0.0, 0.0), look_at=(1.0, 0.0, 0.0), fov=60.0)).reshape(12),
    ])
    renderer.set_cameras(cameras)
    renderer.render()
    host_frame = renderer.outputs[Depth()].read()

    tensor_set = hyperdrone._cuda.upload(cameras[None])  # one set of two cameras
    renderer.set_cameras(tensor_set.view(0))            # dispatches on __dlpack_device__ == kDLCUDA
    renderer.render()
    device_frame = renderer.outputs[Depth()].read()

    assert np.array_equal(host_frame, device_frame)
    assert abs(device_frame[0, 16, 16] - 2.0) < 1e-2
    assert abs(device_frame[1, 16, 16] - 3.0) < 1e-2


def test_renderer_lifecycle():
    # two full create/render/destroy cycles in one process (regression: OWL's PinnedHostMem
    # cudaFree-on-pinned-memory bug killed the second lifecycle on the OptiX backend)
    import gc

    for _ in range(2):
        renderer = render.Renderer(width=16, height=16, batch_size=1, cameras=(CameraSpec("camera", outputs=(Depth(), Probes(),)),), fidelity="low")
        renderer.init(make_scene())
        look_forward(renderer)
        renderer.render()
        assert renderer.outputs[Depth()].read().shape == (1, 16, 16)
        del renderer
        gc.collect()


def test_error_on_wrong_output():
    renderer = render.Renderer(width=16, height=16, batch_size=1, cameras=(CameraSpec("camera", outputs=(Depth(), Probes(),)),), fidelity="low")
    renderer.init(make_scene())
    with pytest.raises(KeyError, match="not configured"):
        renderer.outputs[RGB(format="rgba8")].read()


def write_minimal_glb(path, color=(0.0, 1.0, 0.0), light=None):
    """Two glTF scene-root nodes sharing one quad mesh in the glTF x=2 plane. The FLU
    swizzle (x, -z, y) keeps x, so the first quad sits straight ahead of a camera at the
    origin looking along +X at distance 2; the second sits behind it at x=5. Two roots
    also make Assimp keep a wrapper root, which the assembly loader requires. A light
    colour adds a third, mesh-less root carrying a directional light that shines along +X,
    like the habitat_default_* roots of HSSD scenes."""
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
    if light is not None:
        gltf["scenes"][0]["nodes"].append(2)
        gltf["nodes"].append({"name": "light", "matrix": [0, 0, 1, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 1], "extensions": {"KHR_lights_punctual": {"light": 0}}})
        gltf["extensions"] = {"KHR_lights_punctual": {"lights": [{"type": "directional", "color": list(light), "intensity": 1.0}]}}
        gltf["extensionsUsed"] = ["KHR_lights_punctual"]
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
    renderer = render.Renderer(width=32, height=32, batch_size=1, cameras=(CameraSpec("camera", outputs=(RGB(format="rgba8"), Depth(), Probes(),)),), fidelity="medium")
    renderer.init(scene)
    look_forward(renderer)
    renderer.render()
    frame = renderer.outputs[RGB(format="rgba8")].read()
    depth = renderer.outputs[Depth()].read()
    center = frame[0, 16, 16]
    assert center[1] > center[0]  # green quad
    assert abs(depth[0, 16, 16] - 2.0) < 1e-2  # swizzled to x=+2 in FLU


def test_light_only_root_shades_preserved_instances_like_welded_load(tmp_path):
    glb = tmp_path / "lit.glb"
    write_minimal_glb(glb, light=(0.5, 0.5, 0.5))
    frames = {}
    for preserve_instances in (False, True):
        scene = render.load_scene(glb, fidelity="high", preserve_instances=preserve_instances)
        renderer = render.Renderer(width=32, height=32, batch_size=1, num_probes=0, cameras=(CameraSpec("camera", outputs=(RGB(format="rgba8"), Segmentation())),), fidelity="high")
        renderer.init(scene)
        look_forward(renderer)
        renderer.render()
        frames[preserve_instances] = renderer.outputs[RGB(format="rgba8")].read(), renderer.outputs[Segmentation()].read()
    assert scene.num_instances == 3
    np.testing.assert_array_equal(frames[True][0], frames[False][0])
    assert set(np.unique(frames[True][1]).tolist()) == {0, 0xFFFFFFFF}


def test_glb_assembly_names(tmp_path):
    glb = tmp_path / "quad.glb"
    write_minimal_glb(glb)
    assembly = render.load_assembly(glb, fidelity="low")
    assert assembly.num_objects == 2
    assert assembly.object_names() == ["quad", "quad_far"]
    scene = render.Scene()
    first, count = scene.add_assembly(assembly)
    assert (first, count) == (0, 2)


@pytest.mark.parametrize("load", [render.load_scene, render.load_object, render.load_assembly])
def test_scene_verbosity(load, tmp_path, capfd):
    path = tmp_path / "verbosity.glb"
    write_minimal_glb(path)
    previous = hyperdrone.verbosity()
    try:
        hyperdrone.set_verbosity("warning")
        load(path, fidelity="low")
        assert "#rl_tools::rendering::datasets:" not in capfd.readouterr().err
        hyperdrone.set_verbosity("info")
        load(path, fidelity="low")
        assert "#rl_tools::rendering::datasets: Loaded" in capfd.readouterr().err
    finally:
        hyperdrone.set_verbosity(previous)


def test_verbosity_during_concurrent_scene_loads(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    path = tmp_path / "concurrent.glb"
    write_minimal_glb(path)
    render.load_scene(path, fidelity="low")
    barrier = Barrier(3, timeout=30)

    def load():
        counts = []
        for _ in range(32):
            barrier.wait()
            counts.append(render.load_scene(path, fidelity="low").num_instances)
        return counts

    previous = hyperdrone.verbosity()
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(load)
            second = pool.submit(load)
            for i in range(32):
                barrier.wait()
                hyperdrone.set_verbosity("debug" if i % 2 else "warning")
            assert first.result(timeout=30) == [1] * 32
            assert second.result(timeout=30) == [1] * 32
    finally:
        hyperdrone.set_verbosity(previous)


GLB = Path(os.environ.get("HYPERDRONE_RLTOOLS_ROOT", Path(__file__).resolve().parents[5])) / "tests" / "data" / "ProcTHOR-Train-1.glb"


@pytest.mark.skipif(not GLB.exists(), reason="ProcTHOR test scene not available")
def test_glb_scene():
    scene = render.load_scene(GLB, fidelity="high")
    assert scene.num_objects >= 1
    assert scene.num_instances >= 1
    renderer = render.Renderer(width=64, height=48, batch_size=1, cameras=(CameraSpec("camera", outputs=(RGB(format="rgba8"), Depth(), Probes(),)),), fidelity="high")
    renderer.init(scene)
    bounds = renderer.scene_bounds
    assert bounds["camera_radius"] > 0
    renderer.generate_cameras()
    renderer.render()
    frame = renderer.outputs[RGB(format="rgba8")].read()
    depth = renderer.outputs[Depth()].read()
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


def test_loaded_instance_and_semantic_labels(labelled_glb):
    scene = render.load_scene(labelled_glb, rgb=False, preserve_instances=True)
    assert scene.num_instances == 4
    assert scene.segmentation_taxonomy == 'test'
    assert scene.segmentation_classes == {0: 'Unknown', 21: 'Chair', 162: 'Wall'}
    renderer = render.Renderer(width=24, height=24, batch_size=4, fidelity='low',
                               cameras=(CameraSpec("camera", outputs=(Segmentation(), Segmentation(labels="semantic"), Probes(),)),))
    renderer.init(scene)
    cameras = [renderer.camera(position=(0, 1.5-i, 0), look_at=(2, 1.5-i, 0)) for i in range(4)]
    renderer.set_cameras(np.array(cameras, dtype=np.float32))
    renderer.render()
    instances, classes = renderer.outputs[Segmentation()].read(), renderer.outputs[Segmentation(labels="semantic")].read()
    assert instances[:, 12, 12].tolist() == [0, 1, 2, 3]
    assert classes[:, 12, 12].tolist() == [21, 21, 162, 0]
    misses = instances == 0xffffffff
    assert misses.any()
    np.testing.assert_array_equal(classes == 0xffffffff, misses)
    assert scene.object_name(scene.instance_object(int(instances[1, 12, 12]))) == 'object-1'


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
    renderer = render.Renderer(width=24, height=24, fidelity='low', cameras=(CameraSpec("camera", outputs=(Segmentation(labels="semantic"), Probes(),)),),
                               num_overlays=1, max_overlay_instances=4, max_overlays_per_camera=1)
    renderer.init(scene, asset_pool=pool)
    renderer.attach(0, 0)
    handle = renderer.spawn(0, asset, render.Pose(render.make_transform(position=(-1, 0, 0))))
    camera = renderer.camera(position=(0, 1.5, 0), look_at=(2, 1.5, 0))
    renderer.set_cameras(camera[None])
    renderer.update()
    renderer.render()
    assert renderer.outputs[Segmentation(labels="semantic")].read()[0, 12, 12] == 21
    renderer.set_pose(handle, render.Pose(render.make_transform(position=(-1, 2, 0))))
    renderer.update()
    renderer.render()
    assert renderer.outputs[Segmentation(labels="semantic")].read()[0, 12, 12] == 162


def test_welded_semantics_rejected(labelled_glb):
    scene = render.load_scene(labelled_glb, rgb=False)
    assert scene.num_instances == 1
    renderer = render.Renderer(width=24, height=24, fidelity='low', cameras=(CameraSpec("camera", outputs=(Segmentation(labels="semantic"), Probes(),)),))
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
    renderer = render.Renderer(width=24, height=24, fidelity='low', cameras=(CameraSpec("camera", outputs=(Segmentation(labels="semantic"), Probes(),)),),
                               num_overlays=1, max_overlay_instances=4, max_overlays_per_camera=1)
    renderer.init(scene, asset_pool=pool)
    if pool is not None:
        renderer.attach(0, 0)
        renderer.spawn(0, asset, render.Pose(render.make_transform(position=(0, 0, 0))))
        renderer.update()
    camera = renderer.camera(position=(0, 1.5, 0), look_at=(2, 1.5, 0))
    renderer.set_cameras(camera[None])
    renderer.render()
    assert renderer.outputs[Segmentation(labels="semantic")].read()[0, 12, 12] == 21


def test_failed_semantic_init_can_be_retried(labelled_glb):
    scene = render.load_scene(labelled_glb, rgb=False)
    renderer = render.Renderer(width=24, height=24, fidelity='low', cameras=(CameraSpec("camera", outputs=(Segmentation(labels="semantic"), Probes(),)),))
    with pytest.raises(ValueError, match='welded classes'):
        renderer.init(scene)
    assert renderer._scene is None
    scene.set_object_segmentation_class(0, 21)
    renderer.init(scene)
    camera = renderer.camera(position=(0, 1.5, 0), look_at=(2, 1.5, 0))
    renderer.set_cameras(camera[None])
    renderer.render()
    assert renderer.outputs[Segmentation(labels="semantic")].read()[0, 12, 12] == 21


def test_label_remapping_across_scene_and_pool(labelled_glb):
    assembly = render.load_assembly(labelled_glb, rgb=False)
    source_classes = assembly.segmentation_classes
    scene = render.Scene()
    scene.add_assembly(assembly)
    pool = render.AssetPool()
    asset = pool.add_assembly(assembly)
    mapping = {0: 0, 21: 2147483648, 162: 4294967039}
    classes = {0: 'Unknown', 2147483648: 'Furniture', 4294967039: 'Structure'}
    for target in (scene, assembly, pool):
        for bad_mapping in ({0: 0, 21: 2147483648}, {0: 0, 21: 2147483648, 162: 999}):
            with pytest.raises(ValueError):
                target.remap_segmentation_classes(bad_mapping, taxonomy='custom', classes=classes)
            assert target.segmentation_taxonomy == 'test'
            assert target.segmentation_classes == source_classes
        target.remap_segmentation_classes(mapping, taxonomy='custom', classes=classes)
        assert target.segmentation_taxonomy == 'custom'
        assert target.segmentation_classes == classes
    assert [scene.object_segmentation_class(i) for i in range(4)] == [2147483648, 2147483648, 4294967039, 0]
    assert [assembly.segmentation_class(i) for i in range(4)] == [2147483648, 2147483648, 4294967039, 0]
    assert [pool.object_segmentation_class(asset, i) for i in range(4)] == [2147483648, 2147483648, 4294967039, 0]
    renderer = render.Renderer(width=24, height=24, fidelity='low', cameras=(CameraSpec("camera", outputs=(Segmentation(labels="semantic"), Probes(),)),),
                               num_overlays=1, max_overlay_instances=4, max_overlays_per_camera=1)
    renderer.init(scene, asset_pool=pool)
    renderer.attach(0, 0)
    handle = renderer.spawn(0, asset, render.Pose(render.make_transform(position=(-1, 0, 0))))
    camera = renderer.camera(position=(0, 1.5, 0), look_at=(2, 1.5, 0))
    renderer.set_cameras(camera[None])
    renderer.update()
    renderer.render()
    assert renderer.outputs[Segmentation(labels="semantic")].read()[0, 12, 12] == 2147483648
    renderer.set_pose(handle, render.Pose(render.make_transform(position=(-1, 2, 0))))
    renderer.update()
    renderer.render()
    assert renderer.outputs[Segmentation(labels="semantic")].read()[0, 12, 12] == 4294967039


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
    obj.remap_segmentation_classes({7: 4294967039})
    assert obj.segmentation_taxonomy == ''
    assert obj.segmentation_classes == {}
    assert obj.segmentation_class == 4294967039
    with pytest.raises(ValueError, match='reserved'):
        obj.segmentation_class = 0xffffffff
    assert obj.segmentation_class == 4294967039
    with pytest.raises(ValueError, match='reserved'):
        render.Object(segmentation_class=0xffffffff)


def test_overlay_handles_validate_complete_allocation():
    scene = make_scene(8)
    pool = render.AssetPool()
    asset = pool.add_mesh(make_quad(0))
    renderer = render.Renderer(8, 8, cameras=(CameraSpec("camera", outputs=(Segmentation(), Probes(),)),), num_overlays=2, max_overlay_instances=2, max_overlays_per_camera=1)
    renderer.init(scene, asset_pool=pool)
    first = renderer.spawn(0, asset, render.Pose(render.make_transform(position=(2, 0, 0))))
    neighbor = renderer.spawn(0, asset, render.Pose(render.make_transform(position=(3, 0, 0))))
    renderer.spawn(1, asset, render.Pose(render.make_transform(position=(4, 0, 0))))
    forged = (replace(first, num_parts=2), replace(first, first_slot=neighbor.first_slot), replace(first, overlay=1), replace(first, first_slot=2**64 - 1))
    for placement in forged:
        for verb in (lambda: renderer.despawn(placement), lambda: renderer.set_pose(placement, render.Pose(render.make_transform())), lambda: renderer.set_pose(placement, render.Pose(), render.Pose())):
            with pytest.raises(ValueError, match="allocation"):
                verb()
    with pytest.raises(TypeError, match="OverlayPlacement"):
        renderer.despawn((0, 1, 0))
    other = render.Renderer(8, 8, cameras=(CameraSpec("camera", outputs=(Segmentation(), Probes(),)),), num_overlays=2, max_overlay_instances=2, max_overlays_per_camera=1)
    with pytest.raises(ValueError, match="another renderer"):
        other.despawn(first)
    renderer.despawn(first)
    replacement = renderer.spawn(0, asset, render.Pose(render.make_transform()))
    assert replacement.first_slot == first.first_slot
    assert replacement != first
    with pytest.raises(ValueError, match="stale"):
        renderer.set_pose(first, render.Pose(render.make_transform()))
    class UnreadableCuda:
        def __dlpack_device__(self):
            raise AssertionError("a rejected handle must not consume its tensor")
    with pytest.raises(ValueError, match="stale"):
        renderer.set_pose(first, render.Pose(UnreadableCuda()))
    renderer.set_pose(neighbor, render.Pose(render.make_transform()))
    for raw in ((2**64 - 1, 2, 0), (0, 0, 0), (2, 1, 0)):
        with pytest.raises(ValueError):
            renderer._renderer.despawn(0, raw)


def test_overlay_errors_leave_renderer_usable():
    scene = make_scene(8)
    pool = render.AssetPool()
    with pytest.raises(ValueError, match="at least one part"):
        pool.add_assembly(render.ObjectAssembly())
    asset = pool.add_mesh(make_quad(0))
    renderer = render.Renderer(8, 8, num_overlays=2, max_overlay_instances=1, max_overlays_per_camera=1)
    renderer.init(scene, asset_pool=pool)
    for camera, overlay in ((1, 0), (0, 2)):
        with pytest.raises(ValueError, match="out of range"):
            renderer.attach(camera, overlay)
        with pytest.raises(ValueError, match="out of range"):
            renderer.detach(camera, overlay)
    renderer.attach(0, 0)
    with pytest.raises(RuntimeError, match="capacity"):
        renderer.attach(0, 1)
    for overlay, handle in ((2, asset), (0, asset + 1)):
        with pytest.raises(ValueError, match="out of range"):
            renderer.spawn(overlay, handle, render.Pose(render.make_transform()))
    placement = renderer.spawn(0, asset, render.Pose(render.make_transform()))
    with pytest.raises(RuntimeError, match="capacity"):
        renderer.spawn(0, asset, render.Pose(render.make_transform()))
    renderer.despawn(placement)
    assert renderer.can_spawn(0, asset)


@pytest.mark.parametrize("name", ["width", "height", "batch_size", "motion_blur_samples", "anti_aliasing_grid"])
@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_renderer_dimensions_fail_before_build(name, value):
    options = {"width": 8, "height": 8, name: value}
    with pytest.raises(ValueError, match=name):
        render.Renderer(**options)


def test_depth_is_axial_close_sample_and_misses_are_infinite():
    renderer = render.Renderer(16, 16, cameras=(CameraSpec("camera", outputs=(Depth(), Segmentation(), Probes(),)),), motion_blur_samples=4, anti_aliasing_grid=2)
    renderer.init(make_scene(3))
    camera_open = renderer.camera((0, 0, 0), (1, 0, 0))
    camera_close = renderer.camera((-1, 0, 0), (1, 0, 0))
    renderer.set_motion_blur_cameras(camera_open, camera_close)
    renderer.render()
    depth = renderer.outputs[Depth()].read()
    hit = renderer.outputs[Segmentation()].read() != 0xFFFFFFFF
    assert hit.any() and (~hit).any()
    np.testing.assert_allclose(depth[hit], 4, atol=1e-6)
    assert np.isposinf(depth[~hit]).all()
    renderer.set_cameras(camera_close)
    renderer.render()
    np.testing.assert_array_equal(renderer.outputs[Depth()].read(), depth)


@pytest.mark.skipif(render._backend() != "optix", reason="device camera ordering requires OptiX")
def test_device_cameras_are_retained_and_order_probes():
    renderer = render.Renderer(256, 256, cameras=(CameraSpec("camera", outputs=(RGB(format="rgba8"), Depth(), Probes(),)),), motion_blur_samples=2)
    renderer.init(make_scene(3))
    renderer.generate_probe_directions()
    opening = renderer.camera((0, 0, 0), (1, 0, 0))
    closing = renderer.camera((-1, 0, 0), (1, 0, 0))
    renderer.set_motion_blur_cameras(opening, closing)
    renderer.render()
    expected = renderer.outputs[Depth()].read()
    renderer.probe()
    distances, hits = renderer.outputs[Probes()].read().values()
    from hyperdrone import _cuda
    cuda_open = _cuda.upload(np.asarray(opening).reshape(1, 12)).view()
    cuda_close = _cuda.upload(np.asarray(closing).reshape(1, 12)).view()
    with pytest.raises(ValueError, match="same memory"):
        renderer.set_motion_blur_cameras(cuda_open, closing)
    renderer.set_motion_blur_cameras(cuda_open, cuda_close, stream=np.int64(0))
    assert list(renderer._inputs_in_flight) in ([(cuda_open, cuda_close)], [])
    renderer.render()
    assert not renderer._inputs_in_flight
    np.testing.assert_array_equal(renderer.outputs[Depth()].read(), expected)
    for _ in range(8):
        renderer.set_cameras(opening)
        renderer.render_launch()
        renderer.set_cameras(cuda_close)
        assert list(renderer._inputs_in_flight) in ([(cuda_close,)], [])
        renderer.probe()
        assert not renderer._inputs_in_flight
        actual_distances, actual_hits = renderer.outputs[Probes()].read().values()
        np.testing.assert_array_equal(actual_distances, distances)
        np.testing.assert_array_equal(actual_hits, hits)
    renderer.synchronize()


def test_oversized_asset_does_not_consume_renderer():
    pool = render.AssetPool()
    assembly = render.ObjectAssembly()
    obj = render.Object()
    obj.add_mesh(make_quad(0))
    for _ in range(2):
        assembly.add_part(obj, render.make_transform())
    pool.add_assembly(assembly)
    renderer = render.Renderer(8, 8, num_overlays=1, max_overlay_instances=1, max_overlays_per_camera=1)
    scene = make_scene(4)
    with pytest.raises(ValueError, match="part count"):
        renderer.init(scene, asset_pool=pool)
    assert renderer.init(scene) is None
    look_forward(renderer)
    assert renderer.render() is None
    assert renderer.outputs[RGB(format="rgba8")].read().shape == (1, 8, 8, 4)


@pytest.mark.skipif(render._backend() != "optix", reason="CUDA producer ordering requires OptiX")
def test_cameras_wait_for_nondefault_producer_stream():
    import ctypes
    from hyperdrone import _cuda as cuda

    runtime = ctypes.CDLL("libcudart.so")
    runtime.cudaStreamCreateWithFlags.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
    runtime.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
    runtime.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
    runtime.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    runtime.cudaMallocHost.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    runtime.cudaFree.argtypes = [ctypes.c_void_p]
    runtime.cudaFreeHost.argtypes = [ctypes.c_void_p]
    runtime.cudaMemsetAsync.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t, ctypes.c_void_p]
    runtime.cudaMemcpyAsync.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
    get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
    get_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
    get_pointer.restype = ctypes.c_void_p
    renderer = render.Renderer(16, 16, cameras=(CameraSpec("camera", outputs=(Depth(), Probes(),)),))
    renderer.init(make_scene(3))
    camera = renderer.camera((-1, 0, 0), (1, 0, 0)).reshape(1, 12)
    renderer.set_cameras(camera)
    renderer.render()
    expected = renderer.outputs[Depth()].read()
    initial = renderer.camera((0, 0, 0), (1, 0, 0)).reshape(1, 12)
    source = cuda.upload(initial).view()
    capsule = source.__dlpack__()
    pointer = ctypes.c_void_p.from_address(get_pointer(capsule, b"dltensor")).value
    stream, scratch, pinned = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
    try:
        assert runtime.cudaStreamCreateWithFlags(ctypes.byref(stream), 1) == 0
        assert runtime.cudaMalloc(ctypes.byref(scratch), 64 * 1024 * 1024) == 0
        assert runtime.cudaMallocHost(ctypes.byref(pinned), camera.nbytes) == 0
        ctypes.memmove(pinned, camera.ctypes.data, camera.nbytes)
        for _ in range(32):
            assert runtime.cudaMemsetAsync(scratch, 0, 64 * 1024 * 1024, stream) == 0
        assert runtime.cudaMemcpyAsync(pointer, pinned, camera.nbytes, 1, stream) == 0
        renderer.set_cameras(source, stream=stream.value)
        renderer.render()
        np.testing.assert_array_equal(renderer.outputs[Depth()].read(), expected)
    finally:
        renderer.synchronize()
        if stream:
            runtime.cudaStreamSynchronize(stream)
            runtime.cudaStreamDestroy(stream)
        if scratch:
            runtime.cudaFree(scratch)
        if pinned:
            runtime.cudaFreeHost(pinned)



def test_placement_handles_cannot_cross_processes_by_pickle():
    import pickle
    placement = render.OverlayPlacement(overlay=0, first_slot=0, num_parts=1, _generation=0, _renderer=0)
    with pytest.raises(TypeError, match="cannot be pickled"):
        pickle.dumps(placement)
