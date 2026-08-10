import math
import os
from pathlib import Path

import numpy as np
import pytest

import hyperdrone
from hyperdrone import jit, render


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
    camera = renderer.camera(position=(0.0, 0.0, 0.0), look_at=(1.0, 0.0, 0.0), fov=math.radians(60.0))
    renderer.set_cameras(np.repeat(camera[None, :, :], renderer.num_cameras, axis=0))


def test_depth():
    distance = 2.0
    renderer = render.Renderer(width=64, height=64, num_cameras=1, output="depth", shading="low")
    renderer.init(make_scene(distance))
    look_forward(renderer)
    renderer.render("depth")
    depth = renderer.depth()
    assert depth.shape == (1, 64, 64)
    center = depth[0, 32, 32]
    assert abs(center - distance) < 1e-2


def test_rgb():
    renderer = render.Renderer(width=32, height=32, num_cameras=2, output="rgb", shading="low")
    renderer.init(make_scene())
    look_forward(renderer)
    renderer.render("rgb")
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
    renderer = render.Renderer(width=32, height=32, num_cameras=1, output="segmentation", shading="low")
    renderer.init(scene)
    look_forward(renderer)
    renderer.render("segmentation")
    segmentation = renderer.segmentation()
    left = segmentation[0, 16, 8]    # image x grows toward -Y, so +Y (instance 1) is on the left
    right = segmentation[0, 16, 24]
    assert (left, right) == (1, 0)


def test_collision_probes():
    renderer = render.Renderer(width=16, height=16, num_cameras=1, num_probes=4, output="depth", shading="low")
    renderer.init(make_scene(3.0))
    look_forward(renderer)
    renderer.generate_probe_directions()
    renderer.render("collision")
    distances, hits = renderer.collisions()
    assert distances.shape == (1, 4)
    # probe 0 is the camera forward direction and must hit the wall at ~3m
    assert hits[0, 0] == 1
    assert abs(distances[0, 0] - 3.0) < 1e-2


def test_motion_blur_camera_buffers_and_split_render():
    renderer = render.Renderer(
        width=16,
        height=16,
        num_cameras=1,
        output="depth",
        shading="low",
        motion_blur_samples=2,
        anti_aliasing_grid=2,
    )
    renderer.init(make_scene(2.0))
    camera_open = renderer.camera(position=(0.0, 0.0, 0.0), look_at=(1.0, 0.0, 0.0))
    camera_close = renderer.camera(position=(-1.0, 0.0, 0.0), look_at=(1.0, 0.0, 0.0))
    renderer.set_motion_blur_cameras(camera_open[None], camera_close[None])
    renderer.render_launch("depth")
    renderer.render_sync("depth")
    blurred_depth = renderer.depth()[0, 8, 8]
    assert 2.0 < blurred_depth < 3.0

    renderer.set_cameras(camera_open[None])
    renderer.render("depth")
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
        shading="low",
        motion_blur_samples=2,
        num_overlays=1,
        max_overlay_instances=2,
        max_overlays_per_camera=1,
        dynamic_motion_blur=True,
    )
    renderer.init(scene, asset_pool)
    camera = renderer.camera(position=(0.0, 0.0, 0.0), look_at=(1.0, 0.0, 0.0), fov=math.radians(60.0))
    renderer.set_motion_blur_cameras(camera[None], camera[None])
    renderer.attach(0, 0)
    placement = renderer.spawn(0, asset, render.make_transform(position=(4.0, 0.0, 0.0)))
    renderer.set_transform_pair(0, placement, render.make_transform(position=(2.0, 0.0, 0.0)), render.make_transform(position=(4.0, 0.0, 0.0)))
    renderer.update()
    renderer.render("depth")
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
    renderer.render("depth")
    assert abs(renderer.depth()[0, 0, 0] - 3.0) < 1e-2


def test_overlay_pipeline_and_output_saves(tmp_path):
    scene = make_scene(8.0)
    asset_pool = render.AssetPool()
    dynamic = render.Object(name="dynamic", segmentation_class=23)
    dynamic.add_mesh(make_quad(0.0, half_size=0.35, color=(1.0, 0.0, 0.0)))
    asset = asset_pool.add_object(dynamic)
    assembly_path = tmp_path / "overlay_assembly.glb"
    write_minimal_glb(assembly_path)
    assembly_asset = asset_pool.add_assembly(render.load_assembly(assembly_path, shading="low"))
    assert asset_pool.num_assets == 2

    renderer = render.Renderer(
        width=48,
        height=48,
        num_cameras=2,
        num_probes=2,
        output="rgbd_segmentation",
        shading="low",
        num_overlays=2,
        max_overlay_instances=2,
        max_overlays_per_camera=2,
    )
    renderer.init(scene, asset_pool)
    look_forward(renderer)
    renderer.generate_probe_directions()
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

    outputs = {
        "frame.png": renderer.save_image,
        "depth.png": renderer.save_depth_image,
        "depth.bin": renderer.save_depth_raw,
        "segmentation.png": renderer.save_segmentation_image,
        "probes.bin": renderer.save_probes,
    }
    for name, save in outputs.items():
        path = tmp_path / name
        save(path)
        assert path.stat().st_size > 0


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
        shading="low",
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
    renderer.render("segmentation")
    classes = renderer.segmentation()
    assert np.any(classes == 7)
    assert np.any(classes == 23)


def test_jit_cache_reuse():
    config = render.RendererConfig(
        width=16, height=16, num_cameras=1, num_probes=1, shading=0, output_mode=2,
        motion_blur_samples=1, anti_aliasing_grid=1, num_overlays=0,
        max_overlay_instances=0, max_overlays_per_camera=0, semantic_segmentation=False,
    )
    from hyperdrone.render import _component
    library = jit.build_dir(_component.component()) / "jit" / f"render_{config.key()}.so"
    renderer = render.Renderer(width=16, height=16, num_cameras=1, output="depth", shading="low")
    assert library.exists()
    modification_time = library.stat().st_mtime
    renderer_again = render.Renderer(width=16, height=16, num_cameras=1, output="depth", shading="low")
    assert library.stat().st_mtime == modification_time


def test_zero_copy_host_view():
    renderer = render.Renderer(width=16, height=16, num_cameras=1, output="depth", shading="low")
    renderer.init(make_scene(2.0))
    look_forward(renderer)
    renderer.render("depth")
    first_view = renderer.depth(copy=False)
    assert abs(first_view[0, 8, 8] - 2.0) < 1e-2
    # stepping the camera back must update the earlier view in place (it aliases the
    # renderer's staging buffer)
    camera = renderer.camera(position=(-1.0, 0.0, 0.0), look_at=(1.0, 0.0, 0.0), fov=math.radians(60.0))
    renderer.set_cameras(camera[None])
    renderer.render("depth")
    second_view = renderer.depth(copy=False)
    assert np.shares_memory(first_view, second_view)
    assert abs(first_view[0, 8, 8] - 3.0) < 1e-2


def test_dlpack_export():
    renderer = render.Renderer(width=16, height=16, num_cameras=1, output="rgbd", shading="low")
    renderer.init(make_scene(2.0))
    look_forward(renderer)
    renderer.render("rgb_depth")
    frame_live = renderer.frame_dlpack()
    depth_live = renderer.depth_dlpack()
    device = frame_live.__dlpack_device__()
    if renderer.backend == "optix":
        assert device == (2, 0)  # kDLCUDA
    else:
        assert device[0] == 1  # kDLCPU: live buffer is host-visible, numpy can alias it
        frame_array = np.from_dlpack(frame_live)
        depth_array = np.from_dlpack(depth_live)
        assert frame_array.shape == (1, 16, 16)
        assert np.array_equal(frame_array, renderer.frame_raw())
        assert np.array_equal(depth_array, renderer.depth())


def test_dlpack_camera_input():
    # any DLPack producer works as camera input; numpy's own arrays go through the same path
    renderer = render.Renderer(width=16, height=16, num_cameras=1, output="depth", shading="low")
    renderer.init(make_scene(2.0))
    camera = renderer.camera(position=(0.0, 0.0, 0.0), look_at=(1.0, 0.0, 0.0), fov=math.radians(60.0))

    class DLPackOnly:
        def __init__(self, array):
            self._array = array

        def __dlpack__(self, **kwargs):
            return self._array.__dlpack__(**kwargs)

        def __dlpack_device__(self):
            return self._array.__dlpack_device__()

    renderer.set_cameras(DLPackOnly(np.ascontiguousarray(camera[None])))
    renderer.render("depth")
    assert abs(renderer.depth()[0, 8, 8] - 2.0) < 1e-2


@pytest.mark.skipif(render.backend() != "OPTIX", reason="device camera input requires the OptiX backend")
def test_device_camera_input():
    renderer = render.Renderer(width=32, height=32, num_cameras=2, output="depth", shading="low")
    renderer.init(make_scene(2.0))
    cameras = np.stack([
        np.asarray(renderer.camera(position=(0.0, 0.0, 0.0), look_at=(1.0, 0.0, 0.0), fov=math.radians(60.0))).reshape(12),
        np.asarray(renderer.camera(position=(-1.0, 0.0, 0.0), look_at=(1.0, 0.0, 0.0), fov=math.radians(60.0))).reshape(12),
    ])
    renderer.set_cameras(cameras)
    renderer.render("depth")
    host_frame = renderer.depth()

    tensor_set = hyperdrone.cuda.upload(cameras[None])  # one set of two cameras
    renderer.set_cameras(tensor_set.view(0))            # dispatches on __dlpack_device__ == kDLCUDA
    renderer.render("depth")
    device_frame = renderer.depth()

    assert np.array_equal(host_frame, device_frame)
    assert abs(device_frame[0, 16, 16] - 2.0) < 1e-2
    assert abs(device_frame[1, 16, 16] - 3.0) < 1e-2


def test_renderer_lifecycle():
    # two full create/render/destroy cycles in one process (regression: OWL's PinnedHostMem
    # cudaFree-on-pinned-memory bug killed the second lifecycle on the OptiX backend)
    import gc

    for _ in range(2):
        renderer = render.Renderer(width=16, height=16, num_cameras=1, output="depth", shading="low")
        renderer.init(make_scene())
        look_forward(renderer)
        renderer.render("depth")
        assert renderer.depth().shape == (1, 16, 16)
        del renderer
        gc.collect()


def test_error_on_wrong_output():
    renderer = render.Renderer(width=16, height=16, num_cameras=1, output="depth", shading="low")
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
    scene = render.load_scene(glb, shading="medium")
    assert scene.num_objects == 1
    assert scene.num_instances == 1
    assert scene.num_lights == 3  # neutral fill lights added for light-less files
    renderer = render.Renderer(width=32, height=32, num_cameras=1, output="rgbd", shading="medium")
    renderer.init(scene)
    look_forward(renderer)
    renderer.render("rgb_depth")
    frame = renderer.frame()
    depth = renderer.depth()
    center = frame[0, 16, 16]
    assert center[1] > center[0]  # green quad
    assert abs(depth[0, 16, 16] - 2.0) < 1e-2  # swizzled to x=+2 in FLU


def test_glb_assembly_names(tmp_path):
    glb = tmp_path / "quad.glb"
    write_minimal_glb(glb)
    assembly = render.load_assembly(glb, shading="low")
    assert assembly.num_objects == 2
    assert assembly.object_names() == ["quad", "quad_far"]
    scene = render.Scene()
    first, count = scene.add_assembly(assembly)
    assert (first, count) == (0, 2)


GLB = Path(os.environ.get("HYPERDRONE_RLTOOLS_ROOT", Path(__file__).resolve().parents[5])) / "tests" / "data" / "ProcTHOR-Train-1.glb"


@pytest.mark.skipif(not GLB.exists(), reason="ProcTHOR test scene not available")
def test_glb_scene():
    scene = render.load_scene(GLB, shading="high")
    assert scene.num_objects >= 1
    assert scene.num_instances >= 1
    renderer = render.Renderer(width=64, height=48, num_cameras=1, output="rgbd", shading="high")
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
    assembly = render.load_assembly(GLB, shading="low")
    assert assembly.num_objects >= 1
    names = assembly.object_names()
    assert len(names) == assembly.num_objects
    scene = render.Scene()
    first, count = scene.add_assembly(assembly)
    assert count == assembly.num_parts
    assert scene.num_instances == count
