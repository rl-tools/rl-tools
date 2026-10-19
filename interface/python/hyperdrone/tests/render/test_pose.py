from hyperdrone.camera import CameraSpec, RGB, Depth, Segmentation
import json
import struct

import numpy as np
import pytest

from hyperdrone import render
from test_render import linear_to_rgb8, look_forward, make_quad, make_scene, write_minimal_glb


def pose(x=0.0, y=0.0, z=0.0):
    out = np.eye(4, dtype=np.float32)[:3].copy()
    out[:, 3] = x, y, z
    return out


def assembly(rests=(pose(1), pose(2)), parent=None):
    asset = render.ObjectAssembly()
    for index, (color, rest) in enumerate(zip([(1, 0, 0), (0, 1, 0)], rests)):
        obj = render.Object(name=f"part_{index}")
        obj.add_mesh(make_quad(0, half_size=0.5, color=color))
        asset.add_part(obj, rest, parent=parent if index == 1 else None)
    return asset


def renderer_for(*assets, motion=False):
    renderer = render.Renderer(width=32, height=32, batch_size=1, num_probes=0,
                               cameras=(CameraSpec("camera", outputs=(RGB(format="rgba8"), Depth(), Segmentation(),)),), fidelity="low", num_overlays=2,
                               max_overlay_instances=4, max_overlays_per_camera=2,
                               motion_blur_samples=4 if motion else 1, dynamic_motion_blur=motion)
    pool = render.AssetPool()
    handles = [pool.add_assembly(asset) for asset in assets]
    renderer.init(make_scene(8), asset_pool=pool)
    renderer.attach(0, 0)
    look_forward(renderer)
    return renderer, handles


def frame(renderer):
    renderer.update()
    renderer.render()
    return renderer.outputs[RGB(format="rgba8")].read().copy()


def test_joints_on_assemblies():
    asset = assembly(parent=0)
    assert asset.part_parent(1) == 0 and asset.num_coordinates == 0
    asset.set_joint("part_0", "revolute", axis=(2, 0, 0), pivot=(0, 0.5, 0), coordinate=1)
    asset.set_joint(1, "prismatic", axis=(0, 1, 0))
    assert asset.num_coordinates == 2
    joint, axis, pivot, coordinate = asset.joint(0)
    assert joint == "revolute" and coordinate == 1
    np.testing.assert_array_equal(axis, [2, 0, 0])
    np.testing.assert_array_equal(pivot, [0, 0.5, 0])
    assert asset.joint("part_1")[0] == "prismatic"
    render.AssetPool().add_assembly(asset)
    with pytest.raises(ValueError, match="unknown joint"):
        asset.set_joint(0, "spherical")
    with pytest.raises(ValueError, match="nonzero axis"):
        asset.set_joint(0, "revolute", axis=(0, 0, 0))
    with pytest.raises(ValueError, match="no part named"):
        asset.set_joint("missing", "fixed")
    asset.set_joint(1, "prismatic", axis=(0, 1, 0), coordinate=3)
    with pytest.raises(ValueError, match="coordinates 0..num_coordinates-1"):
        render.AssetPool().add_assembly(asset)
    with pytest.raises(ValueError, match="part index"):
        asset.add_part(render.Object(), pose(), parent=5)


def test_pose_composes_base_scale_and_rest_frames():
    asset = assembly()
    base = render.make_transform(position=(3.0, -0.4, 0.2), orientation_wxyz=(np.cos(0.2), 0, 0, np.sin(0.2)))
    scaled = base.copy()
    scaled[:, :3] *= 1.5
    moved = render.ObjectAssembly()
    for index in range(2):
        obj = render.Object()
        obj.add_mesh(make_quad(0, half_size=0.5, color=[(1, 0, 0), (0, 1, 0)][index]))
        moved.add_part(obj, render.compose_transforms(scaled, asset.part(index)[1]))
    renderer, (handle, moved_handle) = renderer_for(asset, moved)
    placement = renderer.spawn(0, handle, render.Pose(base, scale=1.5))
    posed = frame(renderer)
    renderer.despawn(placement)
    renderer.spawn(0, moved_handle)
    np.testing.assert_array_equal(frame(renderer), posed)


def test_joint_moves_its_part_and_children():
    articulated = assembly(parent=0)
    articulated.set_joint(0, "prismatic", axis=(0, 1, 0))
    renderer, (handle, rigid_handle) = renderer_for(articulated, assembly())
    placement = renderer.spawn(0, handle, render.Pose(pose(1.5), coordinates=[0.4]))
    articulated_frame = frame(renderer)
    with pytest.raises(ValueError, match="coordinates must have shape"):
        renderer.set_pose(placement, render.Pose(coordinates=[0.1, 0.2]))
    renderer.despawn(placement)
    renderer.spawn(0, rigid_handle, render.Pose(pose(1.5, 0.4)))
    np.testing.assert_array_equal(frame(renderer), articulated_frame)


def test_pose_validation():
    renderer, (handle,) = renderer_for(assembly())
    placement = renderer.spawn(0, handle)
    with pytest.raises(ValueError, match="shape \\(0,\\)"):
        renderer.set_pose(placement, render.Pose(coordinates=[0.1]))
    with pytest.raises(ValueError, match="finite positive scale"):
        renderer.set_pose(placement, render.Pose(scale=0))
    with pytest.raises(ValueError, match="finite base"):
        renderer.set_pose(placement, render.Pose(pose(np.inf)))
    with pytest.raises(ValueError, match="transform must have shape"):
        renderer.set_pose(placement, render.Pose(np.zeros(5)))


def test_shutter_pair_matches_the_average_of_its_samples():
    articulated = assembly()
    articulated.set_joint(0, "prismatic", axis=(0, 1, 0))
    renderer, (handle,) = renderer_for(articulated, motion=True)
    placement = renderer.spawn(0, handle, render.Pose(pose(30)))
    static_frames = []
    for t in (np.arange(4) + 0.5) / 4:
        renderer.set_pose(placement, render.Pose(pose(2), coordinates=[-0.7 + 1.4 * t]))
        srgb = frame(renderer)[..., :3].astype(np.float32) / 255
        static_frames.append(np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055)**2.4))
    renderer.set_pose(placement, render.Pose(pose(2), coordinates=[-0.7]), render.Pose(pose(2), coordinates=[0.7]))
    blurred = frame(renderer)
    np.testing.assert_allclose(blurred[..., :3], linear_to_rgb8(np.mean(static_frames, axis=0)), atol=1.1)
    captured_depth = renderer.outputs[Depth()].read().copy()
    renderer.set_pose(placement, render.Pose(pose(2), coordinates=[0.7]))
    sharp = frame(renderer)
    np.testing.assert_array_equal(renderer.outputs[Depth()].read(), captured_depth)  # the close state is what depth sees
    assert np.abs(blurred.astype(np.float32) - sharp.astype(np.float32)).max() > 10


def test_selected_nested_nodes(tmp_path):
    path = tmp_path / "nested.glb"
    write_minimal_glb(path)
    data = path.read_bytes()
    length, _ = struct.unpack_from("<II", data, 12)
    gltf = json.loads(data[20:20 + length])
    gltf["nodes"] = [dict(name="base", mesh=0, translation=[0, 0, 1], children=[1]),
                     dict(name="slider", mesh=0, translation=[0, 2, 0])]
    gltf["scenes"][0]["nodes"] = [0]
    payload = json.dumps(gltf).encode()
    payload += b" " * (-len(payload) % 4)
    tail = data[20 + length:]
    path.write_bytes(struct.pack("<III", 0x46546C67, 2, 20 + len(payload) + len(tail)) +
                     struct.pack("<II", len(payload), 0x4E4F534A) + payload + tail)
    asset = render.load_assembly(path, fidelity="low", part_nodes=["base", "slider"])
    assert asset.num_parts == 2
    assert asset.object_name(0) == "base" and asset.object_name(1) == "slider"
    assert asset.part_parent(0) == -1 and asset.part_parent(1) == 0
    # the loader's swizzle is the single source of truth, so compare the nested part against the
    # welded loader
    welded = render.load_assembly(path, fidelity="low")
    assert welded.num_parts == 1
    np.testing.assert_allclose(asset.part(0)[1], welded.part(0)[1], atol=1e-6)
    asset.set_joint("slider", "prismatic", axis=(0, 1, 0))
    assert asset.num_coordinates == 1
    render.AssetPool().add_assembly(asset)
    with pytest.raises(RuntimeError):
        render.load_assembly(path, part_nodes=["missing"])
    with pytest.raises(RuntimeError):
        render.load_assembly(path, part_nodes=["slider", "base"])  # the enclosing node must come first
    with pytest.raises(RuntimeError):
        render.load_assembly(path, part_nodes=["slider"])  # the base mesh would be dropped
