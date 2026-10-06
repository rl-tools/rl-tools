import json
import struct

import numpy as np
import pytest

from hyperdrone import render
from test_render import look_forward, make_quad, make_scene, write_minimal_glb


def pose(x=0.0, y=0.0, z=0.0):
    out = np.eye(4, dtype=np.float32)[:3].copy()
    out[:, 3] = x, y, z
    return out


def rotation_x(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0]], dtype=np.float32)


def rotation_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0]], dtype=np.float32)


def compose(*transforms):
    out = np.eye(4)
    for transform in transforms:
        out = out @ np.vstack((transform, [0, 0, 0, 1]))
    return out[:3]


def chain():
    # revolute about x at an off-origin pivot, prismatic along y below it, spherical below
    # that; parts map geometry onto links out of order
    return render.Rig([-1, 0, 1], ["revolute", "prismatic", "spherical"],
                      axes=[[2, 0, 0], [0, 3, 0], [0, 0, 1]],
                      origins=[pose(0, 2, 3), pose(1), pose(z=1)],
                      part_links=[1, 0, 2, 0],
                      offsets=[pose(), pose(0, -2, -3), pose(2), pose(z=2)])


def test_forward_kinematics_and_coordinate_layout():
    rig = chain()
    assert rig.num_links == 3 and rig.num_parts == 4 and rig.num_coordinates == 6
    assert rig.coordinate_slices == (slice(0, 1), slice(1, 2), slice(2, 6))
    np.testing.assert_array_equal(rig.default_coordinates, [0, 0, 1, 0, 0, 0])
    quarter = np.pi / 2
    coordinates = [quarter, 0.75, np.sqrt(0.5), 0, 0, np.sqrt(0.5)]
    base = pose(4)
    link0 = compose(base, pose(0, 2, 3), rotation_x(quarter))
    link1 = compose(link0, pose(1), pose(y=0.75))
    link2 = compose(link1, pose(z=1), rotation_z(quarter))
    expected = [link1, compose(link0, pose(0, -2, -3)), compose(link2, pose(2)), compose(link0, pose(z=2))]
    out = np.empty((4, 3, 4), dtype=np.float32)
    assert rig.evaluate(base, coordinates, out=out) is out
    np.testing.assert_allclose(out, expected, atol=1e-6)
    batch = rig.evaluate(np.stack([base, pose(5)]), np.stack([coordinates, rig.default_coordinates]))
    assert batch.shape == (2, 4, 3, 4)
    np.testing.assert_allclose(batch[0], expected, atol=1e-6)
    np.testing.assert_allclose(batch[1], rig.evaluate(pose(5)), atol=0)


def test_sample_keeps_revolutions_and_reports_the_close_pose():
    rig = render.Rig([-1], ["revolute"], axes=[[1, 0, 0]], origins=[pose(y=2)], offsets=[pose(y=-2)])
    samples, close = rig.sample(pose(), pose(4), [0], [4 * np.pi], 4)
    for sample, t in zip(samples, (np.arange(4) + 0.5) / 4):
        np.testing.assert_allclose(sample, rig.evaluate(pose(4 * t), [4 * np.pi * t]), atol=1e-5)
    np.testing.assert_allclose(close, rig.evaluate(pose(4), [4 * np.pi]), atol=1e-5)
    assert not np.allclose(samples[-1], close)
    # the first sample sits at t = 1/8, a quarter turn about x at the pivot y = 2, which swings
    # the part frame's origin from the pivot's foot to (y, z) = (2, -2); the base has moved 0.5
    np.testing.assert_allclose(samples[0, 0][:, 3], [0.5, 2, -2], atol=1e-5)


def test_spherical_shortest_arc_and_normalization():
    rig = render.Rig([-1], ["spherical"])
    samples, close = rig.sample(pose(), pose(), [2, 0, 0, 0], [-1, 0, 0, -1], 1)
    angle = np.pi / 4
    np.testing.assert_allclose(samples[0], rig.evaluate(coordinates=[np.cos(angle / 2), 0, 0, np.sin(angle / 2)]), atol=1e-6)
    np.testing.assert_allclose(close, rig.evaluate(coordinates=[1, 0, 0, 1]), atol=1e-6)


@pytest.mark.parametrize("kwargs", [
    dict(parents=[0]), dict(parents=[1, 0]), dict(parents=[-2]), dict(parents=[]),
    dict(parents=[-1], joints=["hinge"]),
    dict(parents=[-1], joints=["revolute"], axes=[[0, 0, 0]]),
    dict(parents=[-1], part_links=[1]),
    dict(parents=[-1], origins=[pose(float("inf"))]),
])
def test_invalid_models(kwargs):
    with pytest.raises(ValueError):
        render.Rig(**kwargs)


def test_invalid_state_and_output():
    rig = render.Rig([-1], ["spherical"])
    for coordinates in ([0, 0, 0, 0], [float("nan"), 0, 0, 1], [1]):
        with pytest.raises(ValueError):
            rig.evaluate(coordinates=coordinates)
    readonly = np.zeros((1, 3, 4), dtype=np.float32)
    readonly.flags.writeable = False
    for out in (readonly, np.zeros((1, 3, 4)), np.zeros((2, 3, 4), dtype=np.float32), np.zeros((1, 3, 8), dtype=np.float32)[:, :, ::2]):
        with pytest.raises(ValueError, match="out"):
            rig.evaluate(out=out)
    invalid_base = pose()
    invalid_base[0, 3] = float("nan")
    with pytest.raises(ValueError):
        rig.evaluate(invalid_base)
    with pytest.raises(ValueError):
        rig.sample(pose(), pose(), [1, 0, 0, 0], [1, 0, 0, 0], 0)


def assembly():
    asset = render.ObjectAssembly()
    for color, rest in [((1, 0, 0), pose(1)), ((0, 1, 0), pose(2))]:
        obj = render.Object()
        obj.add_mesh(make_quad(0, half_size=0.5, color=color))
        asset.add_part(obj, rest)
    return asset


def setup_renderer(motion=False):
    renderer = render.Renderer(width=32, height=32, num_cameras=1, num_probes=0,
                               output="rgbd_segmentation", fidelity="low", num_overlays=2,
                               max_overlay_instances=4, max_overlays_per_camera=2,
                               motion_blur_samples=4 if motion else 1, dynamic_motion_blur=motion)
    asset = assembly()
    pool = render.AssetPool()
    handle = pool.add_assembly(asset)
    renderer.init(make_scene(8), asset_pool=pool)
    renderer.attach(0, 0)
    renderer.attach(0, 1)
    look_forward(renderer)
    return renderer, asset, handle


def test_from_assembly_reproduces_set_transform():
    renderer, asset, handle = setup_renderer()
    placement = renderer.spawn(0, handle, pose(30))
    renderer.set_transform(0, placement, pose(3, -0.4))
    renderer.update()
    renderer.render()
    expected = renderer.frame()
    rig = render.Rig.from_assembly(asset)
    assert rig.num_parts == 2 and rig.num_coordinates == 0
    renderer.set_transform(0, placement, pose(30))
    renderer.set_transforms(0, placement, rig.evaluate(pose(3, -0.4)))
    renderer.update()
    renderer.render()
    np.testing.assert_array_equal(renderer.frame(), expected)


def test_per_part_and_whole_tensor_submission():
    renderer, asset, handle = setup_renderer()
    first = renderer.spawn(0, handle, pose(30))
    second = renderer.spawn(1, handle, pose(30))
    # ids: the background quad is scene instance 0, overlay o slot s is 1 + 4 * o + s
    renderer.set_transforms(0, first, [pose(3, -0.6), pose(30)])
    renderer.set_transforms(1, second, [pose(30), pose(4, 0.6)])
    renderer.update()
    renderer.render()
    ids = set(np.unique(renderer.segmentation()[0]))
    assert 1 in ids and 6 in ids and 2 not in ids and 5 not in ids
    poses = np.full((2, 4, 3, 4), np.nan, dtype=np.float32)
    poses[0, 0], poses[0, 1] = pose(30), pose(3, -0.6)
    poses[1, 0], poses[1, 1] = pose(4, 0.6), pose(30)
    renderer.set_all_transforms(poses)
    renderer.update()
    renderer.render()
    ids = set(np.unique(renderer.segmentation()[0]))
    assert 2 in ids and 5 in ids and 1 not in ids and 6 not in ids
    with pytest.raises(ValueError):
        renderer.set_transforms(0, first, [pose(3)])
    renderer.despawn(0, first)
    with pytest.raises(ValueError, match="not spawned"):
        renderer.set_transforms(0, first, [pose(3), pose(4)])
    assert renderer.spawn(0, handle, pose(3)) == first


def test_motion_submission_matches_static_average():
    renderer, _, handle = setup_renderer(motion=True)
    placement = renderer.spawn(0, handle, pose(30))
    rig = render.Rig([-1, -1], ["prismatic", "fixed"], axes=[[0, 1, 0], [0, 0, 1]], origins=[pose(3), pose(30)])
    offsets = [-0.7, -0.2, 0.2, 0.7]
    static_frames, static_depths = [], []
    for offset in offsets:
        renderer.set_transforms(0, placement, rig.evaluate(coordinates=[offset]))
        renderer.update()
        renderer.render()
        static_frames.append(renderer.frame().astype(np.float32))
        static_depths.append(renderer.depth())
    samples = np.stack([rig.evaluate(coordinates=[offset]) for offset in offsets])
    close = rig.evaluate(pose(2), coordinates=[0])
    renderer.set_motion_transforms(0, placement, samples, close)
    renderer.update()
    renderer.render()
    np.testing.assert_allclose(renderer.frame().astype(np.float32), np.mean(static_frames, axis=0), atol=1)
    np.testing.assert_allclose(renderer.depth(), np.mean(static_depths, axis=0), atol=1e-5)
    assert renderer.segmentation()[0, 16, 16] == 1  # the close pose is what segmentation sees
    renderer.set_transforms(0, placement, rig.evaluate(coordinates=[0]))
    renderer.update()
    renderer.render()
    sharp = renderer.frame()
    renderer.set_motion_transforms(0, placement, np.repeat(rig.evaluate(coordinates=[0])[None], 4, axis=0), rig.evaluate(coordinates=[0]))
    renderer.update()
    renderer.render()
    np.testing.assert_array_equal(renderer.frame(), sharp)
    with pytest.raises(ValueError):
        renderer.set_motion_transforms(0, placement, samples[:2], close)


def test_rig_sample_feeds_motion_blur():
    renderer, _, handle = setup_renderer(motion=True)
    placement = renderer.spawn(0, handle, pose(30))
    rig = render.Rig([-1, 0], ["prismatic", "fixed"], axes=[[0, 1, 0], [0, 0, 1]], origins=[pose(3), pose(27)])
    samples, close = rig.sample(pose(), pose(), [-0.7], [0.7], renderer.motion_blur_samples)
    renderer.set_motion_transforms(0, placement, samples, close)
    renderer.update()
    renderer.render()
    blurred = renderer.frame().astype(np.float32)
    renderer.set_transforms(0, placement, close)
    renderer.update()
    renderer.render()
    assert np.abs(blurred - renderer.frame().astype(np.float32)).max() > 10


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
    # glTF Y-up lands in FLU: the base sits at z = 1 -> FLU y = -1... the loader's swizzle
    # is the single source of truth, so compare the nested part against the welded loader
    welded = render.load_assembly(path, fidelity="low")
    assert welded.num_parts == 1
    np.testing.assert_allclose(asset.part(0)[1], welded.part(0)[1], atol=1e-6)
    rig = render.Rig.from_assembly(asset, joints={"slider": ("prismatic", (0, 1, 0))})
    assert rig.joints == ("fixed", "prismatic") and rig.num_coordinates == 1
    np.testing.assert_allclose(rig.evaluate(), [asset.part(0)[1], asset.part(1)[1]], atol=1e-6)
    with pytest.raises(RuntimeError):
        render.load_assembly(path, part_nodes=["missing"])
    with pytest.raises(RuntimeError):
        render.load_assembly(path, part_nodes=["slider", "base"])  # the enclosing node must come first
    with pytest.raises(RuntimeError):
        render.load_assembly(path, part_nodes=["slider"])  # the base mesh would be dropped


def test_cuda_inputs():
    if render.backend() != "OPTIX":
        pytest.skip("CUDA transform inputs require OptiX")
    from hyperdrone import cuda
    renderer, _, handle = setup_renderer(motion=True)
    first = renderer.spawn(0, handle, pose(30))
    second = renderer.spawn(0, handle, pose(30))
    renderer.set_transforms(0, first, [pose(3, -0.8), pose(30)])
    renderer.set_transforms(0, second, [pose(4, 0.8), pose(30)])
    renderer.update()
    renderer.render()
    expected = renderer.frame()
    poses = cuda.upload(np.stack([pose(3, -0.8), pose(30)])).view()
    renderer.set_transforms(0, first, poses, stream=renderer.render_stream)
    del poses
    # a later host write to other slots must leave the device submission in place
    renderer.set_transforms(0, second, [pose(4, 0.8), pose(30)])
    renderer.update()
    renderer.render_launch()
    assert renderer._pending_inputs
    renderer.render_sync()
    assert not renderer._pending_inputs
    np.testing.assert_array_equal(renderer.frame(), expected)
    samples = np.stack([np.stack([pose(3, y), pose(30)]) for y in [-0.8, -0.3, 0.3, 0.8]])
    close = np.stack([pose(5), pose(30)])
    renderer.set_motion_transforms(0, first, samples, close)
    renderer.update()
    renderer.render()
    expected = renderer.frame()
    renderer.set_motion_transforms(0, first, cuda.upload(samples).view(), cuda.upload(close).view())
    renderer.update()
    renderer.render()
    np.testing.assert_array_equal(renderer.frame(), expected)
    whole = np.full((2, 4, 3, 4), np.nan, dtype=np.float32)
    whole[0, 0], whole[0, 1], whole[0, 2], whole[0, 3] = pose(3, -0.8), pose(30), pose(4, 0.8), pose(30)
    renderer.set_all_transforms(whole)
    renderer.update()
    renderer.render()
    expected = renderer.frame()
    renderer.set_all_transforms(cuda.upload(whole).view())
    renderer.update()
    renderer.render()
    np.testing.assert_array_equal(renderer.frame(), expected)
