import numpy as np
import pytest

from hyperdrone import Rotorcraft, articulate, render
from hyperdrone.camera import CameraSpec, Segmentation


def pose(x=0.0, y=0.0, z=0.0):
    out = np.eye(4, dtype=np.float32)[:3].copy()
    out[:, 3] = x, y, z
    return out


def quad_xy(center_x, center_y, half_size, name):
    vertices = np.array([[center_x - half_size, center_y - half_size, 0], [center_x + half_size, center_y - half_size, 0],
                         [center_x + half_size, center_y + half_size, 0], [center_x - half_size, center_y + half_size, 0]], dtype=np.float32)
    obj = render.Object(name=name)
    obj.add_mesh(render.Mesh(vertices, np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32), color=(1, 1, 1)))
    return obj


def two_prop_assembly():
    # body plus props at (0.1, 0.1) front-left and (-0.1, 0.1) rear-left, lifted by 0.05
    asset = render.ObjectAssembly()
    asset.add_part(quad_xy(0, 0, 0.1, "body"), pose())
    asset.add_part(quad_xy(0.1, 0.1, 0.05, "prop_0"), pose(z=0.05))
    asset.add_part(quad_xy(-0.1, 0.1, 0.05, "prop_1"), pose(z=0.05))
    return asset


def expect_joint(asset, part, pivot, coordinate):
    joint, axis, actual_pivot, actual_coordinate = asset.joint(part)
    assert joint == "revolute" and actual_coordinate == coordinate
    np.testing.assert_array_equal(axis, [0, 0, 1])
    np.testing.assert_allclose(actual_pivot, pivot, atol=1e-7)


def test_rotorcraft_convention():
    asset = two_prop_assembly()
    with pytest.raises(ValueError, match="articulate"):
        Rotorcraft(asset)
    assert asset.num_coordinates == 0
    articulate(asset)
    rotorcraft = Rotorcraft(asset)
    assert rotorcraft.scale == 1 and rotorcraft.rotor_position_deviation == 0
    assert asset.joint("body")[0] == "fixed" and asset.num_coordinates == 2
    expect_joint(asset, "prop_0", [0.1, 0.1, 0], 0)
    expect_joint(asset, "prop_1", [-0.1, 0.1, 0], 1)
    # Quad-X: the front-left prop turns CW, the rear-left prop CCW
    np.testing.assert_array_equal(rotorcraft.prop_rotor, [0, 1])
    np.testing.assert_array_equal(rotorcraft.prop_spin, [-1, 1])
    np.testing.assert_array_equal(rotorcraft.coordinates([0.5, 0.25]), [-0.5, 0.25])
    np.testing.assert_array_equal(rotorcraft.coordinates([[0.5, 0.25], [1, 2]]), [[-0.5, 0.25], [-1, 2]])
    unconventional = render.ObjectAssembly()
    unconventional.add_part(quad_xy(1, 0, 0.5, "prop_0"), pose())
    with pytest.raises(ValueError, match="body/prop_"):
        articulate(unconventional)
    assert unconventional.joint(0)[0] == "fixed"


def test_rotorcraft_matches_rotors():
    asset = two_prop_assembly()
    articulate(asset)
    rotor_positions = [[-0.2, 0.2, 0], [0.2, 0.2, 0]]
    rotor_torque_directions = [[0, 0, 1], [0, 0, -1]]
    rotorcraft = Rotorcraft(asset, rotor_positions=rotor_positions, rotor_torque_directions=rotor_torque_directions)
    assert rotorcraft.scale == pytest.approx(2) and rotorcraft.rotor_position_deviation == pytest.approx(0, abs=1e-6)
    # the joints are the asset's own; rotor 1 sits over prop_0 and its -z reaction torque means a CCW spin
    expect_joint(asset, "prop_0", [0.1, 0.1, 0], 0)
    expect_joint(asset, "prop_1", [-0.1, 0.1, 0], 1)
    np.testing.assert_array_equal(rotorcraft.prop_rotor, [1, 0])
    np.testing.assert_array_equal(rotorcraft.prop_spin, [1, -1])
    np.testing.assert_array_equal(rotorcraft.coordinates([0.5, 0.25]), [0.25, -0.5])
    with pytest.raises(ValueError, match="together"):
        Rotorcraft(asset, rotor_positions=rotor_positions)
    with pytest.raises(ValueError, match="one-to-one"):
        Rotorcraft(asset, rotor_positions=[[0.1, 0.1, 0]], rotor_torque_directions=[[0, 0, 1]])
    with pytest.raises(ValueError, match="one-to-one"):
        Rotorcraft(asset, rotor_positions=[[0.1, 0.1, 0], [0.1, 0.1, 0]], rotor_torque_directions=rotor_torque_directions)
    with pytest.raises(ValueError, match="one-to-one"):
        Rotorcraft(asset, rotor_positions=rotor_positions, rotor_torque_directions=[[0, 0, 0], [0, 0, 1]])
    with pytest.raises(ValueError, match="shape"):
        Rotorcraft(asset, rotor_positions=[0.1, 0.1], rotor_torque_directions=rotor_torque_directions)


def test_rotorcraft_spins_props_in_the_renderer():
    asset = two_prop_assembly()
    articulate(asset)
    pool = render.AssetPool()
    handle = pool.add_assembly(asset)
    scene = render.Scene()
    scene.add_mesh(render.Mesh(np.array([[5, -5, -5], [5, 5, -5], [5, 5, 5], [5, -5, 5]], np.float32), np.array([[0, 2, 1], [0, 3, 2]], np.int32)))
    renderer = render.Renderer(64, 64, cameras=(CameraSpec("camera", outputs=(Segmentation(),)),), fidelity="low", num_overlays=1, max_overlay_instances=4, max_overlays_per_camera=1)
    renderer.init(scene, asset_pool=pool)
    renderer.attach(0, 0)
    renderer.set_cameras(renderer.camera(position=(0.0, 0.0, 0.0), look_at=(1.0, 0.0, 0.0), fov=40.0)[None])
    # the drone faces the camera from 0.6 ahead (its z towards -x), so the props spin in the image plane
    facing = np.array([[0, 0, -1, 0.6], [0, 1, 0, 0], [1, 0, 0, 0]], np.float32)
    placement = renderer.spawn(0, handle, render.Pose(facing))
    ids = []
    for coordinates in ([0, 0], [np.pi / 4, 0], [0, np.pi / 4]):
        renderer.set_pose(placement, render.Pose(facing), coordinates=coordinates)
        renderer.update()
        renderer.render()
        ids.append(renderer.outputs[Segmentation()].read()[0].copy())
    # the props sit in front of the body, so only the other prop is sure to stay as it was
    prop_0, prop_1 = 2, 3
    for turned, moving, still in ((ids[1], prop_0, prop_1), (ids[2], prop_1, prop_0)):
        assert np.array_equal(turned == still, ids[0] == still)
        assert not np.array_equal(turned == moving, ids[0] == moving)
        assert (turned == moving).sum() == pytest.approx((ids[0] == moving).sum(), rel=0.2)


def test_rotorcraft_x500():
    import conta
    from hyperdrone.examples.data import RIGGED_MODELS
    try:
        path = conta.resolve(RIGGED_MODELS["x500"])
    except Exception as error:  # noqa: BLE001 - no checkout data and no network
        pytest.skip(f"x500 asset unavailable: {error}")
    drone = render.load_assembly(path, fidelity="low")
    convention = render.load_assembly(path, fidelity="low")
    articulate(drone)
    articulate(convention)
    # the x500 registry entry: rotors 0..3 at front-right, rear-right, rear-left, front-left
    arm = 0.176776695296636
    rotor_positions = [[arm, -arm, 0], [-arm, -arm, 0], [-arm, arm, 0], [arm, arm, 0]]
    rotor_torque_directions = [[0, 0, -1], [0, 0, 1], [0, 0, -1], [0, 0, 1]]
    rotorcraft = Rotorcraft(drone, rotor_positions=rotor_positions, rotor_torque_directions=rotor_torque_directions)
    quad_x = Rotorcraft(convention)
    assert drone.num_coordinates == 4
    assert rotorcraft.rotor_position_deviation < 0.1
    props = [part for part in range(drone.num_parts) if drone.object_name(drone.part(part)[0]).startswith("prop_")]
    for rotor, (x, y, _) in enumerate(rotor_positions):
        name = f"prop_{[0, 3, 1, 2][rotor]}"
        joint, axis, pivot, coordinate = drone.joint(name)
        index = next(part for part in range(drone.num_parts) if drone.object_name(drone.part(part)[0]) == name)
        prop = props.index(index)
        assert joint == "revolute" and coordinate == prop and axis[2] == 1
        quad_x_joint, quad_x_axis, quad_x_pivot, quad_x_coordinate = convention.joint(name)
        assert quad_x_joint == joint and quad_x_coordinate == coordinate
        np.testing.assert_array_equal(quad_x_axis, axis)
        np.testing.assert_array_equal(quad_x_pivot, pivot)
        assert rotorcraft.prop_rotor[prop] == rotor
        # the prop turns against the rotor's reaction torque, as the Quad-X convention has it
        assert rotorcraft.prop_spin[prop] == -rotor_torque_directions[rotor][2] == quad_x.prop_spin[prop]
        hub = drone.part(index)[1] @ [*pivot, 1]
        assert np.sign(hub[0]) == np.sign(x) and np.sign(hub[1]) == np.sign(y)
