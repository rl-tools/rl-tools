import numpy as np
import pytest

from hyperdrone import render
from hyperdrone.env import Rotorcraft


def pose(x=0.0, y=0.0, z=0.0):
    out = np.eye(4, dtype=np.float32)[:3].copy()
    out[:, 3] = x, y, z
    return out


def rotation_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0]], dtype=np.float32)


def assembly():
    asset = render.ObjectAssembly()
    for center in (1.0, 2.0):
        asset.add_part(quad_xy(center, 0, 0.5, ""), pose())
    return asset


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


def test_rotorcraft_convention():
    rig = Rotorcraft(two_prop_assembly())
    assert rig.num_links == 3 and rig.num_parts == 3 and rig.num_coordinates == 2
    assert rig.joints == ("fixed", "revolute", "revolute")
    assert rig.coordinate_slices == (slice(0, 0), slice(0, 1), slice(1, 2))
    poses = rig.evaluate(pose(2, 0, 1), [np.pi / 2, np.pi / 2])
    np.testing.assert_allclose(poses[0], pose(2, 0, 1), atol=1e-6)
    # the hubs stay put, the front-left prop turns CW and the rear-left prop CCW (Quad-X)
    np.testing.assert_allclose(poses[1] @ [0.1, 0.1, 0, 1], [2.1, 0.1, 1.05], atol=1e-5)
    np.testing.assert_allclose(poses[1] @ [0.15, 0.1, 0, 1], [2.1, 0.05, 1.05], atol=1e-5)
    np.testing.assert_allclose(poses[2] @ [-0.1, 0.1, 0, 1], [1.9, 0.1, 1.05], atol=1e-5)
    np.testing.assert_allclose(poses[2] @ [-0.05, 0.1, 0, 1], [1.9, 0.15, 1.05], atol=1e-5)
    # at the neutral state the rig reproduces the rest frames
    np.testing.assert_allclose(rig.evaluate(), [two_prop_assembly().part(index)[1] for index in range(3)], atol=1e-6)
    with pytest.raises(ValueError, match="body/prop_"):
        Rotorcraft(assembly())


def test_rotorcraft_matches_rotors():
    asset = two_prop_assembly()
    rotor_positions = [[-0.1, 0.1, 0], [0.1, 0.1, 0]]
    rotor_torque_directions = [[0, 0, 1], [0, 0, -1]]
    rig = Rotorcraft(asset, rotor_positions=rotor_positions, rotor_torque_directions=rotor_torque_directions)
    assert rig.num_coordinates == 2
    # coordinate i is rotor i: rotor 1 sits under prop_0 and its -z reaction torque means a CCW spin
    poses = rig.evaluate(coordinates=[0, np.pi / 2])
    np.testing.assert_allclose(poses[2], asset.part(2)[1], atol=1e-6)
    np.testing.assert_allclose(poses[1] @ [0.15, 0.1, 0, 1], [0.1, 0.15, 0.05], atol=1e-5)
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


def test_rotorcraft_x500():
    import conta
    from hyperdrone.examples.data import RIGGED_MODELS
    try:
        path = conta.resolve(RIGGED_MODELS["x500"])
    except Exception as error:  # noqa: BLE001 - no checkout data and no network
        pytest.skip(f"x500 asset unavailable: {error}")
    drone = render.load_assembly(path, fidelity="low")
    # the x500 registry entry: rotors 0..3 at front-right, rear-right, rear-left, front-left
    arm = 0.176776695296636
    rotor_positions = [[arm, -arm, 0], [-arm, -arm, 0], [-arm, arm, 0], [arm, arm, 0]]
    rotor_torque_directions = [[0, 0, -1], [0, 0, 1], [0, 0, -1], [0, 0, 1]]
    rig = Rotorcraft(drone, rotor_positions=rotor_positions, rotor_torque_directions=rotor_torque_directions)
    assert rig.num_coordinates == 4 and rig.num_parts == drone.num_parts
    rest = rig.evaluate()
    names = [drone.object_name(drone.part(index)[0]) for index in range(drone.num_parts)]
    for rotor, (x, y, _) in enumerate(rotor_positions):
        turned = rig.evaluate(coordinates=np.eye(4)[rotor] * np.pi / 2)
        moved = [index for index in range(drone.num_parts) if not np.allclose(turned[index], rest[index], atol=1e-6)]
        assert len(moved) == 1 and names[moved[0]] == f"prop_{[0, 3, 1, 2][rotor]}"
        low, high = drone.object_bounds(drone.part(moved[0])[0])
        pivot = (low + high) / 2
        pivot[2] = 0
        hub = rest[moved[0]] @ [*pivot, 1]
        assert np.sign(hub[0]) == np.sign(x) and np.sign(hub[1]) == np.sign(y)
        np.testing.assert_allclose(turned[moved[0]] @ [*pivot, 1], hub, atol=1e-5)
        # the prop turns against the rotor's reaction torque
        spin = -rotor_torque_directions[rotor][2]
        tip = rest[moved[0]] @ [*(pivot + [0.05, 0, 0]), 1] - hub
        expected = rotation_z(spin * np.pi / 2)[:, :3] @ tip
        np.testing.assert_allclose(turned[moved[0]] @ [*(pivot + [0.05, 0, 0]), 1] - hub, expected, atol=1e-5)
    convention = Rotorcraft(drone)
    assert convention.joints == rig.joints
    np.testing.assert_allclose(convention.evaluate(), rest, atol=1e-6)
    np.testing.assert_allclose(convention.evaluate(coordinates=[1, 2, 3, 4]), rig.evaluate(coordinates=[1, 4, 2, 3]), atol=1e-5)
