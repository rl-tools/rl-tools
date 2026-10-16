"""Frame-transform unit tests."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from hypersplat.frames import (
    DEFAULT_T_BODY_CAMERA,
    compute_viewmat,
    invert_T,
    pose_to_T,
    quat_xyzw_to_R,
    state_to_T_world_body,
)


def _state(x, q):
    return {"x": np.array(x, dtype=np.float64), "q": np.array(q, dtype=np.float64)}


def test_invert_T_round_trip():
    R = Rotation.from_euler("xyz", [0.1, 0.2, 0.3]).as_matrix()
    t = np.array([1.0, 2.0, 3.0])
    T = pose_to_T(R, t)
    np.testing.assert_allclose(T @ invert_T(T), np.eye(4), atol=1e-10)


def test_default_camera_looks_along_body_x():
    """At identity drone pose, camera (OpenCV) should look along world +X."""
    state = _state([0, 0, 0], [0, 0, 0, 1])
    viewmat = compute_viewmat(state)
    c2w = invert_T(viewmat)
    np.testing.assert_allclose(c2w[:3, 3], np.zeros(3), atol=1e-10)
    # OpenCV: camera looks along its own +z; map +z_camera into world.
    look_in_world = c2w[:3, 2]
    np.testing.assert_allclose(look_in_world, [1.0, 0.0, 0.0], atol=1e-10)
    # Camera "down" (+y_camera) should be world -z (down).
    down_in_world = c2w[:3, 1]
    np.testing.assert_allclose(down_in_world, [0.0, 0.0, -1.0], atol=1e-10)


def test_yaw_90_rotates_camera_view():
    """90° yaw about world +Z makes the camera look along world +Y."""
    q = Rotation.from_euler("z", 90, degrees=True).as_quat()
    state = _state([1, 0, 0.5], q)
    viewmat = compute_viewmat(state)
    c2w = invert_T(viewmat)
    np.testing.assert_allclose(c2w[:3, 3], [1.0, 0.0, 0.5], atol=1e-10)
    look = c2w[:3, 2]
    np.testing.assert_allclose(look, [0.0, 1.0, 0.0], atol=1e-10)


def test_world_origin_projects_to_principal_point():
    """Camera at (-1,0,0) looking toward +X: world origin projects to (cx, cy)."""
    state = _state([-1, 0, 0], [0, 0, 0, 1])
    viewmat = compute_viewmat(state)
    K = np.array([[100.0, 0.0, 40.0], [0.0, 100.0, 25.0], [0.0, 0.0, 1.0]])
    p_world = np.array([0.0, 0.0, 0.0, 1.0])
    p_cam = viewmat @ p_world
    assert p_cam[2] > 0, f"origin should be in front of camera, got z={p_cam[2]}"
    u = K[0, 0] * p_cam[0] / p_cam[2] + K[0, 2]
    v = K[1, 1] * p_cam[1] / p_cam[2] + K[1, 2]
    np.testing.assert_allclose([u, v], [40.0, 25.0], atol=1e-6)


if __name__ == "__main__":
    test_invert_T_round_trip()
    test_default_camera_looks_along_body_x()
    test_yaw_90_rotates_camera_view()
    test_world_origin_projects_to_principal_point()
    print("all frame tests passed")
