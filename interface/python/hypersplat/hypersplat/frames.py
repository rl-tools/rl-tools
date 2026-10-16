"""Coordinate-frame helpers: drone-world (ENU) ↔ body ↔ camera ↔ scene-world.

Conventions:
- Drone world: +X east, +Y north, +Z up (RotorPY ENU). Quaternion order: [x,y,z,w].
- Body: +X forward, +Y left, +Z up (FLU).
- Camera: +X right, +Y down, +Z forward (OpenCV; what gsplat's viewmat consumes).
- Scene world: whatever the .ply was trained in. Mapped via `scene_transform` (4x4),
  defined as drone-world -> scene-world.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def quat_xyzw_to_R(q: np.ndarray) -> np.ndarray:
    """[x, y, z, w] unit quaternion -> 3x3 rotation matrix."""
    return Rotation.from_quat(np.asarray(q)).as_matrix()


def pose_to_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def invert_T(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


# Body (FLU: x-fwd, y-left, z-up) -> camera (OpenCV: x-right, y-down, +z-fwd).
# Columns are the camera basis vectors expressed in body coords:
#   cam +x = body -y (right), cam +y = body -z (down), cam +z = body +x (fwd)
DEFAULT_R_BODY_CAMERA: np.ndarray = np.array(
    [[ 0.0,  0.0, 1.0],
     [-1.0,  0.0, 0.0],
     [ 0.0, -1.0, 0.0]],
    dtype=np.float64,
)


DEFAULT_T_BODY_CAMERA: np.ndarray = pose_to_T(DEFAULT_R_BODY_CAMERA, np.zeros(3))


def state_to_T_world_body(state: dict) -> np.ndarray:
    """RotorPY state dict -> 4x4 body-pose-in-world (ENU)."""
    R = quat_xyzw_to_R(state["q"])
    return pose_to_T(R, np.asarray(state["x"], dtype=np.float64))


def compute_viewmat(
    state: dict,
    scene_transform: np.ndarray | None = None,
    body_to_camera: np.ndarray | None = None,
) -> np.ndarray:
    """Drone state -> world->camera 4x4 viewmat in scene-world coordinates."""
    T_world_body = state_to_T_world_body(state)
    T_body_camera = DEFAULT_T_BODY_CAMERA if body_to_camera is None else np.asarray(body_to_camera, dtype=np.float64)
    T_world_camera = T_world_body @ T_body_camera                 # in ENU world
    if scene_transform is not None:
        T_world_camera = np.asarray(scene_transform, dtype=np.float64) @ T_world_camera
    return invert_T(T_world_camera).astype(np.float32)
