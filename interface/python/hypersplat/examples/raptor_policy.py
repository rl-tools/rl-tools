"""Drive the HyperSplat env with the RAPTOR foundation policy and save a video.

Usage:
    python examples/raptor_policy.py                           # synthetic scene
    python examples/raptor_policy.py --scene scene.ply         # real 3DGS scene
    python examples/raptor_policy.py --start 1 0.5 1.5         # initial offset (drone homes back to origin)

Notes on conventions:

The RAPTOR policy is trained in FLU axes:
    world: x = forward, y = left, z = up
    body:  x = forward, y = left, z = up

RotorPY is already FLU body / ENU world (z up, right-handed) — the numerical
position / velocity / quaternion-derived rotation matrix / body-frame angular velocity
arrays drop in directly without sign flips.

Motor index conventions differ:
    RAPTOR:  [FR, BR, BL, FL]
    RotorPY rotor_pos r1..r4 for the default Crazyflie params is [FL, FR, BR, BL].
We permute on both sides so RAPTOR sees its expected order and RotorPY receives its own.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from foundation_policy import Raptor
from scipy.spatial.transform import Rotation

from hypersplat import HyperSplatEnv
from hypersplat.renderer import Scene


# RAPTOR motor order [FR, BR, BL, FL]  ->  RotorPY order [FL, FR, BR, BL]
RAPTOR_TO_ROTORPY = np.array([3, 0, 1, 2])
ROTORPY_TO_RAPTOR = np.array([1, 2, 3, 0])


# Pre-built rotations that map drone-world ENU (z-up) into a scene-world where
# the named axis points up. Each value is the 3x3 R such that p_scene = R @ p_drone_world.
_AXIS = {
    "+x": np.array([1, 0, 0]), "-x": np.array([-1, 0, 0]),
    "+y": np.array([0, 1, 0]), "-y": np.array([0, -1, 0]),
    "+z": np.array([0, 0, 1]), "-z": np.array([0, 0, -1]),
}


def scene_transform_for_up(scene_up: str, origin: np.ndarray) -> np.ndarray:
    """Build a 4x4 drone-ENU -> scene-world transform given the scene's up-axis."""
    up_scene = _AXIS[scene_up].astype(np.float64)
    # Drone +x maps to whichever scene axis is most "horizontal" — pick scene +x by default,
    # falling back to scene +y if scene_up itself is along +/- x.
    fwd_scene = np.array([1, 0, 0], dtype=np.float64)
    if abs(np.dot(fwd_scene, up_scene)) > 0.9:
        fwd_scene = np.array([0, 1, 0], dtype=np.float64)
    # Right-handed completion: drone_y = cross(drone_z, drone_x), so its scene image is cross(up, fwd).
    left_scene = np.cross(up_scene, fwd_scene)
    R = np.stack([fwd_scene, left_scene, up_scene], axis=1)  # columns = R @ e_x, R @ e_y, R @ e_z
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = origin
    return T


def build_observation(state: dict, prev_action_raptor: np.ndarray) -> np.ndarray:
    """Build the 22-D RAPTOR observation from a RotorPY state dict and previous action."""
    pos = np.asarray(state["x"], dtype=np.float32)
    R_flat = Rotation.from_quat(state["q"]).as_matrix().astype(np.float32).flatten()  # row-major (C-order)
    vel = np.asarray(state["v"], dtype=np.float32)
    omega = np.asarray(state["w"], dtype=np.float32)
    obs = np.concatenate([pos, R_flat, vel, omega, prev_action_raptor.astype(np.float32)])
    assert obs.shape == (22,), obs.shape
    return obs[None, :]  # batch dim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=str, default=None)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--out", type=str, default="raptor.mp4")
    ap.add_argument("--height", type=int, default=200)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--fov", type=float, default=90.0)
    ap.add_argument("--start", type=float, nargs=3, default=(1.0, 0.5, 1.5),
                    metavar=("X", "Y", "Z"), help="initial drone position in ENU (FLU)")
    ap.add_argument("--scene-up", type=str, default=None, choices=list(_AXIS),
                    help="Override: snap the up axis to this scene-frame axis. Defaults to the sidecar JSON if present.")
    ap.add_argument("--scene-rotation", type=float, nargs=9, default=None,
                    metavar=("R00", "R01", "R02", "R10", "R11", "R12", "R20", "R21", "R22"),
                    help="Override: 3x3 row-major rotation taking scene -> upright. Overrides any sidecar JSON.")
    ap.add_argument("--scene-origin", type=float, nargs=3, default=None,
                    metavar=("X", "Y", "Z"), help="Override: drone-world origin in scene coordinates.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    scene = Scene.from_ply(args.scene, device=args.device) if args.scene else Scene.synthetic(device=args.device)
    print(f"scene: {scene.num_gaussians} gaussians on {args.device}; "
          f"calibration {'loaded from sidecar' if scene.scene_transform is not None else 'absent (drone-ENU == scene)'}")

    # Build the scene_transform: explicit CLI overrides > sidecar > None.
    scene_transform = None
    origin = np.asarray(args.scene_origin, dtype=np.float64) if args.scene_origin is not None else None
    if args.scene_rotation is not None:
        R_to_upright = np.asarray(args.scene_rotation, dtype=np.float64).reshape(3, 3)
        scene_transform = np.eye(4, dtype=np.float64)
        scene_transform[:3, :3] = R_to_upright.T
        scene_transform[:3, 3] = origin if origin is not None else 0.0
    elif args.scene_up is not None:
        scene_transform = scene_transform_for_up(args.scene_up, origin if origin is not None else np.zeros(3))
    elif origin is not None and scene.scene_transform is not None:
        # Keep sidecar rotation but override translation only.
        scene_transform = scene.scene_transform.copy()
        scene_transform[:3, 3] = origin

    env = HyperSplatEnv(
        scene=scene,
        obs_mode="dict",
        control_mode="cmd_motor_speeds",
        image_size=(args.height, args.width),
        fov_deg=args.fov,
        sim_rate=100.0,                              # RAPTOR is trained at 100 Hz
        max_episode_steps=args.steps,
        initial_state={"x": np.array(args.start, dtype=np.float64)},
        scene_transform=scene_transform,
        device=args.device,
    )

    policy = Raptor()
    policy.reset()

    obs, info = env.reset(seed=args.seed)
    state = info["state"]
    prev_action_raptor = np.zeros(4, dtype=np.float32)

    frames = [obs["image"]]
    positions = [state["x"].copy()]
    terminated = truncated = False

    for step in range(args.steps):
        raptor_obs = build_observation(state, prev_action_raptor)
        action_raptor = policy.evaluate_step(raptor_obs)[0]                 # (4,) FR,BR,BL,FL in [-1,1]
        action_rotorpy = action_raptor[RAPTOR_TO_ROTORPY]                   # (4,) FL,FR,BR,BL

        obs, reward, terminated, truncated, info = env.step(action_rotorpy)
        state = info["state"]
        prev_action_raptor = action_raptor
        frames.append(obs["image"])
        positions.append(state["x"].copy())
        if terminated or truncated:
            break

    pos_arr = np.stack(positions)
    print(
        f"steps={len(frames) - 1}  start={pos_arr[0]}  end={pos_arr[-1]}  "
        f"final_dist={np.linalg.norm(pos_arr[-1]):.3f} m  "
        f"terminated={terminated} truncated={truncated}"
    )

    out = Path(args.out)
    imageio.mimsave(str(out), frames, fps=int(1 / env.dynamics.dt))
    print(f"wrote {out} ({len(frames)} frames @ {args.width}x{args.height})")


if __name__ == "__main__":
    main()
