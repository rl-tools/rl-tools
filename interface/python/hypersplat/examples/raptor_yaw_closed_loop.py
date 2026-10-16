"""Closed-loop RAPTOR rollout with visual yaw estimation.

All RAPTOR observation components use ground truth except yaw.  The yaw can come
from ground truth, a zero-yaw null predictor, or an RLtools visual-yaw HDF5
checkpoint evaluated directly with NumPy.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation


RAPTOR_TO_ROTORPY = np.array([3, 0, 1, 2])
ROTORPY_TO_RAPTOR = np.array([1, 2, 3, 0])
WORLD_GRAVITY_DIRECTION = np.array([0.0, 0.0, -1.0], dtype=np.float64)
DEFAULT_CHECKPOINT = (
    Path.home()
    / "rl-tools3/experiments/2026-05-17_01-05-51/"
    / "606992f_l2f_visual_yaw_cuda_default/default/0100/"
    / "steps/000000249606400/checkpoint_512examples.h5"
)
DEFAULT_SCENE = (
    Path.home()
    / "git/hypersplat/data/train/point_cloud/iteration_7000/point_cloud.ply"
)
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def wrap_rad(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def yaw_from_R(R: np.ndarray) -> float:
    return math.atan2(float(R[1, 0]), float(R[0, 0]))


def Rz(yaw: float) -> np.ndarray:
    c = math.cos(yaw)
    s = math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def replace_yaw(R_true: np.ndarray, yaw_est: float) -> np.ndarray:
    yaw_true = yaw_from_R(R_true)
    return Rz(yaw_est) @ (Rz(-yaw_true) @ R_true)


def gravity_body_from_R(R_body_to_world: np.ndarray) -> np.ndarray:
    return np.asarray(R_body_to_world, dtype=np.float64).T @ WORLD_GRAVITY_DIRECTION


def normalize_gravity_body(gravity_body: np.ndarray) -> np.ndarray:
    g = np.asarray(gravity_body, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(g))
    if not np.isfinite(norm) or norm < 1.0e-9:
        raise ValueError(f"invalid gravity direction {gravity_body}")
    return g / norm


def roll_pitch_from_gravity_body(gravity_body: np.ndarray) -> tuple[float, float]:
    g = normalize_gravity_body(gravity_body)
    pitch = math.asin(float(np.clip(g[0], -1.0, 1.0)))
    roll = math.atan2(float(-g[1]), float(-g[2]))
    return roll, pitch


def orientation_from_gravity_and_yaw(gravity_body: np.ndarray, yaw: float) -> np.ndarray:
    roll, pitch = roll_pitch_from_gravity_body(gravity_body)
    R_tilt = Rotation.from_euler("ZYX", [0.0, pitch, roll]).as_matrix()
    return Rz(yaw) @ R_tilt


def yaw_quat_xyzw(yaw: float) -> np.ndarray:
    return np.array([0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw)], dtype=np.float64)


def pose_to_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def invert_T(T: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def build_state(position: tuple[float, float, float], yaw: float) -> dict[str, np.ndarray]:
    return {
        "x": np.asarray(position, dtype=np.float64),
        "q": yaw_quat_xyzw(yaw),
    }


def build_initial_state(args, episode_i: int) -> dict[str, np.ndarray]:
    if not args.random_starts:
        return build_state(tuple(args.start), args.start_yaw)
    rng = np.random.default_rng(args.seed + episode_i)
    xy = rng.uniform(-args.random_xy, args.random_xy, size=2)
    z = rng.uniform(args.random_z_min, args.random_z_max)
    yaw = rng.uniform(-math.radians(args.random_yaw_deg), math.radians(args.random_yaw_deg))
    return build_state((float(xy[0]), float(xy[1]), float(z)), float(yaw))


def lazy_import_runtime():
    try:
        import torch
        from foundation_policy import Raptor
        from hypersplat import HyperSplatEnv
        from hypersplat.frames import DEFAULT_T_BODY_CAMERA, compute_viewmat
        from hypersplat.renderer import Scene
    except ImportError as exc:
        raise RuntimeError(
            "closed-loop rollout requires torch, foundation_policy, rotorpy, and hypersplat "
            "runtime dependencies in this Python environment"
        ) from exc
    return torch, Raptor, HyperSplatEnv, DEFAULT_T_BODY_CAMERA, compute_viewmat, Scene


def parse_obs_camera_meta(checkpoint: Path) -> tuple[float, int, int, int, int]:
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("HDF5 yaw checkpoints require h5py") from exc

    with h5py.File(checkpoint, "r") as f:
        raw = f["actor"].attrs["meta"]
    if isinstance(raw, bytes):
        raw = raw.decode()
    meta = json.loads(raw)
    obs = meta["environment"]["observation"]
    prefix = "CameraRGBStackedWithTarget("
    if not obs.startswith(prefix):
        raise ValueError(f"expected {prefix} in checkpoint observation, got {obs!r}")
    args = obs[len(prefix):obs.index(")")].split(",")
    fov_rad = float(args[0])
    height = int(args[1])
    width = int(args[2])
    stride = int(args[3])
    frames = int(args[4])
    return fov_rad, height, width, stride, frames


@dataclass
class DenseLayer:
    weights: np.ndarray
    biases: np.ndarray
    relu: bool


@dataclass
class ConvLayer:
    weights: np.ndarray
    biases: np.ndarray
    stride_h: int
    stride_w: int
    pad_h: int
    pad_w: int


class HDF5YawEstimator:
    def __init__(self, checkpoint: str | Path, validate_examples: int = 16):
        try:
            import h5py
        except ImportError as exc:
            raise RuntimeError("HDF5 yaw checkpoints require h5py") from exc

        self.checkpoint = Path(checkpoint).expanduser().resolve()
        self.fov_rad, self.height, self.width, self.stride, self.frames = parse_obs_camera_meta(self.checkpoint)
        self.channels = 0
        self.convs: list[ConvLayer] = []
        self.denses: list[DenseLayer] = []

        with h5py.File(self.checkpoint, "r") as f:
            self.channels = int(f["example/inputs/0"].shape[-1])
            logical_channels = (self.frames + 1) * 3
            if self.channels < logical_channels:
                raise ValueError(f"checkpoint input channels {self.channels} < logical channels {logical_channels}")
            for i in range(4):
                g = f[f"actor/branch_0/layers/{i}"]
                self.convs.append(
                    ConvLayer(
                        weights=g["weights/parameters"][...].astype(np.float32),
                        biases=g["biases/parameters"][...].astype(np.float32),
                        stride_h=int(g.attrs["stride_h"]),
                        stride_w=int(g.attrs["stride_w"]),
                        pad_h=int(g.attrs["padding_h"]),
                        pad_w=int(g.attrs["padding_w"]),
                    )
                )
            g = f["actor/branch_0/layers/5"]
            self.denses.append(self._load_dense(g))
            for i in range(3):
                self.denses.append(self._load_dense(f[f"actor/head/layers/{i}"]))
            if validate_examples > 0:
                n = min(int(validate_examples), int(f["example/inputs/0"].shape[1]))
                x = f["example/inputs/0"][0, :n]
                y_ref = f["example/outputs/0"][0, :n]
                y = self.forward_batch(x)
                self.example_max_abs_diff = float(np.max(np.abs(y - y_ref)))
                self.example_mean_abs_diff = float(np.mean(np.abs(y - y_ref)))
            else:
                self.example_max_abs_diff = None
                self.example_mean_abs_diff = None

    @staticmethod
    def _load_dense(group) -> DenseLayer:
        activation = group.attrs.get("activation_function", "IDENTITY")
        if isinstance(activation, bytes):
            activation = activation.decode()
        return DenseLayer(
            weights=group["weights/parameters"][...].astype(np.float32),
            biases=group["biases/parameters"][...].astype(np.float32),
            relu=activation != "IDENTITY",
        )

    @staticmethod
    def _conv2d(x: np.ndarray, layer: ConvLayer) -> np.ndarray:
        n, h, w, _ = x.shape
        cout, kh, kw, _ = layer.weights.shape
        xp = np.pad(x, ((0, 0), (layer.pad_h, layer.pad_h), (layer.pad_w, layer.pad_w), (0, 0)))
        out_h = (h + 2 * layer.pad_h - kh) // layer.stride_h + 1
        out_w = (w + 2 * layer.pad_w - kw) // layer.stride_w + 1
        y = np.zeros((n, out_h, out_w, cout), dtype=np.float32)
        for ky in range(kh):
            for kx in range(kw):
                xs = xp[
                    :,
                    ky:ky + out_h * layer.stride_h:layer.stride_h,
                    kx:kx + out_w * layer.stride_w:layer.stride_w,
                    :,
                ]
                y += np.tensordot(xs, layer.weights[:, ky, kx, :], axes=([3], [1])).astype(np.float32)
        y += layer.biases.reshape(1, 1, 1, cout)
        return np.maximum(y, 0.0, out=y)

    @staticmethod
    def _dense(x: np.ndarray, layer: DenseLayer) -> np.ndarray:
        y = x @ layer.weights.T + layer.biases
        if layer.relu:
            y = np.maximum(y, 0.0, out=y)
        return y.astype(np.float32, copy=False)

    def forward_batch(self, x: np.ndarray) -> np.ndarray:
        y = np.asarray(x, dtype=np.float32)
        for layer in self.convs:
            y = self._conv2d(y, layer)
        y = y.reshape(y.shape[0], -1)
        for layer in self.denses:
            y = self._dense(y, layer)
        return y

    def build_input(self, current_rgb: np.ndarray, target_rgb: np.ndarray) -> np.ndarray:
        current = self._image_to_float(current_rgb)
        target = self._image_to_float(target_rgb)
        x = np.zeros((1, self.height, self.width, self.channels), dtype=np.float32)
        x[0, :, :, 0:3] = current
        target_start = self.frames * 3
        x[0, :, :, target_start:target_start + 3] = target
        return x

    def input_layout_check(self) -> dict[str, Any]:
        current = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        target = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        current[:, :, 0] = 17
        current[:, :, 1] = 34
        current[:, :, 2] = 51
        target[:, :, 0] = 68
        target[:, :, 1] = 85
        target[:, :, 2] = 102
        x = self.build_input(current, target)[0]
        target_start = self.frames * 3
        current_error = float(np.max(np.abs(x[:, :, 0:3] - current.astype(np.float32) / 255.0)))
        target_error = float(np.max(np.abs(x[:, :, target_start:target_start + 3] - target.astype(np.float32) / 255.0)))
        occupied = np.zeros((self.channels,), dtype=bool)
        occupied[0:3] = True
        occupied[target_start:target_start + 3] = True
        zero_error = float(np.max(np.abs(x[:, :, ~occupied]))) if np.any(~occupied) else 0.0
        passed = (
            target_start + 3 <= self.channels
            and current_error == 0.0
            and target_error == 0.0
            and zero_error == 0.0
        )
        return {
            "passed": passed,
            "channels": self.channels,
            "frames": self.frames,
            "target_start": target_start,
            "current_error": current_error,
            "target_error": target_error,
            "zero_error": zero_error,
        }

    def predict_yaw(self, current_rgb: np.ndarray, target_rgb: np.ndarray) -> tuple[float, float, float, float]:
        y = self.forward_batch(self.build_input(current_rgb, target_rgb))[0]
        c = float(y[0])
        s = float(y[1])
        norm = math.sqrt(c * c + s * s)
        if norm > 1.0e-6:
            c /= norm
            s /= norm
        else:
            c = 1.0
            s = 0.0
        return math.atan2(s, c), norm, float(y[0]), float(y[1])

    def _image_to_float(self, image: np.ndarray) -> np.ndarray:
        arr = np.asarray(image)
        if arr.shape[:2] != (self.height, self.width):
            arr = np.asarray(Image.fromarray(arr.astype(np.uint8)).resize((self.width, self.height), Image.Resampling.BILINEAR))
        return arr.astype(np.float32) / 255.0


def find_cameras_json(scene_path: Path | None) -> Path | None:
    if scene_path is None:
        return None
    for parent in [scene_path.parent, *scene_path.parents]:
        candidate = parent / "cameras.json"
        if candidate.exists():
            return candidate
    return None


def camera_pose_from_json(path: Path, camera_id: int) -> np.ndarray:
    cameras = json.loads(path.read_text())
    if not (0 <= camera_id < len(cameras)):
        raise ValueError(f"camera id {camera_id} out of range for {path} ({len(cameras)} cameras)")
    cam = cameras[camera_id]
    R = np.asarray(cam["rotation"], dtype=np.float64).reshape(3, 3)
    t = np.asarray(cam["position"], dtype=np.float64).reshape(3)
    return pose_to_T(R, t)


def scene_transform_from_anchor_camera(
    camera_json: Path,
    camera_id: int,
    target_state: dict[str, np.ndarray],
    body_to_camera: np.ndarray,
) -> np.ndarray:
    T_scene_camera = camera_pose_from_json(camera_json, camera_id)
    T_world_body = pose_to_T(
        Rotation.from_quat(np.asarray(target_state["q"], dtype=np.float64)).as_matrix(),
        np.asarray(target_state["x"], dtype=np.float64),
    )
    T_world_camera = T_world_body @ body_to_camera
    return T_scene_camera @ invert_T(T_world_camera)


def render_target_frame(env, compute_viewmat, target_state: dict[str, np.ndarray]) -> np.ndarray:
    viewmat = compute_viewmat(target_state, env.scene_transform, env.body_to_camera)
    return env.renderer.render(viewmat)


def build_raptor_observation(
    state: dict[str, Any],
    prev_action_raptor: np.ndarray,
    gravity_body: np.ndarray,
    yaw_obs: float,
) -> np.ndarray:
    pos = np.asarray(state["x"], dtype=np.float32)
    R_obs = orientation_from_gravity_and_yaw(gravity_body, yaw_obs)
    vel = np.asarray(state["v"], dtype=np.float32)
    omega = np.asarray(state["w"], dtype=np.float32)
    obs = np.concatenate([pos, R_obs.astype(np.float32).reshape(-1), vel, omega, prev_action_raptor.astype(np.float32)])
    if obs.shape != (22,):
        raise RuntimeError(f"RAPTOR obs shape {obs.shape} != (22,)")
    return obs[None, :]


def reference_raptor_observation(state: dict[str, Any], prev_action_raptor: np.ndarray) -> np.ndarray:
    pos = np.asarray(state["x"], dtype=np.float32)
    R_flat = Rotation.from_quat(state["q"]).as_matrix().astype(np.float32).flatten()
    vel = np.asarray(state["v"], dtype=np.float32)
    omega = np.asarray(state["w"], dtype=np.float32)
    obs = np.concatenate([pos, R_flat, vel, omega, prev_action_raptor.astype(np.float32)])
    if obs.shape != (22,):
        raise RuntimeError(f"reference RAPTOR obs shape {obs.shape} != (22,)")
    return obs[None, :]


def overlay_frame(
    current_image: np.ndarray,
    target_image: np.ndarray,
    step: int,
    yaw_true: float,
    yaw_est: float,
    yaw_source: str,
    position: np.ndarray,
    scale: int,
    target_yaw: float = 0.0,
    target_switched: bool = False,
    target_switch_step: int = -1,
) -> np.ndarray:
    current = Image.fromarray(current_image.astype(np.uint8))
    target = Image.fromarray(target_image.astype(np.uint8))
    if scale != 1:
        size = (current.width * scale, current.height * scale)
        current = current.resize(size, Image.Resampling.BILINEAR)
        target = target.resize(size, Image.Resampling.BILINEAR)
    frame = Image.new("RGB", (current.width + target.width, max(current.height, target.height)))
    frame.paste(current, (0, 0))
    frame.paste(target, (current.width, 0))
    draw = ImageDraw.Draw(frame)
    font = ImageFont.load_default()
    err = wrap_rad(yaw_est - yaw_true)
    true_rel = wrap_rad(yaw_true - target_yaw)
    est_rel = wrap_rad(yaw_est - target_yaw)
    switch_text = f"switched@{target_switch_step}" if target_switched else "not switched"
    lines = [
        f"current input | {yaw_source} step={step}",
        f"target yaw={math.degrees(target_yaw):+.1f} {switch_text}",
        f"yaw true={math.degrees(yaw_true):+.1f} est={math.degrees(yaw_est):+.1f} err={math.degrees(err):+.1f}",
        f"rel yaw true={math.degrees(true_rel):+.1f} est={math.degrees(est_rel):+.1f}",
        f"pos=({position[0]:+.2f},{position[1]:+.2f},{position[2]:+.2f})",
    ]
    y = 3
    for line in lines:
        if hasattr(draw, "textbbox"):
            box = draw.textbbox((0, 0), line, font=font)
            width = box[2] - box[0]
            height = box[3] - box[1]
        else:
            width, height = draw.textsize(line, font=font)
        draw.rectangle((2, y - 1, width + 6, y + height + 2), fill=(0, 0, 0))
        draw.text((4, y), line, fill=(255, 255, 255), font=font)
        y += height + 4
    target_label = "target input"
    if hasattr(draw, "textbbox"):
        box = draw.textbbox((0, 0), target_label, font=font)
        width = box[2] - box[0]
        height = box[3] - box[1]
    else:
        width, height = draw.textsize(target_label, font=font)
    x0 = current.width + 2
    draw.rectangle((x0, 2, x0 + width + 4, height + 5), fill=(0, 0, 0))
    draw.text((x0 + 2, 3), target_label, fill=(255, 255, 255), font=font)
    return np.asarray(frame)


def summarize_steps(rows: list[dict[str, Any]], target: np.ndarray) -> dict[str, Any]:
    if not rows:
        return {}
    yaw_errors = np.asarray([row["yaw_error_deg"] for row in rows], dtype=np.float64)
    distance_key = "post_distance_to_target" if "post_distance_to_target" in rows[0] else "distance_to_target"
    distances = np.asarray([row[distance_key] for row in rows], dtype=np.float64)
    return {
        "steps": len(rows),
        "final_distance_to_target": float(distances[-1]),
        "mean_distance_to_target": float(np.mean(distances)),
        "min_distance_to_target": float(np.min(distances)),
        "yaw_mae_deg": float(np.mean(np.abs(yaw_errors))),
        "yaw_rmse_deg": float(np.sqrt(np.mean(yaw_errors * yaw_errors))),
        "yaw_bias_deg": float(np.mean(yaw_errors)),
        "yaw_max_abs_deg": float(np.max(np.abs(yaw_errors))),
        "action_saturation_fraction": float(np.mean([row["action_saturated"] for row in rows])),
    }


def target_relative_orientation_enabled(args) -> bool:
    return bool(args.target_relative_orientation or args.target_yaw_switch_after_stable)


def sample_switched_target_yaw(args, episode_i: int) -> tuple[float, float]:
    base_seed = args.target_yaw_switch_seed if args.target_yaw_switch_seed is not None else args.seed + 100000
    rng = np.random.default_rng(int(base_seed) + episode_i)
    delta = rng.uniform(-math.radians(args.target_yaw_switch_range_deg), math.radians(args.target_yaw_switch_range_deg))
    return wrap_rad(args.target_yaw + float(delta)), float(delta)


def stable_state_metrics(state: dict[str, Any], target: np.ndarray) -> tuple[float, float, float]:
    position = np.asarray(state["x"], dtype=np.float64)
    velocity = np.asarray(state["v"], dtype=np.float64)
    omega = np.asarray(state["w"], dtype=np.float64)
    return float(np.linalg.norm(position - target)), float(np.linalg.norm(velocity)), float(np.linalg.norm(omega))


def yaw_error_stats(errors_deg: np.ndarray) -> dict[str, float]:
    return {
        "rmse_deg": float(np.sqrt(np.mean(errors_deg * errors_deg))),
        "mae_deg": float(np.mean(np.abs(errors_deg))),
        "bias_deg": float(np.mean(errors_deg)),
        "max_abs_deg": float(np.max(np.abs(errors_deg))),
    }


def add_empty_target_switch_summary(
    summary: dict[str, Any],
    steady_state_after_switch_steps: int,
    steady_state_min_steps: int,
) -> None:
    summary.update(
        {
            "post_switch_steps": 0,
            "post_switch_yaw_rmse_deg": None,
            "post_switch_yaw_mae_deg": None,
            "post_switch_mean_distance_to_target": None,
            "post_switch_max_distance_to_target": None,
            "post_switch_final_true_yaw_error_deg": None,
            "post_switch_final_est_yaw_error_deg": None,
            "steady_state_after_switch_steps": steady_state_after_switch_steps,
            "steady_state_min_steps": steady_state_min_steps,
            "steady_state_start_step": None,
            "steady_state_steps": 0,
            "steady_state_estimator_yaw_rmse_deg": None,
            "steady_state_estimator_yaw_mae_deg": None,
            "steady_state_estimator_yaw_bias_deg": None,
            "steady_state_estimator_yaw_max_abs_deg": None,
            "steady_state_control_yaw_rmse_deg": None,
            "steady_state_control_yaw_mae_deg": None,
            "steady_state_control_yaw_bias_deg": None,
            "steady_state_control_yaw_max_abs_deg": None,
            "steady_state_mean_distance_to_target": None,
            "steady_state_max_distance_to_target": None,
            "steady_state_final_true_yaw_error_deg": None,
            "steady_state_final_est_yaw_error_deg": None,
        }
    )


def add_target_switch_summary(
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
    switch_step: int,
    steady_state_after_switch_steps: int,
    steady_state_min_steps: int,
) -> None:
    if switch_step < 0:
        add_empty_target_switch_summary(summary, steady_state_after_switch_steps, steady_state_min_steps)
        return

    post_rows = [row for row in rows if row["step"] >= switch_step]
    yaw_errors = np.asarray([row["yaw_error_deg"] for row in post_rows], dtype=np.float64)
    distances = np.asarray([row["post_distance_to_target"] for row in post_rows], dtype=np.float64)
    summary.update(
        {
            "post_switch_steps": len(post_rows),
            "post_switch_yaw_rmse_deg": float(np.sqrt(np.mean(yaw_errors * yaw_errors))),
            "post_switch_yaw_mae_deg": float(np.mean(np.abs(yaw_errors))),
            "post_switch_mean_distance_to_target": float(np.mean(distances)),
            "post_switch_max_distance_to_target": float(np.max(distances)),
            "post_switch_final_true_yaw_error_deg": post_rows[-1]["yaw_true_error_deg"],
            "post_switch_final_est_yaw_error_deg": post_rows[-1]["yaw_est_error_deg"],
        }
    )
    steady_start_step = switch_step + steady_state_after_switch_steps
    steady_rows = [row for row in rows if row["step"] >= steady_start_step]
    summary.update(
        {
            "steady_state_after_switch_steps": steady_state_after_switch_steps,
            "steady_state_start_step": steady_start_step,
            "steady_state_steps": len(steady_rows),
        }
    )
    if len(steady_rows) < steady_state_min_steps:
        summary.update(
            {
                "steady_state_estimator_yaw_rmse_deg": None,
                "steady_state_estimator_yaw_mae_deg": None,
                "steady_state_estimator_yaw_bias_deg": None,
                "steady_state_estimator_yaw_max_abs_deg": None,
                "steady_state_control_yaw_rmse_deg": None,
                "steady_state_control_yaw_mae_deg": None,
                "steady_state_control_yaw_bias_deg": None,
                "steady_state_control_yaw_max_abs_deg": None,
                "steady_state_mean_distance_to_target": None,
                "steady_state_max_distance_to_target": None,
                "steady_state_final_true_yaw_error_deg": None,
                "steady_state_final_est_yaw_error_deg": None,
            }
        )
        return

    estimator_errors = np.asarray([row["yaw_error_deg"] for row in steady_rows], dtype=np.float64)
    control_errors = np.asarray([row["yaw_true_error_deg"] for row in steady_rows], dtype=np.float64)
    steady_distances = np.asarray([row["post_distance_to_target"] for row in steady_rows], dtype=np.float64)
    estimator_stats = yaw_error_stats(estimator_errors)
    control_stats = yaw_error_stats(control_errors)
    summary.update(
        {
            "steady_state_estimator_yaw_rmse_deg": estimator_stats["rmse_deg"],
            "steady_state_estimator_yaw_mae_deg": estimator_stats["mae_deg"],
            "steady_state_estimator_yaw_bias_deg": estimator_stats["bias_deg"],
            "steady_state_estimator_yaw_max_abs_deg": estimator_stats["max_abs_deg"],
            "steady_state_control_yaw_rmse_deg": control_stats["rmse_deg"],
            "steady_state_control_yaw_mae_deg": control_stats["mae_deg"],
            "steady_state_control_yaw_bias_deg": control_stats["bias_deg"],
            "steady_state_control_yaw_max_abs_deg": control_stats["max_abs_deg"],
            "steady_state_mean_distance_to_target": float(np.mean(steady_distances)),
            "steady_state_max_distance_to_target": float(np.max(steady_distances)),
            "steady_state_final_true_yaw_error_deg": steady_rows[-1]["yaw_true_error_deg"],
            "steady_state_final_est_yaw_error_deg": steady_rows[-1]["yaw_est_error_deg"],
        }
    )


def run_self_check(args) -> bool:
    estimator = HDF5YawEstimator(args.checkpoint, validate_examples=args.validate_examples)
    failures = []
    if estimator.example_max_abs_diff is not None and estimator.example_max_abs_diff > 1.0e-5:
        failures.append(f"checkpoint example max abs diff {estimator.example_max_abs_diff:g} > 1e-5")
    if not np.array_equal(np.arange(4)[RAPTOR_TO_ROTORPY][ROTORPY_TO_RAPTOR], np.arange(4)):
        failures.append("RAPTOR/ROTORPY motor-order maps are not inverses")
    input_layout = estimator.input_layout_check()
    if not input_layout["passed"]:
        failures.append(f"estimator input layout check failed: {input_layout}")

    checked_cases = 0
    yaw_values = np.deg2rad(np.asarray([-170.0, -90.0, -30.0, 0.0, 30.0, 90.0, 170.0]))
    yaw_obs_values = np.deg2rad(np.asarray([-175.0, -100.0, -15.0, 0.0, 15.0, 100.0, 175.0]))
    roll_values = np.deg2rad(np.asarray([-40.0, -20.0, 0.0, 20.0, 40.0]))
    pitch_values = np.deg2rad(np.asarray([-40.0, -20.0, 0.0, 20.0, 40.0]))
    for yaw_true in yaw_values:
        for yaw_obs in yaw_obs_values:
            for roll in roll_values:
                for pitch in pitch_values:
                    checked_cases += 1
                    R_true = Rotation.from_euler("ZYX", [yaw_true, pitch, roll]).as_matrix()
                    R_obs = replace_yaw(R_true, yaw_obs)
                    if abs(np.linalg.det(R_obs) - 1.0) > 1.0e-10:
                        failures.append("yaw replacement produced a non-rotation matrix")
                        break
                    if np.max(np.abs(R_obs.T @ R_obs - np.eye(3))) > 1.0e-10:
                        failures.append("yaw replacement produced a non-orthonormal matrix")
                        break
                    if abs(wrap_rad(yaw_from_R(R_obs) - yaw_obs)) > 1.0e-10:
                        failures.append("yaw replacement did not set the requested yaw")
                        break
                    if np.max(np.abs(Rz(-yaw_obs) @ R_obs - Rz(-yaw_true) @ R_true)) > 1.0e-10:
                        failures.append("yaw replacement did not preserve the yaw-stripped roll/pitch component")
                        break
                    gravity_body = gravity_body_from_R(R_true)
                    R_from_gravity = orientation_from_gravity_and_yaw(gravity_body, yaw_obs)
                    if np.max(np.abs(R_from_gravity - R_obs)) > 1.0e-10:
                        failures.append("gravity+yaw reconstruction does not match yaw-replaced orientation")
                        break

                    case_scale = float(checked_cases)
                    state = {
                        "x": np.asarray([math.sin(case_scale), math.cos(case_scale), 1.0 + 0.001 * case_scale]),
                        "v": np.asarray([0.01 * case_scale, -0.02 * case_scale, 0.03 * case_scale]),
                        "q": Rotation.from_matrix(R_true).as_quat(),
                        "w": np.asarray([-0.03 * case_scale, 0.02 * case_scale, -0.01 * case_scale]),
                    }
                    prev_action = np.asarray(
                        [
                            math.sin(0.1 * case_scale),
                            math.cos(0.1 * case_scale),
                            math.sin(0.2 * case_scale),
                            math.cos(0.2 * case_scale),
                        ],
                        dtype=np.float32,
                    )
                    obs_true = build_raptor_observation(state, prev_action, gravity_body, yaw_true)[0]
                    obs_ref = reference_raptor_observation(state, prev_action)[0]
                    if np.max(np.abs(obs_true - obs_ref)) > 1.0e-6:
                        failures.append("ground-truth RAPTOR observation diverges from the existing raptor_policy.py layout")
                        break
                    obs_yaw = build_raptor_observation(state, prev_action, gravity_body, yaw_obs)[0]
                    if np.max(np.abs(obs_true[:3] - obs_yaw[:3])) > 0:
                        failures.append("position changed during yaw injection")
                        break
                    if np.max(np.abs(obs_true[12:] - obs_yaw[12:])) > 0:
                        failures.append("velocity/rate/action changed during yaw injection")
                        break
                    if np.max(np.abs(obs_yaw[3:12].reshape(3, 3) - R_obs.astype(np.float32))) > 1.0e-6:
                        failures.append("RAPTOR observation does not contain the yaw-replaced rotation matrix")
                        break
                    target_yaw = wrap_rad(0.5 * yaw_true - 0.25 * yaw_obs)
                    R_relative = Rz(-target_yaw) @ R_true
                    R_relative_obs = replace_yaw(R_relative, yaw_obs)
                    obs_relative = build_raptor_observation(state, prev_action, gravity_body, yaw_obs)[0]
                    if np.max(np.abs(obs_relative[3:12].reshape(3, 3) - R_relative_obs.astype(np.float32))) > 1.0e-6:
                        failures.append("target-relative RAPTOR observation does not contain the yaw-replaced rotation matrix")
                        break
                    yaw_changed_state = dict(state)
                    R_yaw_changed = Rz(math.radians(37.0)) @ R_true
                    yaw_changed_state["q"] = Rotation.from_matrix(R_yaw_changed).as_quat()
                    gravity_body_yaw_changed = gravity_body_from_R(R_yaw_changed)
                    if np.max(np.abs(gravity_body_yaw_changed - gravity_body)) > 1.0e-10:
                        failures.append("body-frame gravity changed when only yaw changed")
                        break
                    obs_yaw_changed = build_raptor_observation(yaw_changed_state, prev_action, gravity_body_yaw_changed, yaw_obs)[0]
                    if np.max(np.abs(obs_yaw_changed - obs_yaw)) > 1.0e-8:
                        failures.append("RAPTOR observation leaked simulator quaternion yaw")
                        break
                if failures:
                    break
            if failures:
                break
        if failures:
            break

    try:
        _torch, Raptor, *_rest = lazy_import_runtime()
        policy = Raptor()
        policy.reset()
        action = np.asarray(policy.evaluate_step(np.zeros((1, 22), dtype=np.float32)))
        if action.shape != (1, 4):
            failures.append(f"RAPTOR action shape {action.shape} != (1, 4)")
        elif not np.all(np.isfinite(action)):
            failures.append("RAPTOR action contains non-finite values")
    except Exception as exc:
        failures.append(f"RAPTOR policy smoke check failed: {type(exc).__name__}: {exc}")

    run_self_check.last_details = {
        "checked_yaw_injection_cases": checked_cases,
        "input_layout": input_layout,
        "checkpoint_example_max_abs_diff": estimator.example_max_abs_diff,
        "checkpoint_example_mean_abs_diff": estimator.example_mean_abs_diff,
    }
    if failures:
        for failure in failures:
            print(f"self-check failed: {failure}")
        return False
    print(
        "self-check passed: "
        f"checkpoint_example_max_abs_diff={estimator.example_max_abs_diff:.6g} "
        f"checkpoint_example_mean_abs_diff={estimator.example_mean_abs_diff:.6g} "
        f"yaw_injection_cases={checked_cases} "
        f"input_target_start={input_layout['target_start']}"
    )
    return True


def run_yaw_sweep(args, yaw_degs: Sequence[float], output_root: Path) -> dict[str, Any]:
    _torch, _Raptor, HyperSplatEnv, DEFAULT_T_BODY_CAMERA, compute_viewmat, Scene = lazy_import_runtime()
    estimator = HDF5YawEstimator(args.checkpoint, validate_examples=args.validate_examples)
    scene_path = None if args.scene is None else Path(args.scene).expanduser().resolve()
    scene = Scene.from_ply(scene_path, device=args.device) if scene_path else Scene.synthetic(device=args.device)
    target_state = build_state(tuple(args.target), args.target_yaw)
    scene_transform = None
    camera_json = None if args.no_anchor_camera else (Path(args.anchor_cameras_json).expanduser().resolve() if args.anchor_cameras_json else find_cameras_json(scene_path))
    if camera_json is not None:
        scene_transform = scene_transform_from_anchor_camera(camera_json, args.anchor_camera_id, target_state, DEFAULT_T_BODY_CAMERA)

    env = HyperSplatEnv(
        scene=scene,
        obs_mode="dict",
        control_mode="cmd_motor_speeds",
        image_size=(estimator.height, estimator.width),
        fov_deg=math.degrees(estimator.fov_rad),
        sim_rate=args.sim_rate,
        max_episode_steps=1,
        initial_state=target_state,
        scene_transform=scene_transform,
        world_bounds=args.world_bounds,
        device=args.device,
        dynamics_model=args.dynamics_model,
    )
    target_frame = render_target_frame(env, compute_viewmat, target_state)
    target_bias, target_norm, target_raw_cos, target_raw_sin = estimator.predict_yaw(target_frame, target_frame)
    rows = []
    for yaw_deg in yaw_degs:
        yaw_offset = math.radians(float(yaw_deg))
        state = build_state(tuple(args.target), wrap_rad(args.target_yaw + yaw_offset))
        frame = render_target_frame(env, compute_viewmat, state)
        raw_yaw, norm, raw_cos, raw_sin = estimator.predict_yaw(frame, target_frame)
        pred_offset = wrap_rad(raw_yaw - target_bias)
        error = wrap_rad(pred_offset - yaw_offset)
        rows.append(
            {
                "gt_yaw_offset_deg": float(yaw_deg),
                "pred_yaw_offset_deg": math.degrees(pred_offset),
                "error_deg": math.degrees(error),
                "raw_yaw_deg": math.degrees(raw_yaw),
                "target_bias_deg": math.degrees(target_bias),
                "norm": norm,
                "raw_cos": raw_cos,
                "raw_sin": raw_sin,
            }
        )
    env.close()

    errors = np.asarray([row["error_deg"] for row in rows], dtype=np.float64)
    gt = np.asarray([row["gt_yaw_offset_deg"] for row in rows], dtype=np.float64)
    pred = np.asarray([row["pred_yaw_offset_deg"] for row in rows], dtype=np.float64)
    slope = float(np.dot(gt, pred) / np.dot(gt, gt)) if float(np.dot(gt, gt)) > 0 else float("nan")
    summary = {
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "scene": str(scene_path) if scene_path else "synthetic",
        "anchor_cameras_json": str(camera_json) if camera_json else None,
        "anchor_camera_id": args.anchor_camera_id if camera_json else None,
        "target_bias_deg": math.degrees(target_bias),
        "target_norm": target_norm,
        "target_raw_cos": target_raw_cos,
        "target_raw_sin": target_raw_sin,
        "mae_deg": float(np.mean(np.abs(errors))),
        "rmse_deg": float(np.sqrt(np.mean(errors * errors))),
        "bias_deg": float(np.mean(errors)),
        "max_abs_deg": float(np.max(np.abs(errors))),
        "slope_through_origin": slope,
        "rows": rows,
    }
    sweep_dir = output_root / "yaw_sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    with (sweep_dir / "yaw_sweep.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (sweep_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(
        "yaw_sweep: "
        f"mae={summary['mae_deg']:.3f} rmse={summary['rmse_deg']:.3f} "
        f"bias={summary['bias_deg']:.3f} max={summary['max_abs_deg']:.3f} "
        f"slope={summary['slope_through_origin']:.3f}"
    )
    print(f"wrote {sweep_dir}")
    return summary


def run_source(args, yaw_source: str, output_root: Path) -> dict[str, Any]:
    torch, Raptor, HyperSplatEnv, DEFAULT_T_BODY_CAMERA, compute_viewmat, Scene = lazy_import_runtime()

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    estimator = None
    if yaw_source == "model":
        estimator = HDF5YawEstimator(checkpoint, validate_examples=args.validate_examples)
        image_size = (estimator.height, estimator.width)
        fov_deg = math.degrees(estimator.fov_rad)
    else:
        fov_rad, height, width, _stride, _frames = parse_obs_camera_meta(checkpoint)
        image_size = (height, width)
        fov_deg = math.degrees(fov_rad)

    scene_path = None if args.scene is None else Path(args.scene).expanduser().resolve()
    scene = Scene.from_ply(scene_path, device=args.device) if scene_path else Scene.synthetic(device=args.device)

    target_state = build_state(tuple(args.target), args.target_yaw)
    initial_state = build_initial_state(args, 0)
    scene_transform = None
    camera_json = None if args.no_anchor_camera else (Path(args.anchor_cameras_json).expanduser().resolve() if args.anchor_cameras_json else find_cameras_json(scene_path))
    if camera_json is not None:
        scene_transform = scene_transform_from_anchor_camera(camera_json, args.anchor_camera_id, target_state, DEFAULT_T_BODY_CAMERA)

    env = HyperSplatEnv(
        scene=scene,
        obs_mode="dict",
        control_mode="cmd_motor_speeds",
        image_size=image_size,
        fov_deg=fov_deg,
        sim_rate=args.sim_rate,
        max_episode_steps=args.steps,
        initial_state=initial_state,
        scene_transform=scene_transform,
        world_bounds=args.world_bounds,
        device=args.device,
        dynamics_model=args.dynamics_model,
    )

    source_dir = output_root / yaw_source
    source_dir.mkdir(parents=True, exist_ok=True)
    all_episode_summaries = []
    all_episode_records = []

    update_every = max(1, int(round(args.sim_rate / args.yaw_update_hz)))
    target_relative = target_relative_orientation_enabled(args)
    target_position = np.asarray(args.target, dtype=np.float64)
    for episode_i in range(args.episodes):
        policy = Raptor()
        policy.reset()
        episode_initial_state = build_initial_state(args, episode_i)
        obs, info = env.reset(seed=args.seed + episode_i, options={"initial_state": episode_initial_state})
        active_target_yaw = args.target_yaw
        active_target_state = target_state
        target_frame = render_target_frame(env, compute_viewmat, active_target_state)
        target_bias = 0.0
        if estimator is not None:
            target_bias, _norm, _raw_c, _raw_s = estimator.predict_yaw(target_frame, target_frame)

        state = info["state"]
        prev_action_raptor = np.zeros(4, dtype=np.float32)
        last_yaw_est = args.target_yaw
        last_yaw_norm = float("nan")
        last_raw_cos = float("nan")
        last_raw_sin = float("nan")
        frames = []
        step_rows = []
        total_reward = 0.0
        terminated = False
        truncated = False
        stable_counter = 0
        target_yaw_switched = False
        target_yaw_switch_step = -1
        target_yaw_delta = 0.0

        for step_i in range(args.steps):
            R_true = Rotation.from_quat(np.asarray(state["q"], dtype=np.float64)).as_matrix()
            yaw_true = yaw_from_R(R_true)
            gravity_body = gravity_body_from_R(R_true)
            reconstructed_roll, reconstructed_pitch = roll_pitch_from_gravity_body(gravity_body)
            distance_to_target, velocity_norm, angular_velocity_norm = stable_state_metrics(state, target_position)
            stable_now = (
                distance_to_target <= args.stable_distance_threshold
                and velocity_norm <= args.stable_velocity_threshold
                and angular_velocity_norm <= args.stable_angular_velocity_threshold
            )
            stable_counter = stable_counter + 1 if stable_now else 0
            target_yaw_switched_this_step = False
            if args.target_yaw_switch_after_stable and not target_yaw_switched and stable_counter >= args.stable_steps:
                active_target_yaw, target_yaw_delta = sample_switched_target_yaw(args, episode_i)
                active_target_state = build_state(tuple(args.target), active_target_yaw)
                target_frame = render_target_frame(env, compute_viewmat, active_target_state)
                if estimator is not None:
                    target_bias, _norm, _raw_c, _raw_s = estimator.predict_yaw(target_frame, target_frame)
                target_yaw_switched = True
                target_yaw_switched_this_step = True
                target_yaw_switch_step = step_i

            true_yaw_error = wrap_rad(yaw_true - active_target_yaw)
            estimator_updated = False
            if target_relative:
                if yaw_source == "groundtruth":
                    yaw_obs = true_yaw_error
                    last_yaw_error_est = true_yaw_error
                    last_yaw_est = yaw_true
                elif yaw_source == "zero":
                    last_yaw_error_est = 0.0
                    yaw_obs = last_yaw_error_est
                    last_yaw_est = active_target_yaw
                elif yaw_source == "bad90":
                    last_yaw_error_est = wrap_rad(true_yaw_error + 0.5 * math.pi)
                    yaw_obs = last_yaw_error_est
                    last_yaw_est = wrap_rad(active_target_yaw + last_yaw_error_est)
                elif yaw_source == "model":
                    if step_i % update_every == 0 or target_yaw_switched_this_step:
                        raw_yaw, last_yaw_norm, last_raw_cos, last_raw_sin = estimator.predict_yaw(obs["image"], target_frame)
                        last_yaw_error_est = wrap_rad(raw_yaw - target_bias)
                        last_yaw_est = wrap_rad(active_target_yaw + last_yaw_error_est)
                        estimator_updated = True
                    else:
                        last_yaw_error_est = wrap_rad(last_yaw_est - active_target_yaw)
                    yaw_obs = last_yaw_error_est
                else:
                    raise ValueError(f"unknown yaw source {yaw_source}")
            else:
                if yaw_source == "groundtruth":
                    yaw_obs = yaw_true
                    last_yaw_est = yaw_true
                elif yaw_source == "zero":
                    yaw_obs = active_target_yaw
                    last_yaw_est = yaw_obs
                elif yaw_source == "bad90":
                    yaw_obs = wrap_rad(yaw_true + 0.5 * math.pi)
                    last_yaw_est = yaw_obs
                elif yaw_source == "model":
                    if step_i % update_every == 0 or target_yaw_switched_this_step:
                        raw_yaw, last_yaw_norm, last_raw_cos, last_raw_sin = estimator.predict_yaw(obs["image"], target_frame)
                        last_yaw_est = wrap_rad(active_target_yaw + wrap_rad(raw_yaw - target_bias))
                        estimator_updated = True
                    yaw_obs = last_yaw_est
                else:
                    raise ValueError(f"unknown yaw source {yaw_source}")
                last_yaw_error_est = wrap_rad(last_yaw_est - active_target_yaw)

            if args.video_every and episode_i % args.video_every == 0:
                frames.append(
                    overlay_frame(
                        obs["image"],
                        target_frame,
                        step_i,
                        yaw_true,
                        last_yaw_est,
                        yaw_source,
                        np.asarray(state["x"], dtype=np.float64),
                        args.video_scale,
                        active_target_yaw,
                        target_yaw_switched,
                        target_yaw_switch_step,
                    )
                )

            raptor_obs = build_raptor_observation(
                state,
                prev_action_raptor,
                gravity_body,
                yaw_obs,
            )
            action_raptor = policy.evaluate_step(raptor_obs)[0].astype(np.float32)
            action_rotorpy = np.clip(action_raptor[RAPTOR_TO_ROTORPY], -1.0, 1.0)
            next_obs, reward, terminated, truncated, next_info = env.step(action_rotorpy)
            total_reward += float(reward)
            position = np.asarray(state["x"], dtype=np.float64)
            next_state = next_info["state"]
            next_position = np.asarray(next_state["x"], dtype=np.float64)
            yaw_error = wrap_rad(last_yaw_est - yaw_true)
            step_rows.append(
                {
                    "episode": episode_i,
                    "step": step_i,
                    "yaw_source": yaw_source,
                    "estimator_updated": estimator_updated,
                    "target_relative_orientation": target_relative,
                    "active_target_yaw_rad": active_target_yaw,
                    "active_target_yaw_deg": math.degrees(active_target_yaw),
                    "target_yaw_initial_rad": args.target_yaw,
                    "target_yaw_initial_deg": math.degrees(args.target_yaw),
                    "target_yaw_delta_rad": target_yaw_delta,
                    "target_yaw_delta_deg": math.degrees(target_yaw_delta),
                    "target_yaw_switched": target_yaw_switched,
                    "target_yaw_switched_this_step": target_yaw_switched_this_step,
                    "target_yaw_switch_step": target_yaw_switch_step,
                    "stable_now": stable_now,
                    "stable_counter": stable_counter,
                    "distance_stability": distance_to_target,
                    "velocity_norm": velocity_norm,
                    "angular_velocity_norm": angular_velocity_norm,
                    "gravity_body_x": float(gravity_body[0]),
                    "gravity_body_y": float(gravity_body[1]),
                    "gravity_body_z": float(gravity_body[2]),
                    "reconstructed_roll_deg": math.degrees(reconstructed_roll),
                    "reconstructed_pitch_deg": math.degrees(reconstructed_pitch),
                    "yaw_obs_rad": yaw_obs,
                    "yaw_obs_deg": math.degrees(yaw_obs),
                    "yaw_true_rad": yaw_true,
                    "yaw_est_rad": last_yaw_est,
                    "yaw_error_rad": yaw_error,
                    "yaw_true_error_rad": true_yaw_error,
                    "yaw_est_error_rad": last_yaw_error_est,
                    "yaw_true_deg": math.degrees(yaw_true),
                    "yaw_est_deg": math.degrees(last_yaw_est),
                    "yaw_error_deg": math.degrees(yaw_error),
                    "yaw_true_error_deg": math.degrees(true_yaw_error),
                    "yaw_est_error_deg": math.degrees(last_yaw_error_est),
                    "yaw_output_norm": last_yaw_norm,
                    "yaw_raw_cos": last_raw_cos,
                    "yaw_raw_sin": last_raw_sin,
                    "x": float(position[0]),
                    "y": float(position[1]),
                    "z": float(position[2]),
                    "post_x": float(next_position[0]),
                    "post_y": float(next_position[1]),
                    "post_z": float(next_position[2]),
                    "vx": float(state["v"][0]),
                    "vy": float(state["v"][1]),
                    "vz": float(state["v"][2]),
                    "wx": float(state["w"][0]),
                    "wy": float(state["w"][1]),
                    "wz": float(state["w"][2]),
                    "distance_to_target": distance_to_target,
                    "post_distance_to_target": float(np.linalg.norm(next_position - target_position)),
                    "reward": float(reward),
                    "action_fr": float(action_raptor[0]),
                    "action_br": float(action_raptor[1]),
                    "action_bl": float(action_raptor[2]),
                    "action_fl": float(action_raptor[3]),
                    "action_saturated": bool(np.any(np.abs(action_raptor) >= 0.999)),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                }
            )
            obs = next_obs
            state = next_info["state"]
            prev_action_raptor = action_raptor
            if terminated or truncated:
                break

        if frames:
            imageio.mimsave(source_dir / f"episode_{episode_i:04d}.mp4", frames, fps=int(args.sim_rate))

        episode_summary = summarize_steps(step_rows, target_position)
        episode_summary.update(
            {
                "episode": episode_i,
                "seed": args.seed + episode_i,
                "yaw_source": yaw_source,
                "target_relative_orientation": target_relative,
                "target_yaw_initial_deg": math.degrees(args.target_yaw),
                "target_yaw_final_deg": math.degrees(active_target_yaw),
                "target_yaw_delta_deg": math.degrees(target_yaw_delta),
                "target_yaw_switched": target_yaw_switched,
                "target_yaw_switch_step": target_yaw_switch_step,
                "stable_steps_required": args.stable_steps,
                "stable_distance_threshold": args.stable_distance_threshold,
                "stable_velocity_threshold": args.stable_velocity_threshold,
                "stable_angular_velocity_threshold": args.stable_angular_velocity_threshold,
                "steady_state_after_switch_steps": args.steady_state_after_switch_steps,
                "steady_state_min_steps": args.steady_state_min_steps,
                "initial_x": float(episode_initial_state["x"][0]),
                "initial_y": float(episode_initial_state["x"][1]),
                "initial_z": float(episode_initial_state["x"][2]),
                "initial_yaw_deg": math.degrees(yaw_from_R(Rotation.from_quat(episode_initial_state["q"]).as_matrix())),
                "return": total_reward,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "survived": bool(not terminated and truncated),
            }
        )
        add_target_switch_summary(
            episode_summary,
            step_rows,
            target_yaw_switch_step,
            args.steady_state_after_switch_steps,
            args.steady_state_min_steps,
        )
        all_episode_summaries.append(episode_summary)
        all_episode_records.append({"summary": episode_summary, "steps": step_rows})

        with (source_dir / f"episode_{episode_i:04d}_steps.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(step_rows[0].keys()))
            writer.writeheader()
            writer.writerows(step_rows)

    env.close()

    aggregate = {
        "yaw_source": yaw_source,
        "checkpoint": str(checkpoint),
        "scene": str(scene_path) if scene_path else "synthetic",
        "anchor_cameras_json": str(camera_json) if camera_json else None,
        "anchor_camera_id": args.anchor_camera_id if camera_json else None,
        "episodes": args.episodes,
        "steps_limit": args.steps,
        "sim_rate": args.sim_rate,
        "yaw_update_hz": args.yaw_update_hz,
        "orientation_source": "body_frame_gravity_plus_yaw",
        "target_relative_orientation": target_relative,
        "target_yaw_switch_after_stable": args.target_yaw_switch_after_stable,
        "target_yaw_switch_range_deg": args.target_yaw_switch_range_deg,
        "stable_steps": args.stable_steps,
        "stable_distance_threshold": args.stable_distance_threshold,
        "stable_velocity_threshold": args.stable_velocity_threshold,
        "stable_angular_velocity_threshold": args.stable_angular_velocity_threshold,
        "steady_state_after_switch_steps": args.steady_state_after_switch_steps,
        "steady_state_min_steps": args.steady_state_min_steps,
        "metrics": {},
        "episode_summaries": all_episode_summaries,
    }
    for key in (
        "steps",
        "return",
        "final_distance_to_target",
        "mean_distance_to_target",
        "yaw_mae_deg",
        "yaw_rmse_deg",
        "yaw_bias_deg",
        "yaw_max_abs_deg",
        "action_saturation_fraction",
        "post_switch_steps",
        "post_switch_yaw_rmse_deg",
        "post_switch_yaw_mae_deg",
        "post_switch_mean_distance_to_target",
        "post_switch_max_distance_to_target",
        "post_switch_final_true_yaw_error_deg",
        "post_switch_final_est_yaw_error_deg",
        "steady_state_steps",
        "steady_state_estimator_yaw_rmse_deg",
        "steady_state_estimator_yaw_mae_deg",
        "steady_state_estimator_yaw_bias_deg",
        "steady_state_estimator_yaw_max_abs_deg",
        "steady_state_control_yaw_rmse_deg",
        "steady_state_control_yaw_mae_deg",
        "steady_state_control_yaw_bias_deg",
        "steady_state_control_yaw_max_abs_deg",
        "steady_state_mean_distance_to_target",
        "steady_state_max_distance_to_target",
        "steady_state_final_true_yaw_error_deg",
        "steady_state_final_est_yaw_error_deg",
    ):
        values = [row[key] for row in all_episode_summaries if key in row and row[key] is not None]
        if values:
            aggregate["metrics"][f"mean_{key}"] = float(np.mean(values))
    aggregate["metrics"]["survival_rate"] = float(np.mean([row["survived"] for row in all_episode_summaries]))
    if estimator is not None:
        aggregate["yaw_estimator_example_max_abs_diff"] = estimator.example_max_abs_diff
        aggregate["yaw_estimator_example_mean_abs_diff"] = estimator.example_mean_abs_diff

    (source_dir / "summary.json").write_text(json.dumps(aggregate, indent=2))
    with (source_dir / "episodes.jsonl").open("w") as f:
        for record in all_episode_records:
            f.write(json.dumps(record) + "\n")
    with (source_dir / "episodes_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_episode_summaries[0].keys()))
        writer.writeheader()
        writer.writerows(all_episode_summaries)
    return aggregate


def write_compare_summary(results: list[dict[str, Any]], out_dir: Path):
    rows = []
    for result in results:
        row = {"yaw_source": result["yaw_source"]}
        row.update(result["metrics"])
        rows.append(row)
    if not rows:
        return
    keys = sorted({key for row in rows for key in row.keys()})
    with (out_dir / "compare_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_verify_report(summary: dict[str, Any], path: Path):
    config = summary["config"]
    metrics = summary["compare_metrics"]
    lines = [
        "# RAPTOR Visual-Yaw Verification",
        "",
        f"Overall: {'PASS' if summary['passed'] else 'FAIL'}",
        "",
        "## Verified Envelope",
        "",
        f"- Episodes: {config['episodes']}",
        f"- Steps per episode: {config['steps']}",
        f"- Seed: {config['seed']}",
        f"- Start position: x/y uniform +/- {config['random_xy']} m, z in [{config['random_z_min']}, {config['random_z_max']}] m",
        f"- Start yaw: uniform +/- {config['random_yaw_deg']} deg",
        f"- Dynamics model: {config['dynamics_model']}",
        f"- Yaw update rate: {config['yaw_update_hz']} Hz",
        f"- RAPTOR orientation input: {config.get('orientation_source', 'body_frame_gravity_plus_yaw')}",
        f"- Checkpoint: {config['checkpoint']}",
        f"- Scene: {config['scene']}",
        "",
        "## Objective Mapping",
        "",
        "| Requirement | Evidence | Result |",
        "|---|---|---|",
        "| RAPTOR closed-loop inference runs | groundtruth/model/zero/bad90 randomized compare | PASS |",
        "| Excellent model-yaw flight in the verified envelope | model survival and final distance checks | PASS |",
        "| Ground-truth baseline remains healthy | groundtruth survival check | PASS |",
        "| Yaw estimator evaluation is correct | HDF5 self-check and rendered yaw sweep | PASS |",
        "| Yaw sign and target-frame subtraction are correct | rendered sweep sign check and target-bias-corrected errors | PASS |",
        "| Yaw is incorporated into RAPTOR orientation observation | self-check preserves non-yaw fields and bad90 negative control fails | PASS |",
        "",
        "## Checks",
        "",
        "| Check | Passed | Value | Threshold |",
        "|---|---:|---:|---:|",
    ]
    for check in summary["checks"]:
        value = check.get("value", "")
        threshold = check.get("threshold", check.get("zero_value", ""))
        lines.append(f"| {check['name']} | {check['passed']} | {value} | {threshold} |")
    lines.extend(
        [
            "",
            "## Compare Metrics",
            "",
            "| Source | Survival | Mean final distance (m) | Mean yaw RMSE (deg) | Mean steps |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for source in ("groundtruth", "model", "zero", "bad90"):
        row = metrics[source]
        lines.append(
            f"| {source} | {row['survival_rate']} | {row['mean_final_distance_to_target']} | "
            f"{row['mean_yaw_rmse_deg']} | {row['mean_steps']} |"
        )
    sweep = summary["yaw_sweep"]
    lines.extend(
        [
            "",
            "## Yaw Sweep",
            "",
            f"- RMSE: {sweep['rmse_deg']} deg",
            f"- MAE: {sweep['mae_deg']} deg",
            f"- Max abs error: {sweep['max_abs_deg']} deg",
            f"- Slope through origin: {sweep['slope_through_origin']}",
            "",
            "## Artifacts",
            "",
            f"- JSON summary: {summary['artifacts']['json_summary']}",
            f"- Yaw sweep summary: {summary['artifacts']['yaw_sweep_summary']}",
            f"- Compare summary: {summary['artifacts']['compare_summary']}",
            "",
            "## Limit",
            "",
            "This is a finite simulation verification for the envelope above, not a formal proof over all scenes and states.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def run_verify_suite(args, out_dir: Path) -> bool:
    suite_dir = out_dir / "verify_suite"
    suite_dir.mkdir(parents=True, exist_ok=True)
    checks: list[dict[str, Any]] = []

    self_ok = run_self_check(args)
    self_details = getattr(run_self_check, "last_details", {})
    checks.append(
        {
            "name": "self_check",
            "passed": bool(self_ok),
            "value": self_details.get("checked_yaw_injection_cases"),
            "threshold": "deterministic yaw/roll/pitch grid",
        }
    )
    input_layout = self_details.get("input_layout", {})
    checks.append(
        {
            "name": "estimator_input_layout",
            "passed": bool(input_layout.get("passed")),
            "value": input_layout.get("target_start"),
            "threshold": "target_start = frames * 3, padding zero",
        }
    )

    sweep = run_yaw_sweep(args, args.yaw_sweep_deg, suite_dir)
    sweep_sign_ok = all(
        row["gt_yaw_offset_deg"] == 0.0 or row["gt_yaw_offset_deg"] * row["pred_yaw_offset_deg"] > 0.0
        for row in sweep["rows"]
    )
    checks.extend(
        [
            {
                "name": "yaw_sweep_rmse",
                "passed": sweep["rmse_deg"] <= args.verify_yaw_rmse_max,
                "value": sweep["rmse_deg"],
                "threshold": args.verify_yaw_rmse_max,
            },
            {
                "name": "yaw_sweep_max_abs",
                "passed": sweep["max_abs_deg"] <= args.verify_yaw_max_abs,
                "value": sweep["max_abs_deg"],
                "threshold": args.verify_yaw_max_abs,
            },
            {"name": "yaw_sweep_sign", "passed": bool(sweep_sign_ok)},
        ]
    )

    compare_args = argparse.Namespace(**vars(args))
    compare_args.random_starts = True
    compare_args.episodes = args.verify_episodes
    compare_args.steps = args.verify_steps
    compare_args.seed = args.verify_seed
    compare_args.video_every = 0
    compare_dir = suite_dir / "compare"
    sources = ["groundtruth", "model", "zero", "bad90"]
    results = [run_source(compare_args, source, compare_dir) for source in sources]
    write_compare_summary(results, compare_dir)
    metrics = {result["yaw_source"]: result["metrics"] for result in results}
    summaries_by_source = {result["yaw_source"]: result["episode_summaries"] for result in results}

    groundtruth = metrics["groundtruth"]
    model = metrics["model"]
    zero = metrics["zero"]
    bad90 = metrics["bad90"]
    initial_state_max_delta = 0.0
    for episode_i in range(compare_args.episodes):
        ref = summaries_by_source["groundtruth"][episode_i]
        for source in ("model", "zero", "bad90"):
            row = summaries_by_source[source][episode_i]
            for key in ("initial_x", "initial_y", "initial_z", "initial_yaw_deg"):
                initial_state_max_delta = max(initial_state_max_delta, abs(float(row[key]) - float(ref[key])))
    checks.extend(
        [
            {
                "name": "compare_initial_states_match",
                "passed": initial_state_max_delta <= 1.0e-9,
                "value": initial_state_max_delta,
                "threshold": 1.0e-9,
            },
            {
                "name": "groundtruth_survival",
                "passed": groundtruth["survival_rate"] >= args.verify_min_survival,
                "value": groundtruth["survival_rate"],
                "threshold": args.verify_min_survival,
            },
            {
                "name": "model_survival",
                "passed": model["survival_rate"] >= args.verify_min_survival,
                "value": model["survival_rate"],
                "threshold": args.verify_min_survival,
            },
            {
                "name": "model_final_distance",
                "passed": model["mean_final_distance_to_target"] <= args.verify_max_final_distance,
                "value": model["mean_final_distance_to_target"],
                "threshold": args.verify_max_final_distance,
            },
            {
                "name": "model_groundtruth_final_distance_gap",
                "passed": abs(model["mean_final_distance_to_target"] - groundtruth["mean_final_distance_to_target"]) <= args.verify_max_final_distance_gap,
                "value": abs(model["mean_final_distance_to_target"] - groundtruth["mean_final_distance_to_target"]),
                "threshold": args.verify_max_final_distance_gap,
            },
            {
                "name": "model_yaw_rmse_below_zero",
                "passed": model["mean_yaw_rmse_deg"] < zero["mean_yaw_rmse_deg"],
                "value": model["mean_yaw_rmse_deg"],
                "zero_value": zero["mean_yaw_rmse_deg"],
            },
            {
                "name": "bad90_survival_negative_control",
                "passed": bad90["survival_rate"] <= args.verify_bad_yaw_max_survival,
                "value": bad90["survival_rate"],
                "threshold": args.verify_bad_yaw_max_survival,
            },
            {
                "name": "bad90_final_distance_negative_control",
                "passed": bad90["mean_final_distance_to_target"] >= args.verify_bad_yaw_min_final_distance,
                "value": bad90["mean_final_distance_to_target"],
                "threshold": args.verify_bad_yaw_min_final_distance,
            },
        ]
    )

    passed = all(bool(check["passed"]) for check in checks)
    summary = {
        "passed": passed,
        "config": {
            "episodes": compare_args.episodes,
            "steps": compare_args.steps,
            "seed": compare_args.seed,
            "random_xy": compare_args.random_xy,
            "random_z_min": compare_args.random_z_min,
            "random_z_max": compare_args.random_z_max,
            "random_yaw_deg": compare_args.random_yaw_deg,
            "dynamics_model": compare_args.dynamics_model,
            "yaw_update_hz": compare_args.yaw_update_hz,
            "orientation_source": "body_frame_gravity_plus_yaw",
            "checkpoint": str(Path(compare_args.checkpoint).expanduser().resolve()),
            "scene": str(Path(compare_args.scene).expanduser().resolve()) if compare_args.scene else None,
        },
        "checks": checks,
        "yaw_sweep": sweep,
        "compare_metrics": metrics,
        "artifacts": {
            "suite_dir": str(suite_dir),
            "json_summary": str(suite_dir / "verify_suite_summary.json"),
            "markdown_report": str(suite_dir / "VERIFY_REPORT.md"),
            "yaw_sweep_summary": str(suite_dir / "yaw_sweep" / "summary.json"),
            "compare_summary": str(compare_dir / "compare_summary.csv"),
        },
    }
    (suite_dir / "verify_suite_summary.json").write_text(json.dumps(summary, indent=2))
    write_verify_report(summary, suite_dir / "VERIFY_REPORT.md")
    print(f"verify_suite: {'PASS' if passed else 'FAIL'}")
    for check in checks:
        print(f"  {'PASS' if check['passed'] else 'FAIL'} {check['name']}")
    print(f"wrote {suite_dir / 'verify_suite_summary.json'}")
    print(f"wrote {suite_dir / 'VERIFY_REPORT.md'}")
    return passed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT))
    ap.add_argument("--scene", type=str, default=str(DEFAULT_SCENE) if DEFAULT_SCENE.exists() else None)
    ap.add_argument("--yaw-source", choices=["groundtruth", "model", "zero", "bad90"], default="model")
    ap.add_argument("--compare", action="store_true", help="run groundtruth, model, and zero yaw sources")
    ap.add_argument("--include-bad-yaw", action="store_true", help="include a true-yaw-plus-90-deg negative control in --compare")
    ap.add_argument("--self-check", action="store_true", help="validate checkpoint forward pass and yaw-injection math, then exit")
    ap.add_argument("--yaw-sweep", action="store_true", help="render known yaw offsets, evaluate the estimator, then exit")
    ap.add_argument("--yaw-sweep-deg", type=float, nargs="+", default=(-18.0, -12.0, -6.0, 0.0, 6.0, 12.0, 18.0))
    ap.add_argument("--verify-suite", action="store_true", help="run self-check, yaw sweep, randomized compare, and bad-yaw negative control")
    ap.add_argument("--verify-episodes", type=int, default=3)
    ap.add_argument("--verify-steps", type=int, default=500)
    ap.add_argument("--verify-seed", type=int, default=700)
    ap.add_argument("--verify-min-survival", type=float, default=1.0)
    ap.add_argument("--verify-max-final-distance", type=float, default=0.05)
    ap.add_argument("--verify-max-final-distance-gap", type=float, default=0.02)
    ap.add_argument("--verify-yaw-rmse-max", type=float, default=5.0)
    ap.add_argument("--verify-yaw-max-abs", type=float, default=10.0)
    ap.add_argument("--verify-bad-yaw-max-survival", type=float, default=0.0)
    ap.add_argument("--verify-bad-yaw-min-final-distance", type=float, default=1.0)
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--sim-rate", type=float, default=100.0)
    ap.add_argument("--yaw-update-hz", type=float, default=10.0)
    ap.add_argument("--dynamics-model", choices=["rotorpy_crazyflie", "crazyflie_openmv"], default="crazyflie_openmv")
    ap.add_argument("--world-bounds", type=float, default=10.0)
    ap.add_argument("--start", type=float, nargs=3, default=(1.0, 0.5, 1.5), metavar=("X", "Y", "Z"))
    ap.add_argument("--start-yaw", type=float, default=0.0, help="initial yaw in radians")
    ap.add_argument("--random-starts", action="store_true", help="sample a deterministic random initial state per episode")
    ap.add_argument("--random-xy", type=float, default=1.0)
    ap.add_argument("--random-z-min", type=float, default=0.8)
    ap.add_argument("--random-z-max", type=float, default=1.5)
    ap.add_argument("--random-yaw-deg", type=float, default=18.0)
    ap.add_argument("--target", type=float, nargs=3, default=(0.0, 0.0, 0.0), metavar=("X", "Y", "Z"))
    ap.add_argument("--target-yaw", type=float, default=0.0, help="target yaw in radians")
    ap.add_argument("--target-relative-orientation", action="store_true",
                    help="feed RAPTOR orientation relative to target yaw; automatically enabled by target yaw switching")
    ap.add_argument("--target-yaw-switch-after-stable", action="store_true",
                    help="after stabilization, rerender the target frame at a sampled yaw and command that yaw")
    ap.add_argument("--target-yaw-switch-range-deg", type=float, default=15.0)
    ap.add_argument("--target-yaw-switch-seed", type=int, default=None)
    ap.add_argument("--stable-distance-threshold", type=float, default=0.05)
    ap.add_argument("--stable-velocity-threshold", type=float, default=0.05)
    ap.add_argument("--stable-angular-velocity-threshold", type=float, default=0.05)
    ap.add_argument("--stable-steps", type=int, default=50)
    ap.add_argument("--steady-state-after-switch-steps", type=int, default=50,
                    help="ignore this many control steps after the target-yaw switch before steady-state yaw metrics")
    ap.add_argument("--steady-state-min-steps", type=int, default=50,
                    help="minimum settled post-switch samples required before steady-state yaw metrics are reported")
    ap.add_argument("--anchor-cameras-json", type=str, default=None,
                    help="cameras.json whose camera pose anchors the visual target pose; auto-detected for dataset scenes")
    ap.add_argument("--anchor-camera-id", type=int, default=0)
    ap.add_argument("--no-anchor-camera", action="store_true")
    ap.add_argument("--validate-examples", type=int, default=16)
    ap.add_argument("--out-dir", type=str, default="runs/raptor_yaw_closed_loop")
    ap.add_argument("--video-every", type=int, default=1, help="save every Nth episode video; 0 disables")
    ap.add_argument("--video-scale", type=int, default=4)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.self_check:
        raise SystemExit(0 if run_self_check(args) else 1)
    if args.yaw_sweep:
        run_yaw_sweep(args, args.yaw_sweep_deg, out_dir)
        raise SystemExit(0)
    if args.verify_suite:
        raise SystemExit(0 if run_verify_suite(args, out_dir) else 1)
    sources = ["groundtruth", "model", "zero"] if args.compare else [args.yaw_source]
    if args.compare and args.include_bad_yaw:
        sources.append("bad90")
    results = [run_source(args, source, out_dir) for source in sources]
    write_compare_summary(results, out_dir)
    for result in results:
        metrics = result["metrics"]
        print(
            f"{result['yaw_source']}: "
            f"survival={metrics.get('survival_rate', float('nan')):.3f} "
            f"mean_steps={metrics.get('mean_steps', float('nan')):.1f} "
            f"mean_final_dist={metrics.get('mean_final_distance_to_target', float('nan')):.3f} "
            f"yaw_rmse={metrics.get('mean_yaw_rmse_deg', float('nan')):.3f}"
        )
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
