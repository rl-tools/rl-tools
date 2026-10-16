"""RLtools L2F-visual checkpoint/TFLite helpers for HyperSplat evaluation."""
from __future__ import annotations

import json
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from hypersplat.frames import compute_viewmat


L2F_TO_ROTORPY = np.array([3, 0, 1, 2])
GRAVITY_WORLD = np.array([0.0, 0.0, -9.81], dtype=np.float32)
WORLD_Z_SOURCES = ("groundtruth", "mahony")


@dataclass(frozen=True)
class CameraStackSpec:
    fov_rad: float
    height: int
    width: int
    stride: int
    frames: int
    with_target: bool
    channels_per_frame: int = 3

    @property
    def logical_image_inputs(self) -> int:
        return self.frames + (1 if self.with_target else 0)

    @property
    def logical_channels(self) -> int:
        return self.logical_image_inputs * self.channels_per_frame

    @property
    def fov_deg(self) -> float:
        return math.degrees(self.fov_rad)


@dataclass(frozen=True)
class StateObservationSpec:
    orientation: str
    angular_velocity: bool
    linear_acceleration_body_history: int
    action_history: int

    @property
    def dim(self) -> int:
        dim = 0
        if self.orientation == "OrientationWorldZ":
            dim += 3
        else:
            raise NotImplementedError(f"unsupported orientation observation {self.orientation!r}")
        if self.angular_velocity:
            dim += 3
        if self.linear_acceleration_body_history:
            dim += 3 * self.linear_acceleration_body_history
        if self.action_history:
            dim += 4 * self.action_history
        return dim


@dataclass(frozen=True)
class L2FVisualSpec:
    camera: CameraStackSpec
    state: StateObservationSpec
    output: str
    raw_observation: str


def split_top_level(text: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    start = 0
    for i, ch in enumerate(text):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(text[start:i].strip())
            start = i + 1
    parts.append(text[start:].strip())
    return parts


def parse_observation_slug(observation: str) -> L2FVisualSpec:
    parts = split_top_level(observation)
    if len(parts) != 2:
        raise ValueError(f"expected image and state observation, got {observation!r}")

    image_slug, state_slug = parts
    m = re.match(r"^CameraRGB(Stacked)?(WithTarget)?\(([^)]*)\)$", image_slug)
    if not m:
        raise ValueError(f"unsupported camera observation {image_slug!r}")
    stacked = m.group(1) is not None
    with_target = m.group(2) is not None
    args = [a.strip() for a in m.group(3).split(",")]
    if stacked:
        if len(args) != 5:
            raise ValueError(f"expected CameraRGBStacked args fov,H,W,stride,N, got {args}")
        fov_rad = float(args[0])
        height = int(args[1])
        width = int(args[2])
        stride = int(args[3])
        frames = int(args[4])
    else:
        if len(args) != 3:
            raise ValueError(f"expected CameraRGB args fov,H,W, got {args}")
        fov_rad = float(args[0])
        height = int(args[1])
        width = int(args[2])
        stride = 1
        frames = 1

    tokens = state_slug.split(".")
    if not tokens:
        raise ValueError(f"empty state observation in {observation!r}")
    orientation = tokens[0]
    angular_velocity = "AngularVelocity" in tokens
    accel_hist = 0
    action_hist = 0
    for token in tokens:
        if token.startswith("LinearAccelerationBodyFrameHistory("):
            accel_hist = _parse_int_arg(token, "LinearAccelerationBodyFrameHistory")
        elif token.startswith("ActionHistory("):
            action_hist = _parse_int_arg(token, "ActionHistory")

    state = StateObservationSpec(
        orientation=orientation,
        angular_velocity=angular_velocity,
        linear_acceleration_body_history=accel_hist,
        action_history=action_hist,
    )
    camera = CameraStackSpec(
        fov_rad=fov_rad,
        height=height,
        width=width,
        stride=stride,
        frames=frames,
        with_target=with_target,
    )
    return L2FVisualSpec(
        camera=camera,
        state=state,
        output="",
        raw_observation=observation,
    )


def _parse_int_arg(token: str, name: str) -> int:
    m = re.match(rf"^{re.escape(name)}\((\d+)\)$", token)
    if not m:
        raise ValueError(f"malformed {name} token {token!r}")
    return int(m.group(1))


def load_hdf5_actor_meta(checkpoint_path: str | Path) -> dict[str, Any]:
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("reading .h5 checkpoint metadata requires h5py") from exc

    with h5py.File(checkpoint_path, "r") as f:
        if "actor" not in f or "meta" not in f["actor"].attrs:
            raise ValueError(f"{checkpoint_path} does not contain actor.attrs['meta']")
        raw = f["actor"].attrs["meta"]
    if isinstance(raw, bytes):
        raw = raw.decode()
    return json.loads(raw)


def load_example_meta(meta_path: str | Path) -> dict[str, Any]:
    return json.loads(Path(meta_path).read_text())


def spec_from_actor_meta(actor_meta: dict[str, Any]) -> L2FVisualSpec:
    env_meta = actor_meta.get("environment", {})
    observation = env_meta.get("observation")
    if not observation:
        raise ValueError("actor metadata missing environment.observation")
    spec = parse_observation_slug(observation)
    return L2FVisualSpec(
        camera=spec.camera,
        state=spec.state,
        output=str(env_meta.get("output", "")),
        raw_observation=spec.raw_observation,
    )


def spec_from_checkpoint_or_meta(path: str | Path) -> L2FVisualSpec:
    path = Path(path)
    if path.suffix == ".h5":
        return spec_from_actor_meta(load_hdf5_actor_meta(path))
    meta = load_example_meta(path)
    actor_meta = meta.get("actor_meta")
    if not actor_meta:
        raise ValueError(f"{path} does not contain actor_meta")
    return spec_from_actor_meta(actor_meta)


def default_converter_paths() -> tuple[Path, Path]:
    rl_tools = Path.home() / "rl-tools2"
    return rl_tools / ".venv" / "bin" / "python", rl_tools / "tools" / "hdf5_to_tflite.py"


def ensure_tflite_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    quantize: bool = False,
    force: bool = False,
    converter_python: str | Path | None = None,
    converter_script: str | Path | None = None,
    extra_args: Sequence[str] = (),
) -> tuple[Path, Path]:
    """Convert an RLtools HDF5 checkpoint if needed and return (tflite, meta)."""
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    if checkpoint.suffix != ".h5":
        raise ValueError(f"expected .h5 checkpoint, got {checkpoint}")
    base = checkpoint.with_suffix("")
    tflite_path = Path(str(base) + (".int8.tflite" if quantize else ".tflite"))
    meta_path = Path(str(base) + ".example_meta.json")

    if not force and tflite_path.exists() and meta_path.exists():
        newest_out = min(tflite_path.stat().st_mtime, meta_path.stat().st_mtime)
        if newest_out >= checkpoint.stat().st_mtime:
            return tflite_path, meta_path

    default_python, default_script = default_converter_paths()
    python = Path(converter_python) if converter_python is not None else default_python
    script = Path(converter_script) if converter_script is not None else default_script
    if not python.exists():
        raise FileNotFoundError(f"converter python not found: {python}")
    if not script.exists():
        raise FileNotFoundError(f"converter script not found: {script}")

    cmd = [str(python), str(script), *extra_args]
    if quantize:
        cmd += ["--quantize", "int8"]
    cmd.append(str(checkpoint))
    subprocess.run(cmd, cwd=str(script.parents[1]), check=True)
    if not tflite_path.exists():
        raise FileNotFoundError(f"converter did not produce {tflite_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"converter did not produce {meta_path}")
    return tflite_path, meta_path


class TFLitePolicy:
    """Small wrapper around TensorFlow Lite with converter metadata ordering."""

    def __init__(self, tflite_path: str | Path, meta_path: str | Path):
        self.tflite_path = Path(tflite_path)
        self.meta_path = Path(meta_path)
        self.meta = load_example_meta(self.meta_path)
        self.inputs_meta = sorted(self.meta["inputs"], key=lambda x: int(x["keras_input_index"]))
        self._interpreter = self._make_interpreter()
        self._allocated_shape: tuple[tuple[int, ...], ...] | None = None
        self._input_details: list[dict[str, Any]] = []
        self._output_details: list[dict[str, Any]] = []

    def _make_interpreter(self):
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            try:
                import tensorflow as tf
            except ImportError as exc:
                raise RuntimeError(
                    "TFLite inference requires either tensorflow or tflite_runtime "
                    "in the Python environment running HyperSplat."
                ) from exc
            Interpreter = tf.lite.Interpreter
        return Interpreter(model_path=str(self.tflite_path))

    @property
    def actor_meta(self) -> dict[str, Any]:
        return self.meta.get("actor_meta", {})

    @property
    def spec(self) -> L2FVisualSpec:
        return spec_from_actor_meta(self.actor_meta)

    def input_feature_shapes_by_keras(self) -> list[tuple[int, ...]]:
        shapes = []
        for item in self.inputs_meta:
            shape = tuple(int(x) for x in item["shape"])
            shapes.append(shape[1:])
        return shapes

    def predict(self, keras_inputs: Sequence[np.ndarray]) -> np.ndarray:
        if len(keras_inputs) != len(self.inputs_meta):
            raise ValueError(f"got {len(keras_inputs)} inputs, expected {len(self.inputs_meta)}")
        batched_by_keras = [self._with_batch(x) for x in keras_inputs]
        by_tflite = [None] * len(batched_by_keras)
        for keras_i, item in enumerate(self.inputs_meta):
            tflite_i = int(item["tflite_input_index"])
            by_tflite[tflite_i] = batched_by_keras[keras_i]
        if any(x is None for x in by_tflite):
            raise RuntimeError("incomplete TFLite input mapping")
        return self._invoke_tflite([x for x in by_tflite if x is not None])

    def validate_companion_examples(self, limit: int = 1, atol: float = 1e-4) -> dict[str, float]:
        self._ensure_allocated_from_meta(batch=max(1, limit))
        tflite_inputs = []
        for item in sorted(self.meta["inputs"], key=lambda x: int(x["tflite_input_index"])):
            shape = tuple(int(x) for x in item["shape"])
            path = self.meta_path.with_name(item["path"])
            arr = np.fromfile(path, dtype=np.float32).reshape(shape)
            tflite_inputs.append(arr[:limit])

        output_meta = self.meta.get("int8_output") or self.meta["output"]
        y_path = self.meta_path.with_name(output_meta["path"])
        y_ref = np.fromfile(y_path, dtype=np.float32).reshape(tuple(output_meta["shape"]))[:limit]
        y_pred = self._invoke_tflite(tflite_inputs)
        diff = np.abs(y_pred.reshape(y_ref.shape) - y_ref)
        max_abs = float(diff.max())
        mean_abs = float(diff.mean())
        if max_abs > atol:
            raise RuntimeError(
                f"companion example validation failed: max_abs={max_abs:.6g} > {atol:.6g}"
            )
        return {"max_abs": max_abs, "mean_abs": mean_abs}

    @staticmethod
    def _with_batch(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.float32)
        return arr[None, ...] if arr.ndim in (1, 3) else arr

    def _ensure_allocated(self, inputs: Sequence[np.ndarray]):
        shapes = tuple(tuple(int(x) for x in arr.shape) for arr in inputs)
        if self._allocated_shape == shapes:
            return
        details = self._interpreter.get_input_details()
        if len(details) != len(inputs):
            raise RuntimeError(f"TFLite model has {len(details)} inputs, got {len(inputs)}")
        for detail, arr in zip(details, inputs):
            if tuple(int(x) for x in detail["shape"]) != tuple(arr.shape):
                self._interpreter.resize_tensor_input(detail["index"], list(arr.shape))
        self._interpreter.allocate_tensors()
        self._input_details = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()
        self._allocated_shape = shapes

    def _ensure_allocated_from_meta(self, batch: int):
        inputs = []
        for item in sorted(self.meta["inputs"], key=lambda x: int(x["tflite_input_index"])):
            feature = tuple(int(x) for x in item["shape"][1:])
            inputs.append(np.zeros((batch, *feature), dtype=np.float32))
        self._ensure_allocated(inputs)

    def _invoke_tflite(self, tflite_inputs: Sequence[np.ndarray]) -> np.ndarray:
        self._ensure_allocated(tflite_inputs)
        for detail, arr in zip(self._input_details, tflite_inputs):
            arr = np.asarray(arr, dtype=np.float32)
            if detail["dtype"] == np.int8:
                scale, zero = detail["quantization"]
                if scale == 0:
                    raise RuntimeError(f"input {detail['name']} has zero quantization scale")
                q = np.round(arr / scale + zero).clip(-128, 127).astype(np.int8)
                self._interpreter.set_tensor(detail["index"], q)
            else:
                self._interpreter.set_tensor(detail["index"], arr.astype(detail["dtype"]))
        self._interpreter.invoke()
        out_detail = self._output_details[0]
        out = self._interpreter.get_tensor(out_detail["index"])
        if out_detail["dtype"] == np.int8:
            scale, zero = out_detail["quantization"]
            out = (out.astype(np.float32) - zero) * scale
        return out.astype(np.float32)


def clamp(v, lo, hi):
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


class MahonyFilter:
    def __init__(self, kp=0.4, ki=0.001, max_bias_rad_s=0.1,
                 accel_gate_lo_g=0.75, accel_gate_hi_g=1.25, g_ref=9.80665):
        self.q0 = 1.0
        self.q1 = 0.0
        self.q2 = 0.0
        self.q3 = 0.0
        self.bx = 0.0
        self.by = 0.0
        self.bz = 0.0
        self.kp = kp
        self.ki = ki
        self.max_bias = max_bias_rad_s
        self.g_ref = g_ref
        self.a_lo = accel_gate_lo_g * g_ref
        self.a_hi = accel_gate_hi_g * g_ref
        self.initialized = False

    def _seed_from_accel(self, ax, ay, az):
        norm = math.sqrt(ax * ax + ay * ay + az * az)
        if norm < 1e-9:
            return
        ux = ax / norm
        uy = ay / norm
        uz = az / norm
        if uz < -0.999999:
            self.q0, self.q1, self.q2, self.q3 = 0.0, 1.0, 0.0, 0.0
            return
        w = 1.0 + uz
        x = uy
        y = -ux
        z = 0.0
        qn = math.sqrt(w * w + x * x + y * y + z * z)
        self.q0 = w / qn
        self.q1 = x / qn
        self.q2 = y / qn
        self.q3 = z / qn

    def update(self, ax, ay, az, gx, gy, gz, dt):
        if not self.initialized:
            self._seed_from_accel(ax, ay, az)
            self.initialized = True

        a_norm = math.sqrt(ax * ax + ay * ay + az * az)
        use_accel = a_norm > 1e-6 and self.a_lo <= a_norm <= self.a_hi

        if use_accel:
            inv = 1.0 / a_norm
            ahx = ax * inv
            ahy = ay * inv
            ahz = az * inv

            vx = 2.0 * (self.q1 * self.q3 - self.q0 * self.q2)
            vy = 2.0 * (self.q2 * self.q3 + self.q0 * self.q1)
            vz = self.q0 * self.q0 - self.q1 * self.q1 - self.q2 * self.q2 + self.q3 * self.q3

            ex = ahy * vz - ahz * vy
            ey = ahz * vx - ahx * vz
            ez = ahx * vy - ahy * vx

            self.bx -= self.ki * ex * dt
            self.by -= self.ki * ey * dt
            self.bz -= self.ki * ez * dt
            self.bx = clamp(self.bx, -self.max_bias, self.max_bias)
            self.by = clamp(self.by, -self.max_bias, self.max_bias)
            self.bz = clamp(self.bz, -self.max_bias, self.max_bias)

            wx = gx - self.bx + self.kp * ex
            wy = gy - self.by + self.kp * ey
            wz = gz - self.bz + self.kp * ez
        else:
            wx = gx - self.bx
            wy = gy - self.by
            wz = gz - self.bz

        q0, q1, q2, q3 = self.q0, self.q1, self.q2, self.q3
        dq0 = 0.5 * (-q1 * wx - q2 * wy - q3 * wz)
        dq1 = 0.5 * ( q0 * wx + q2 * wz - q3 * wy)
        dq2 = 0.5 * ( q0 * wy - q1 * wz + q3 * wx)
        dq3 = 0.5 * ( q0 * wz + q1 * wy - q2 * wx)

        q0 += dq0 * dt
        q1 += dq1 * dt
        q2 += dq2 * dt
        q3 += dq3 * dt

        qn = math.sqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3)
        if qn > 1e-12:
            inv_q = 1.0 / qn
            self.q0 = q0 * inv_q
            self.q1 = q1 * inv_q
            self.q2 = q2 * inv_q
            self.q3 = q3 * inv_q
        else:
            self.q0, self.q1, self.q2, self.q3 = 1.0, 0.0, 0.0, 0.0

    def orientation_world_z(self):
        x = 2.0 * (self.q1 * self.q3 - self.q0 * self.q2)
        y = 2.0 * (self.q2 * self.q3 + self.q0 * self.q1)
        z = self.q0 * self.q0 - self.q1 * self.q1 - self.q2 * self.q2 + self.q3 * self.q3
        return x, y, z


class L2FVisualObservationBuilder:
    """Build RLtools L2F-visual policy inputs from a HyperSplat environment."""

    def __init__(
        self,
        env,
        spec: L2FVisualSpec,
        *,
        target_state: dict[str, np.ndarray] | None = None,
        gyro_bias: Sequence[float] | np.ndarray | None = None,
        world_z_source: str = "groundtruth",
        vision_delay_steps: int = 0,
    ):
        if world_z_source not in WORLD_Z_SOURCES:
            raise ValueError(f"world_z_source {world_z_source!r} not in {WORLD_Z_SOURCES}")
        vision_delay_steps = int(vision_delay_steps)
        if vision_delay_steps < 0:
            raise ValueError(f"vision_delay_steps must be >= 0, got {vision_delay_steps}")
        self.env = env
        self.spec = spec
        self.world_z_source = world_z_source
        self.vision_delay_steps = vision_delay_steps
        self.target_state = target_state or {
            "x": np.zeros(3, dtype=np.float64),
            "q": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        }
        self.gyro_bias = (
            np.zeros(3, dtype=np.float32)
            if gyro_bias is None
            else np.asarray(gyro_bias, dtype=np.float32)
        )
        if self.gyro_bias.shape != (3,):
            raise ValueError(f"gyro_bias shape {self.gyro_bias.shape} != (3,)")
        self.frames: list[np.ndarray] = []
        self.target_frame = np.zeros(
            (spec.camera.height, spec.camera.width, spec.camera.channels_per_frame),
            dtype=np.float32,
        )
        self.step_index = 0
        self.state: dict[str, np.ndarray] | None = None
        self.prev_velocity: np.ndarray | None = None
        self.mahony = MahonyFilter()
        self.mahony_world_z = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        self.linear_accel_history = np.zeros(
            (max(spec.state.linear_acceleration_body_history, 1), 3), dtype=np.float32
        )
        self.action_history = np.zeros((max(spec.state.action_history, 1), 4), dtype=np.float32)

    def reset(self, obs: dict[str, Any], info: dict[str, Any]):
        self.frames = [self._image_to_float(obs["image"])]
        self.step_index = 0
        self.state = info["state"]
        self.prev_velocity = np.asarray(self.state["v"], dtype=np.float32).copy()
        self.mahony = MahonyFilter()
        self.mahony_world_z = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        accel_body = self._linear_accel_body(self.state, self.prev_velocity)
        self.linear_accel_history[:] = accel_body
        self._update_mahony(accel_body)
        self.action_history[:] = 0
        self.target_frame = self._render_target_frame()

    def advance(self, obs: dict[str, Any], info: dict[str, Any], previous_action_l2f: np.ndarray):
        if self.state is None or self.prev_velocity is None:
            raise RuntimeError("observation builder must be reset before advance")
        prev_v = np.asarray(self.state["v"], dtype=np.float32)
        self.state = info["state"]
        self.prev_velocity = prev_v
        self.step_index += 1
        self.frames.append(self._image_to_float(obs["image"]))

        accel_body = self._linear_accel_body(self.state, prev_v)
        if self.linear_accel_history.shape[0] > 1:
            self.linear_accel_history[1:] = self.linear_accel_history[:-1]
        self.linear_accel_history[0] = accel_body
        self._update_mahony(accel_body)

        action = np.clip(np.asarray(previous_action_l2f, dtype=np.float32).reshape(4), -1.0, 1.0)
        if self.action_history.shape[0] > 1:
            self.action_history[1:] = self.action_history[:-1]
        self.action_history[0] = action

    def keras_inputs(self, feature_shapes_by_keras: Sequence[Sequence[int]]) -> list[np.ndarray]:
        logical_images = self._logical_images()
        state = self._state_observation()

        image_cursor = 0
        result: list[np.ndarray] = []
        for raw_shape in feature_shapes_by_keras:
            shape = tuple(int(x) for x in raw_shape)
            if len(shape) == 3 and shape[-1] == 3:
                if image_cursor >= len(logical_images):
                    raise ValueError("model expects more split image inputs than observation provides")
                arr = logical_images[image_cursor]
                image_cursor += 1
            elif len(shape) == 3 and shape[:2] == (self.spec.camera.height, self.spec.camera.width):
                arr = self._combined_image(logical_images, channels=shape[-1])
            elif len(shape) == 1 and shape[0] == state.shape[0]:
                arr = state
            else:
                raise ValueError(f"cannot satisfy model input shape {shape}")
            if tuple(arr.shape) != shape:
                raise ValueError(f"built input shape {arr.shape} does not match expected {shape}")
            result.append(arr.astype(np.float32, copy=False))
        return result

    def _logical_images(self) -> list[np.ndarray]:
        if not self.frames:
            raise RuntimeError("observation builder has no frames; call reset first")
        images = []
        episode_start = 0
        for frame_i in range(self.spec.camera.frames):
            back = frame_i * self.spec.camera.stride
            desired = self.step_index - self.vision_delay_steps - back
            if desired < episode_start:
                desired = episode_start
            images.append(self.frames[desired])
        if self.spec.camera.with_target:
            images.append(self.target_frame)
        return images

    def _combined_image(self, logical_images: Sequence[np.ndarray], channels: int | None = None) -> np.ndarray:
        channels = channels if channels is not None else self._default_padded_image_channels()
        if channels < self.spec.camera.logical_channels:
            raise ValueError(
                f"combined image needs at least {self.spec.camera.logical_channels} channels, got {channels}"
            )
        out = np.zeros((self.spec.camera.height, self.spec.camera.width, channels), dtype=np.float32)
        for i, image in enumerate(logical_images):
            start = i * self.spec.camera.channels_per_frame
            out[:, :, start:start + self.spec.camera.channels_per_frame] = image
        return out

    def _default_padded_image_channels(self) -> int:
        # The common RLtools fast path pads 33 logical channels to 40.
        return ((self.spec.camera.logical_channels + 7) // 8) * 8

    def _state_observation(self) -> np.ndarray:
        if self.state is None:
            raise RuntimeError("observation builder must be reset before building state")
        pieces: list[np.ndarray] = []
        if self.spec.state.orientation == "OrientationWorldZ":
            pieces.append(self._orientation_world_z_body(self.state))
        else:
            raise NotImplementedError(
                f"unsupported state orientation {self.spec.state.orientation!r}"
            )
        if self.spec.state.angular_velocity:
            pieces.append(self._angular_velocity_body(self.state))
        if self.spec.state.linear_acceleration_body_history:
            n = self.spec.state.linear_acceleration_body_history
            pieces.append(self.linear_accel_history[:n].reshape(-1))
        if self.spec.state.action_history:
            n = self.spec.state.action_history
            pieces.append(self.action_history[:n].reshape(-1))
        obs = np.concatenate(pieces).astype(np.float32)
        if obs.shape != (self.spec.state.dim,):
            raise RuntimeError(f"state obs shape {obs.shape} != ({self.spec.state.dim},)")
        return obs

    @staticmethod
    def _image_to_float(image: np.ndarray) -> np.ndarray:
        return np.asarray(image, dtype=np.float32) / 255.0

    def _render_target_frame(self) -> np.ndarray:
        viewmat = compute_viewmat(
            self.target_state,
            self.env.scene_transform,
            self.env.body_to_camera,
        )
        return self._image_to_float(self.env.renderer.render(viewmat))

    def _linear_accel_body(self, state: dict[str, Any], prev_velocity: np.ndarray) -> np.ndarray:
        velocity = np.asarray(state["v"], dtype=np.float32)
        dt = float(self.env.dynamics.dt)
        accel_world = (velocity - prev_velocity) / dt - GRAVITY_WORLD
        rot = Rotation.from_quat(np.asarray(state["q"], dtype=np.float64)).as_matrix()
        return (rot.T @ accel_world.astype(np.float64)).astype(np.float32)

    def _angular_velocity_body(self, state: dict[str, Any]) -> np.ndarray:
        return np.asarray(state["w"], dtype=np.float32).reshape(3) + self.gyro_bias

    def _orientation_world_z_body(self, state: dict[str, Any]) -> np.ndarray:
        if self.world_z_source == "mahony":
            return self.mahony_world_z.copy()
        return self._world_z_body(state)

    def _update_mahony(self, accel_body: np.ndarray):
        if self.world_z_source != "mahony":
            return
        if self.state is None:
            raise RuntimeError("observation builder must be reset before updating Mahony filter")
        if self.env is None:
            raise RuntimeError("Mahony filter update requires an environment with dynamics.dt")
        accel = np.asarray(accel_body, dtype=np.float32).reshape(3)
        gyro = self._angular_velocity_body(self.state)
        self.mahony.update(
            float(accel[0]),
            float(accel[1]),
            float(accel[2]),
            float(gyro[0]),
            float(gyro[1]),
            float(gyro[2]),
            float(self.env.dynamics.dt),
        )
        self.mahony_world_z = np.asarray(self.mahony.orientation_world_z(), dtype=np.float32)

    @staticmethod
    def _world_z_body(state: dict[str, Any]) -> np.ndarray:
        rot = Rotation.from_quat(np.asarray(state["q"], dtype=np.float64)).as_matrix()
        return (rot.T @ np.array([0.0, 0.0, 1.0])).astype(np.float32)


def l2f_action_to_rotorpy(action: np.ndarray) -> np.ndarray:
    """Map RLtools/RAPTOR order [FR, BR, BL, FL] to RotorPY [FL, FR, BR, BL]."""
    action = np.asarray(action, dtype=np.float32).reshape(4)
    return np.clip(action, -1.0, 1.0)[L2F_TO_ROTORPY].astype(np.float32)


def load_scene_for_eval(scene_path: str | None, device: str):
    from hypersplat.renderer import Scene

    return Scene.from_ply(scene_path, device=device) if scene_path else Scene.synthetic(device=device)
