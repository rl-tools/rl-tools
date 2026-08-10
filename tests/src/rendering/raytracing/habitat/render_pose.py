#!/usr/bin/env -S conda run --no-capture-output -n habitat-new python
"""Render RGB images from the raytracing-golden or arbitrary camera poses.

The camera pose is camera-to-world. Habitat convention is +X right, +Y up,
and -Z forward. Use ``--pose-convention opencv`` when a matrix/quaternion uses
the OpenCV camera convention (+X right, +Y down, +Z forward).

Examples:
    # Render all 12 rl-tools raytracing-golden poses with their 256x256,
    # 80-degree-FOV camera configuration.
    ./render_pose.py --output habitat_golden_poses

    # Render one golden pose.
    ./render_pose.py --golden-pose 01 --output habitat_pose_01.png

    # Render a custom pose and camera.
    conda run --no-capture-output -n habitat-new python render_pose.py \
        --position 1.0 1.5 2.0 --look-at 0.0 1.0 0.0 \
        --width 640 --height 480 \
        --intrinsics 525.0 525.0 319.5 239.5 \
        --output render.png

    conda run --no-capture-output -n habitat-new python render_pose.py \
        --camera-to-world 1 0 0 1  0 1 0 1.5  0 0 1 2  0 0 0 1 \
        --pose-convention opencv \
        --intrinsics 525 0 319.5  0 525 239.5  0 0 1
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Sequence, Tuple

# ProcTHOR's many compressed textures otherwise emit one importer message per
# texture. Explicitly set either variable before launching to override this.
os.environ.setdefault("HABITAT_SIM_LOG", "quiet")
os.environ.setdefault("MAGNUM_LOG", "quiet")

import habitat_sim
import magnum as mn
import numpy as np
import quaternion as qt
from PIL import Image


DEFAULT_DATASET = Path(
    "/home/jonas/mono/rl-tools/src/rendering/procthor2glb/data/"
    "ai2thor-hab/ai2thor-hab/ai2thor-hab.scene_dataset_config.json"
)
DEFAULT_SCENE = "ProcTHOR-Train-1"
SENSOR_UUID = "rgb"
GOLDEN_WIDTH = 2048
GOLDEN_HEIGHT = 2048
GOLDEN_HORIZONTAL_FOV_DEGREES = 80.0

# Exact FLU-frame cases used to create
# ~/rl-tools2/tests/data/rendering_raytracing_golden at rl-tools2 commit
# 4b4410607. FLU is +X forward, +Y left, +Z up. Each value is
# (position, look_at, up).
GOLDEN_POSES_FLU = {
    "00": ((3.4, 2.6, 9.2), (-7.94, -8.73, 1.30), (0.0, 0.0, 1.0)),
    "01": (
        (-6.28, -4.18, 1.99),
        (-5.4566, -4.6867, 1.7344),
        (0.2178, -0.1338, 0.9668),
    ),
    "02": (
        (-6.28, -4.18, 1.99),
        (-5.7733, -3.3566, 1.7344),
        (0.1338, 0.2178, 0.9668),
    ),
    "03": ((-10.0, -1.5, 1.2), (-9.42, -0.92, 0.62), (0.0, 0.0, 1.0)),
    "04": (
        (-8.623972, -13.78547, 1.378925),
        (-7.87335, -14.38384, 1.098727),
        (0.2190995, -0.1746609, 0.9599422),
    ),
    "05": (
        (-8.851608, -14.26024, 1.378925),
        (-9.560421, -14.81557, 0.9439588),
        (-0.3423963, -0.2682541, 0.9004468),
    ),
    "06": (
        (-9.23895, -14.42166, 1.378925),
        (-9.527512, -13.48674, 1.172421),
        (-0.06090194, 0.1973191, 0.9784458),
    ),
    "07": (
        (-6.75248, -11.95607, 1.378925),
        (-5.834495, -11.78127, 1.022909),
        (0.3497314, 0.06659576, 0.9344801),
    ),
    "08": (
        (-5.578073, -11.17941, 1.378925),
        (-6.497158, -10.9725, 1.043552),
        (-0.3271845, 0.07365467, 0.9420856),
    ),
    "09": (
        (-6.62411, -8.327263, 1.378925),
        (-6.43543, -7.359907, 1.209742),
        (0.0323884, 0.1660538, 0.9855847),
    ),
    "10": (
        (-6.529247, -6.471247, 1.378925),
        (-7.501519, -6.577228, 1.170464),
        (-0.2072329, -0.02258914, 0.9780308),
    ),
    "11": (
        (-6.843887, -4.858216, 1.378925),
        (-6.422187, -3.968547, 1.203832),
        (0.07499529, 0.1582194, 0.9845518),
    ),
}

# Convert the golden scene's FLU world coordinates to Habitat's Y-up world:
# (x, y, z)_FLU -> (x, z, -y)_Habitat.
FLU_TO_HABITAT = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
    dtype=np.float64,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render the rl-tools raytracing-golden camera poses or an arbitrary "
            "pose in a Habitat scene."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Habitat scene dataset configuration JSON.",
    )
    parser.add_argument(
        "--scene",
        default=DEFAULT_SCENE,
        help="Scene handle from the dataset, or a path to a scene asset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Output image for one pose, or output directory for all golden poses. "
            "Defaults to render.png or habitat_golden_poses, respectively."
        ),
    )
    parser.add_argument("--width", type=int, default=GOLDEN_WIDTH)
    parser.add_argument("--height", type=int, default=GOLDEN_HEIGHT)
    parser.add_argument(
        "--intrinsics",
        type=float,
        nargs="+",
        metavar="K",
        help=(
            "Either 'fx fy cx cy' or the nine row-major values of a 3x3 "
            "intrinsic matrix. Defaults to the goldens' centered 80-degree "
            "horizontal FOV."
        ),
    )
    parser.add_argument(
        "--intrinsics-file",
        type=Path,
        help="3x3 intrinsic matrix in .npy, JSON, or whitespace/CSV text format.",
    )

    pose = parser.add_argument_group("camera pose")
    pose.add_argument(
        "--golden-pose",
        choices=(*GOLDEN_POSES_FLU.keys(), "all"),
        help=(
            "Render one or all exact rl-tools raytracing-golden poses. Defaults "
            "to all when no custom pose is supplied."
        ),
    )
    pose.add_argument(
        "--position",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="Camera position in world coordinates (defaults to 0 1.5 0).",
    )
    orientation = pose.add_mutually_exclusive_group()
    orientation.add_argument(
        "--quaternion",
        type=float,
        nargs=4,
        metavar=("QX", "QY", "QZ", "QW"),
        help="Camera-to-world quaternion in x y z w order.",
    )
    orientation.add_argument(
        "--rotation-matrix",
        type=float,
        nargs=9,
        metavar=("R00", "R01", "R02", "R10", "R11", "R12", "R20", "R21", "R22"),
        help="Row-major camera-to-world rotation matrix.",
    )
    orientation.add_argument(
        "--look-at",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="World-space target point toward which the camera looks.",
    )
    matrix_pose = pose.add_mutually_exclusive_group()
    matrix_pose.add_argument(
        "--camera-to-world",
        type=float,
        nargs=16,
        metavar="T",
        help="Four row-major rows of a 4x4 camera-to-world matrix.",
    )
    matrix_pose.add_argument(
        "--camera-to-world-file",
        type=Path,
        help="4x4 camera-to-world matrix in .npy, JSON, or whitespace/CSV text format.",
    )
    pose.add_argument(
        "--pose-convention",
        choices=("habitat", "opencv"),
        default="habitat",
        help="Local camera-axis convention used by quaternion/matrix pose inputs.",
    )
    pose.add_argument(
        "--up",
        type=float,
        nargs=3,
        default=(0.0, 1.0, 0.0),
        metavar=("X", "Y", "Z"),
        help="Approximate world up direction used with --look-at.",
    )

    rendering = parser.add_argument_group("rendering")
    rendering.add_argument("--near", type=float, default=0.05)
    rendering.add_argument("--far", type=float, default=1000.0)
    rendering.add_argument("--gpu-device", type=int, default=0)
    rendering.add_argument(
        "--disable-frustum-culling",
        action="store_true",
        help="Disable render frustum culling (useful only for debugging).",
    )

    args = parser.parse_args()
    custom_pose_supplied = any(
        value is not None
        for value in (
            args.position,
            args.quaternion,
            args.rotation_matrix,
            args.look_at,
            args.camera_to_world,
            args.camera_to_world_file,
        )
    )
    if args.golden_pose is not None and custom_pose_supplied:
        parser.error("--golden-pose cannot be combined with a custom pose")
    if args.golden_pose is None and not custom_pose_supplied:
        args.golden_pose = "all"
    if args.camera_to_world is not None or args.camera_to_world_file is not None:
        if args.position is not None or any(
            value is not None
            for value in (args.quaternion, args.rotation_matrix, args.look_at)
        ):
            parser.error(
                "--camera-to-world/--camera-to-world-file cannot be combined with "
                "--position or another orientation option"
            )
    if args.intrinsics is not None and args.intrinsics_file is not None:
        parser.error("use only one of --intrinsics and --intrinsics-file")
    if args.width <= 0 or args.height <= 0:
        parser.error("--width and --height must be positive")
    if args.near <= 0.0 or args.far <= args.near:
        parser.error("clipping planes must satisfy 0 < near < far")
    return args


def load_matrix(path: Path, shape: Tuple[int, int]) -> np.ndarray:
    if not path.is_file():
        raise ValueError(f"matrix file does not exist: {path}")
    if path.suffix.lower() == ".npy":
        values = np.load(path, allow_pickle=False)
    else:
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".json":
            values = np.asarray(json.loads(text), dtype=np.float64)
        else:
            values = np.fromstring(text.replace(",", " "), sep=" ")
    values = np.asarray(values, dtype=np.float64)
    if values.size != math.prod(shape):
        raise ValueError(
            f"expected {math.prod(shape)} values in {path}, found {values.size}"
        )
    values = values.reshape(shape)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"matrix contains non-finite values: {path}")
    return values


def intrinsic_matrix(args: argparse.Namespace) -> np.ndarray:
    if args.intrinsics_file is not None:
        intrinsic = load_matrix(args.intrinsics_file, (3, 3))
    elif args.intrinsics is None:
        focal = args.width / (
            2.0 * math.tan(math.radians(GOLDEN_HORIZONTAL_FOV_DEGREES) / 2.0)
        )
        intrinsic = np.array(
            [
                [focal, 0.0, args.width / 2.0],
                [0.0, focal, args.height / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
    elif len(args.intrinsics) == 4:
        fx, fy, cx, cy = args.intrinsics
        intrinsic = np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
    elif len(args.intrinsics) == 9:
        intrinsic = np.asarray(args.intrinsics, dtype=np.float64).reshape(3, 3)
    else:
        raise ValueError("--intrinsics expects either 4 or 9 values")

    if not np.all(np.isfinite(intrinsic)):
        raise ValueError("intrinsic matrix contains non-finite values")
    if intrinsic[0, 0] <= 0.0 or intrinsic[1, 1] <= 0.0:
        raise ValueError("fx and fy must be positive")
    if not math.isclose(float(intrinsic[1, 0]), 0.0, abs_tol=1e-8):
        raise ValueError("intrinsic matrix element K[1,0] must be zero")
    if not np.allclose(intrinsic[2], (0.0, 0.0, 1.0), atol=1e-8):
        raise ValueError("the last row of the intrinsic matrix must be [0, 0, 1]")
    return intrinsic


def validate_rotation(rotation: np.ndarray) -> np.ndarray:
    if not np.all(np.isfinite(rotation)):
        raise ValueError("rotation contains non-finite values")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4):
        raise ValueError("rotation matrix is not orthonormal")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-4):
        raise ValueError("rotation matrix must have determinant +1")
    return rotation


def look_at_rotation(
    position: np.ndarray, target: Sequence[float], up: Sequence[float]
) -> np.ndarray:
    forward = np.asarray(target, dtype=np.float64) - position
    forward_norm = np.linalg.norm(forward)
    if forward_norm < 1e-10:
        raise ValueError("--look-at target must differ from --position")
    forward /= forward_norm

    approximate_up = np.asarray(up, dtype=np.float64)
    up_norm = np.linalg.norm(approximate_up)
    if up_norm < 1e-10:
        raise ValueError("--up must be nonzero")
    approximate_up /= up_norm

    right = np.cross(forward, approximate_up)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-8:
        raise ValueError("--up must not be parallel to the viewing direction")
    right /= right_norm
    camera_up = np.cross(right, forward)
    # Columns are the world-space Habitat camera axes: right, up, backward.
    return np.column_stack((right, camera_up, -forward))


def camera_pose(args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray]:
    if args.camera_to_world_file is not None:
        transform = load_matrix(args.camera_to_world_file, (4, 4))
    elif args.camera_to_world is not None:
        transform = np.asarray(args.camera_to_world, dtype=np.float64).reshape(4, 4)
    else:
        transform = None

    if transform is not None:
        if not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0), atol=1e-8):
            raise ValueError("camera-to-world matrix must end with [0, 0, 0, 1]")
        position = transform[:3, 3].copy()
        rotation = transform[:3, :3].copy()
    else:
        position = np.asarray(
            args.position if args.position is not None else (0.0, 1.5, 0.0),
            dtype=np.float64,
        )
        if args.look_at is not None:
            rotation = look_at_rotation(position, args.look_at, args.up)
        elif args.rotation_matrix is not None:
            rotation = np.asarray(args.rotation_matrix, dtype=np.float64).reshape(3, 3)
        elif args.quaternion is not None:
            quaternion = np.asarray(args.quaternion, dtype=np.float64)
            norm = np.linalg.norm(quaternion)
            if not np.isfinite(norm) or norm < 1e-10:
                raise ValueError("quaternion must be finite and nonzero")
            quaternion /= norm
            rotation = qt.as_rotation_matrix(
                qt.quaternion(quaternion[3], *quaternion[:3])
            )
        else:
            rotation = np.eye(3, dtype=np.float64)

    if not np.all(np.isfinite(position)):
        raise ValueError("position contains non-finite values")

    # Convert an OpenCV local camera frame to Habitat's local camera frame.
    if args.pose_convention == "opencv" and args.look_at is None:
        rotation = rotation @ np.diag((1.0, -1.0, -1.0))
    return position, validate_rotation(rotation)


def golden_camera_pose(pose_id: str) -> Tuple[np.ndarray, np.ndarray]:
    position_flu, target_flu, up_flu = GOLDEN_POSES_FLU[pose_id]
    position = FLU_TO_HABITAT @ np.asarray(position_flu, dtype=np.float64)
    target = FLU_TO_HABITAT @ np.asarray(target_flu, dtype=np.float64)
    up = FLU_TO_HABITAT @ np.asarray(up_flu, dtype=np.float64)
    return position, validate_rotation(look_at_rotation(position, target, up))


def selected_camera_poses(
    args: argparse.Namespace,
) -> Sequence[Tuple[str, np.ndarray, np.ndarray]]:
    if args.golden_pose is None:
        position, rotation = camera_pose(args)
        return (("custom", position, rotation),)
    pose_ids = (
        tuple(GOLDEN_POSES_FLU.keys())
        if args.golden_pose == "all"
        else (args.golden_pose,)
    )
    return tuple(
        (pose_id, *golden_camera_pose(pose_id)) for pose_id in pose_ids
    )


def output_paths(args: argparse.Namespace, pose_ids: Sequence[str]) -> Sequence[Path]:
    if len(pose_ids) == 1:
        return (args.output if args.output is not None else Path("render.png"),)
    output_directory = (
        args.output if args.output is not None else Path("habitat_golden_poses")
    )
    return tuple(output_directory / f"{pose_id}.png" for pose_id in pose_ids)


def projection_from_intrinsics(
    intrinsic: np.ndarray, width: int, height: int, near: float, far: float
) -> np.ndarray:
    """Map an OpenCV K matrix to Habitat/OpenGL clip coordinates."""
    fx, skew, cx = intrinsic[0]
    fy, cy = intrinsic[1, 1], intrinsic[1, 2]
    return np.array(
        [
            [2.0 * fx / width, -2.0 * skew / width, 1.0 - 2.0 * cx / width, 0.0],
            [0.0, 2.0 * fy / height, 2.0 * cy / height - 1.0, 0.0],
            [0.0, 0.0, -(far + near) / (far - near), -2.0 * far * near / (far - near)],
            [0.0, 0.0, -1.0, 0.0],
        ],
        dtype=np.float32,
    )


def render(args: argparse.Namespace) -> None:
    if not args.dataset.is_file():
        raise ValueError(f"dataset configuration does not exist: {args.dataset}")

    intrinsic = intrinsic_matrix(args)
    poses = selected_camera_poses(args)
    outputs = output_paths(args, tuple(pose_id for pose_id, _, _ in poses))

    sensor_spec = habitat_sim.CameraSensorSpec()
    sensor_spec.uuid = SENSOR_UUID
    sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_spec.resolution = [args.height, args.width]
    sensor_spec.position = [0.0, 0.0, 0.0]
    sensor_spec.orientation = [0.0, 0.0, 0.0]
    sensor_spec.near = args.near
    sensor_spec.far = args.far
    sensor_spec.hfov = math.degrees(
        2.0 * math.atan(args.width / (2.0 * intrinsic[0, 0]))
    )

    agent_config = habitat_sim.agent.AgentConfiguration()
    agent_config.sensor_specifications = [sensor_spec]

    simulator_config = habitat_sim.SimulatorConfiguration()
    simulator_config.scene_dataset_config_file = str(args.dataset)
    simulator_config.scene_id = args.scene
    simulator_config.enable_physics = False
    simulator_config.gpu_device_id = args.gpu_device
    simulator_config.frustum_culling = not args.disable_frustum_culling

    configuration = habitat_sim.Configuration(simulator_config, [agent_config])
    with habitat_sim.Simulator(configuration) as simulator:
        projection = projection_from_intrinsics(
            intrinsic, args.width, args.height, args.near, args.far
        )
        # Habitat-Sim 0.3.3 exposes the Python sensor wrappers through
        # ``_sensors``; newer builds expose the same mapping as ``sensors``.
        sensor_wrappers = (
            simulator.sensors if hasattr(simulator, "sensors") else simulator._sensors
        )
        sensor_wrapper = sensor_wrappers[SENSOR_UUID]
        sensor_object = getattr(sensor_wrapper, "sensor_object", None)
        if sensor_object is None:
            sensor_object = sensor_wrapper._sensor_object
        camera = sensor_object.render_camera
        camera.projection_matrix = mn.Matrix4(projection)

        for (pose_id, position, rotation), output in zip(poses, outputs):
            state = habitat_sim.AgentState()
            state.position = position
            state.rotation = qt.from_rotation_matrix(rotation)
            simulator.get_agent(0).set_state(state)

            rgba = np.asarray(
                simulator.get_sensor_observations()[SENSOR_UUID], dtype=np.uint8
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            image = Image.fromarray(rgba)
            if output.suffix.lower() in (".jpg", ".jpeg"):
                image = image.convert("RGB")
            image.save(output)
            print(
                f"Wrote {output.resolve()} ({args.width}x{args.height}, pose {pose_id})"
            )

    print(f"Scene: {args.scene}")
    if args.golden_pose is not None:
        print(
            "Golden configuration: "
            f"{GOLDEN_HORIZONTAL_FOV_DEGREES:g}-degree horizontal FOV, "
            f"{len(poses)} pose(s), FLU-to-Habitat world conversion enabled"
        )
    print(f"Intrinsics:\n{np.array2string(intrinsic, precision=8)}")


def main() -> None:
    args = parse_args()
    try:
        render(args)
    except (OSError, RuntimeError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error


if __name__ == "__main__":
    main()
