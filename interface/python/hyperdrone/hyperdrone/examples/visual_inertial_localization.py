"""The visual-inertial localization benchmark from Python. The task flies a waypoint route
through a scene at IMU rate and streams what an estimator gets — a camera frame every
`frame_stride` steps, an IMU sample after every step — next to the ground truth to score
it. The RAPTOR policy (pip install foundation-policy) is the autopilot, exactly as in the
C++ harness; the estimator slot holds IMU dead reckoning (the C++ demo's blind baseline):
swap in your own and compare the absolute trajectory error.

  python -m hyperdrone.examples.visual_inertial_localization
  python -m hyperdrone.examples.visual_inertial_localization --instances 4 --width 320 --height 240 --video vio.mp4
"""
import argparse
import time

import numpy as np
from foundation_policy import Raptor

from hyperdrone.env import EnvConfig, MultiEnvironment
from hyperdrone.examples.data import procthor_scene_path

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", default=None, help="GLB scene (default: the ProcTHOR test scene)")
parser.add_argument("--instances", type=int, default=2)
parser.add_argument("--width", type=int, default=160)
parser.add_argument("--height", type=int, default=120)
parser.add_argument("--steps", type=int, default=None, help="IMU steps to fly (default: the episode length)")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--video", default=None, help="record the onboard frames to this path (mp4)")
arguments = parser.parse_args()

GRAVITY = np.array([0.0, 0.0, -9.81])
TARGET_POSITION_ERROR_CLIP = 1.0  # tasks::visual_inertial_localization::Specification::TARGET_POSITION_ERROR_CLIP

env = MultiEnvironment(
    [arguments.model or procthor_scene_path()],
    config=EnvConfig(instances=arguments.instances, cam_width=arguments.width, cam_height=arguments.height,
                     shading="high", preset="x500_fpv_imu", task="visual_inertial_localization"),
    seed=arguments.seed,
)
steps = arguments.steps or env.episode_step_limit
print(f"{env.config_string} imu_rate={1 / env.dt:.0f}Hz frame_rate={1 / (env.dt * env.frame_stride):.0f}Hz")


def block(vector, layout, name):
    offset, size = layout.blocks[name]
    return vector[:, offset:offset + size]


def quaternion_multiply(a, b):
    w = a[:, 0] * b[:, 0] - a[:, 1] * b[:, 1] - a[:, 2] * b[:, 2] - a[:, 3] * b[:, 3]
    x = a[:, 0] * b[:, 1] + a[:, 1] * b[:, 0] + a[:, 2] * b[:, 3] - a[:, 3] * b[:, 2]
    y = a[:, 0] * b[:, 2] - a[:, 1] * b[:, 3] + a[:, 2] * b[:, 0] + a[:, 3] * b[:, 1]
    z = a[:, 0] * b[:, 3] + a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1] + a[:, 3] * b[:, 0]
    return np.stack([w, x, y, z], axis=1)


def rotation_matrix(quaternion):
    w, x, y, z = quaternion.T
    return np.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], axis=1).reshape(-1, 3, 3)


class DeadReckoning:
    """IMU-only estimator in the episode frame: identity pose and zero velocity at t=0 (the
    protocol hands the estimator no ground truth), gyro-integrated attitude, accelerometer
    integrated twice — the C++ baseline in tasks/visual_inertial_localization/baseline.h."""

    def __init__(self, count, dt):
        self.dt = dt
        self.position = np.zeros((count, 3))
        self.orientation = np.tile([1.0, 0.0, 0.0, 0.0], (count, 1))
        self.velocity = np.zeros((count, 3))

    def step(self, accelerometer, gyroscope):
        rate = np.concatenate([np.zeros((len(gyroscope), 1)), gyroscope], axis=1)
        self.orientation += 0.5 * quaternion_multiply(self.orientation, rate) * self.dt
        self.orientation /= np.linalg.norm(self.orientation, axis=1, keepdims=True)
        acceleration = np.einsum("nij,nj->ni", rotation_matrix(self.orientation), accelerometer) + GRAVITY
        velocity_next = self.velocity + acceleration * self.dt
        self.position += (self.velocity + velocity_next) * self.dt / 2
        self.velocity = velocity_next


class TrajectoryMetrics:
    """Absolute trajectory error and rotation error of poses expressed relative to the
    episode start, plus the ground-truth distance traveled (metrics.h)."""

    def __init__(self, count):
        self.count = 0
        self.position_squared_error = np.zeros(count)
        self.rotation_error_sum = np.zeros(count)
        self.rotation_error_max = np.zeros(count)
        self.distance_traveled = np.zeros(count)
        self.previous_position = None

    def accumulate(self, truth_position, truth_rotation, estimate_position, estimate_rotation):
        self.position_squared_error += np.sum((estimate_position - truth_position) ** 2, axis=1)
        cosine = (np.einsum("nij,nij->n", truth_rotation, estimate_rotation) - 1) / 2
        rotation_error = np.arccos(np.clip(cosine, -1.0, 1.0))
        self.rotation_error_sum += rotation_error
        self.rotation_error_max = np.maximum(self.rotation_error_max, rotation_error)
        if self.previous_position is not None:
            self.distance_traveled += np.linalg.norm(truth_position - self.previous_position, axis=1)
        self.previous_position = truth_position.copy()
        self.count += 1

    def ate(self):
        return np.sqrt(self.position_squared_error / self.count)


# the autopilot sees [clamped position error to the current waypoint | R | v | omega | last
# action] — the privileged observation re-targeted on the task's waypoint block
policy = Raptor()
policy.reset()
layout = env.observation_layout_privileged
layout_imu = env.observation_layout_imu


def autopilot(state, previous_action):
    error = np.clip(block(state, layout, "position") - block(state, layout, "waypoint"), -TARGET_POSITION_ERROR_CLIP, TARGET_POSITION_ERROR_CLIP)
    observation = np.concatenate([
        error,
        block(state, layout, "orientation_rotation_matrix"),
        block(state, layout, "linear_velocity"),
        block(state, layout, "angular_velocity"),
        previous_action,
    ], axis=1)
    return policy.evaluate_step(observation).astype(np.float32)


video_writer = None
if arguments.video:
    import imageio

    video_writer = imageio.get_writer(arguments.video, fps=round(1 / (env.dt * env.frame_stride)))

reset_mask = np.ones(env.total_instances, dtype=np.uint8)
no_reset = np.zeros(env.total_instances, dtype=np.uint8)
env.reset(reset_mask)
state = env.observe_privileged()
origin_position = block(state, layout, "position").copy()
origin_rotation = block(state, layout, "orientation_rotation_matrix").reshape(-1, 3, 3).copy()
estimator = DeadReckoning(env.total_instances, env.dt)
metrics = TrajectoryMetrics(env.total_instances)
previous_action = np.zeros((env.total_instances, env.action_dim), dtype=np.float32)
frames_rendered = 0

start = time.perf_counter()
for step in range(steps):
    env.render(reset_mask)
    reset_mask = no_reset
    if step % env.frame_stride == 0:
        frames = env.frames()  # (instances, height, width, 3) in [0, 1]: the estimator's camera input
        frames_rendered += 1
        if video_writer is not None:
            video_writer.append_data((np.clip(np.concatenate(list(frames), axis=0), 0, 1) * 255).astype(np.uint8))
    action = autopilot(state, previous_action)
    env.step(action)
    previous_action = action
    imu = env.observe_imu()  # the estimator's IMU sample for this step
    estimator.step(block(imu, layout_imu, "accelerometer"), block(imu, layout_imu, "gyroscope"))
    state = env.observe_privileged()  # ground truth after the step, relative to the episode start below
    relative_position = np.einsum("nji,nj->ni", origin_rotation, block(state, layout, "position") - origin_position)
    relative_rotation = origin_rotation.transpose(0, 2, 1) @ block(state, layout, "orientation_rotation_matrix").reshape(-1, 3, 3)
    metrics.accumulate(relative_position, relative_rotation, estimator.position, rotation_matrix(estimator.orientation))
elapsed = time.perf_counter() - start

if video_writer is not None:
    video_writer.close()
    print(f"saved {arguments.video}")
print(f"{steps} IMU steps x {env.total_instances} instances, {frames_rendered} frames at {env.cam_width}x{env.cam_height} "
      f"in {elapsed:.1f}s ({steps * env.total_instances / elapsed:.0f} steps/s)")
for instance in range(env.total_instances):
    print(f"instance {instance}: dead reckoning ATE {metrics.ate()[instance]:.3f} m | "
          f"rot mean {metrics.rotation_error_sum[instance] / metrics.count:.4f} rad max {metrics.rotation_error_max[instance]:.4f} rad | "
          f"drift {metrics.ate()[instance] / max(metrics.distance_traveled[instance], 1e-9):.4f} m/m | "
          f"distance {metrics.distance_traveled[instance]:.2f} m")
if env.terminated().any():
    print("warning: at least one instance terminated during the episode")
env.close()
