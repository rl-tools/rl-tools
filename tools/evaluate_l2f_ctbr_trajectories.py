#!/usr/bin/env python3
import argparse
import gzip
import json
import math
from pathlib import Path


def mean(values):
    return sum(values) / len(values) if values else float("nan")


def percentile(values, q):
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lo = math.floor(index)
    hi = math.ceil(index)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - index) + ordered[hi] * (index - lo)


def summarize(values):
    return {
        "mean": mean(values),
        "median": percentile(values, 0.5),
        "p90": percentile(values, 0.9),
        "p95": percentile(values, 0.95),
        "max": max(values) if values else float("nan"),
    }


def roll_pitch_to_world_z_body(roll, pitch):
    sin_roll = math.sin(roll)
    cos_roll = math.cos(roll)
    sin_pitch = math.sin(pitch)
    cos_pitch = math.cos(pitch)
    return [-sin_pitch, cos_pitch * sin_roll, cos_pitch * cos_roll]


def quaternion_to_world_z_body(q):
    w, x, y, z = q
    return [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]


def rotate_body_z_to_world(q):
    w, x, y, z = q
    return [2 * x * z + 2 * w * y, 2 * y * z - 2 * w * x, 1 - 2 * x * x - 2 * y * y]


def norm(v):
    return math.sqrt(sum(x * x for x in v))


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def clamp(x, lo, hi):
    return min(max(x, lo), hi)


def load_trajectories(path):
    if path.is_dir():
        candidates = sorted(path.glob("**/trajectories.json.gz"))
        if not candidates:
            candidates = sorted(path.glob("**/trajectories.json"))
        if not candidates:
            raise FileNotFoundError(f"no trajectories file found under {path}")
        path = candidates[-1]
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as f:
            return path, json.load(f)
    with path.open() as f:
        return path, json.load(f)


def evaluate(episodes):
    tilt_distance = []
    tilt_angle_deg = []
    yaw_rate_abs = []
    thrust_g_abs = []
    d_action_sq = []
    returns = []
    lengths = []
    terminated_count = 0

    for episode in episodes:
        params = episode["parameters"]
        gravity = params["dynamics"]["gravity"]
        gravity_norm = norm(gravity)
        trajectory = episode["trajectory"]
        ep_return = 0.0
        ep_length = 0
        terminated = False
        for step_i, step in enumerate(trajectory):
            if terminated:
                break
            ep_return += float(step["reward"])
            ep_length += 1
            if step_i + 1 >= len(trajectory):
                terminated = bool(step["terminated"])
                break

            state = step["state"]
            next_state = trajectory[step_i + 1]["state"]
            setpoint = state["attitude_setpoint"]
            target_world_z_body = roll_pitch_to_world_z_body(setpoint["roll"], setpoint["pitch"])
            current_world_z_body = quaternion_to_world_z_body(next_state["orientation"])
            diff = [a - b for a, b in zip(current_world_z_body, target_world_z_body)]
            tilt_distance.append(norm(diff))
            denom = norm(target_world_z_body) * norm(current_world_z_body)
            if denom > 0:
                tilt_angle_deg.append(math.degrees(math.acos(clamp(dot(target_world_z_body, current_world_z_body) / denom, -1.0, 1.0))))

            yaw_rate_abs.append(abs(next_state["angular_velocity"][2] - setpoint["yaw_rate"]))

            body_z_world = rotate_body_z_to_world(next_state["orientation"])
            specific_force_world = [a - g for a, g in zip(next_state["linear_acceleration"], gravity)]
            actual_thrust_g = dot(specific_force_world, body_z_world) / gravity_norm
            thrust_g_abs.append(abs(actual_thrust_g - setpoint["thrust_g"]))

            last_action = state.get("last_action")
            if last_action is not None:
                d_action_sq.append(sum((a - b) * (a - b) for a, b in zip(step["action"], last_action)))

            terminated = bool(step["terminated"])
        returns.append(ep_return)
        lengths.append(ep_length)
        terminated_count += 1 if terminated else 0

    return {
        "episodes": len(episodes),
        "transitions": len(tilt_distance),
        "terminated_share": terminated_count / len(episodes) if episodes else float("nan"),
        "return": summarize(returns),
        "episode_length": summarize(lengths),
        "tilt_distance": summarize(tilt_distance),
        "tilt_angle_deg": summarize(tilt_angle_deg),
        "yaw_rate_abs_rad_s": summarize(yaw_rate_abs),
        "thrust_g_abs": summarize(thrust_g_abs),
        "d_action_sq": summarize(d_action_sq),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    path, episodes = load_trajectories(args.path)
    metrics = evaluate(episodes)
    print(f"file: {path}")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
