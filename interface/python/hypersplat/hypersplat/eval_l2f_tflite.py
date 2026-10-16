"""Evaluate RLtools L2F-visual TFLite policies in HyperSplat."""
from __future__ import annotations

import argparse
import json
import time
from copy import deepcopy
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from rotorpy.vehicles.crazyflie_params import quad_params as rotorpy_crazyflie_params
try:
    from tqdm.auto import tqdm
except ImportError:
    class tqdm:
        def __init__(self, iterable=None, *args, **kwargs):
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def update(self, n=1):
            pass

        def set_postfix(self, *args, **kwargs):
            pass

        @staticmethod
        def write(message):
            print(message)

from hypersplat.dynamics import DYNAMICS_MODELS, crazyflie_openmv_summary
from hypersplat import HyperSplatEnv
from hypersplat.l2f_visual_tflite import (
    L2FVisualObservationBuilder,
    TFLitePolicy,
    WORLD_Z_SOURCES,
    ensure_tflite_from_checkpoint,
    l2f_action_to_rotorpy,
    load_scene_for_eval,
)

RLTOOLS_CRAZYFLIE_YAW_TORQUE_PER_THRUST = 4.665e-3


def _initial_state_from_args(args):
    if args.start is None:
        return args.initial_state
    return {"x": np.asarray(args.start, dtype=np.float64)}


def _target_state_from_args(args):
    yaw = float(args.target_yaw)
    return {
        "x": np.asarray(args.target, dtype=np.float64),
        "q": np.array([0.0, 0.0, np.sin(yaw / 2.0), np.cos(yaw / 2.0)], dtype=np.float64),
    }


def _gyro_bias_from_args(args) -> np.ndarray:
    gyro_bias = np.asarray(args.gyro_bias, dtype=np.float32)
    if gyro_bias.shape != (3,):
        raise ValueError(f"--gyro-bias must contain exactly 3 values, got {gyro_bias}")
    return gyro_bias


def _vision_delay_steps_from_args(args) -> int:
    vision_delay_steps = int(args.vision_delay_steps)
    if vision_delay_steps < 0:
        raise ValueError(f"--vision-delay-steps must be >= 0, got {vision_delay_steps}")
    return vision_delay_steps


def _quad_params_from_args(args) -> tuple[dict | None, dict]:
    rotorpy_override_selected = [
        args.rotorpy_k_m is not None,
        args.rotorpy_yaw_torque_per_thrust is not None,
        bool(args.rotorpy_match_rltools_yaw),
    ]
    if args.dynamics_model == "crazyflie_openmv":
        if any(rotorpy_override_selected):
            raise ValueError("RotorPY yaw overrides require --dynamics-model rotorpy_crazyflie")
        return None, crazyflie_openmv_summary()

    params = deepcopy(rotorpy_crazyflie_params)
    default_k_eta = float(params["k_eta"])
    default_k_m = float(params["k_m"])

    if sum(rotorpy_override_selected) > 1:
        raise ValueError(
            "use only one of --rotorpy-k-m, --rotorpy-yaw-torque-per-thrust, "
            "or --rotorpy-match-rltools-yaw"
        )

    mode = "rotorpy_default"
    yaw_torque_per_thrust = default_k_m / default_k_eta
    if args.rotorpy_match_rltools_yaw:
        mode = "rltools_crazyflie"
        yaw_torque_per_thrust = RLTOOLS_CRAZYFLIE_YAW_TORQUE_PER_THRUST
        params["k_m"] = yaw_torque_per_thrust * default_k_eta
    elif args.rotorpy_yaw_torque_per_thrust is not None:
        mode = "yaw_torque_per_thrust"
        yaw_torque_per_thrust = float(args.rotorpy_yaw_torque_per_thrust)
        params["k_m"] = yaw_torque_per_thrust * default_k_eta
    elif args.rotorpy_k_m is not None:
        mode = "k_m"
        params["k_m"] = float(args.rotorpy_k_m)
        yaw_torque_per_thrust = float(params["k_m"]) / default_k_eta

    config = {
        "model": "rotorpy_crazyflie",
        "mode": mode,
        "k_eta": default_k_eta,
        "k_m": float(params["k_m"]),
        "yaw_torque_per_thrust": float(yaw_torque_per_thrust),
        "default_k_m": default_k_m,
        "default_yaw_torque_per_thrust": default_k_m / default_k_eta,
        "rltools_crazyflie_yaw_torque_per_thrust": RLTOOLS_CRAZYFLIE_YAW_TORQUE_PER_THRUST,
    }
    return params, config


def _resolve_policy_artifacts(args) -> tuple[Path, Path]:
    if args.tflite:
        tflite = Path(args.tflite)
        if args.meta:
            meta = Path(args.meta)
        elif str(tflite).endswith(".int8.tflite"):
            meta = Path(str(tflite)[: -len(".int8.tflite")] + ".example_meta.json")
        else:
            meta = Path(str(tflite)[: -len(".tflite")] + ".example_meta.json")
        if not tflite.exists():
            raise FileNotFoundError(tflite)
        if not meta.exists():
            raise FileNotFoundError(f"meta JSON not found; pass --meta explicitly: {meta}")
        return tflite, meta

    if not args.checkpoint:
        raise ValueError("pass either --checkpoint or --tflite")
    converter_args = list(args.converter_arg)
    if args.example_bin_limit is not None:
        converter_args += ["--example-bin-limit", str(args.example_bin_limit)]
    return ensure_tflite_from_checkpoint(
        args.checkpoint,
        quantize=args.quantize == "int8",
        force=args.force_convert,
        converter_python=args.converter_python,
        converter_script=args.converter_script,
        extra_args=converter_args,
    )


def run(args) -> dict:
    tflite_path, meta_path = _resolve_policy_artifacts(args)
    policy = TFLitePolicy(tflite_path, meta_path)
    spec = policy.spec
    quad_params, dynamics_config = _quad_params_from_args(args)

    if args.validate_examples:
        validation = policy.validate_companion_examples(limit=args.validate_examples, atol=args.example_tolerance)
        print(
            "companion validation: "
            f"max_abs={validation['max_abs']:.6g} mean_abs={validation['mean_abs']:.6g}"
        )

    scene = load_scene_for_eval(args.scene, args.device)
    print(
        f"policy={tflite_path} scene={'synthetic' if args.scene is None else args.scene} "
        f"image={spec.camera.height}x{spec.camera.width} fov={spec.camera.fov_deg:.3f}deg"
    )
    print(
        "dynamics: "
        + " ".join(f"{key}={value:.8g}" if isinstance(value, float) else f"{key}={value}"
                   for key, value in dynamics_config.items())
    )

    env = HyperSplatEnv(
        scene=scene,
        obs_mode="dict",
        control_mode="cmd_motor_speeds",
        image_size=(spec.camera.height, spec.camera.width),
        fov_deg=spec.camera.fov_deg,
        sim_rate=args.sim_rate,
        max_episode_steps=args.steps,
        initial_state=_initial_state_from_args(args),
        world_bounds=args.world_bounds,
        device=args.device,
        quad_params=quad_params,
        dynamics_model=args.dynamics_model,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    episode_records = []
    summary_rows = []

    input_shapes = policy.input_feature_shapes_by_keras()
    target_state = _target_state_from_args(args)
    gyro_bias = _gyro_bias_from_args(args)
    vision_delay_steps = _vision_delay_steps_from_args(args)
    if np.any(gyro_bias):
        print(f"gyro_bias_rad_s={gyro_bias.tolist()}")
    if args.world_z_source == "mahony":
        print("world_z_source=mahony")
    if vision_delay_steps:
        print(f"vision_delay_steps={vision_delay_steps}")
    started = time.time()
    progress_disable = True if args.no_progress else None

    with tqdm(
        total=args.episodes,
        desc="episodes",
        unit="episode",
        dynamic_ncols=True,
        disable=progress_disable,
    ) as episode_bar:
        for episode_i in range(args.episodes):
            seed = args.seed + episode_i
            obs, info = env.reset(seed=seed)
            builder = L2FVisualObservationBuilder(
                env,
                spec,
                target_state=target_state,
                gyro_bias=gyro_bias,
                world_z_source=args.world_z_source,
                vision_delay_steps=vision_delay_steps,
            )
            builder.reset(obs, info)

            frames = [obs["image"]]
            episode_steps = []
            total_reward = 0.0
            terminated = False
            truncated = False

            with tqdm(
                total=args.steps,
                desc=f"episode {episode_i}",
                unit="step",
                leave=False,
                dynamic_ncols=True,
                disable=progress_disable,
            ) as step_bar:
                for step_i in range(args.steps):
                    keras_inputs = builder.keras_inputs(input_shapes)
                    action_l2f = policy.predict(keras_inputs).reshape(-1)[:4].astype(np.float32)
                    action_rotorpy = l2f_action_to_rotorpy(action_l2f)

                    next_obs, reward, terminated, truncated, next_info = env.step(action_rotorpy)
                    total_reward += float(reward)

                    state = next_info["state"]
                    position = np.asarray(state["x"], dtype=np.float32)
                    target = np.asarray(target_state["x"], dtype=np.float32)
                    episode_steps.append(
                        {
                            "step": step_i,
                            "reward": float(reward),
                            "terminated": bool(terminated),
                            "truncated": bool(truncated),
                            "position": position.tolist(),
                            "velocity": np.asarray(state["v"], dtype=np.float32).tolist(),
                            "quat_xyzw": np.asarray(state["q"], dtype=np.float32).tolist(),
                            "body_rates": np.asarray(state["w"], dtype=np.float32).tolist(),
                            "action_l2f": action_l2f.tolist(),
                            "action_rotorpy": action_rotorpy.tolist(),
                            "distance_to_target": float(np.linalg.norm(position - target)),
                            "action_saturated": bool(np.any(np.abs(action_l2f) > 1.0)),
                        }
                    )

                    frames.append(next_obs["image"])
                    builder.advance(next_obs, next_info, action_l2f)
                    obs = next_obs
                    info = next_info
                    step_bar.update(1)
                    if terminated or truncated:
                        break

            steps = len(episode_steps)
            final_dist = episode_steps[-1]["distance_to_target"] if episode_steps else None
            saturated = sum(1 for row in episode_steps if row["action_saturated"])
            summary = {
                "episode": episode_i,
                "seed": seed,
                "steps": steps,
                "return": total_reward,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "survived": bool(not terminated and truncated),
                "final_distance_to_target": final_dist,
                "action_saturation_fraction": saturated / max(steps, 1),
            }
            summary_rows.append(summary)
            episode_records.append({"summary": summary, "trajectory": episode_steps})

            postfix = {"steps": steps, "return": f"{total_reward:.3f}"}
            if final_dist is not None:
                postfix["final_dist"] = f"{final_dist:.3f}"
            episode_bar.set_postfix(postfix)
            episode_bar.update(1)

            if args.video_every and episode_i % args.video_every == 0:
                video_path = out_dir / f"episode_{episode_i:04d}.mp4"
                imageio.mimsave(video_path, frames, fps=int(args.sim_rate))
                tqdm.write(f"episode {episode_i}: steps={steps} return={total_reward:.3f} wrote {video_path}")
            else:
                tqdm.write(
                    f"episode {episode_i}: steps={steps} return={total_reward:.3f} "
                    f"terminated={terminated} truncated={truncated} final_dist={final_dist}"
                )

    env.close()

    survival = sum(1 for row in summary_rows if row["survived"]) / max(len(summary_rows), 1)
    mean_steps = float(np.mean([row["steps"] for row in summary_rows])) if summary_rows else 0.0
    mean_return = float(np.mean([row["return"] for row in summary_rows])) if summary_rows else 0.0
    mean_final_dist = float(np.mean([row["final_distance_to_target"] for row in summary_rows if row["final_distance_to_target"] is not None])) if summary_rows else 0.0
    result = {
        "policy": str(tflite_path),
        "meta": str(meta_path),
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "scene": args.scene,
        "episodes": args.episodes,
        "steps_limit": args.steps,
        "seed": args.seed,
        "elapsed_s": time.time() - started,
        "observation": spec.raw_observation,
        "output": spec.output,
        "dynamics": dynamics_config,
        "rotorpy": dynamics_config if dynamics_config.get("model") == "rotorpy_crazyflie" else None,
        "gyro_bias_rad_s": gyro_bias.tolist(),
        "world_z_source": args.world_z_source,
        "vision_delay_steps": vision_delay_steps,
        "metrics": {
            "survival_rate": survival,
            "mean_steps": mean_steps,
            "mean_return": mean_return,
            "mean_final_distance_to_target": mean_final_dist,
        },
        "episodes_detail": episode_records,
    }

    (out_dir / "summary.json").write_text(json.dumps(result, indent=2))
    with (out_dir / "episodes.jsonl").open("w") as f:
        for record in episode_records:
            f.write(json.dumps(record) + "\n")
    print(
        f"summary: survival={survival:.3f} mean_steps={mean_steps:.1f} "
        f"mean_return={mean_return:.3f} mean_final_dist={mean_final_dist:.3f}"
    )
    print(f"wrote {out_dir / 'summary.json'}")
    return result


def main():
    ap = argparse.ArgumentParser()
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=str, help="RLtools .h5 checkpoint to convert/cache")
    source.add_argument("--tflite", type=str, help="existing .tflite or .int8.tflite model")
    ap.add_argument("--meta", type=str, default=None, help="converter .example_meta.json; inferred when possible")
    ap.add_argument("--quantize", choices=["none", "int8"], default="none", help="convert checkpoint to this TFLite flavor")
    ap.add_argument("--force-convert", action="store_true")
    ap.add_argument("--converter-python", type=str, default=None)
    ap.add_argument("--converter-script", type=str, default=None)
    ap.add_argument("--converter-arg", action="append", default=[], help="extra argument passed to hdf5_to_tflite.py; repeatable")
    ap.add_argument("--example-bin-limit", type=int, default=16, help="samples written to converter companion bins; does not limit int8 calibration")
    ap.add_argument("--validate-examples", type=int, default=1, help="number of companion examples to verify before rollout; 0 disables")
    ap.add_argument("--example-tolerance", type=float, default=1e-4)

    ap.add_argument("--scene", type=str, default=None, help="3DGS .ply scene; omitted uses synthetic scene")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--sim-rate", type=float, default=100.0)
    ap.add_argument("--world-bounds", type=float, default=10.0)
    ap.add_argument("--dynamics-model", choices=DYNAMICS_MODELS, default="crazyflie_openmv",
                    help="quadrotor dynamics backend for policy rollout")
    ap.add_argument("--rotorpy-k-m", type=float, default=None,
                    help="RotorPY only: override yaw moment coefficient k_m wrt rotor speed squared")
    ap.add_argument("--rotorpy-yaw-torque-per-thrust", type=float, default=None,
                    help="RotorPY only: override k_m/k_eta yaw torque per thrust ratio")
    ap.add_argument("--rotorpy-match-rltools-yaw", action="store_true",
                    help="RotorPY only: set k_m/k_eta to RLtools crazyflie.h value 0.004665")
    ap.add_argument("--gyro-bias", type=float, nargs=3, default=(0.0, 0.0, 0.0), metavar=("X", "Y", "Z"),
                    help="constant bias added to the AngularVelocity policy observation, in rad/s")
    ap.add_argument("--world-z-source", choices=WORLD_Z_SOURCES, default="groundtruth",
                    help="source for the OrientationWorldZ policy observation")
    ap.add_argument("--vision-delay-steps", type=int, default=0,
                    help="camera observation delay in sim/control ticks")
    ap.add_argument("--initial-state", choices=["hover", "random"], default="random")
    ap.add_argument("--start", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
    ap.add_argument("--target", type=float, nargs=3, default=(0.0, 0.0, 0.0), metavar=("X", "Y", "Z"))
    ap.add_argument("--target-yaw", type=float, default=0.0, help="target yaw in radians")
    ap.add_argument("--out-dir", type=str, default="runs/l2f_tflite")
    ap.add_argument("--video-every", type=int, default=0, help="save every Nth episode video; 0 disables")
    ap.add_argument("--no-progress", action="store_true", help="disable tqdm progress bars")
    args = ap.parse_args()
    try:
        run(args)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
