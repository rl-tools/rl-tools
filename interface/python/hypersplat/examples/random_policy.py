"""End-to-end smoke test: reset, run random actions, save the camera stream as a GIF.

Usage:
    python examples/random_policy.py                              # synthetic toy scene
    python examples/random_policy.py --scene /path/to/scene.ply
"""
from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

from hypersplat import HyperSplatEnv
from hypersplat.renderer import Scene


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=str, default=None)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--out", type=str, default="rollout.gif")
    ap.add_argument("--height", type=int, default=50)
    ap.add_argument("--width", type=int, default=80)
    ap.add_argument("--fov", type=float, default=90.0)
    ap.add_argument("--control-mode", type=str, default="cmd_motor_speeds")
    ap.add_argument("--initial-state", type=str, default="hover", choices=["hover", "random"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.scene:
        scene = Scene.from_ply(args.scene, device=args.device)
    else:
        scene = Scene.synthetic(device=args.device)
    print(f"scene: {scene.num_gaussians} gaussians on {args.device}")

    env = HyperSplatEnv(
        scene=scene,
        obs_mode="dict",
        control_mode=args.control_mode,
        image_size=(args.height, args.width),
        fov_deg=args.fov,
        initial_state=args.initial_state,
        max_episode_steps=args.steps,
        device=args.device,
    )

    rng = np.random.default_rng(args.seed)
    obs, info = env.reset(seed=args.seed)
    frames = [obs["image"]]
    rewards = []
    terminated = truncated = False
    for _ in range(args.steps):
        action = rng.uniform(-1.0, 1.0, size=env.action_space.shape).astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        frames.append(obs["image"])
        rewards.append(reward)
        if terminated or truncated:
            break

    out = Path(args.out)
    imageio.mimsave(str(out), frames, fps=24)
    print(
        f"episode summary: steps={len(rewards)} return={sum(rewards):.3f} "
        f"terminated={terminated} truncated={truncated}"
    )
    print(f"wrote {out} ({len(frames)} frames @ {args.width}x{args.height})")


if __name__ == "__main__":
    main()
