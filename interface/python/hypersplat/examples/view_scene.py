"""Render a static orbit around the scene; sanity-checks the renderer in isolation.

Usage:
    python examples/view_scene.py                              # synthetic toy scene
    python examples/view_scene.py --scene /path/to/scene.ply   # real 3DGS scene
"""
from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

from hypersplat.renderer import Renderer, Scene, make_intrinsics


def look_at(cam_pos: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Build a camera-to-world 4x4 in OpenCV convention (+z forward, +y down)."""
    forward = target - cam_pos
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)  # OpenCV: y-axis is down
    R = np.stack([right, down, forward], axis=1)  # columns = camera basis in world
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = R
    c2w[:3, 3] = cam_pos
    return c2w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=str, default=None, help="Path to a 3DGS .ply (optional)")
    ap.add_argument("--out", type=str, default="orbit.mp4")
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--height", type=int, default=200)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--fov", type=float, default=70.0)
    ap.add_argument("--radius", type=float, default=6.0)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.scene:
        print(f"loading {args.scene} on {args.device} ...")
        scene = Scene.from_ply(args.scene, device=args.device)
    else:
        print(f"using synthetic scene on {args.device}")
        scene = Scene.synthetic(device=args.device)
    print(f"  {scene.num_gaussians} gaussians, sh_degree={scene.sh_degree}")

    K = make_intrinsics((args.height, args.width), args.fov)
    renderer = Renderer(scene, (args.height, args.width), K)

    target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)  # ENU world up

    frames = []
    for i in range(args.frames):
        theta = 2 * np.pi * i / args.frames
        cam_pos = np.array(
            [args.radius * np.cos(theta), args.radius * np.sin(theta), 1.5],
            dtype=np.float32,
        )
        c2w = look_at(cam_pos, target, up)
        viewmat = np.linalg.inv(c2w)
        img = renderer.render(viewmat)
        frames.append(img)

    out = Path(args.out)
    imageio.mimsave(str(out), frames, fps=24)
    print(f"wrote {out} ({len(frames)} frames @ {args.width}x{args.height})")


if __name__ == "__main__":
    main()
