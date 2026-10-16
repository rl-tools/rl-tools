# hypersplat

A small single-drone quadrotor simulator combining **RotorPY** dynamics with **gsplat** Gaussian-splat camera rendering, exposed through a `gymnasium.Env` API.

## Install

```bash
.venv/bin/pip install -e . plyfile imageio
```

(The venv is expected to already contain `rotorpy`, `gsplat`, `torch`, `gymnasium`.)

## Usage

```python
import numpy as np
from hypersplat import HyperSplatEnv

env = HyperSplatEnv(
    scene_path="scene.ply",         # 3DGS .ply (INRIA / nerfstudio export); or pass scene=Scene.synthetic()
    obs_mode="dict",                # "dict" | "image" | "state"
    control_mode="cmd_motor_speeds",
    image_size=(50, 80),            # (H, W)
    fov_deg=90.0,                   # horizontal FOV; or pass camera_intrinsics={fx,fy,cx,cy}
    sim_rate=100,                   # Hz
    max_episode_steps=500,
    initial_state="hover",          # "hover" | "random" | dict
)

obs, info = env.reset(seed=0)
for _ in range(100):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    if terminated or truncated:
        break
env.close()
```

## Examples

```bash
.venv/bin/python examples/view_scene.py                # orbit a synthetic scene -> orbit.gif
.venv/bin/python examples/random_policy.py             # full env rollout with rendering -> rollout.gif
.venv/bin/python examples/random_policy.py --scene path/to/scene.ply
```

## Conventions

- **Drone world frame**: ENU (`+X` east, `+Y` north, `+Z` up) — RotorPY native.
- **Body frame**: FLU (`+X` forward, `+Y` left, `+Z` up).
- **Camera frame**: OpenCV (`+X` right, `+Y` down, `+Z` forward) — what gsplat consumes.
- Default camera mounting points the camera along body `+X`. Override via `body_to_camera`.
- For most pretrained scenes, the `.ply` world frame won't match ENU; calibrate `scene_transform` (a 4×4 ENU→scene mapping) once per scene.

## Layout

```
hypersplat/
  env.py        HyperSplatEnv (gymnasium.Env)
  dynamics.py   RotorPY Multirotor adapter (normalized [-1,1] actions)
  renderer.py   gsplat rasterization + .ply / synthetic loader
  frames.py     coordinate-frame transforms
  rewards.py    default hover reward
examples/       view_scene.py, random_policy.py
tests/          test_frames.py, test_env.py
```
