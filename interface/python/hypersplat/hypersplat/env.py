"""`HyperSplatEnv` — gymnasium.Env composing quadrotor dynamics with gsplat rendering."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Optional

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from hypersplat.dynamics import ACTION_DIMS, Dynamics
from hypersplat.frames import compute_viewmat
from hypersplat.renderer import Renderer, Scene, make_intrinsics
from hypersplat.rewards import hover_reward


RewardFn = Callable[[dict, np.ndarray], float]


class HyperSplatEnv(gym.Env):
    """Single-drone quadrotor env with a 3DGS-rendered onboard camera.

    See `pyproject.toml` and the project README for usage. Brief recap:

        env = HyperSplatEnv(scene_path="scene.ply", obs_mode="dict")
        obs, info = env.reset(seed=0)
        obs, reward, terminated, truncated, info = env.step(action)
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        scene_path: Optional[str] = None,
        *,
        scene: Optional[Scene] = None,
        obs_mode: str = "dict",
        control_mode: str = "cmd_motor_speeds",
        image_size: tuple[int, int] = (50, 80),
        fov_deg: float = 90.0,
        camera_intrinsics: Optional[dict] = None,
        sim_rate: float = 100.0,
        max_episode_steps: int = 500,
        reward_fn: Optional[RewardFn] = None,
        initial_state: Any = "hover",
        scene_transform: Optional[np.ndarray] = None,
        body_to_camera: Optional[np.ndarray] = None,
        world_bounds: float = 10.0,
        device: str = "cuda",
        quad_params: Optional[dict] = None,
        aero: bool = True,
        dynamics_model: str = "rotorpy_crazyflie",
        seed: Optional[int] = None,
    ):
        super().__init__()
        if obs_mode not in ("dict", "image", "state"):
            raise ValueError(f"obs_mode must be one of dict|image|state, got {obs_mode!r}")
        if scene is None and scene_path is None:
            raise ValueError("must supply either `scene_path` or a preloaded `scene`")

        self.obs_mode = obs_mode
        self.image_size = tuple(image_size)
        self.max_episode_steps = int(max_episode_steps)
        self.world_bounds = float(world_bounds)
        self.body_to_camera = None if body_to_camera is None else np.asarray(body_to_camera, dtype=np.float64)
        self.reward_fn: RewardFn = reward_fn if reward_fn is not None else hover_reward
        self._initial_state_spec = initial_state

        # ---- physics
        self.dynamics = Dynamics(
            control_mode=control_mode,
            quad_params=quad_params,
            sim_rate=sim_rate,
            aero=aero,
            dynamics_model=dynamics_model,
        )

        # ---- renderer
        self.scene = scene if scene is not None else Scene.from_ply(scene_path, device=device)

        # Default scene_transform comes from the scene's sidecar JSON; explicit arg overrides.
        if scene_transform is not None:
            self.scene_transform = np.asarray(scene_transform, dtype=np.float64)
        else:
            self.scene_transform = self.scene.scene_transform
        K = make_intrinsics(self.image_size, fov_deg, camera_intrinsics)
        self.renderer = Renderer(self.scene, self.image_size, K)

        # ---- spaces
        H, W = self.image_size
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(ACTION_DIMS[control_mode],), dtype=np.float32,
        )
        image_space = spaces.Box(low=0, high=255, shape=(H, W, 3), dtype=np.uint8)
        state_space = spaces.Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32)
        if obs_mode == "image":
            self.observation_space = image_space
        elif obs_mode == "state":
            self.observation_space = state_space
        else:  # dict
            self.observation_space = spaces.Dict({"image": image_space, "state": state_space})

        # ---- bookkeeping
        self._np_random: np.random.Generator
        if seed is not None:
            self._np_random, _ = gym.utils.seeding.np_random(seed)
        else:
            self._np_random, _ = gym.utils.seeding.np_random(None)
        self._step_count = 0
        self._last_image: Optional[np.ndarray] = None

    # ---------- gymnasium API ---------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._np_random, _ = gym.utils.seeding.np_random(seed)

        spec = (options or {}).get("initial_state", self._initial_state_spec)
        init = self._build_initial_state(spec)
        self.dynamics.reset(init)
        self._step_count = 0
        self._render_current()
        return self._make_obs(), self._info()

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        state = self.dynamics.step(action)
        self._step_count += 1
        self._render_current()

        reward = float(self.reward_fn(state, action))
        terminated = self._is_terminated(state)
        truncated = self._step_count >= self.max_episode_steps
        return self._make_obs(), reward, terminated, truncated, self._info()

    def render(self) -> Optional[np.ndarray]:
        return None if self._last_image is None else self._last_image.copy()

    def close(self):
        # gsplat tensors are freed when self.scene is GC'd; nothing else holds resources.
        pass

    # ---------- internals -------------------------------------------------
    def _build_initial_state(self, spec: Any) -> dict:
        if isinstance(spec, dict):
            base = self.dynamics.default_initial_state()
            base.update({k: np.asarray(v, dtype=np.float64) for k, v in spec.items()})
            return base
        if spec == "hover":
            return self.dynamics.default_initial_state()
        if spec == "random":
            init = self.dynamics.default_initial_state()
            init["x"] = self._np_random.uniform(-1.0, 1.0, size=3)
            init["x"][2] = self._np_random.uniform(0.5, 2.0)  # spawn above ground
            yaw = self._np_random.uniform(-np.pi, np.pi)
            init["q"] = np.array([0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)])
            return init
        raise ValueError(f"unknown initial_state spec {spec!r}")

    def _render_current(self):
        viewmat = compute_viewmat(self.dynamics.state, self.scene_transform, self.body_to_camera)
        with torch.no_grad():
            self._last_image = self.renderer.render(viewmat)

    def _make_obs(self):
        if self.obs_mode == "image":
            return self._last_image
        if self.obs_mode == "state":
            return self.dynamics.flat_state()
        return {"image": self._last_image, "state": self.dynamics.flat_state()}

    def _info(self) -> dict:
        return {
            "image": self._last_image,
            "step": self._step_count,
            "state": deepcopy(self.dynamics.state),
        }

    def _is_terminated(self, state: dict) -> bool:
        x = np.asarray(state["x"])
        if not np.all(np.isfinite(x)):
            return True
        return bool(np.max(np.abs(x)) > self.world_bounds)
