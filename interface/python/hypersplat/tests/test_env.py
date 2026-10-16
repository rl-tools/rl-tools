"""Gymnasium API conformance + end-to-end smoke."""
from __future__ import annotations

import numpy as np
import torch
from gymnasium.utils.env_checker import check_env

from hypersplat import HyperSplatEnv
from hypersplat.dynamics import Dynamics, crazyflie_openmv_summary
from hypersplat.renderer import Scene


def _scene():
    return Scene.synthetic(num_gaussians=500, device="cuda" if torch.cuda.is_available() else "cpu")


def test_check_env_dict():
    env = HyperSplatEnv(scene=_scene(), obs_mode="dict", max_episode_steps=5)
    check_env(env, skip_render_check=True)


def test_check_env_image():
    env = HyperSplatEnv(scene=_scene(), obs_mode="image", max_episode_steps=5)
    check_env(env, skip_render_check=True)


def test_check_env_state():
    env = HyperSplatEnv(scene=_scene(), obs_mode="state", max_episode_steps=5)
    check_env(env, skip_render_check=True)


def test_step_and_reset():
    env = HyperSplatEnv(scene=_scene(), obs_mode="dict", max_episode_steps=5)
    obs, info = env.reset(seed=42)
    assert obs["image"].shape == (50, 80, 3)
    assert obs["state"].shape == (13,)
    assert "image" in info and "state" in info
    for _ in range(5):
        a = env.action_space.sample()
        obs, r, term, trunc, info = env.step(a)
        assert isinstance(r, float)
    assert trunc, "should truncate after max_episode_steps"


def test_termination_on_out_of_bounds():
    env = HyperSplatEnv(
        scene=_scene(), obs_mode="state", max_episode_steps=10000, world_bounds=2.0,
    )
    env.reset()
    # Force the dynamics state outside the bounds.
    env.dynamics.state["x"] = np.array([100.0, 0.0, 0.0])
    obs, r, term, trunc, info = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
    assert term, "should terminate when |x| exceeds world_bounds"


def test_action_dim_changes_with_control_mode():
    env_vel = HyperSplatEnv(scene=_scene(), control_mode="cmd_vel", max_episode_steps=5)
    assert env_vel.action_space.shape == (3,)
    env_motor = HyperSplatEnv(scene=_scene(), control_mode="cmd_motor_speeds", max_episode_steps=5)
    assert env_motor.action_space.shape == (4,)


def test_crazyflie_openmv_dynamics_backend_step():
    dyn = Dynamics(dynamics_model="crazyflie_openmv")
    summary = crazyflie_openmv_summary()
    state = dyn.reset()
    np.testing.assert_allclose(state["rotor_speeds"], summary["hover_command"])

    hover_action = np.full(4, 2 * summary["hover_command"] - 1, dtype=np.float32)
    next_state = dyn.step(hover_action)
    assert np.all(np.isfinite(dyn.flat_state()))
    assert next_state["v"][2] > 0.0
    assert next_state["v"][2] < 0.01


if __name__ == "__main__":
    test_check_env_dict()
    test_check_env_image()
    test_check_env_state()
    test_step_and_reset()
    test_termination_on_out_of_bounds()
    test_action_dim_changes_with_control_mode()
    test_crazyflie_openmv_dynamics_backend_step()
    print("all env tests passed")
