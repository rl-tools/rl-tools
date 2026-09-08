"""hyperdrone.gym — optional Gymnasium VectorEnv adapter over hyperdrone.env.

A leaf module: the core packages never import gymnasium, and gym-side API churn stays
isolated here. Requires the optional dependency: pip install "hyperdrone[gym]".
"""
import numpy as np

try:
    import gymnasium
except ImportError as error:
    raise ImportError("hyperdrone.gym needs gymnasium — pip install 'hyperdrone[gym]'") from error

from ..env import EnvConfig, MultiEnvironment

__all__ = ["VectorEnv"]


class VectorEnv(gymnasium.vector.VectorEnv):
    """Same-step autoreset vectorized environment over MultiEnvironment.

    Instances that terminate or hit the step limit reset within the same step, so the
    returned observation is the first of the new episode. No final_observation is surfaced;
    the environment renders once per step. `truncations` flags time limits that are not
    terminations. Seeding is fixed at construction (the C++ environment owns its RNG);
    reset(seed=...) is rejected to keep determinism claims honest.
    """

    render_mode = None

    def __init__(self, scenes, config=None, seed=0, drone_asset=None):
        self._env = MultiEnvironment(scenes, config=config, seed=seed, drone_asset=drone_asset)
        self.num_envs = self._env.total_instances
        self.single_observation_space = gymnasium.spaces.Box(
            0.0, 1.0, shape=(self._env.observation_dim,), dtype=np.float32
        )
        self.single_action_space = gymnasium.spaces.Box(
            -1.0, 1.0, shape=(self._env.action_dim,), dtype=np.float32
        )
        self.observation_space = gymnasium.vector.utils.batch_space(self.single_observation_space, self.num_envs)
        self.action_space = gymnasium.vector.utils.batch_space(self.single_action_space, self.num_envs)
        self._steps = np.zeros(self.num_envs, dtype=np.int64)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            raise ValueError("hyperdrone: the environment seed is fixed at construction — pass seed= to VectorEnv()")
        self._env.reset()
        self._steps[:] = 0
        return self._env.observe(), {}

    def step(self, actions):
        self._env.step(np.asarray(actions, dtype=np.float32))
        rewards = self._env.rewards()
        terminations = self._env.terminated()
        self._steps += 1
        truncations = (self._env.episode_step_limit > 0) & (self._steps >= self._env.episode_step_limit) & ~terminations
        reset_mask = terminations | truncations
        if reset_mask.any():
            self._env.reset(reset_mask)
            self._steps[reset_mask] = 0
        return self._env.observe(), rewards, terminations, truncations, {}

    def close_extras(self, **kwargs):
        self._env.close()
