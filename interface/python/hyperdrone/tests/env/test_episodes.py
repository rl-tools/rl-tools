import numpy as np
import pytest

pytest.importorskip("gymnasium")
from hyperdrone import gym


class BatchEnvironment:
    total_instances = 2
    observation_dim = 2
    action_dim = 1
    episode_step_limit = 3

    def __init__(self, *args, **kwargs):
        self.age = np.zeros(2, dtype=np.int64)
        self.episodes = np.zeros(2, dtype=np.int64)
        self.terminations = np.zeros(2, dtype=bool)

    def reset(self, mask=None):
        mask = np.ones(2, dtype=bool) if mask is None else mask
        self.age[mask] = 0
        self.episodes[mask] += 1

    def step(self, actions):
        self.age += 1
        self.terminations = actions[:, 0] > 0

    def observe(self):
        return np.column_stack((self.age, self.episodes)).astype(np.float32)

    def rewards(self):
        return np.array([1, 2], dtype=np.float32)

    def terminated(self):
        return self.terminations.copy()

    def close(self):
        pass


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setattr(gym, "MultiEnvironment", BatchEnvironment)
    environment = gym.VectorEnv("unused")
    environment.reset()
    yield environment
    environment.close()


def test_same_step_autoreset_and_independent_time_limits(env):
    observations, rewards, terminated, truncated, _ = env.step([[1], [0]])
    np.testing.assert_array_equal(terminated, [True, False])
    np.testing.assert_array_equal(truncated, [False, False])
    np.testing.assert_array_equal(observations, [[0, 2], [1, 1]])
    np.testing.assert_array_equal(rewards, [1, 2])

    env.step([[0], [0]])
    observations, _, terminated, truncated, _ = env.step([[0], [0]])
    np.testing.assert_array_equal(terminated, [False, False])
    np.testing.assert_array_equal(truncated, [False, True])
    np.testing.assert_array_equal(observations, [[2, 2], [0, 2]])

    observations, _, _, truncated, _ = env.step([[0], [0]])
    np.testing.assert_array_equal(truncated, [True, False])
    np.testing.assert_array_equal(observations, [[0, 3], [1, 2]])


def test_termination_at_time_limit(env):
    env.step([[0], [0]])
    env.step([[0], [0]])
    observations, _, terminated, truncated, _ = env.step([[1], [0]])
    np.testing.assert_array_equal(terminated, [True, False])
    np.testing.assert_array_equal(truncated, [False, True])
    np.testing.assert_array_equal(observations, [[0, 2], [0, 2]])


def test_explicit_reset_restarts_time_limits(env):
    env.step([[0], [0]])
    env.step([[0], [0]])
    observations, _ = env.reset()
    np.testing.assert_array_equal(observations, [[0, 2], [0, 2]])
    for _ in range(2):
        _, _, _, truncated, _ = env.step([[0], [0]])
        assert not truncated.any()
    _, _, _, truncated, _ = env.step([[0], [0]])
    assert truncated.all()


def test_zero_disables_time_limit(env):
    env._env.episode_step_limit = 0
    for _ in range(5):
        _, _, _, truncated, _ = env.step([[0], [0]])
        assert not truncated.any()


def test_reset_rejects_reseeding(env):
    with pytest.raises(ValueError, match="seed is fixed"):
        env.reset(seed=1)
