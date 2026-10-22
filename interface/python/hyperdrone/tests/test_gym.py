import os
from pathlib import Path

import numpy as np
import pytest

gymnasium = pytest.importorskip("gymnasium")

from hyperdrone import Config
from hyperlux.camera import CameraSpec, Depth, RGB
from hyperdrone.camera import ImageObservation
from hyperlux._jit import source_root


@pytest.fixture(scope="module")
def scene_directory(tmp_path_factory):
    source = Path(os.environ.get("HYPERDRONE_TEST_SCENE_DIR", source_root() / "tests" / "data")) / "ProcTHOR-Train-1.glb"
    if not source.exists():
        pytest.skip(f"no ProcTHOR test scene at {source}")
    directory = tmp_path_factory.mktemp("scenes")
    (directory / source.name).symlink_to(source)
    return directory


def test_vector_env_api(scene_directory):
    from hyperdrone.gym import VectorEnv

    config = Config(num_worlds=1, instances_per_world=2, width=16, height=16, cameras=(CameraSpec("front", (RGB("float32"),)), CameraSpec("down", (Depth(),))), observation=ImageObservation(("front",)))
    env = VectorEnv(scene_directory, config=config, seed=11)
    try:
        assert env.num_envs == 2
        observations, infos = env.reset()
        assert observations.shape == (2, 16 * 16 * 3)
        assert env.observation_space.contains(observations)
        for _ in range(3):
            actions = np.zeros((env.num_envs, env.single_action_space.shape[0]), dtype=np.float32)
            observations, rewards, terminations, truncations, infos = env.step(actions)
            assert observations.shape == (2, 16 * 16 * 3)
            assert rewards.shape == (2,)
            assert terminations.dtype == bool and truncations.dtype == bool
        with pytest.raises(ValueError, match="seed"):
            env.reset(seed=1)
    finally:
        env.close()


def test_vector_env_rejects_missing_policy_observation(monkeypatch):
    from hyperdrone import gym

    closed = []

    class NonvisualEnvironment:
        has_image_observation = False

        def __init__(self, scenes, **kwargs):
            assert kwargs["config"].observation is None

        def close(self):
            closed.append(True)

    monkeypatch.setattr(gym, "HyperDrone", NonvisualEnvironment)
    config = Config(cameras=(CameraSpec("depth", (Depth(),)),), observation=None)
    with pytest.raises(ValueError, match="requires a configured policy observation"):
        gym.VectorEnv([], config=config)
    assert closed == [True]
