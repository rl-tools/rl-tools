import os
from pathlib import Path

import numpy as np
import pytest

from hyperdrone.env import EnvConfig, MultiEnvironment
from hyperdrone.jit import source_root


@pytest.fixture(scope="module")
def scene_directory(tmp_path_factory):
    # a directory holding exactly the scenes the environment should schedule over
    source = Path(os.environ.get("HYPERDRONE_TEST_SCENE_DIR", source_root() / "tests" / "data")) / "ProcTHOR-Train-1.glb"
    if not source.exists():
        pytest.skip(f"no ProcTHOR test scene at {source}")
    directory = tmp_path_factory.mktemp("scenes")
    (directory / source.name).symlink_to(source)
    return directory


CONFIG = EnvConfig(num_environments=1, instances=2, cam_width=32, cam_height=32, shading="low")


def rollout(scene_directory, seed, steps=3):
    env = MultiEnvironment(scene_directory, config=CONFIG, seed=seed)
    try:
        assert env.total_instances == 2
        assert env.action_dim == 4
        assert env.observation_dim == 32 * 32 * 3
        mask = np.ones(env.total_instances, dtype=np.uint8)
        env.reset(mask)
        env.render(mask)
        record = [env.observe().copy()]
        rng = np.random.default_rng(seed)
        no_reset = np.zeros(env.total_instances, dtype=np.uint8)
        for _ in range(steps):
            actions = rng.uniform(-1, 1, size=(env.total_instances, env.action_dim)).astype(np.float32)
            env.step(actions)
            env.render(no_reset)
            record.append(env.observe().copy())
            record.append(env.rewards().copy())
            record.append(env.terminated().copy())
            record.append(env.observe_privileged().copy())
        return record
    finally:
        env.close()


def test_verb_sequence_shapes_and_determinism(scene_directory):
    first = rollout(scene_directory, 1337)
    second = rollout(scene_directory, 1337)
    assert len(first) == len(second)
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)
    frames = first[0]
    assert frames.shape == (2, 32 * 32 * 3)
    assert np.all(frames >= 0) and np.all(frames <= 1)
    assert frames.std() > 0, "the rendered observation should not be blank"


def test_frames_and_rotate_scene(scene_directory):
    env = MultiEnvironment(scene_directory, config=CONFIG, seed=7)
    try:
        mask = np.ones(env.total_instances, dtype=np.uint8)
        env.reset(mask)
        env.render(mask)
        frames = env.frames()
        assert frames.shape == (2, 32, 32, 3)
        env.rotate_scene()
        env.reset(mask)
        env.render(mask)
        assert env.frames().shape == (2, 32, 32, 3)
    finally:
        env.close()
