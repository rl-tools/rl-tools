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


def test_config_validation_errors():
    with pytest.raises(ValueError, match="unknown preset"):
        EnvConfig(preset="nonexistent")
    with pytest.raises(ValueError, match="unknown task"):
        EnvConfig(task="nonexistent")
    with pytest.raises(ValueError, match="SELF_VISIBLE"):
        EnvConfig(n_agents=2)
    with pytest.raises(ValueError, match="spec_header supersedes"):
        EnvConfig(task="target_frame", spec_header=__file__)
    with pytest.raises(ValueError, match="needs drone_asset"):
        MultiEnvironment("/nonexistent", config=EnvConfig(preset="x500_fpv", n_agents=1))


def test_observation_layout(scene_directory):
    env = MultiEnvironment(scene_directory, config=CONFIG, seed=0)
    try:
        layout = env.observation_layout
        assert layout.axis == "channel"
        assert layout.shape == (32, 32, 3)
        assert layout.blocks == {"image": (0, 3)}
        privileged = env.observation_layout_privileged
        assert privileged.axis == "flat"
        assert privileged.shape == (18,)
        assert privileged.blocks["position"] == (0, 3)
        assert privileged.blocks["orientation_rotation_matrix"] == (3, 9)
        assert privileged.blocks["linear_velocity"] == (12, 3)
        assert privileged.blocks["angular_velocity"] == (15, 3)
    finally:
        env.close()


def drone_asset_path():
    source = Path(os.environ.get("HYPERDRONE_TEST_SCENE_DIR", source_root() / "tests" / "data")) / "x500.glb"
    if not source.exists():
        pytest.skip(f"no drone asset at {source}")
    return source


def smoke_rollout(env):
    mask = np.ones(env.total_instances, dtype=np.uint8)
    env.reset(mask)
    env.render(mask)
    observations = env.observe()
    assert observations.shape == (env.total_instances, env.observation_dim)
    actions = np.zeros((env.total_instances, env.action_dim), dtype=np.float32)
    env.step(actions)
    env.render(np.zeros(env.total_instances, dtype=np.uint8))
    assert np.isfinite(env.rewards()).all()
    assert env.observe_privileged().shape == (env.total_instances, env.observation_dim_privileged)


def test_x500_target_frame(scene_directory):
    config = EnvConfig(instances=2, cam_width=16, cam_height=16, preset="x500_fpv", task="target_frame")
    env = MultiEnvironment(scene_directory, config=config, seed=3, drone_asset=drone_asset_path())
    try:
        assert env.observation_dim == 16 * 16 * 6
        layout = env.observation_layout
        assert layout.shape == (16, 16, 6)
        assert layout.blocks == {"image_stack": (0, 3), "target_image": (3, 3)}
        smoke_rollout(env)
    finally:
        env.close()


def test_moving_gate(scene_directory):
    config = EnvConfig(instances=2, cam_width=16, cam_height=16, task="moving_gate")
    with pytest.raises(ValueError, match="needs gate_asset"):
        MultiEnvironment(scene_directory, config=config, seed=4)
    env = MultiEnvironment(scene_directory, config=config, seed=4, gate_asset=drone_asset_path())
    try:
        privileged = env.observation_layout_privileged
        assert privileged.blocks["gate_state"] == (18, 8)
        assert env.observation_dim_privileged == 26
        smoke_rollout(env)
    finally:
        env.close()


def test_spec_header_escape_hatch(scene_directory):
    header = Path(__file__).parent / "user_spec_header.h"
    config = EnvConfig(spec_header=str(header))
    env = MultiEnvironment(scene_directory, config=config, seed=5, drone_asset=drone_asset_path())
    try:
        assert env.instances_per_environment == 2
        assert env.cam_width == 16 and env.cam_height == 16
        assert env.observation_layout.blocks == {"observation": (0, env.observation_dim)}
        smoke_rollout(env)
    finally:
        env.close()
