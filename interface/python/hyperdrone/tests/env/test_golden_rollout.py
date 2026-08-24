"""The cross-language stability anchor: the C++ generator (tests/src/rl/environments/
hyperdrone/python_golden.cpp, compiled from the exact shim TU) records a seeded rollout;
this test replays it through the binding and compares bit-exactly. Any C++ change that
alters environment semantics fails here the same day; regenerating the golden is a
deliberate, reviewed act. The golden is generated on the GENERIC backend — run this suite
with HYPERDRONE_RENDER_BACKEND=GENERIC (rendering is pinned byte-identical across
backends by the raytracing golden corpus, so other backends are expected to match too).
"""
import json
import os
from pathlib import Path

import numpy as np
import pytest

from hyperdrone.env import EnvConfig, MultiEnvironment
from hyperdrone.jit import source_root

GOLDEN = Path(__file__).parent / "golden"

CONFIG = EnvConfig(num_environments=1, instances=2, cam_width=16, cam_height=16, shading="low", history_length=1)


def observe_pair(env):
    return env.observe(), env.observe_privileged()


def golden_actions(step, total, action_dim):
    actions = np.empty((total, action_dim), dtype=np.float32)
    for instance in range(total):
        for dim in range(action_dim):
            actions[instance, dim] = np.float32(-1.0 + 2.0 * (((step * 131 + instance * 31 + dim * 7) % 101) / 100.0))
    return actions


@pytest.fixture(scope="module")
def scene_directory(tmp_path_factory):
    manifest = json.loads((GOLDEN / "manifest.json").read_text())
    source = Path(os.environ.get("HYPERDRONE_TEST_SCENE_DIR", source_root() / "tests" / "data")) / manifest["scene"]
    if not source.exists():
        pytest.skip(f"no golden scene at {source}")
    directory = tmp_path_factory.mktemp("golden_scene")
    (directory / source.name).symlink_to(source)
    return directory


def test_golden_rollout(scene_directory):
    if not (GOLDEN / "manifest.json").exists():
        pytest.skip("golden rollout not generated (build and run test_rl_environments_hyperdrone_python_golden)")
    manifest = json.loads((GOLDEN / "manifest.json").read_text())
    env = MultiEnvironment(scene_directory, config=CONFIG, seed=manifest["seed"])
    try:
        assert env.config_string == manifest["config_string"], (
            "the binding's configuration drifted from the golden's — regenerate deliberately"
        )
        total, steps = env.total_instances, manifest["steps"]
        assert total == manifest["total_instances"]

        rewards, terminated = [], []
        mask = np.ones(total, dtype=np.uint8)
        none = np.zeros(total, dtype=np.uint8)
        env.reset(mask)
        env.render(mask)
        observations = [observe_pair(env)]
        for step in range(steps):
            env.step(golden_actions(step, total, env.action_dim))
            env.render(none)
            observations.append(observe_pair(env))
            rewards.append(env.rewards())
            terminated.append(env.terminated())

        golden_observations = np.fromfile(GOLDEN / "observations.bin", dtype=np.float32)
        golden_privileged = np.fromfile(GOLDEN / "observations_privileged.bin", dtype=np.float32)
        golden_rewards = np.fromfile(GOLDEN / "rewards.bin", dtype=np.float32)
        golden_terminated = np.fromfile(GOLDEN / "terminated.bin", dtype=np.uint8)

        actual_observations = np.concatenate([pair[0].ravel() for pair in observations])
        actual_privileged = np.concatenate([pair[1].ravel() for pair in observations])
        np.testing.assert_array_equal(actual_observations, golden_observations)
        np.testing.assert_array_equal(actual_privileged, golden_privileged)
        np.testing.assert_array_equal(np.concatenate(rewards), golden_rewards)
        np.testing.assert_array_equal(np.concatenate(terminated).astype(np.uint8), golden_terminated)

        assert manifest["observation_layout"].strip().splitlines()[0].startswith("shape")
    finally:
        env.close()
