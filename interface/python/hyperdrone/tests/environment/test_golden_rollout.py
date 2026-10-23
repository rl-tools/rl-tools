"""The C++ generator records a seeded rollout for bit-exact replay on its GENERIC host build. Run with HYPERLUX_BACKEND=GENERIC; other renderer backends are qualified against measured bounds by the raytracing corpus, not claimed byte-identical. Regeneration is deliberate and reviewed."""
import json
import os
from pathlib import Path

import numpy as np
import pytest

from hyperdrone import Config, HyperDrone
from hyperdrone._environment import ObservationLayout
from hyperlux._jit import source_root

GOLDEN = source_root() / "tests" / "data" / "rl" / "environments" / "hyperdrone" / "python_golden"

# the generator's World runs on the host; a CUDA World draws from a different RNG engine
CONFIG = Config(num_worlds=1, instances_per_world=2, width=16, height=16, fidelity="low", anti_aliasing_grid=1, device="cpu")


def observe_pair(env):
    return env.observe(), env.observe_privileged()


def golden_actions(step, total, action_dim):
    actions = np.empty((total, action_dim), dtype=np.float32)
    for instance in range(total):
        for dim in range(action_dim):
            actions[instance, dim] = np.float32(-1.0 + 2.0 * (((step * 131 + instance * 31 + dim * 7) % 101) / 100.0))
    return actions


def layout_from_json(layout):
    return ObservationLayout(shape=tuple(layout["shape"]), axis=layout["axis"], blocks={name: tuple(block) for name, block in layout["blocks"].items()})


@pytest.fixture(scope="module")
def scene_directory(tmp_path_factory):
    if not (GOLDEN / "manifest.json").exists():
        pytest.skip(f"no golden rollout at {GOLDEN}")
    manifest = json.loads((GOLDEN / "manifest.json").read_text())
    source = Path(os.environ.get("HYPERDRONE_TEST_SCENE_DIR", source_root() / "tests" / "data")) / manifest["scene"]
    if not source.exists():
        pytest.skip(f"no golden scene at {source}")
    directory = tmp_path_factory.mktemp("golden_scene")
    (directory / source.name).symlink_to(source)
    return directory


def test_golden_rollout(scene_directory):
    manifest = json.loads((GOLDEN / "manifest.json").read_text())
    scalar_defines = {name: value for name, value in CONFIG.defines().items() if name != "HYPERLUX_CAMERA_BACKEND_REQUIREMENT"}
    manifest["defines"].setdefault("HYPERDRONE_ENVIRONMENT_AA_GRID", 1)
    assert scalar_defines == {name: manifest["defines"][name] for name in scalar_defines}, "the test configuration drifted from the golden's — regenerate deliberately"
    env = HyperDrone(scene_directory, config=CONFIG, seed=manifest["seed"])
    try:
        assert env.observation_layout == layout_from_json(manifest["observation_layout"])
        assert env.observation_layout_privileged == layout_from_json(manifest["observation_layout_privileged"])
        total, steps = env.total_instances, manifest["steps"]
        assert total == manifest["total_instances"]

        rewards, terminated = [], []
        mask = np.ones(total, dtype=np.uint8)
        env.reset(mask)
        observations = [observe_pair(env)]
        for step in range(steps):
            env.step(golden_actions(step, total, env.action_dim))
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
    finally:
        env.close()
