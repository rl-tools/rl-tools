import os
from pathlib import Path

import numpy as np
import pytest

from hyperdrone.env import EnvConfig, MultiEnvironment
from hyperdrone.jit import source_root


@pytest.fixture(scope="module")
def scene_directory(tmp_path_factory):
    source = Path(os.environ.get("HYPERDRONE_TEST_SCENE_DIR", source_root() / "tests" / "data")) / "ProcTHOR-Train-1.glb"
    if not source.exists():
        pytest.skip(f"no ProcTHOR test scene at {source}")
    directory = tmp_path_factory.mktemp("scenes")
    (directory / source.name).symlink_to(source)
    return directory


def _rollout_step(env, actions):
    env.begin_step()
    flags_begin = env.episode_flags()
    env.render(flags_begin["reset"])
    env.step(actions)
    env.end_step()
    return flags_begin, env.episode_flags()


def test_episode_bookkeeping(scene_directory):
    config = EnvConfig(num_environments=1, instances=2, cam_width=16, cam_height=16)
    env = MultiEnvironment(scene_directory, config=config, seed=3)
    try:
        step_limit = 3
        env.set_step_limit(step_limit)
        actions = np.zeros((env.total_instances, env.action_dim), dtype=np.float32)
        previous_truncated = np.ones(env.total_instances, dtype=bool)  # every instance starts due
        truncations = 0
        for step_i in range(8):
            flags_begin, flags_end = _rollout_step(env, actions)
            # the applied reset mask is the previous step's truncation
            assert np.array_equal(flags_begin["reset"], previous_truncated)
            assert np.all(flags_begin["episode_step"][flags_begin["reset"]] == 0)
            # terminated implies truncated; a time limit truncates exactly at the limit
            assert np.all(flags_end["truncated"] | ~flags_end["terminated"])
            time_limit = flags_end["truncated"] & ~flags_end["terminated"]
            assert np.array_equal(time_limit, flags_end["episode_step"] == step_limit)
            assert np.all(flags_end["end_reason"][time_limit] == MultiEnvironment.END_REASON_TIME_LIMIT)
            previous_truncated = flags_end["truncated"]
            truncations += int(flags_end["truncated"].sum())
        assert truncations >= 2 * env.total_instances

        # forced reset of a single instance applies at the next begin_step
        env.set_step_limit(0)
        _rollout_step(env, actions)
        _rollout_step(env, actions)
        mask = np.zeros(env.total_instances, dtype=np.uint8)
        mask[0] = 1
        env.force_reset(mask)
        flags_begin, _ = _rollout_step(env, actions)
        assert flags_begin["reset"][0]
    finally:
        env.close()
