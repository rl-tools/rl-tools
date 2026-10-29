import hyperlux
import copy
import json
from contextlib import closing
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import conta
import hyperdrone
from hyperdrone import Config, HyperDrone
from hyperlux.camera import CameraSpec, RGB, Flow, FlowStatus
from hyperdrone.camera import ImageObservation
from hyperlux._jit import source_root


def float32(value):
    """The binding stores float32 and prints it with 9 significant digits; compare at that precision."""
    if isinstance(value, dict):
        return {key: float32(item) for key, item in value.items()}
    if isinstance(value, list):
        return [float32(item) for item in value]
    if isinstance(value, float):
        return float(np.float32(value))
    return value


def pinned_parameters(env, drone, pose):
    """A replayed agent flies its drone-table nominal with the pose pinned as the initialization center."""
    nominal = copy.deepcopy(env.drones[drone]["parameters"])
    nominal["dynamics"]["mdp"]["init"].update(position=list(pose[:3]), orientation=list(pose[3:]), max_position=0, max_angle=0)
    return float32({**nominal, "drone": drone})


@pytest.fixture(scope="module")
def scene_directory(tmp_path_factory):
    # a directory holding exactly the scenes the environment should schedule over
    source = Path(os.environ.get("HYPERDRONE_TEST_SCENE_DIR", source_root() / "tests" / "data")) / "ProcTHOR-Train-1.glb"
    if not source.exists():
        pytest.skip(f"no ProcTHOR test scene at {source}")
    directory = tmp_path_factory.mktemp("scenes")
    (directory / source.name).symlink_to(source)
    return directory


CONFIG = Config(num_worlds=1, instances_per_world=2, width=32, height=32, fidelity="low")


@pytest.mark.parametrize("name", ["num_worlds", "instances_per_world", "width", "height", "num_agents", "num_camera_agents"])
@pytest.mark.parametrize("value", [0, -1, 2.5, True])
def test_positive_integer_configuration(name, value):
    with pytest.raises(ValueError, match=f"{name} must be a positive integer"):
        Config(**{name: value})


def test_camera_agent_configuration():
    config = Config(preset="x500_fpv", num_agents=2)
    assert config.num_camera_agents == 1
    assert config.defines()["HYPERDRONE_ENVIRONMENT_N_CAMERA_AGENTS"] == 1
    assert config.key() == replace(config, num_camera_agents=1).key()
    assert config.key() != replace(config, num_camera_agents=2).key()
    with pytest.raises(ValueError, match="num_camera_agents must not exceed num_agents"):
        replace(config, num_camera_agents=3)
    with pytest.raises(ValueError, match="spec_header supersedes.*num_camera_agents"):
        replace(config, num_camera_agents=2, spec_header=__file__)


@pytest.mark.parametrize("value", [0, 5, 2.0, True, "2", None])
def test_invalid_anti_aliasing_grid(value):
    with pytest.raises(ValueError, match="anti_aliasing_grid"):
        Config(anti_aliasing_grid=value)


def test_render_quality_defaults_and_overrides(scene_directory):
    config = Config(instances_per_world=2, width=32, height=32, device="cpu")
    assert config.key() == replace(config, fidelity="veryhigh", anti_aliasing_grid=2).key()
    assert len({replace(config, fidelity=fidelity).key() for fidelity in hyperlux.FIDELITY}) == 4
    aliased = replace(config, anti_aliasing_grid=1)
    assert config.key() != aliased.key()
    frames, states = [], []
    for settings in (config, aliased):
        with closing(HyperDrone(scene_directory, config=settings, seed=7)) as env:
            env.reset()
            env.observe()
            frames.append(env.frames())
            states.append(env.observe_privileged())
            assert frames[-1].shape == (2, 32, 32, 3)
            assert np.isfinite(frames[-1]).all() and frames[-1].std() > 0
    np.testing.assert_array_equal(states[0], states[1])
    assert not np.array_equal(frames[0], frames[1])


@pytest.mark.parametrize("task", ["target_frame", "moving_gate", "visual_inertial_localization"])
def test_tasks_reject_multiple_agents(task):
    with pytest.raises(ValueError, match="single-agent"):
        Config(task=task, preset="x500_fpv_imu", num_agents=2)


def test_reset_guard_and_repeated_acquisition(scene_directory):
    env = HyperDrone(scene_directory, config=CONFIG)
    try:
        actions = np.zeros((env.total_instances, env.action_dim), dtype=np.float32)
        verbs = [env.observe, env.observe_privileged, env.instance_parameters, env.spawn_points, lambda: env.step(actions)]
        for verb in verbs:
            with pytest.raises(RuntimeError, match="reset every instance"):
                verb()
        env.reset([True, False])
        with pytest.raises(RuntimeError, match="reset every instance"):
            env.observe()
        env.reset([False, True])
        first = env.observe()
        env.observe()
        np.testing.assert_array_equal(env.observe(), first)
        env.step(actions)
        second = env.observe()
        env.observe()
        np.testing.assert_array_equal(env.observe(), second)
    finally:
        env.close()


def test_privileged_reads_do_not_change_trajectory(scene_directory):
    parameters = {"dynamics": {"mdp": {"observation_noise": {"position": 0.05, "orientation": 0.05, "linear_velocity": 0.05, "angular_velocity": 0.05}, "action_noise": {"normalized_rpm": 0.05}}}}
    records = []
    for reads in (0, 1, 3):
        env = HyperDrone(scene_directory, config=CONFIG, seed=987, parameters=parameters)
        try:
            env.reset()
            actions = np.zeros((env.total_instances, env.action_dim), dtype=np.float32)
            for _ in range(4):
                for _ in range(reads):
                    env.observe_privileged()
                env.step(actions)
            records.append(env.observe_privileged())
        finally:
            env.close()
    for record in records[1:]:
        np.testing.assert_array_equal(record, records[0])


def test_environment_verbosity(scene_directory, capfd, monkeypatch):
    previous = hyperlux.verbosity()
    try:
        for level in ("warning", "debug"):
            hyperlux.set_verbosity(level)
            environment = HyperDrone(scene_directory, config=CONFIG, seed=0)
            try:
                backend = environment.render_backend
                with monkeypatch.context() as context:
                    context.setenv("HYPERLUX_BACKEND", "INVALID")
                    assert environment.render_backend == backend
                output = capfd.readouterr().err
                if level == "warning":
                    assert "#rl_tools::hyperlux::" not in output
                else:
                    assert f"#rl_tools::hyperlux: backend={backend}" in output
                    assert "#rl_tools::hyperlux::datasets: Loaded" in output
            finally:
                environment.close()
                capfd.readouterr()
    finally:
        hyperlux.set_verbosity(previous)


def rollout(scene_directory, seed, steps=3):
    env = HyperDrone(scene_directory, config=CONFIG, seed=seed)
    try:
        assert env.total_instances == 2
        assert env.action_dim == 4
        assert env.observation_dim == 32 * 32 * 3
        mask = np.ones(env.total_instances, dtype=np.uint8)
        env.reset(mask)
        env.observe()
        record = [env.observe().copy()]
        rng = np.random.default_rng(seed)
        no_reset = np.zeros(env.total_instances, dtype=np.uint8)
        for _ in range(steps):
            actions = rng.uniform(-1, 1, size=(env.total_instances, env.action_dim)).astype(np.float32)
            env.step(actions)
            env.observe()
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


def test_configurations_keep_independent_state(scene_directory):
    small = HyperDrone(scene_directory, config=Config(instances_per_world=2, width=16, height=16, fidelity="low"), seed=5)
    large = HyperDrone(scene_directory, config=CONFIG, seed=5)
    try:
        assert type(small._environment) is not type(large._environment)
        for environment in (small, large):
            environment.reset()
            environment.observe()
        assert small.frames().shape == (2, 16, 16, 3)
        assert large.frames().shape == (2, 32, 32, 3)
        state = large.observe_privileged()
        small.step(np.zeros((2, small.action_dim), dtype=np.float32))
        np.testing.assert_array_equal(large.observe_privileged(), state)
        small.close()
        large.step(np.zeros((2, large.action_dim), dtype=np.float32))
        assert not np.array_equal(large.observe_privileged(), state)
    finally:
        large.close()


def test_motion_blur_shutter_reset_and_determinism(scene_directory):
    config = replace(CONFIG, motion_blur_samples=8)
    assert config.key() != CONFIG.key()
    records = []
    for _ in range(2):
        env = HyperDrone(scene_directory, config=config, seed=5)
        try:
            env.reset()
            env.observe()
            for _ in range(10):
                env.step(np.zeros((env.total_instances, env.action_dim), dtype=np.float32))
            env.observe()
            blurred = env.observe()
            state = env.observe_privileged()
            env.reset(spawn_points=env.spawn_points())
            sharp = env.observe()
            np.testing.assert_array_equal(env.observe_privileged()[:, :12], state[:, :12])
            assert np.isfinite(blurred).all()
            assert not np.array_equal(blurred, sharp)
            records.append((blurred, sharp))
        finally:
            env.close()
    for first, second in zip(*records):
        np.testing.assert_array_equal(first, second)


def test_native_binding_rejects_invalid_buffers(scene_directory):
    environment = HyperDrone(scene_directory, config=CONFIG)
    try:
        with pytest.raises(ValueError, match="buffer size"):
            environment._environment.reset(np.ones(1, dtype=bool))
        with pytest.raises(ValueError, match="buffer size"):
            environment._environment.step(np.zeros(1, dtype=np.float32))
        with pytest.raises(TypeError):
            environment._environment.observe(np.empty((2, environment.observation_dim), dtype=np.float64))
        environment.reset()
        environment.observe()
        assert np.isfinite(environment.observe()).all()
    finally:
        environment.close()


def test_frames_and_rotate_scene(scene_directory):
    env = HyperDrone(scene_directory, config=CONFIG, seed=7)
    try:
        mask = np.ones(env.total_instances, dtype=np.uint8)
        env.reset(mask)
        env.observe()
        frames = env.frames()
        assert frames.shape == (2, 32, 32, 3)
        env.rotate_scene()
        env.reset(mask)
        env.observe()
        assert env.frames().shape == (2, 32, 32, 3)
    finally:
        env.close()


def nominal_rollout(scene_directory, parameters=None):
    env = HyperDrone(scene_directory, config=CONFIG, seed=3, parameters=parameters)
    try:
        mask = np.ones(env.total_instances, dtype=np.uint8)
        env.reset(mask)
        env.observe()
        frames = env.observe().copy()
        hold = np.zeros((env.total_instances, env.action_dim), dtype=np.float32)
        for _ in range(10):
            env.step(hold)
        states = env.observe_privileged().copy()
        env.reset(mask)
        return env.parameters, frames, states
    finally:
        env.close()


def test_nominal_parameters(scene_directory):
    default_parameters, default_frames, default_states = nominal_rollout(scene_directory)
    # an empty override still round-trips every parameter through JSON, which must be lossless
    explicit_parameters, explicit_frames, explicit_states = nominal_rollout(scene_directory, {})
    assert explicit_parameters == default_parameters
    np.testing.assert_array_equal(explicit_frames, default_frames)
    np.testing.assert_array_equal(explicit_states, default_states)

    x500_parameters = nominal_rollout(scene_directory, {"dynamics": {"dynamics": {"model": "x500"}}})[0]
    assert x500_parameters["dynamics"]["dynamics"]["mass"] != default_parameters["dynamics"]["dynamics"]["mass"]
    J = np.diag([0.02, 0.03, 0.04])
    pitch = 0.3
    camera = {"fov": 90, "mount": {"pose": {"position": [0.1, 0, 0.05], "orientation": [np.cos(pitch / 2), 0, np.sin(pitch / 2), 0]}}}
    parameters, frames, states = nominal_rollout(scene_directory, {"dynamics": {"dynamics": {"model": "x500", "mass": 2.0, "J": J}, "integration": {"dt": 0.02}}, "cameras": {"front": camera}})
    dynamics = parameters["dynamics"]["dynamics"]
    assert dynamics["mass"] == pytest.approx(2.0)
    np.testing.assert_allclose(dynamics["J"], J)
    np.testing.assert_allclose(dynamics["J_inv"], np.linalg.inv(J), rtol=1e-6)
    overridden = ("mass", "J", "J_inv")
    assert {key: value for key, value in dynamics.items() if key not in overridden} == {key: value for key, value in x500_parameters["dynamics"]["dynamics"].items() if key not in overridden}
    assert parameters["dynamics"]["integration"]["dt"] == pytest.approx(0.02)
    assert parameters["cameras"]["front"]["fov"] == 90
    np.testing.assert_allclose(parameters["cameras"]["front"]["mount"]["pose"]["orientation"], camera["mount"]["pose"]["orientation"], rtol=1e-6)
    assert not np.array_equal(frames, default_frames)
    assert not np.array_equal(states, default_states)


def test_nominal_parameter_validation(scene_directory):
    with pytest.raises(ValueError, match="unknown dynamics model"):
        HyperDrone(scene_directory, config=CONFIG, parameters={"dynamics": {"dynamics": {"model": "zeppelin"}}})
    with pytest.raises(ValueError, match="unknown parameter"):
        HyperDrone(scene_directory, config=CONFIG, parameters={"dynamics": {"dynamics": {"weight": 1.0}}})
    with pytest.raises(ValueError, match="must have shape"):
        HyperDrone(scene_directory, config=CONFIG, parameters={"cameras": {"front": {"mount": {"pose": {"orientation": [1, 0, 0]}}}}})


def test_instance_parameters(scene_directory):
    env = HyperDrone(scene_directory, config=CONFIG, seed=3)
    reference = HyperDrone(scene_directory, config=CONFIG, seed=3)
    try:
        all_instances = np.ones(env.total_instances, dtype=np.uint8)
        no_reset = np.zeros(env.total_instances, dtype=np.uint8)
        for e in (env, reference):
            e.reset(all_instances)
            e.observe()
        # the field-of-view check below needs instance 0 to look at structure rather than a
        # blank wall; redraw the placement (both environments stay in lockstep) until it does
        for _ in range(16):
            if env.observe()[0].std() > 0.05:
                break
            for e in (env, reference):
                e.reset(all_instances)
                e.observe()
        else:
            pytest.skip("no structured view for instance 0 within 16 placements")
        nominal = env.parameters
        flying_nominal = {**nominal, "drone": 0}
        current = env.instance_parameters()
        assert current == [flying_nominal] * env.total_instances
        mass = nominal["dynamics"]["dynamics"]["mass"]
        env.set_instance_parameters({"dynamics": {"dynamics": {"mass": 3 * mass}}, "cameras": {"front": {"fov": 30}}}, mask=np.array([1, 0], dtype=np.uint8))
        updated = env.instance_parameters()
        assert updated[1] == flying_nominal
        assert updated[0]["dynamics"]["dynamics"]["mass"] == pytest.approx(3 * mass)
        assert updated[0]["cameras"]["front"]["fov"] == 30

        frames, states = [], []
        for e in (env, reference):
            e.observe()
            frames.append(e.observe().copy())
            hold = np.zeros((e.total_instances, e.action_dim), dtype=np.float32)
            for _ in range(10):
                e.step(hold)
            states.append(e.observe_privileged().copy())
        np.testing.assert_array_equal(frames[0][1], frames[1][1])
        assert not np.array_equal(frames[0][0], frames[1][0])
        np.testing.assert_array_equal(states[0][1], states[1][1])
        assert not np.array_equal(states[0][0], states[1][0])

        env.reset(all_instances)
        assert env.instance_parameters() == [flying_nominal] * env.total_instances
        env.set_instance_parameters([{"cameras": {"front": {"fov": 50}}}, {"dynamics": {"dynamics": {"model": "x500"}}}])
        updated = env.instance_parameters()
        assert updated[0]["cameras"]["front"]["fov"] == 50
        assert updated[1]["dynamics"]["dynamics"]["mass"] != mass
        with pytest.raises(ValueError, match="shared by all instances"):
            env.set_instance_parameters({"dynamics": {"integration": {"dt": 0.1}}})
        with pytest.raises(ValueError, match="overrides for"):
            env.set_instance_parameters([{}])
    finally:
        env.close()
        reference.close()


def test_gym_autoreset_observations_match_explicit_render(scene_directory, monkeypatch):
    pytest.importorskip("gymnasium")
    from hyperdrone.gym import VectorEnv

    env = VectorEnv(scene_directory, config=CONFIG, seed=7)
    reference = HyperDrone(scene_directory, config=CONFIG, seed=7)
    try:
        monkeypatch.setattr(env._env, "episode_step_limit", 1)
        reset_mask = np.ones(reference.total_instances, dtype=np.uint8)
        reference.reset()
        reference.observe()
        observations, _ = env.reset()
        np.testing.assert_array_equal(observations, reference.observe())

        actions = np.zeros((env.num_envs, reference.action_dim), dtype=np.float32)
        reference.step(actions)
        expected_rewards, expected_terminated = reference.rewards(), reference.terminated()
        reference.reset()
        reference.observe()

        observations, rewards, terminated, truncated, _ = env.step(actions)
        np.testing.assert_array_equal(rewards, expected_rewards)
        np.testing.assert_array_equal(terminated, expected_terminated)
        np.testing.assert_array_equal(truncated, ~expected_terminated)
        np.testing.assert_array_equal(observations, reference.observe())
        np.testing.assert_array_equal(observations, env._env.observe())
    finally:
        reference.close()
        env.close()


def test_config_validation_errors():
    for samples in (0, -1, 3, 2.5, True):
        with pytest.raises(ValueError, match="motion_blur_samples"):
            Config(motion_blur_samples=samples)
    with pytest.raises(ValueError, match="unknown preset"):
        Config(preset="nonexistent")
    for cameras in ((), ("front",), (CameraSpec("front", (RGB("float32"),)), CameraSpec("front", (RGB("float32"),)))):
        with pytest.raises(ValueError):
            Config(cameras=cameras)
    profiles = (CameraSpec("front", (RGB("float32"),)), CameraSpec("chase", (RGB("float32"),)))
    with pytest.raises(ValueError, match="one observation camera"):
        Config(task="target_frame", cameras=profiles, observation=ImageObservation(("front", "chase")))
    Config(task="target_frame", cameras=profiles, observation=ImageObservation(("front",)))
    with pytest.raises(ValueError, match="spec_header supersedes"):
        Config(cameras=profiles, spec_header=__file__)
    with pytest.raises(ValueError, match="unknown task"):
        Config(task="nonexistent")
    with pytest.raises(ValueError, match="SELF_VISIBLE"):
        Config(num_agents=2)
    with pytest.raises(ValueError, match="spec_header supersedes"):
        Config(task="target_frame", spec_header=__file__)
    with pytest.raises(ValueError, match="unknown device"):
        Config(device="tpu")
    with pytest.raises(ValueError, match="single-agent"):
        Config(preset="x500_fpv", task="target_frame", num_agents=2)
    with pytest.raises(ValueError, match="single-agent"):
        Config(preset="x500_fpv", task="moving_gate", num_agents=2)
    with pytest.raises(ValueError, match="no CUDA verbs"):
        Config(preset="x500_fpv_imu", task="visual_inertial_localization", device="cuda")
    assert Config(spec_header=__file__).world_device == "cpu"
    with pytest.raises(ValueError, match="asset per drone"):
        HyperDrone("/nonexistent", config=Config(preset="x500_fpv", num_agents=1))


def test_observation_layout(scene_directory):
    env = HyperDrone(scene_directory, config=CONFIG, seed=0)
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


@pytest.mark.parametrize("num_agents", [1, 2])
def test_position_commands_use_scene_frame_for_every_agent(scene_directory, num_agents):
    config = Config(instances_per_world=2, width=16, height=16, fidelity="low",
                    control="position", preset="crazyflie" if num_agents == 1 else "x500_fpv",
                    num_agents=num_agents, device="cpu")
    drones = None if num_agents == 1 else [{"asset": drone_asset_path(), "parameters": {}}]
    parameters = {"dynamics": {"mdp": {"init": {
        "max_linear_velocity": 0, "max_angular_velocity": 0, "min_rpm": 0, "max_rpm": 0,
    }}}}
    with closing(HyperDrone(scene_directory, config=config, seed=29, drones=drones, parameters=parameters)) as env:
        env.reset()
        points = env.spawn_points()
        anchor = np.asarray(points[0]["scene_translation"], dtype=np.float32)
        local = np.array([[0.1, 0.15, 0.2, 0.15], [-0.2, 0.1, 0.15, -0.2]], dtype=np.float32)[:num_agents]
        commands = np.empty((2, num_agents, 4), dtype=np.float32)
        for index, point in enumerate(points):
            translation = anchor + np.array([0.2 * index, -0.1 * index, 0], dtype=np.float32)
            point["scene_translation"] = translation.tolist()
            point["scene_yaw"] = [1, 0] if index == 0 else [0, 1]
            point["poses"] = [[0, 0, 0, 1, 0, 0, 0] for _ in range(num_agents)]
            point["drone"] = [0] * num_agents
            rotation = np.eye(3) if index == 0 else np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
            commands[index, :, :3] = local[:, :3] @ rotation.T + translation
            commands[index, :, 3] = local[:, 3] + index * np.pi / 2
        env.reset(spawn_points=points)
        assert env.control == "position"
        assert env.action_dim == 4 * num_agents
        assert np.isneginf(env.action_low).all() and np.isposinf(env.action_high).all()
        initial = env.observe_privileged()
        for _ in range(6):
            env.step(commands.reshape(2, -1))
            state = env.observe_privileged()
            np.testing.assert_allclose(state[0], state[1], atol=2e-4, rtol=2e-5)
        assert not np.array_equal(state, initial)
        env.reset([True, False], spawn_points=[points[0]])
        np.testing.assert_array_equal(env.observe_privileged()[1], state[1])


def test_ctbr_commands_reach_every_agent_current_step(scene_directory):
    config = Config(instances_per_world=2, num_agents=2, preset="x500_fpv_imu",
                    width=16, height=16, fidelity="low", control="ctbr", actuation="ideal", device="cpu")
    with closing(HyperDrone(scene_directory, config=config, drone_asset=drone_asset_path(), seed=31)) as env:
        env.reset()
        assert env.control == "ctbr" and env.actuation == "ideal"
        assert env.action_dim == 8
        np.testing.assert_array_equal(env.action_low, -np.ones(8))
        np.testing.assert_array_equal(env.action_high, np.ones(8))
        commands = np.empty((2, 2, 4), dtype=np.float32)
        rate_limit = np.empty((2, 2, 3), dtype=np.float32)
        for instance, parameters in enumerate(env.instance_parameters()):
            for agent, values in enumerate(parameters["agents"]):
                ideal = values["dynamics"]["ideal_rates"]
                commands[instance, agent, 0] = 2 / ideal["thrust_to_weight"] - 1
                rate_limit[instance, agent] = ideal["rate_limit"]
        commands[:, :, 1:] = [[0.02, -0.025, 0.03], [-0.035, 0.01, -0.015]]
        for _ in range(2):
            env.step(commands.reshape(2, -1))
            observation = env.observe_privileged()
            for agent in range(2):
                offset, width = env.observation_layout_privileged.blocks[f"agent{agent}/angular_velocity"]
                np.testing.assert_allclose(observation[:, offset:offset + width], commands[:, agent, 1:] * rate_limit[:, agent], rtol=1e-5, atol=1e-6)
            if env.observation_dim_imu:
                assert np.isfinite(env.observe_imu()).all()
            commands[:, :, 1:] *= -1
        before = env.instance_parameters()
        env.set_instance_parameters({"agents": [{"dynamics": {"ideal_rates": {"thrust_to_weight": 3.0}}}, {}]}, mask=[True, False])
        after = env.instance_parameters()
        assert after[0]["agents"][0]["dynamics"]["ideal_rates"]["thrust_to_weight"] == 3.0
        assert after[0]["agents"][1] == before[0]["agents"][1]
        assert after[1] == before[1]


def test_position_parameter_mutation_resets_only_selected_instances(scene_directory):
    config = Config(instances_per_world=2, width=16, height=16, fidelity="low",
                    control="position", device="cpu")
    parameters = {"dynamics": {"mdp": {"init": {
        "max_linear_velocity": 0, "max_angular_velocity": 0, "min_rpm": 0, "max_rpm": 0,
    }}}}
    with closing(HyperDrone(scene_directory, config=config, seed=37, parameters=parameters)) as env:
        env.reset()
        points = env.spawn_points()
        commands = np.empty((2, 4), dtype=np.float32)
        for instance, point in enumerate(points):
            point["poses"] = [[0, 0, 0, 1, 0, 0, 0]]
            point["scene_yaw"] = [1, 0]
            commands[instance, :3] = np.asarray(point["scene_translation"]) + [0.2, -0.1, 0.3]
            commands[instance, 3] = 0.2
        mass = env.parameters["dynamics"]["dynamics"]["mass"]
        records = []
        for mask in ([True, False], [True, True], [False, False]):
            env.reset(spawn_points=points)
            for _ in range(12):
                env.step(commands)
            env.set_instance_parameters({"dynamics": {"dynamics": {"mass": mass}}}, mask)
            for _ in range(3):
                env.step(commands)
            records.append(env.observe_privileged())
        np.testing.assert_array_equal(records[0][0], records[1][0])
        np.testing.assert_array_equal(records[0][1], records[2][1])
        assert not np.array_equal(records[0][0], records[2][0])


def rigged_asset_path(name):
    source = Path(os.environ.get("HYPERDRONE_TEST_SCENE_DIR", source_root() / "tests" / "data")) / f"{name}.glb"
    if not source.exists():
        pytest.skip(f"no drone asset at {source}")
    return source


def drone_asset_path():
    return rigged_asset_path("x500")


CONFIG_DRONES = Config(instances_per_world=4, width=16, height=16, preset="x500_fpv")
CONFIG_MULTI_AGENT = Config(instances_per_world=2, num_agents=7, num_camera_agents=7, width=16, height=16, fidelity="low", preset="x500_fpv")


@pytest.mark.parametrize("camera_agents", [1, 2])
def test_multi_camera_model_parameters_and_replay(scene_directory, camera_agents):
    config = replace(CONFIG_MULTI_AGENT, num_worlds=2, num_agents=2, num_camera_agents=camera_agents, cameras=(CameraSpec("front", (RGB("float32"),)), CameraSpec("chase", (RGB("float32"),))), observation=ImageObservation(("front", "chase")))
    drones = [{"asset": rigged_asset_path(name), "parameters": {
        "dynamics": {"dynamics": {"model": name}},
        "cameras": {"front": {"fov": fov}, "chase": {"fov": fov + 10, "mount": {"pose": {"position": [-1, 0, 0.5]}}}},
    }} for name, fov in (("x500", 60), ("crazyflie", 80))]
    initial = {"dynamics": {"mdp": {"init": {"max_linear_velocity": 0, "max_angular_velocity": 0, "min_rpm": 0, "max_rpm": 0}}}}
    env = HyperDrone([scene_directory / "ProcTHOR-Train-1.glb"] * 2, config=config, drones=drones, parameters=initial, seed=7)
    try:
        assert (env.num_agents, env.num_camera_agents, env.num_cameras_per_agent, env.num_views) == (2, camera_agents, 2, camera_agents * 2)
        assert env.action_dim == 8
        env.reset()
        points = env.spawn_points()
        for index, point in enumerate(points):
            point["drone"] = [index % 2, (index + 1) % 2]
        env.reset(spawn_points=points)
        baseline = env.instance_parameters()
        for instance, point in zip(baseline, points):
            for agent, drone, pose in zip(instance["agents"], point["drone"], point["poses"]):
                assert float32(agent) == pinned_parameters(env, drone, pose)
        frames = env.frames().copy()
        assert frames.shape == (4, camera_agents * 2 * 16, 16, 3)
        views = frames.reshape(4, camera_agents, 2, 16, 16, 3)
        assert all(not np.array_equal(agent[0], agent[1]) for instance in views for agent in instance)
        mask = np.array([False, False, False, True])
        env.set_instance_parameters({"agents": [{}, {"drone": 0, "cameras": {"front": {}, "chase": {"fov": 40}}}]}, mask)
        updated = env.instance_parameters()
        assert updated[:3] == baseline[:3]
        assert updated[3]["agents"][0] == baseline[3]["agents"][0]
        assert updated[3]["agents"][1]["cameras"]["front"] == env.drones[0]["parameters"]["cameras"]["front"]
        assert updated[3]["agents"][1]["cameras"]["chase"]["fov"] == 40
        valid = {"agents": [{"cameras": {"front": {"fov": 25}, "chase": {}}}, {}]}
        for cameras in ([{}], {"missing": {}}, {"chase": None}, {"chase": {"mount": {"pose": {"position": [0, 0]}}}}):
            overrides = [valid, {"agents": [{}, {}]}, {"agents": [{}, {}]}, {"agents": [{}, {"cameras": cameras}]}]
            with pytest.raises(ValueError):
                env.set_instance_parameters(overrides)
            assert env.instance_parameters() == updated
        with pytest.raises(ValueError, match="unknown parameter"):
            env.set_instance_parameters({"agents": [{"camera": {}}, {}]})
        native = env._indexed_cameras(copy.deepcopy(updated))
        native[0]["agents"][0]["cameras"][0]["fov"] = 25
        native[-1]["agents"][-1]["cameras"].pop()
        with pytest.raises(ValueError, match="per camera"):
            env._environment.set_instance_parameters(np.ones(env.total_instances, dtype=bool), json.dumps(native))
        assert env.instance_parameters() == updated
        for _ in range(3):
            env.step(np.zeros((env.total_instances, env.action_dim), dtype=np.float32))
        state = env.observe_privileged().copy()
        assert state.shape == (4, 36)
        env.reset(mask, spawn_points=[points[3]])
        np.testing.assert_array_equal(env.observe_privileged()[:3], state[:3])
        assert env.instance_parameters()[3] == baseline[3]
        np.testing.assert_array_equal(env.frames()[3], frames[3])
        env.reset(spawn_points=points)
        assert env.instance_parameters() == baseline
        np.testing.assert_array_equal(env.frames(), frames)
    finally:
        env.close()


def mixed_drones():
    return [{"asset": rigged_asset_path(name), "parameters": {"dynamics": {"dynamics": {"model": name}}, "cameras": {"front": {"fov": fov}}}}
            for name, fov in (("x500", 90), ("crazyflie", 70))]


def test_multi_agent_parameters(scene_directory):
    env = HyperDrone(scene_directory, config=CONFIG_MULTI_AGENT, seed=5, drones=mixed_drones())
    try:
        env.reset()
        table = env.drones
        sampled = env.instance_parameters()
        assert all(set(instance) == {"agents"} and len(instance["agents"]) == env.num_agents for instance in sampled)
        assert any(len({agent["drone"] for agent in instance["agents"]}) > 1 for instance in sampled)
        for instance in sampled:
            for agent in instance["agents"]:
                assert agent == {**table[agent["drone"]]["parameters"], "drone": agent["drone"]}
        choices = [index % len(table) for index in range(env.num_agents)]
        env.set_instance_parameters({"agents": [{"drone": drone} for drone in choices]}, mask=[True, False])
        selected = env.instance_parameters()
        assert selected[1] == sampled[1]
        assert selected[0]["agents"] == [{**table[drone]["parameters"], "drone": drone} for drone in choices]
        overrides = [{"agents": [{"cameras": {"front": {"fov": 45}}}] + [{}] * (env.num_agents - 1)},
                     {"agents": [{}] * env.num_agents}]
        env.set_instance_parameters(overrides)
        changed = env.instance_parameters()
        assert changed[0]["agents"][0]["cameras"]["front"]["fov"] == 45
        assert changed[0]["agents"][1:] == selected[0]["agents"][1:]
        assert changed[1] == selected[1]
        valid = {"agents": [{"cameras": {"front": {"fov": 30}}}] + [{}] * (env.num_agents - 1)}
        for invalid in ({"drone": 0}, {"agents": [{}]}, {"agents": [{}] * (env.num_agents - 1) + [None]},
                        {"agents": [{}] * (env.num_agents - 1) + [{"drone": len(table)}]},
                        {"agents": [{}] * (env.num_agents - 1) + [{"dynamics": {"integration": {"dt": 0.1}}}]}):
            with pytest.raises(ValueError):
                env.set_instance_parameters([valid, invalid])
            assert env.instance_parameters() == changed
        native_updates = env._indexed_cameras(copy.deepcopy(changed))
        native_updates[0]["agents"][0]["cameras"][0]["fov"] = 30
        native_updates[1]["agents"][-1]["dynamics"]["integration"]["dt"] *= 2
        with pytest.raises(ValueError, match="same integration timestep"):
            env._environment.set_instance_parameters(np.ones(env.total_instances, dtype=bool), json.dumps(native_updates))
        assert env.instance_parameters() == changed
        states = env.observe_privileged()
        assert states.shape == (env.total_instances, env.num_agents * 18)
        assert env.action_dim == env.num_agents * 4
        assert env.frames().shape == (env.total_instances, env.num_agents * env.height, env.width, 3)
        env.step(np.zeros((env.total_instances, env.action_dim), dtype=np.float32))
        assert np.isfinite(env.observe_privileged()).all()
        assert np.isfinite(env.rewards()).all()
    finally:
        env.close()


@pytest.mark.parametrize("motion_blur_samples", [1, 4])
def test_multi_agent_spawn_replay(scene_directory, motion_blur_samples):
    parameters = {"dynamics": {"mdp": {"init": {"max_linear_velocity": 0, "max_angular_velocity": 0, "min_rpm": 0, "max_rpm": 0}}}}
    reference = None
    for seed in (1, 123):
        env = HyperDrone(scene_directory, config=replace(CONFIG_MULTI_AGENT, motion_blur_samples=motion_blur_samples), seed=seed, drones=mixed_drones(), parameters=parameters)
        try:
            if reference is None:
                env.reset()
                points = env.spawn_points()
                for point in points:
                    assert "drones" not in point
                    point["drone"] = [index % 2 for index in range(env.num_agents)]
            env.reset(spawn_points=points)
            assert env.spawn_points() == points
            for instance, point in zip(env.instance_parameters(), points):
                assert [agent["drone"] for agent in instance["agents"]] == point["drone"]
                for agent, pose in zip(instance["agents"], point["poses"]):
                    assert float32(agent) == pinned_parameters(env, agent["drone"], pose)
            initial = env.frames().copy()
            record = [initial, env.observe_privileged()]
            for _ in range(3):
                env.step(np.zeros((env.total_instances, env.action_dim), dtype=np.float32))
                record.extend((env.frames().copy(), env.observe_privileged()))
            if reference is None:
                reference = record
            else:
                for actual, expected in zip(record, reference):
                    np.testing.assert_array_equal(actual, expected)
            state = env.observe_privileged()
            env.reset([True, False], spawn_points=[points[0]])
            np.testing.assert_array_equal(env.observe_privileged()[1], state[1])
            np.testing.assert_array_equal(env.frames()[0], initial[0])
            state = env.observe_privileged()
            sampled = env.instance_parameters()
            for model_indices in ([0], 0, [0] * (env.num_agents - 1) + [-1], [0] * (env.num_agents - 1) + [2], [0.0] * env.num_agents):
                invalid = copy.deepcopy(points)
                invalid[-1]["drone"] = model_indices
                with pytest.raises(ValueError):
                    env.reset(spawn_points=invalid)
                with pytest.raises(ValueError):
                    env._environment.reset(None, json.dumps(invalid))
                np.testing.assert_array_equal(env.observe_privileged(), state)
                assert env.instance_parameters() == sampled
            for poses in (points[0]["poses"][:-1], points[0]["poses"] + [points[0]["poses"][0]], [[0] * 6] * env.num_agents):
                invalid = copy.deepcopy(points)
                invalid[-1]["poses"] = poses
                with pytest.raises(ValueError, match="pose"):
                    env._environment.reset(None, json.dumps(invalid))
                np.testing.assert_array_equal(env.observe_privileged(), state)
                assert env.instance_parameters() == sampled
        finally:
            env.close()


@pytest.mark.parametrize("task", [None, "target_frame"])
def test_explicit_spawns_ignore_seed_and_preserve_mask(scene_directory, task):
    config = Config(instances_per_world=2, width=16, height=16, preset="x500_fpv", motion_blur_samples=8, task=task)
    drones = [{"asset": rigged_asset_path(name), "parameters": {"dynamics": {"dynamics": {"model": name}}}} for name in ("x500", "crazyflie")]
    parameters = {"dynamics": {"mdp": {"init": {"max_linear_velocity": 0, "max_angular_velocity": 0, "min_rpm": 0, "max_rpm": 0}}}}
    reference = None
    for seed in (1, 123):
        env = HyperDrone(scene_directory, config=config, seed=seed, drones=drones, parameters=parameters)
        try:
            if reference is None:
                env.reset()
                points = env.spawn_points()
                for i, point in enumerate(points):
                    point["drone"] = [i]
            env.reset(spawn_points=points)
            assert env.spawn_points() == points
            assert [p["drone"] for p in env.instance_parameters()] == [0, 1]
            assert [p["dynamics"]["dynamics"]["mass"] for p in env.instance_parameters()] == [d["parameters"]["dynamics"]["dynamics"]["mass"] for d in env.drones]
            initial = env.frames().copy()
            record = [initial, env.observe(), env.observe_privileged()]
            for _ in range(3):
                env.step(np.zeros((2, env.action_dim), dtype=np.float32))
                record.extend((env.frames().copy(), env.observe(), env.observe_privileged()))
            if reference is None:
                reference = record
            else:
                for actual, expected in zip(record, reference):
                    np.testing.assert_array_equal(actual, expected)
            state = env.observe_privileged()
            env.reset([True, False], spawn_points=[points[0]])
            np.testing.assert_array_equal(env.observe_privileged()[1], state[1])
            np.testing.assert_array_equal(env.frames()[0], initial[0])
            env.reset([False, False], spawn_points=[])
            state = env.observe_privileged()
            for override in ({"scene": "wrong-room"}, {"drone": 2}, {"drone": -1}, {"poses": [[0] * 7]}, {"shutter_fraction": float("nan")}):
                invalid = copy.deepcopy(points)
                invalid[0].update(override)
                with pytest.raises(ValueError):
                    env.reset(spawn_points=invalid)
                np.testing.assert_array_equal(env.observe_privileged(), state)
            with pytest.raises(ValueError, match="one spawn"):
                env.reset(spawn_points=points[:1])
        finally:
            env.close()


@pytest.mark.parametrize("task", [None, "target_frame"])
def test_native_spawn_validation_is_atomic(scene_directory, task):
    config = replace(CONFIG, num_worlds=2, width=16, height=16, task=task)
    scenes = [scene_directory / "ProcTHOR-Train-1.glb"] * 2
    with closing(HyperDrone(scenes, config=config, seed=471)) as env, closing(HyperDrone(scenes, config=config, seed=471)) as control:
        env.reset()
        control.reset()
        for override in ({"scene_translation": [1e100, 0, 0]}, {"scene_translation": [0] * 4},
                         {"scene_yaw": [0, 0]}, {"scene_yaw": [1, 0, 0]},
                         {"poses": [[0] * 7]}, {"shutter_fraction": 1.1}, {"drone": [-1]}, {"drone": [1 << 63]}):
            invalid = copy.deepcopy(env.spawn_points())
            invalid[-1].update(override)
            state = env.observe_privileged()
            parameters = env.instance_parameters()
            with pytest.raises(ValueError):
                env._environment.reset(None, json.dumps(invalid))
            np.testing.assert_array_equal(env.observe_privileged(), state)
            assert env.instance_parameters() == parameters
            env.reset()
            control.reset()
            np.testing.assert_array_equal(env.observe_privileged(), control.observe_privileged())
            assert env.instance_parameters() == control.instance_parameters()
        env.reset([False] * env.total_instances)
        env.reset([False] * env.total_instances, spawn_points=[])
        env.reset()
        control.reset()
        np.testing.assert_array_equal(env.observe_privileged(), control.observe_privileged())
        assert env.instance_parameters() == control.instance_parameters()


def test_drones_table(scene_directory):
    drones = [
        {"asset": rigged_asset_path("x500")},
        {"asset": rigged_asset_path("crazyflie"), "parameters": {"dynamics": {"dynamics": {"model": "crazyflie"}}, "cameras": {"front": {"fov": 70}}}},
        {"asset": rigged_asset_path("x500"), "parameters": {"dynamics": {"dynamics": {"model": "x500_sim"}}}},
    ]
    env = HyperDrone(scene_directory, config=CONFIG_DRONES, seed=5, drones=drones)
    try:
        assert env.max_drone_types == 8
        table = env.drones
        assert len(table) == 3
        assert table[0]["parameters"] == env.parameters
        assert table[1]["parameters"]["cameras"]["front"]["fov"] == 70
        assert table[1]["parameters"]["dynamics"]["dynamics"]["mass"] != table[0]["parameters"]["dynamics"]["dynamics"]["mass"]
        assert table[2]["parameters"]["dynamics"]["dynamics"]["rotor_positions"] != table[0]["parameters"]["dynamics"]["dynamics"]["rotor_positions"]
        seen = set()
        for _ in range(16):
            env.reset()
            for instance in env.instance_parameters():
                seen.add(instance["drone"])
                nominal = table[instance["drone"]]["parameters"]
                assert instance["dynamics"]["dynamics"]["mass"] == pytest.approx(nominal["dynamics"]["dynamics"]["mass"])
                assert instance["cameras"]["front"]["fov"] == pytest.approx(nominal["cameras"]["front"]["fov"])
        assert seen == {0, 1, 2}
        smoke_rollout(env)
        switched = np.array([1, 0, 0, 0], dtype=bool)
        env.set_instance_parameters({"drone": 1, "cameras": {"front": {"fov": 40}}}, mask=switched)
        instance = env.instance_parameters()[0]
        assert instance["drone"] == 1
        assert instance["dynamics"] == table[1]["parameters"]["dynamics"]
        assert instance["cameras"]["front"]["fov"] == 40
        smoke_rollout(env)
        with pytest.raises(ValueError, match="index the drones table"):
            env.set_instance_parameters({"drone": 3}, mask=switched)
        positions = np.array(table[0]["parameters"]["dynamics"]["dynamics"]["rotor_positions"])
        with pytest.raises(ValueError, match="quadrant"):
            env.set_instance_parameters({"drone": 0, "dynamics": {"dynamics": {"rotor_positions": positions[[1, 0, 2, 3]].tolist()}}}, mask=switched)
        before_invalid_geometry = env.instance_parameters()
        for scale in (1e-200, 1e200):
            with pytest.raises(ValueError, match="finite positive wheelbase"):
                env.set_instance_parameters({"drone": 0, "dynamics": {"dynamics": {"rotor_positions": (positions * scale).tolist()}}}, mask=switched)
            assert env.instance_parameters() == before_invalid_geometry
        with pytest.raises(ValueError, match="sampled at reset"):
            HyperDrone(scene_directory, config=CONFIG_DRONES, seed=0, drones=[{"asset": drones[0]["asset"], "parameters": {"drone": 1}}])
    finally:
        env.close()


def test_drones_resolved_nominals(scene_directory):
    asset = rigged_asset_path("x500")
    drones = [
        {"asset": asset, "parameters": {"dynamics": {"dynamics": {"mass": 0.1}}, "cameras": {"front": {"fov": 60}}}},
        {"asset": asset},
    ]
    env = HyperDrone(scene_directory, config=CONFIG_DRONES, seed=5, parameters={"cameras": {"front": {"fov": 90}}}, drones=drones)
    try:
        table = env.drones
        assert table[0]["parameters"] == env.parameters
        assert [entry["parameters"]["cameras"]["front"]["fov"] for entry in table] == [60, 90]
        env.reset()
        for instance in env.instance_parameters():
            assert instance == {**table[instance["drone"]]["parameters"], "drone": instance["drone"]}
        table[0]["parameters"]["cameras"]["front"]["fov"] = 20
        assert env.drones[0]["parameters"]["cameras"]["front"]["fov"] == 60
        assert env.parameters["cameras"]["front"]["fov"] == 60
    finally:
        env.close()


def test_drone_asset_is_a_one_entry_table(scene_directory):
    asset = rigged_asset_path("x500")
    frames = []
    for arguments in ({"drone_asset": asset}, {"drones": [{"asset": asset}]}):
        env = HyperDrone(scene_directory, config=CONFIG_DRONES, seed=7, **arguments)
        try:
            assert [entry["asset"] for entry in env.drones] == [str(asset)]
            env.reset()
            env.observe()
            frames.append(env.observe().copy())
            assert all(instance["drone"] == 0 for instance in env.instance_parameters())
        finally:
            env.close()
    np.testing.assert_array_equal(frames[0], frames[1])


def test_drones_validation(scene_directory):
    asset = rigged_asset_path("x500")
    with pytest.raises(ValueError, match="either"):
        HyperDrone(scene_directory, config=CONFIG_DRONES, seed=0, drone_asset=asset, drones=[{"asset": asset}])
    with pytest.raises(ValueError, match="holds 8"):
        HyperDrone(scene_directory, config=CONFIG_DRONES, seed=0, drones=[{"asset": asset}] * 9)
    with pytest.raises(ValueError, match="asset per drone"):
        HyperDrone(scene_directory, config=CONFIG_DRONES, seed=0, drones=[{"asset": asset}, {}])
    with pytest.raises(ValueError, match="SELF_VISIBLE world"):
        HyperDrone(scene_directory, config=CONFIG, seed=0, drones=[{"asset": asset}])
    with pytest.raises(ValueError, match="shared by all instances"):
        HyperDrone(scene_directory, config=CONFIG_DRONES, seed=0, drones=[{"asset": asset, "parameters": {"dynamics": {"integration": {"dt": 0.1}}}}])
    with pytest.raises(ValueError, match="unknown drone entry"):
        HyperDrone(scene_directory, config=CONFIG_DRONES, seed=0, drones=[{"asset": asset, "mass": 1}])


def smoke_rollout(env):
    mask = np.ones(env.total_instances, dtype=np.uint8)
    env.reset(mask)
    env.observe()
    observations = env.observe()
    assert observations.shape == (env.total_instances, env.observation_dim)
    actions = np.zeros((env.total_instances, env.action_dim), dtype=np.float32)
    env.step(actions)
    env.observe()
    assert np.isfinite(env.rewards()).all()
    assert env.observe_privileged().shape == (env.total_instances, env.observation_dim_privileged)


def test_x500_target_frame(scene_directory):
    config = Config(instances_per_world=2, width=16, height=16, preset="x500_fpv", task="target_frame")
    env = HyperDrone(scene_directory, config=config, seed=3, drone_asset=drone_asset_path())
    try:
        assert env.observation_dim == 16 * 16 * 6
        layout = env.observation_layout
        assert layout.shape == (16, 16, 6)
        assert layout.blocks == {"image_stack": (0, 3), "target_image": (3, 3)}
        smoke_rollout(env)
        with pytest.raises(ValueError, match="target setup"):
            env.set_instance_parameters({"cameras": {"front": {"fov": 60}}})
    finally:
        env.close()


@pytest.mark.parametrize("cameras", [1, 2])
def test_moving_gate(scene_directory, cameras):
    profiles = tuple(CameraSpec(f"camera{index}", (RGB("float32"),)) for index in range(cameras))
    config = Config(instances_per_world=2, width=16, height=16, task="moving_gate", cameras=profiles, observation=ImageObservation(tuple(camera.name for camera in profiles)))
    with pytest.raises(ValueError, match="needs assets"):
        HyperDrone(scene_directory, config=config, seed=4)
    env = HyperDrone(scene_directory, config=config, seed=4, assets={"gate": drone_asset_path()})
    try:
        privileged = env.observation_layout_privileged
        assert privileged.blocks["gate_state"] == (18, 8)
        assert env.observation_dim_privileged == 26
        smoke_rollout(env)
        assert env.frames().shape == (2, cameras * 16, 16, 3)
    finally:
        env.close()


def check_vil_spawn_replay(env):
    points = env.spawn_points()
    offset = np.array([100.0, -100.0, 25.0])
    for point in points:
        cosine, sine = point["scene_yaw"]
        scene_offset = np.array([cosine * offset[0] - sine * offset[1], sine * offset[0] + cosine * offset[1], offset[2]])
        point["scene_translation"] = (np.asarray(point["scene_translation"]) - scene_offset).tolist()
        point["poses"][0][:3] = (np.asarray(point["poses"][0][:3]) + offset).tolist()
    env.reset(spawn_points=points)
    state = env.observe_privileged()
    np.testing.assert_allclose(state[:, :3], [point["poses"][0][:3] for point in points])
    assert np.linalg.norm(state[:, 18:21] - state[:, :3], axis=1).max() < 20
    env.step()
    assert not env.terminated().any()
    assert np.isfinite(env.observe_privileged()).all()


def test_visual_inertial_localization(scene_directory):
    with pytest.raises(ValueError, match="x500_fpv_imu"):
        Config(task="visual_inertial_localization")
    with pytest.raises(ValueError, match="asset per drone"):
        HyperDrone(scene_directory, config=Config(preset="x500_fpv_imu"), seed=6)
    config = Config(instances_per_world=2, width=16, height=16, preset="x500_fpv_imu", task="visual_inertial_localization")
    env = HyperDrone(scene_directory, config=config, seed=6)
    try:
        assert env.action_dim == 0, "the task flies itself"
        assert env.frame_stride == 4
        assert env.dt == pytest.approx(0.005)
        assert "imu" in env.parameters["dynamics"]
        assert env.observation_dim_imu == 8
        assert env.observation_layout_imu.blocks == {
            "accelerometer": (0, 3), "gyroscope": (3, 3), "frame_age": (6, 1), "new_frame": (7, 1),
        }
        assert env.observation_layout_privileged.blocks["waypoint"] == (18, 3)
        assert env.observation_dim_privileged == 21
        mask = np.ones(env.total_instances, dtype=np.uint8)
        env.reset(mask)
        np.testing.assert_array_equal(env.observe_imu()[:, 6], 0)
        before = env.observe_privileged()
        with pytest.raises(ValueError, match="all-or-none"):
            env.reset(np.array([1, 0], dtype=np.uint8))
        np.testing.assert_array_equal(env.observe_privileged(), before)
        frames = []
        for step in range(2 * env.frame_stride + 1):
            env.observe()
            frames.append(env.frames())
            env.step()
            imu = env.observe_imu()
            assert imu.shape == (2, 8)
            assert np.isfinite(imu).all()
            frame_age = (step + 1) % env.frame_stride
            np.testing.assert_allclose(imu[:, 6], frame_age / env.frame_stride, atol=1e-6)
            assert (imu[:, 7] == (1.0 if frame_age == 0 else 0.0)).all()
        # the camera only advances on frame boundaries: identical within a stride, and fresh
        # once the autopilot has moved the drone toward its first waypoint
        for frame in frames[1:env.frame_stride]:
            np.testing.assert_array_equal(frame, frames[0])
        for _ in range(60):
            env.observe()
            env.step()
        assert not np.array_equal(env.frames(), frames[0])
        state = env.observe_privileged()
        assert np.isfinite(state).all()
        assert np.linalg.norm(state[:, 0:3] - state[:, 18:21], axis=1).max() < 100, "the route stays in the scene"
        check_vil_spawn_replay(env)
    finally:
        env.close()


def test_spec_header_escape_hatch(scene_directory):
    header = Path(__file__).parent / "user_spec_header.h"
    config = Config(spec_header=str(header))
    env = HyperDrone(scene_directory, config=config, seed=5, drone_asset=drone_asset_path())
    try:
        assert env.self_visible
        assert env.max_drone_types == 3
        assert env.instances_per_world == 2
        assert env.width == 16 and env.height == 16
        assert env.observation_layout.blocks == {"image": (0, 3)}
        smoke_rollout(env)
    finally:
        env.close()


def test_spec_header_custom_controller(scene_directory):
    directory = Path(__file__).parent
    with closing(HyperDrone(scene_directory, config=Config(spec_header=str(directory / "user_controller_spec_header.h")), seed=5, drone_asset=drone_asset_path())) as custom, \
         closing(HyperDrone(scene_directory, config=Config(spec_header=str(directory / "user_spec_header.h")), seed=5, drone_asset=drone_asset_path())) as direct:
        assert custom.control == "collective"
        assert custom.action_dim == 1
        np.testing.assert_array_equal(custom.action_low, [0])
        np.testing.assert_array_equal(custom.action_high, [1])
        for environment in (custom, direct):
            environment.reset()
        commands = np.array([[0.3], [0.8]], dtype=np.float32)
        for _ in range(3):
            custom.step(commands)
            direct.step(np.repeat(2 * commands - 1, 4, axis=1))
            np.testing.assert_array_equal(custom.observe_privileged(), direct.observe_privileged())
            np.testing.assert_array_equal(custom.rewards(), direct.rewards())


def test_spec_header_drone_table(scene_directory):
    header = Path(__file__).parent / "user_spec_header.h"
    config = Config(spec_header=str(header))
    asset = drone_asset_path()
    drones = [{"asset": asset, "parameters": {"cameras": {"camera0": {"fov": fov}}}} for fov in (50, 70, 90)]
    env = HyperDrone(scene_directory, config=config, seed=5, drones=drones)
    try:
        assert env.self_visible
        assert env.max_drone_types == 3
        assert len(env.drones) == env.max_drone_types
        seen = set()
        for _ in range(16):
            env.reset()
            for instance in env.instance_parameters():
                seen.add(instance["drone"])
                assert instance["cameras"]["camera0"]["fov"] == (50, 70, 90)[instance["drone"]]
        assert seen == {0, 1, 2}
        smoke_rollout(env)
    finally:
        env.close()
    with pytest.raises(ValueError, match="MAX_DRONES in spec_header"):
        HyperDrone(scene_directory, config=config, drones=drones + drones[:1])
    with pytest.raises(ValueError, match="asset per drone"):
        HyperDrone(scene_directory, config=config, drones=[{}])


def scene_source():
    source = Path(os.environ.get("HYPERDRONE_TEST_SCENE_DIR", source_root() / "tests" / "data")) / "ProcTHOR-Train-1.glb"
    if not source.exists():
        pytest.skip(f"no ProcTHOR test scene at {source}")
    return source


def conta_store(monkeypatch, tmp_path, *blobs):
    # a local conta store served over file://, so resolution runs the full download
    # protocol without network; the cache is the shared layout both languages use
    store = tmp_path / "conta_store"
    store.mkdir()
    hashes = []
    for blob in blobs:
        digest = conta.sha1_file(blob)
        (store / digest).symlink_to(blob)
        hashes.append(digest)
    monkeypatch.delenv("CONTA_ROOT", raising=False)
    monkeypatch.setenv("CONTA_CACHE", str(tmp_path / "conta_cache"))
    monkeypatch.setenv("CONTA_URL", store.as_uri() + "/")
    return hashes


def test_scenes_reference_list(tmp_path):
    source = scene_source()
    directory = tmp_path / "scenes"
    directory.mkdir()
    (directory / "a.glb").symlink_to(source)
    (directory / "b.glb").symlink_to(source)
    from_directory = rollout(directory, 21)
    from_references = rollout([directory / "a.glb", directory / "b.glb"], 21)
    assert len(from_directory) == len(from_references)
    for a, b in zip(from_directory, from_references):
        np.testing.assert_array_equal(a, b)


def test_scenes_conta_reference(monkeypatch, tmp_path):
    source = scene_source()
    (scene_hash,) = conta_store(monkeypatch, tmp_path, source)
    from_path = rollout([source], 22)
    from_conta = rollout([{"description": source.name, "hash": scene_hash}], 22)
    for a, b in zip(from_path, from_conta):
        np.testing.assert_array_equal(a, b)
    cached = Path(tmp_path / "conta_cache" / scene_hash)
    assert cached.is_file()
    for a, b in zip(from_conta, rollout([f"conta:{scene_hash}"], 22)):
        np.testing.assert_array_equal(a, b)


def test_scenes_preflight_errors(tmp_path):
    source = scene_source()
    with pytest.raises(ValueError, match="scene reference"):
        HyperDrone([source], config=Config(num_worlds=2), seed=0)
    with pytest.raises(ValueError, match="not found"):
        HyperDrone(["/nonexistent.glb"], config=Config(), seed=0)
    with pytest.raises(ValueError, match=r"\.glb scene"):
        HyperDrone(tmp_path, config=Config(), seed=0)


def test_drone_asset_conta_reference(monkeypatch, tmp_path):
    scene = scene_source()
    asset = drone_asset_path()
    scene_hash, asset_hash = conta_store(monkeypatch, tmp_path, scene, asset)
    config = Config(instances_per_world=2, width=16, height=16, preset="x500_fpv", task="target_frame")
    env = HyperDrone([f"conta:{scene_hash}"], config=config, seed=3, drone_asset=f"conta:{asset_hash}")
    try:
        smoke_rollout(env)
    finally:
        env.close()


@pytest.mark.parametrize("history_length", [1, 3])
def test_step_clock_is_independent_of_image_reads(scene_directory, history_length, tmp_path):
    import hyperlux
    header = tmp_path / "history_world.h"
    header.write_text("""#pragma once
#include <rl_tools/rl/environments/hyperdrone/presets.h>
#include <cstddef>
namespace hyperdrone_user {
    struct SPEC: rl_tools::rl::environments::hyperdrone::presets::X500FPV<float, std::size_t> {
        static constexpr std::size_t N_CAMERA_AGENTS = 1;
        static constexpr std::size_t INSTANCES_PER_WORLD = 2;
        static constexpr std::size_t CAM_WIDTH = 16;
        static constexpr std::size_t CAM_HEIGHT = 16;
        static constexpr std::size_t HISTORY_LENGTH = HISTORY_COUNT;
        static constexpr bool ENABLE_MOTION_BLUR = true;
        static constexpr std::size_t MOTION_BLUR_SAMPLES = 4;
        static constexpr bool SELF_VISIBLE = false;
        using SHADING = rl_tools::hyperlux::Low;
    };
    using WORLD = rl_tools::rl::environments::hyperdrone::World<SPEC>;
}
""".replace("HISTORY_COUNT", str(history_length)))
    config = Config(spec_header=str(header), device="cuda" if hyperlux.backend() == "optix" else "cpu")
    records = []
    for read_every_step in (False, True):
        with closing(HyperDrone(scene_directory, config=config, seed=319)) as env:
            env.reset()
            actions = np.full((env.total_instances, env.action_dim), 0.1, dtype=np.float32)
            for step in range(8):
                if step == 3:
                    env.reset([True, False])
                env.step(actions)
                if read_every_step:
                    image = env.observe()
                    env.observe()
                    np.testing.assert_array_equal(env.observe(), image)
            records.append((env.observe(), env.observe_privileged()))
    for actual, expected in zip(records[0], records[1]):
        np.testing.assert_array_equal(actual, expected)


def test_vil_cadence_does_not_depend_on_reads(scene_directory):
    config = Config(instances_per_world=2, width=16, height=16, preset="x500_fpv_imu", task="visual_inertial_localization")
    records = []
    for read_every_step in (False, True):
        with closing(HyperDrone(scene_directory, config=config, seed=419)) as env:
            env.reset()
            for _ in range(9):
                env.step()
                if read_every_step:
                    image = env.observe()
                    env.observe()
                    np.testing.assert_array_equal(env.observe(), image)
            records.append((env.observe(), env.observe_privileged(), env.observe_imu()))
    for actual, expected in zip(records[0], records[1]):
        np.testing.assert_array_equal(actual, expected)


def test_custom_vil_rejects_partial_reset_before_sampling(scene_directory, tmp_path):
    header = tmp_path / "vil_world.h"
    header.write_text("""#pragma once
#include <rl_tools/rl/environments/hyperdrone/presets.h>
#include <rl_tools/rl/environments/hyperdrone/tasks/visual_inertial_localization/operations_cpu.h>
#include <cstddef>
namespace hyperdrone_user {
    struct SPEC: rl_tools::rl::environments::hyperdrone::presets::X500FPVIMU<float, std::size_t> {
        static constexpr std::size_t N_CAMERA_AGENTS = 1;
        static constexpr std::size_t INSTANCES_PER_WORLD = 2;
        static constexpr std::size_t CAM_WIDTH = 16;
        static constexpr std::size_t CAM_HEIGHT = 16;
        static constexpr bool SELF_VISIBLE = false;
        using SHADING = rl_tools::hyperlux::Low;
    };
    using BASE = rl_tools::rl::environments::hyperdrone::World<SPEC>;
    using TASK = rl_tools::rl::environments::hyperdrone::tasks::visual_inertial_localization::Specification<BASE>;
    using WORLD = rl_tools::rl::environments::hyperdrone::tasks::visual_inertial_localization::World<TASK>;
}
""")
    records = []
    for reject_first in (False, True):
        with closing(HyperDrone(scene_directory, config=Config(spec_header=str(header)), seed=37)) as env:
            if reject_first:
                with pytest.raises(ValueError, match="all-or-none"):
                    env.reset([True, False])
            env.reset()
            records.append(env.observe_privileged())
            check_vil_spawn_replay(env)
    np.testing.assert_array_equal(records[0], records[1])


def test_flow_reference_does_not_depend_on_image_reads(scene_directory):
    import hyperlux
    config = replace(CONFIG, width=16, height=16, cameras=(CameraSpec("front", (RGB("float32"), Flow(), FlowStatus())),))
    records = []
    for read_every_step in (False, True):
        with closing(HyperDrone(scene_directory, config=config, seed=541)) as env:
            env.reset()
            env.observe()
            actions = np.full((env.total_instances, env.action_dim), 0.1, dtype=np.float32)
            for step in range(4):
                if step == 2:
                    env.reset([True, False])
                env.step(actions)
                if read_every_step:
                    env.observe()
            image, flow, status = env.observe(), env.outputs[Flow()].read(), env.outputs[FlowStatus()].read()
            assert np.any(status == hyperlux.FlowStatus.VALID)
            records.append((image, flow, status))
    for actual, expected in zip(records[0], records[1]):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_flow_reference_survives_same_step_reacquisition(scene_directory, device):
    import hyperlux
    if device == "cuda" and hyperlux.backend() != "optix":
        pytest.skip("CUDA Worlds require OptiX")
    config = replace(CONFIG, width=16, height=16, cameras=(CameraSpec("front", (RGB("float32"), Flow(), FlowStatus())),), device=device)
    with closing(HyperDrone(scene_directory, config=config, seed=541)) as env:
        env.reset()
        env.observe()
        actions = np.full((env.total_instances, env.action_dim), 0.1, dtype=np.float32)
        for _ in range(4):
            env.step(actions)
        before = (env.observe(), env.outputs[Flow()].read(), env.outputs[FlowStatus()].read())
        assert np.max(np.abs(before[1][1])) > 0.01
        for operation in (
            lambda: env.reset([False, False]),
            lambda: env.set_instance_parameters({}, [False, False]),
            lambda: env.set_instance_parameters({}, [True, False]),
        ):
            operation()
            for actual, expected in zip((env.observe(), env.outputs[Flow()].read(), env.outputs[FlowStatus()].read()), before):
                np.testing.assert_array_equal(actual, expected)
        for operation in (
            lambda: env.set_instance_parameters({"cameras": {"front": {"fov": 75.0}}}, [True, False]),
            lambda: env.reset([True, False]),
        ):
            operation()
            after = (env.observe(), env.outputs[Flow()].read(), env.outputs[FlowStatus()].read())
            for actual, expected in zip(after, before):
                np.testing.assert_array_equal(actual[1], expected[1])
            for actual, expected in zip((env.observe(), env.outputs[Flow()].read(), env.outputs[FlowStatus()].read()), after):
                np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_motion_blur_reset_survives_same_step_reacquisition(scene_directory, device):
    import hyperlux
    if device == "cuda" and hyperlux.backend() != "optix":
        pytest.skip("CUDA Worlds require OptiX")
    config = replace(CONFIG, width=16, height=16, motion_blur_samples=4, device=device)
    with closing(HyperDrone(scene_directory, config=config, seed=673)) as env:
        env.reset()
        actions = np.full((env.total_instances, env.action_dim), 0.1, dtype=np.float32)
        for _ in range(8):
            env.step(actions)
        previous = env.observe()
        env.reset([True, False])
        before = env.observe()
        assert not np.array_equal(before[0], previous[0])
        env.reset([False, False])
        np.testing.assert_array_equal(env.observe(), before)
        env.set_instance_parameters({}, [True, False])
        np.testing.assert_array_equal(env.observe(), before)


def test_spawn_points_accept_scene_references(scene_directory):
    parameters = {"dynamics": {"mdp": {"init": {"max_linear_velocity": 0, "max_angular_velocity": 0, "min_rpm": 0, "max_rpm": 0}}}}
    env = HyperDrone(scene_directory, config=CONFIG, seed=5, parameters=parameters)
    try:
        env.reset()
        points = env.spawn_points()
        state = env.observe_privileged()
        for reference in (lambda scene: f"conta:{scene}", lambda scene: {"description": "scene", "hash": scene}):
            env.reset(spawn_points=[{**point, "scene": reference(point["scene"])} for point in points])
            assert env.spawn_points() == points
            np.testing.assert_array_equal(env.observe_privileged(), state)
        with pytest.raises(ValueError, match="does not match"):
            env.reset(spawn_points=[{**point, "scene": "conta:" + "0" * 40} for point in points])
        with pytest.raises(ValueError, match="content hash"):
            env.reset(spawn_points=[{**point, "scene": 7} for point in points])
        np.testing.assert_array_equal(env.observe_privileged(), state)
    finally:
        env.close()

