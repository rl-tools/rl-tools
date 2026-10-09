import copy
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import conta
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


def test_configurations_keep_independent_state(scene_directory):
    small = MultiEnvironment(scene_directory, config=EnvConfig(instances=2, cam_width=16, cam_height=16, shading="low"), seed=5)
    large = MultiEnvironment(scene_directory, config=CONFIG, seed=5)
    try:
        assert type(small._environment) is not type(large._environment)
        for environment in (small, large):
            environment.reset()
            environment.render()
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
        env = MultiEnvironment(scene_directory, config=config, seed=5)
        try:
            env.reset()
            env.render()
            for _ in range(10):
                env.step(np.zeros((env.total_instances, env.action_dim), dtype=np.float32))
            env.render()
            blurred = env.observe()
            state = env.observe_privileged()
            env.render(np.ones(env.total_instances, dtype=bool))
            sharp = env.observe()
            np.testing.assert_array_equal(env.observe_privileged(), state)
            assert np.isfinite(blurred).all()
            assert not np.array_equal(blurred, sharp)
            records.append((blurred, sharp))
        finally:
            env.close()
    for first, second in zip(*records):
        np.testing.assert_array_equal(first, second)


def test_native_binding_rejects_invalid_buffers(scene_directory):
    environment = MultiEnvironment(scene_directory, config=CONFIG)
    try:
        with pytest.raises(ValueError, match="buffer size"):
            environment._environment.reset(np.ones(1, dtype=bool))
        with pytest.raises(ValueError, match="buffer size"):
            environment._environment.step(np.zeros(1, dtype=np.float32))
        with pytest.raises(TypeError):
            environment._environment.observe(np.empty((2, environment.observation_dim), dtype=np.float64))
        environment.reset()
        environment.render()
        assert np.isfinite(environment.observe()).all()
    finally:
        environment.close()


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


def nominal_rollout(scene_directory, parameters=None):
    env = MultiEnvironment(scene_directory, config=CONFIG, seed=3, parameters=parameters)
    try:
        mask = np.ones(env.total_instances, dtype=np.uint8)
        env.reset(mask)
        env.render(mask)
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
    parameters, frames, states = nominal_rollout(scene_directory, {"dynamics": {"dynamics": {"model": "x500", "mass": 2.0, "J": J}, "integration": {"dt": 0.02}}, "camera": camera})
    dynamics = parameters["dynamics"]["dynamics"]
    assert dynamics["mass"] == pytest.approx(2.0)
    np.testing.assert_allclose(dynamics["J"], J)
    np.testing.assert_allclose(dynamics["J_inv"], np.linalg.inv(J), rtol=1e-6)
    overridden = ("mass", "J", "J_inv")
    assert {key: value for key, value in dynamics.items() if key not in overridden} == {key: value for key, value in x500_parameters["dynamics"]["dynamics"].items() if key not in overridden}
    assert parameters["dynamics"]["integration"]["dt"] == pytest.approx(0.02)
    assert parameters["camera"]["fov"] == 90
    np.testing.assert_allclose(parameters["camera"]["mount"]["pose"]["orientation"], camera["mount"]["pose"]["orientation"], rtol=1e-6)
    assert not np.array_equal(frames, default_frames)
    assert not np.array_equal(states, default_states)


def test_nominal_parameter_validation(scene_directory):
    with pytest.raises(ValueError, match="unknown dynamics model"):
        MultiEnvironment(scene_directory, config=CONFIG, parameters={"dynamics": {"dynamics": {"model": "zeppelin"}}})
    with pytest.raises(ValueError, match="unknown parameter"):
        MultiEnvironment(scene_directory, config=CONFIG, parameters={"dynamics": {"dynamics": {"weight": 1.0}}})
    with pytest.raises(ValueError, match="must have shape"):
        MultiEnvironment(scene_directory, config=CONFIG, parameters={"camera": {"mount": {"pose": {"orientation": [1, 0, 0]}}}})


def test_instance_parameters(scene_directory):
    env = MultiEnvironment(scene_directory, config=CONFIG, seed=3)
    reference = MultiEnvironment(scene_directory, config=CONFIG, seed=3)
    try:
        with pytest.raises(RuntimeError, match="reset"):
            env.instance_parameters()
        all_instances = np.ones(env.total_instances, dtype=np.uint8)
        no_reset = np.zeros(env.total_instances, dtype=np.uint8)
        for e in (env, reference):
            e.reset(all_instances)
            e.render(all_instances)
        nominal = env.parameters
        flying_nominal = {**nominal, "drone": 0}
        current = env.instance_parameters()
        assert current == [flying_nominal] * env.total_instances
        mass = nominal["dynamics"]["dynamics"]["mass"]
        env.set_instance_parameters({"dynamics": {"dynamics": {"mass": 3 * mass}}, "camera": {"fov": 30}}, mask=np.array([1, 0], dtype=np.uint8))
        updated = env.instance_parameters()
        assert updated[1] == flying_nominal
        assert updated[0]["dynamics"]["dynamics"]["mass"] == pytest.approx(3 * mass)
        assert updated[0]["camera"]["fov"] == 30

        frames, states = [], []
        for e in (env, reference):
            e.render(no_reset)
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
        env.set_instance_parameters([{"camera": {"fov": 50}}, {"dynamics": {"dynamics": {"model": "x500"}}}])
        updated = env.instance_parameters()
        assert updated[0]["camera"]["fov"] == 50
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
    reference = MultiEnvironment(scene_directory, config=CONFIG, seed=7)
    try:
        monkeypatch.setattr(env._env, "episode_step_limit", 1)
        reset_mask = np.ones(reference.total_instances, dtype=np.uint8)
        reference.reset()
        reference.render(reset_mask)
        observations, _ = env.reset()
        np.testing.assert_array_equal(observations, reference.observe())

        actions = np.zeros((env.num_envs, reference.action_dim), dtype=np.float32)
        reference.step(actions)
        expected_rewards, expected_terminated = reference.rewards(), reference.terminated()
        reference.reset()
        reference.render(reset_mask)

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
            EnvConfig(motion_blur_samples=samples)
    with pytest.raises(ValueError, match="unknown preset"):
        EnvConfig(preset="nonexistent")
    with pytest.raises(ValueError, match="unknown task"):
        EnvConfig(task="nonexistent")
    with pytest.raises(ValueError, match="SELF_VISIBLE"):
        EnvConfig(n_agents=2)
    with pytest.raises(ValueError, match="spec_header supersedes"):
        EnvConfig(task="target_frame", spec_header=__file__)
    with pytest.raises(ValueError, match="asset per drone"):
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


def rigged_asset_path(name):
    source = Path(os.environ.get("HYPERDRONE_TEST_SCENE_DIR", source_root() / "tests" / "data")) / f"{name}.glb"
    if not source.exists():
        pytest.skip(f"no drone asset at {source}")
    return source


def drone_asset_path():
    return rigged_asset_path("x500")


CONFIG_DRONES = EnvConfig(instances=4, cam_width=16, cam_height=16, preset="x500_fpv", max_drones=4)


@pytest.mark.parametrize("task", [None, "target_frame"])
def test_explicit_spawns_ignore_seed_and_preserve_mask(scene_directory, task):
    config = EnvConfig(instances=2, cam_width=16, cam_height=16, preset="x500_fpv", max_drones=2, motion_blur_samples=8, task=task)
    drones = [{"asset": rigged_asset_path(name), "parameters": {"dynamics": {"dynamics": {"model": name}}}} for name in ("x500", "crazyflie")]
    parameters = {"dynamics": {"mdp": {"init": {"max_linear_velocity": 0, "max_angular_velocity": 0, "min_rpm": 0, "max_rpm": 0}}}}
    reference = None
    for seed in (1, 123):
        env = MultiEnvironment(scene_directory, config=config, seed=seed, drones=drones, parameters=parameters)
        try:
            if reference is None:
                with pytest.raises(RuntimeError, match="every instance has been reset"):
                    env.spawn_points()
                env.reset()
                points = env.spawn_points()
                for i, point in enumerate(points):
                    point["drone"] = i
            env.reset(spawn_points=points)
            assert env.spawn_points() == points
            assert [p["drone"] for p in env.instance_parameters()] == [0, 1]
            assert [p["dynamics"]["dynamics"]["mass"] for p in env.instance_parameters()] == [d["parameters"]["dynamics"]["dynamics"]["mass"] for d in env.drones]
            initial = env.frames().copy()
            record = [initial, env.observe_privileged()]
            for _ in range(3):
                env.step(np.zeros((2, env.action_dim), dtype=np.float32))
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


def test_drones_table(scene_directory):
    drones = [
        {"asset": rigged_asset_path("x500")},
        {"asset": rigged_asset_path("crazyflie"), "parameters": {"dynamics": {"dynamics": {"model": "crazyflie"}}, "camera": {"fov": 70}}},
        {"asset": rigged_asset_path("x500"), "parameters": {"dynamics": {"dynamics": {"model": "x500_sim"}}}},
    ]
    env = MultiEnvironment(scene_directory, config=CONFIG_DRONES, seed=5, drones=drones)
    try:
        assert env.max_drones == 4
        table = env.drones
        assert len(table) == 3
        assert table[0]["parameters"] == env.parameters
        assert table[1]["parameters"]["camera"]["fov"] == 70
        assert table[1]["parameters"]["dynamics"]["dynamics"]["mass"] != table[0]["parameters"]["dynamics"]["dynamics"]["mass"]
        assert table[2]["parameters"]["dynamics"]["dynamics"]["rotor_positions"] != table[0]["parameters"]["dynamics"]["dynamics"]["rotor_positions"]
        seen = set()
        for _ in range(16):
            env.reset()
            for instance in env.instance_parameters():
                seen.add(instance["drone"])
                nominal = table[instance["drone"]]["parameters"]
                assert instance["dynamics"]["dynamics"]["mass"] == pytest.approx(nominal["dynamics"]["dynamics"]["mass"])
                assert instance["camera"]["fov"] == pytest.approx(nominal["camera"]["fov"])
        assert seen == {0, 1, 2}
        smoke_rollout(env)
        with pytest.raises(ValueError, match="sampled at reset"):
            env.set_instance_parameters({"drone": 1})
    finally:
        env.close()


def test_drones_resolved_nominals(scene_directory):
    asset = rigged_asset_path("x500")
    drones = [
        {"asset": asset, "parameters": {"dynamics": {"dynamics": {"mass": 0.1}}, "camera": {"fov": 60}}},
        {"asset": asset},
    ]
    env = MultiEnvironment(scene_directory, config=CONFIG_DRONES, seed=5, parameters={"camera": {"fov": 90}}, drones=drones)
    try:
        table = env.drones
        assert table[0]["parameters"] == env.parameters
        assert [entry["parameters"]["camera"]["fov"] for entry in table] == [60, 90]
        env.reset()
        for instance in env.instance_parameters():
            assert instance == {**table[instance["drone"]]["parameters"], "drone": instance["drone"]}
        table[0]["parameters"]["camera"]["fov"] = 20
        assert env.drones[0]["parameters"]["camera"]["fov"] == 60
        assert env.parameters["camera"]["fov"] == 60
    finally:
        env.close()


def test_drone_asset_is_a_one_entry_table(scene_directory):
    asset = rigged_asset_path("x500")
    frames = []
    for arguments in ({"drone_asset": asset}, {"drones": [{"asset": asset}]}):
        env = MultiEnvironment(scene_directory, config=CONFIG_DRONES, seed=7, **arguments)
        try:
            assert [entry["asset"] for entry in env.drones] == [str(asset)]
            env.reset()
            env.render(np.ones(env.total_instances, dtype=bool))
            frames.append(env.observe().copy())
            assert all(instance["drone"] == 0 for instance in env.instance_parameters())
        finally:
            env.close()
    np.testing.assert_array_equal(frames[0], frames[1])


def test_drones_validation(scene_directory):
    asset = rigged_asset_path("x500")
    with pytest.raises(ValueError, match="either"):
        MultiEnvironment(scene_directory, config=CONFIG_DRONES, seed=0, drone_asset=asset, drones=[{"asset": asset}])
    with pytest.raises(ValueError, match="max_drones"):
        MultiEnvironment(scene_directory, config=CONFIG_DRONES, seed=0, drones=[{"asset": asset}] * 5)
    with pytest.raises(ValueError, match="asset per drone"):
        MultiEnvironment(scene_directory, config=CONFIG_DRONES, seed=0, drones=[{"asset": asset}, {}])
    with pytest.raises(ValueError, match="SELF_VISIBLE world"):
        MultiEnvironment(scene_directory, config=CONFIG, seed=0, drones=[{"asset": asset}])
    with pytest.raises(ValueError, match="shared by all instances"):
        MultiEnvironment(scene_directory, config=CONFIG_DRONES, seed=0, drones=[{"asset": asset, "parameters": {"dynamics": {"integration": {"dt": 0.1}}}}])
    with pytest.raises(ValueError, match="unknown drone entry"):
        MultiEnvironment(scene_directory, config=CONFIG_DRONES, seed=0, drones=[{"asset": asset, "mass": 1}])


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


def test_visual_inertial_localization(scene_directory):
    with pytest.raises(ValueError, match="x500_fpv_imu"):
        EnvConfig(task="visual_inertial_localization")
    with pytest.raises(ValueError, match="asset per drone"):
        MultiEnvironment(scene_directory, config=EnvConfig(preset="x500_fpv_imu"), seed=6)
    config = EnvConfig(instances=2, cam_width=16, cam_height=16, preset="x500_fpv_imu", task="visual_inertial_localization")
    env = MultiEnvironment(scene_directory, config=config, seed=6)
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
        with pytest.raises(ValueError, match="all-or-none"):
            env.render(np.array([1, 0], dtype=np.uint8))
        frames = []
        for step in range(2 * env.frame_stride + 1):
            env.render(mask if step == 0 else np.zeros(env.total_instances, dtype=np.uint8))
            frames.append(env.frames())
            env.step()
            imu = env.observe_imu()
            assert imu.shape == (2, 8)
            assert np.isfinite(imu).all()
            frame_age = step % env.frame_stride
            np.testing.assert_allclose(imu[:, 6], frame_age / env.frame_stride, atol=1e-6)
            assert (imu[:, 7] == (1.0 if frame_age == 0 else 0.0)).all()
        # the camera only advances on frame boundaries: identical within a stride, and fresh
        # once the autopilot has moved the drone toward its first waypoint
        for frame in frames[1:env.frame_stride]:
            np.testing.assert_array_equal(frame, frames[0])
        for _ in range(60):
            env.render(np.zeros(env.total_instances, dtype=np.uint8))
            env.step()
        assert not np.array_equal(env.frames(), frames[0])
        state = env.observe_privileged()
        assert np.isfinite(state).all()
        assert np.linalg.norm(state[:, 0:3] - state[:, 18:21], axis=1).max() < 100, "the route stays in the scene"
    finally:
        env.close()


def test_spec_header_escape_hatch(scene_directory):
    header = Path(__file__).parent / "user_spec_header.h"
    config = EnvConfig(spec_header=str(header))
    env = MultiEnvironment(scene_directory, config=config, seed=5, drone_asset=drone_asset_path())
    try:
        assert env.self_visible
        assert env.max_drones == 3
        assert env.instances_per_environment == 2
        assert env.cam_width == 16 and env.cam_height == 16
        assert env.observation_layout.blocks == {"observation": (0, env.observation_dim)}
        smoke_rollout(env)
    finally:
        env.close()


def test_spec_header_drone_table(scene_directory):
    header = Path(__file__).parent / "user_spec_header.h"
    config = EnvConfig(spec_header=str(header))
    asset = drone_asset_path()
    drones = [{"asset": asset, "parameters": {"camera": {"fov": fov}}} for fov in (50, 70, 90)]
    env = MultiEnvironment(scene_directory, config=config, seed=5, drones=drones)
    try:
        assert env.self_visible
        assert env.max_drones == 3
        assert len(env.drones) == env.max_drones
        seen = set()
        for _ in range(16):
            env.reset()
            for instance in env.instance_parameters():
                seen.add(instance["drone"])
                assert instance["camera"]["fov"] == (50, 70, 90)[instance["drone"]]
        assert seen == {0, 1, 2}
        smoke_rollout(env)
    finally:
        env.close()
    with pytest.raises(ValueError, match="MAX_DRONES in spec_header"):
        MultiEnvironment(scene_directory, config=config, drones=drones + drones[:1])
    with pytest.raises(ValueError, match="asset per drone"):
        MultiEnvironment(scene_directory, config=config, drones=[{}])


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
        MultiEnvironment([source], config=EnvConfig(num_environments=2), seed=0)
    with pytest.raises(ValueError, match="not found"):
        MultiEnvironment(["/nonexistent.glb"], config=EnvConfig(), seed=0)
    with pytest.raises(ValueError, match=r"\.glb scene"):
        MultiEnvironment(tmp_path, config=EnvConfig(), seed=0)


def test_drone_asset_conta_reference(monkeypatch, tmp_path):
    scene = scene_source()
    asset = drone_asset_path()
    scene_hash, asset_hash = conta_store(monkeypatch, tmp_path, scene, asset)
    config = EnvConfig(instances=2, cam_width=16, cam_height=16, preset="x500_fpv", task="target_frame")
    env = MultiEnvironment([f"conta:{scene_hash}"], config=config, seed=3, drone_asset=f"conta:{asset_hash}")
    try:
        smoke_rollout(env)
    finally:
        env.close()
