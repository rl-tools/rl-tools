from dataclasses import replace

import numpy as np
import pytest

from hyperdrone import render
from hyperdrone.env import EnvConfig, MultiEnvironment
from test_multi_environment import rigged_asset_path, scene_directory


OUTPUTS = ("rgb", "depth", "segmentation", "normals", "flow")
CONFIG = EnvConfig(instances_per_environment=2, width=16, height=16, fidelity="low", output=OUTPUTS)


def test_environment_output_config():
    assert CONFIG.key() == replace(CONFIG, output=tuple(reversed(OUTPUTS))).key()
    assert CONFIG.key() != replace(CONFIG, output="rgb").key()
    assert CONFIG.key() != replace(CONFIG, num_cameras_per_agent=2).key()
    with pytest.raises(ValueError, match="require rgb"):
        replace(CONFIG, output="depth")
    with pytest.raises(ValueError, match="segmentation"):
        replace(CONFIG, output="rgb", semantic_segmentation=True)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("task,cameras", [(None, 1), (None, 2), ("target_frame", 1)])
def test_environment_capture_and_partial_reset(scene_directory, device, task, cameras):
    if device == "cuda" and render._backend() != "optix":
        pytest.skip("CUDA Worlds require OptiX")
    parameters = {"cameras": [{}] + [{"fov": 100, "mount": {"pose": {"position": [0.1, 0.1, 0.1]}}}] * (cameras - 1)}
    env = MultiEnvironment(scene_directory, config=replace(CONFIG, device=device, task=task, num_cameras_per_agent=cameras), parameters=parameters, seed=9)
    try:
        env.reset()
        assert env.outputs == OUTPUTS
        assert env.image_channels == 3
        assert env.segmentation_mode == "instance"
        assert np.unique(env.segmentation()).size > 1
        normals = env.normals()
        np.testing.assert_allclose(np.linalg.norm(normals, axis=-1), 1, atol=2e-5)
        np.testing.assert_allclose(env.flow(), 0, atol=2e-4)
        for name, channels, dtype in (("depth", 1, np.float32), ("segmentation", 1, np.uint32),
                                      ("normals", 3, np.float32), ("flow", 2, np.float32),
                                      ("flow_status", 1, np.uint32)):
            accessor = getattr(env, name)
            value = accessor()
            assert value.shape == (2, cameras, 16, 16) + ((channels,) if channels > 1 else ())
            assert value.dtype == dtype
            out = np.empty_like(value)
            assert accessor(out=out) is out
            np.testing.assert_array_equal(value, out)
            with pytest.raises(ValueError, match="shape"):
                accessor(out=out.reshape(-1))
            with pytest.raises(TypeError):
                accessor(out=out.astype(np.float64))
            if cameras > 1:
                with pytest.raises(ValueError, match="buffer size"):
                    accessor(out=out[:, :1].copy())
        actions = np.zeros((2, env.action_dim), dtype=np.float32)
        for _ in range(15):
            env.step(actions)
        moved = env.flow()
        valid = env.flow_status() == render.FlowStatus.VALID
        assert np.max(np.abs(moved[valid])) > 0.01
        for _ in range(5):
            env.step(actions)
        env.reset(np.array([True, False]))
        flow = env.flow()
        np.testing.assert_allclose(flow[0], 0, atol=2e-4)
        assert np.max(np.abs(flow[1][env.flow_status()[1] == render.FlowStatus.VALID])) > 0.001
        np.testing.assert_array_equal(flow, env.flow())
        env.observe()
        np.testing.assert_array_equal(flow, env.flow())
        env.rotate_scene()
        with pytest.raises(RuntimeError, match="reset every instance"):
            env.flow()
        env.reset([True, False])
        with pytest.raises(RuntimeError, match="reset every instance"):
            env.observe()
        env.reset([False, True])
        np.testing.assert_allclose(env.flow(), 0, atol=2e-4)
    finally:
        env.close()


def test_auxiliary_outputs_preserve_rgb_observations(scene_directory):
    config = replace(CONFIG, output=("rgb", "depth", "normals", "flow"), device="cpu")
    base = MultiEnvironment(scene_directory, config=replace(config, output="rgb"), seed=3)
    extra = MultiEnvironment(scene_directory, config=config, seed=3)
    try:
        for env in (base, extra):
            env.reset()
        np.testing.assert_array_equal(base.observe(), extra.observe())
        np.testing.assert_array_equal(base.observe_privileged(), extra.observe_privileged())
        with pytest.raises(RuntimeError, match="no depth"):
            base.depth()
    finally:
        base.close()
        extra.close()


def test_environment_cuda_image_outputs(scene_directory):
    if render._backend() != "optix":
        pytest.skip("CUDA arrays require OptiX")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("PyTorch needs CUDA")
    env = MultiEnvironment([scene_directory / "ProcTHOR-Train-1.glb"] * 2, config=replace(CONFIG, num_environments=2, num_cameras_per_agent=2, device="cuda"), seed=3)
    try:
        env.reset()
        stream = torch.cuda.Stream()
        env.stream = stream.cuda_stream
        for name in ("depth", "segmentation", "normals", "flow", "flow_status"):
            accessor = getattr(env, name)
            expected = accessor()
            with torch.cuda.stream(stream):
                out = torch.empty(expected.shape, dtype=torch.from_numpy(expected).dtype, device="cuda")
                assert accessor(out=out) is out
                np.testing.assert_array_equal(out.cpu().numpy(), expected)
    finally:
        env.close()


@pytest.mark.parametrize("semantic", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_custom_world_multi_agent_motion_blur(scene_directory, tmp_path, semantic, device):
    if device == "cuda" and render._backend() != "optix":
        pytest.skip("CUDA Worlds require OptiX")
    from pathlib import Path

    asset = rigged_asset_path("x500")
    header = tmp_path / "outputs.h"
    source = (Path(__file__).parent / "user_spec_header.h").read_text()
    source = source.replace("        static constexpr TI MAX_DRONES = 3;", f"""
        static constexpr TI N_AGENTS = 2;
        static constexpr TI N_CAMERAS_PER_AGENT = 2;
        static constexpr TI MAX_ENTITY_SLOTS_PER_INSTANCE = 16;
        static constexpr bool ENABLE_MOTION_BLUR = true;
        static constexpr TI MOTION_BLUR_SAMPLES = 2;
        using OUTPUTS = rl_tools::rendering::raytracing::outputs::List<rl_tools::rendering::raytracing::outputs::RGB<rl_tools::rendering::raytracing::outputs::format::Float32>, rl_tools::rendering::raytracing::outputs::Depth<>, rl_tools::rendering::raytracing::outputs::Segmentation<rl_tools::rendering::raytracing::outputs::format::UInt32, rl_tools::utils::typing::conditional_t<({'true' if semantic else 'false'}), rl_tools::rendering::raytracing::outputs::labels::Semantic, rl_tools::rendering::raytracing::outputs::labels::Instance>>, rl_tools::rendering::raytracing::outputs::Normals<>, rl_tools::rendering::raytracing::outputs::Flow<>, rl_tools::rendering::raytracing::outputs::FlowStatus<>, rl_tools::rendering::raytracing::outputs::Probes>;
""")
    header.write_text(source)
    with pytest.raises(ValueError, match="spec_header defines"):
        EnvConfig(spec_header=str(header), output=OUTPUTS)
    env = MultiEnvironment(scene_directory, config=EnvConfig(spec_header=str(header), device=device), drone_asset=str(asset), seed=4)
    try:
        assert env.outputs == OUTPUTS
        assert env.segmentation_mode == ("semantic" if semantic else "instance")
        assert (env.num_agents, env.num_camera_agents, env.num_cameras_per_agent, env.num_views) == (2, 1, 2, 2)
        assert env.action_dim == 8
        env.reset()
        initial_spawns = env.spawn_points()
        assert env.observe_privileged().shape == (2, 36)
        assert all(len(point["poses"]) == 2 for point in initial_spawns)
        assert env.flow().shape == (2, 2, 16, 16, 2)
        np.testing.assert_allclose(env.flow(), 0, atol=1e-3)
        for _ in range(5):
            env.step(np.ones((2, env.action_dim), dtype=np.float32))
        assert np.any(np.abs(env.flow()) > 0.001)
        env.reset(np.array([True, False]), spawn_points=initial_spawns[:1])
        np.testing.assert_allclose(env.flow()[0], 0, atol=1e-3)
        assert not np.isnan(env.depth()).any()
        assert np.isfinite(env.normals()).all()
    finally:
        env.close()


@pytest.mark.parametrize("cameras", [1, 2])
def test_flow_between_intermittent_captures(scene_directory, cameras):
    config = replace(CONFIG, preset="x500_fpv_imu", task="visual_inertial_localization", device="cpu", num_cameras_per_agent=cameras)
    env = MultiEnvironment(scene_directory, config=config, seed=5)
    try:
        env.reset()
        captured = env.flow()
        assert captured.shape == (2, cameras, 16, 16, 2)
        for step in range(1, env.frame_stride * 3 + 1):
            env.step()
            flow = env.flow()
            if step % env.frame_stride:
                np.testing.assert_array_equal(flow, captured)
            else:
                assert np.any(np.abs(flow) > 1e-4)
                captured = flow
        env.reset()
        np.testing.assert_allclose(env.flow(), 0, atol=2e-4)
    finally:
        env.close()
