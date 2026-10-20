from contextlib import closing
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from hyperdrone import render
from hyperdrone.camera import CameraSpec, RGB, Depth, Segmentation, Normals, Flow, FlowStatus, Probes, ImageObservation
from hyperdrone import Config, HyperDrone
from test_hyperdrone import rigged_asset_path, scene_directory


OUTPUTS = (RGB("float32"), Depth(), Segmentation(), Normals(), Flow(), FlowStatus())
CONFIG = Config(instances_per_world=2, width=16, height=16, fidelity="low", cameras=(CameraSpec("front", outputs=OUTPUTS),))
# At 4–6 mm depth, float reprojection after an identity reset reaches 1.52e-3 pixels on OptiX.
RESET_FLOW_ATOL = 2e-3


def profiles(count, outputs=OUTPUTS):
    return tuple(CameraSpec(name, outputs=outputs) for name in ("front", "rear")[:count])


def test_environment_output_config():
    assert CONFIG.key() == replace(CONFIG, cameras=profiles(1, tuple(reversed(OUTPUTS)))).key()
    assert CONFIG.key() != replace(CONFIG, cameras=profiles(1, (RGB("float32"),))).key()
    assert CONFIG.key() != replace(CONFIG, cameras=profiles(2)).key()
    assert CONFIG.key() == replace(CONFIG, cameras=(CameraSpec("renamed", outputs=OUTPUTS),), observation=ImageObservation(cameras=("renamed",))).key()
    with pytest.raises(ValueError, match="must declare RGB"):
        replace(CONFIG, cameras=profiles(1, (Depth(),)))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("task,cameras", [(None, 1), (None, 2), ("target_frame", 2)])
def test_environment_capture_and_partial_reset(scene_directory, device, task, cameras):
    if device == "cuda" and render._backend() != "optix":
        pytest.skip("CUDA Worlds require OptiX")
    parameters = {"cameras": {"rear": {"fov": 100, "mount": {"pose": {"position": [0.1, 0.1, 0.1]}}}}} if cameras == 2 else {}
    with closing(HyperDrone(scene_directory, config=replace(CONFIG, device=device, task=task, cameras=profiles(cameras)), parameters=parameters, seed=9)) as env:
        env.reset()
        with pytest.raises(RuntimeError, match="render or observe"):
            env.outputs[Depth()].read()
        env.render()
        assert set(env.outputs) == set(OUTPUTS)
        segmentation = env.outputs[Segmentation()].read()
        assert np.unique(segmentation).size > 1
        normals = env.outputs[Normals()].read()
        hit = segmentation != np.iinfo(np.uint32).max
        np.testing.assert_allclose(np.linalg.norm(normals[hit], axis=-1), 1, atol=2e-5)
        np.testing.assert_array_equal(normals[~hit], 0)
        np.testing.assert_allclose(env.outputs[Flow()].read(), 0, atol=RESET_FLOW_ATOL)
        for descriptor, channels, dtype in ((Depth(), 1, np.float32), (Segmentation(), 1, np.uint32), (Normals(), 3, np.float32), (Flow(), 2, np.float32), (FlowStatus(), 1, np.uint32)):
            handle = env.outputs[descriptor]
            value = handle.read()
            assert value.shape == (2, cameras, 16, 16) + ((channels,) if channels > 1 else ())
            assert value.dtype == dtype
            out = np.empty_like(value)
            assert handle.read(out=out) is out
            np.testing.assert_array_equal(value, out)
            with pytest.raises(ValueError, match="shape and dtype"):
                handle.read(out=out.reshape(-1))
            with pytest.raises(ValueError, match="shape and dtype"):
                handle.read(out=out.astype(np.float64))
        actions = np.zeros((2, env.action_dim), dtype=np.float32)
        for _ in range(15):
            env.step(actions)
        with pytest.raises(RuntimeError, match="render or observe"):
            env.outputs[Flow()].read()
        with pytest.raises(RuntimeError, match="render or observe"):
            env.outputs[Flow()].blocks
        env.render()
        moved = env.outputs[Flow()].read()
        valid = env.outputs[FlowStatus()].read() == render.FlowStatus.VALID
        assert np.max(np.abs(moved[valid])) > 0.01
        for _ in range(5):
            env.step(actions)
        env.reset(np.array([True, False]))
        with pytest.raises(RuntimeError, match="render or observe"):
            env.outputs[Depth()].read()
        env.render()
        flow = env.outputs[Flow()].read()
        np.testing.assert_allclose(flow[0], 0, atol=RESET_FLOW_ATOL)
        assert np.max(np.abs(flow[1][env.outputs[FlowStatus()].read()[1] == render.FlowStatus.VALID])) > 0.001
        np.testing.assert_array_equal(flow, env.outputs[Flow()].read())
        env.observe()
        np.testing.assert_array_equal(flow, env.outputs[Flow()].read())
        if task == "target_frame":
            for _ in range(40):
                env.step(actions)
            env.render()
            before = env.observe_privileged()
            env.reset([False, False])
            np.testing.assert_array_equal(env.observe_privileged(), before)
            with pytest.raises(RuntimeError, match="render or observe"):
                env.outputs[Depth()].read()
            env.render()
            assert env.outputs[Depth()].read().shape == (2, cameras, 16, 16)
        env.rotate_scene()
        with pytest.raises(RuntimeError, match="reset every instance"):
            env.render()
        env.reset([True, False])
        with pytest.raises(RuntimeError, match="reset every instance"):
            env.observe()
        env.reset([False, True])
        env.render()
        np.testing.assert_allclose(env.outputs[Flow()].read(), 0, atol=RESET_FLOW_ATOL)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_auxiliary_outputs_preserve_policy_observations(scene_directory, device):
    if device == "cuda" and render._backend() != "optix":
        pytest.skip("CUDA Worlds require OptiX")
    config = replace(CONFIG, device=device, observation=ImageObservation(cameras=("front",)))
    mixed = (CameraSpec("front", outputs=OUTPUTS), CameraSpec("diagnostic", outputs=(RGB("float32"), RGB(format="rgba8"))), CameraSpec("down", outputs=(Depth(),)))
    with closing(HyperDrone(scene_directory, config=replace(config, cameras=profiles(1, (RGB("float32"),))), seed=3)) as base, closing(HyperDrone(scene_directory, config=replace(config, cameras=mixed), seed=3)) as extra:
        for env in (base, extra):
            env.reset()
        assert base.observation_dim == extra.observation_dim == 16 * 16 * 3
        assert extra.num_observation_views == 1
        for _ in range(3):
            np.testing.assert_array_equal(base.observe(), extra.observe())
            np.testing.assert_array_equal(base.observe_privileged(), extra.observe_privileged())
            for env in (base, extra):
                env.step(np.zeros((2, env.action_dim), dtype=np.float32))
        with pytest.raises(KeyError):
            base.outputs[Depth()]
        assert extra.outputs[RGB("float32")].layout.cameras == ((0, "front"), (0, "diagnostic"))
        assert extra.outputs[Depth()].layout.cameras == ((0, "front"), (0, "down"))
        assert extra.outputs[RGB(format="rgba8")].layout.shape == (2, 1, 16, 16, 4)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_raw_only_sparse_outputs_and_blocks(scene_directory, device):
    if device == "cuda" and render._backend() != "optix":
        pytest.skip("CUDA Worlds require OptiX")
    cameras = (CameraSpec("left", outputs=(Depth(), FlowStatus(direction="forward"))), CameraSpec("display", outputs=(RGB(format="rgba8"),)), CameraSpec("right", outputs=(Depth(), Probes())))
    config = replace(CONFIG, num_worlds=2, cameras=cameras, observation=None, device=device)
    env = HyperDrone([scene_directory / "ProcTHOR-Train-1.glb"] * 2, config=config, seed=4)
    try:
        assert env.observation_dim == 0 and not env.has_image_observation
        env.reset()
        with pytest.raises(RuntimeError, match="no policy observation"):
            env.observe()
        env.render()
        assert Flow() not in env.outputs and FlowStatus(direction="forward") in env.outputs
        handle = env.outputs[Depth()]
        assert handle.layout.shape == (4, 2, 16, 16)
        assert handle.layout.cameras == ((0, "left"), (0, "right"))
        expected = handle.read()
        blocks = handle.blocks
        assert len(blocks) == 2
        if handle.layout.device[0] == 1:
            views = [np.from_dlpack(block) for block in blocks]
            np.testing.assert_array_equal(np.concatenate(views), expected)
        rgba = env.outputs[RGB(format="rgba8")]
        np.testing.assert_array_equal(rgba.read().reshape(4, 1, 16, 16, 4), rgba.packed.read().view(np.uint8).reshape(4, 1, 16, 16, 4))
        with pytest.raises(RuntimeError, match="probe before"):
            env.outputs[Probes()].distance.read()
        env.probe()
        probes = env.outputs[Probes()]
        assert probes.distance.layout.cameras == ((0, "right"),)
        assert probes.distance.layout.shape[:2] == (4, 1)
        assert probes.hit.read().dtype == np.int32
        probe_snapshot = probes.distance.read()
        if probes.distance.layout.device[0] == 1:
            np.testing.assert_array_equal(np.concatenate([np.from_dlpack(block) for block in probes.distance.blocks]), probe_snapshot)
        env.step(np.zeros((env.total_instances, env.action_dim), dtype=np.float32))
        with pytest.raises(RuntimeError, match="render the camera poses"):
            env.probe()
        with pytest.raises(RuntimeError, match="render and probe"):
            probes.distance.read()
        with pytest.raises(RuntimeError, match="render or observe"):
            handle.read()
        with pytest.raises(RuntimeError, match="render or observe"):
            handle.blocks
        env.render()
        with pytest.raises(RuntimeError, match="render and probe"):
            probes.distance.read()
        env.probe()
        probes.distance.read()
        expected = handle.read()
        env.rotate_scene()
        env.close()
        if handle.layout.device[0] == 1:
            np.testing.assert_array_equal(np.concatenate([np.from_dlpack(block) for block in blocks]), expected)
        with pytest.raises(RuntimeError, match="closed"):
            handle.read()
    finally:
        env.close()


def test_environment_cuda_image_outputs(scene_directory):
    if render._backend() != "optix":
        pytest.skip("CUDA arrays require OptiX")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("PyTorch needs CUDA")
    config = replace(CONFIG, num_worlds=2, cameras=profiles(2), device="cuda")
    with closing(HyperDrone([scene_directory / "ProcTHOR-Train-1.glb"] * 2, config=config, seed=3)) as env:
        env.reset()
        producer = torch.cuda.Stream()
        consumer = torch.cuda.Stream()
        env.stream = producer.cuda_stream
        env.render()
        for descriptor in OUTPUTS:
            handle = env.outputs[descriptor]
            expected = handle.read()
            with torch.cuda.stream(consumer):
                out = torch.empty(expected.shape, dtype=torch.from_numpy(expected).dtype, device="cuda")
                assert handle.read(out=out, stream=consumer.cuda_stream) is out
                np.testing.assert_array_equal(out.cpu().numpy(), expected)
                views = [torch.from_dlpack(block) for block in handle.blocks]
                np.testing.assert_array_equal(torch.cat(views).cpu().numpy(), expected)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_custom_world_multi_agent_motion_blur(scene_directory, tmp_path, device):
    if device == "cuda" and render._backend() != "optix":
        pytest.skip("CUDA Worlds require OptiX")
    for instances_per_world in (2, 9):
        _check_custom_world_multi_agent_motion_blur(scene_directory, tmp_path, device, instances_per_world)


def _check_custom_world_multi_agent_motion_blur(scene_directory, tmp_path, device, instances_per_world):
    asset = rigged_asset_path("x500")
    header = tmp_path / f"outputs_{instances_per_world}.h"
    source = (Path(__file__).parent / "user_spec_header.h").read_text()
    source = source.replace("        static constexpr TI MAX_DRONES = 3;", """
        static constexpr TI N_AGENTS = 2;
        static constexpr TI N_CAMERA_AGENTS = 2;
        static constexpr TI MAX_ENTITY_SLOTS_PER_INSTANCE = 16;
        static constexpr bool ENABLE_MOTION_BLUR = true;
        static constexpr TI MOTION_BLUR_SAMPLES = 2;
        using OUTPUT_PROFILE = rl_tools::rendering::raytracing::outputs::List<rl_tools::rendering::raytracing::outputs::RGB<rl_tools::rendering::raytracing::outputs::format::Float32>, rl_tools::rendering::raytracing::outputs::Depth<>, rl_tools::rendering::raytracing::outputs::Normals<>, rl_tools::rendering::raytracing::outputs::Flow<>, rl_tools::rendering::raytracing::outputs::FlowStatus<>>;
        using DEPTH_PROFILE = rl_tools::rendering::raytracing::outputs::List<rl_tools::rendering::raytracing::outputs::Depth<>>;
        using CAMERA_OUTPUTS = rl_tools::rl::environments::hyperdrone::CameraOutputs<DEPTH_PROFILE, OUTPUT_PROFILE, DEPTH_PROFILE>;
        using OBSERVATION = rl_tools::rl::environments::hyperdrone::ImageObservation<1>;
""").replace("        static constexpr TI N_CAMERA_AGENTS = 1;", "")
    source = source.replace("INSTANCES_PER_WORLD = 2;", f"INSTANCES_PER_WORLD = {instances_per_world};")
    header.write_text(source)
    with closing(HyperDrone([scene_directory / "ProcTHOR-Train-1.glb"] * 2, config=Config(spec_header=str(header), num_worlds=2, device=device), drone_asset=str(asset), seed=4)) as env:
        assert env.total_instances == 2 * instances_per_world
        assert (env.num_agents, env.num_camera_agents, env.num_cameras_per_agent, env.num_views) == (2, 2, 3, 6)
        assert env.outputs[RGB("float32")].layout.cameras == ((0, "camera1"), (1, "camera1"))
        assert env.outputs[Depth()].layout.cameras == tuple((agent, f"camera{mount}") for agent in range(2) for mount in range(3))
        assert env.action_dim == 8
        env.reset()
        initial_spawns = env.spawn_points()
        assert env.observe_privileged().shape == (env.total_instances, 36)
        assert all(len(point["poses"]) == 2 for point in initial_spawns)
        env.render()
        assert env.outputs[Flow()].read().shape == (env.total_instances, 2, 16, 16, 2)
        np.testing.assert_allclose(env.outputs[Flow()].read(), 0, atol=RESET_FLOW_ATOL)
        for _ in range(5):
            env.step(np.ones((env.total_instances, env.action_dim), dtype=np.float32))
        env.render()
        moved = env.outputs[Flow()].read()
        status = env.outputs[FlowStatus()].read()
        assert np.any(np.abs(moved[status == render.FlowStatus.VALID]) > 0.001)
        mask = np.zeros(env.total_instances, dtype=bool)
        mask[::instances_per_world] = True
        if instances_per_world > 2:
            mask[instances_per_world - 1::instances_per_world] = True
        env.reset(mask, spawn_points=[spawn for spawn, selected in zip(initial_spawns, mask) if selected])
        env.render()
        flow = env.outputs[Flow()].read()
        np.testing.assert_allclose(flow[mask], 0, atol=RESET_FLOW_ATOL)
        np.testing.assert_array_equal(flow[~mask], moved[~mask])
        np.testing.assert_array_equal(env.outputs[FlowStatus()].read()[~mask], status[~mask])
        assert not np.isnan(env.outputs[Depth()].read()).any()
        assert np.isfinite(env.outputs[Normals()].read()).all()


@pytest.mark.parametrize("cameras", [1, 2])
def test_flow_between_intermittent_captures(scene_directory, cameras):
    config = replace(CONFIG, preset="x500_fpv_imu", task="visual_inertial_localization", device="cpu", cameras=profiles(cameras))
    with closing(HyperDrone(scene_directory, config=config, seed=5)) as env:
        env.reset()
        env.render()
        captured = env.outputs[Flow()].read()
        assert captured.shape == (2, cameras, 16, 16, 2)
        for step in range(1, env.frame_stride * 3 + 1):
            env.step()
            env.render()
            flow = env.outputs[Flow()].read()
            if step % env.frame_stride:
                np.testing.assert_array_equal(flow, captured)
            else:
                assert np.any(np.abs(flow) > 1e-4)
                captured = flow
        env.reset()
        env.render()
        np.testing.assert_allclose(env.outputs[Flow()].read(), 0, atol=2e-4)


@pytest.mark.parametrize("acquisition", ["render", "observe"])
def test_intermittent_capture_preserves_outputs_and_probes(scene_directory, acquisition):
    config = replace(CONFIG, preset="x500_fpv_imu", task="visual_inertial_localization", device="cpu", cameras=profiles(1, (*OUTPUTS, Probes())))
    with closing(HyperDrone(scene_directory, config=config, seed=5)) as env:
        acquire = getattr(env, acquisition)
        depth = env.outputs[Depth()]
        probes = env.outputs[Probes()].distance
        env.reset()
        env.step()
        with pytest.raises(RuntimeError, match="render or observe"):
            depth.read()
        with pytest.raises(RuntimeError, match="render the camera poses"):
            env.probe()
        acquire()
        captured = depth.read()
        env.probe()
        probed = probes.read()
        for _ in range(2, env.frame_stride):
            env.step()
            np.testing.assert_array_equal(depth.read(), captured)
            np.testing.assert_array_equal(probes.read(), probed)
            assert len(depth.blocks) == len(probes.blocks) == 1
            acquire()
            np.testing.assert_array_equal(depth.read(), captured)
            np.testing.assert_array_equal(probes.read(), probed)
            env.probe()
            np.testing.assert_array_equal(probes.read(), probed)
        env.step()
        with pytest.raises(RuntimeError, match="render or observe"):
            depth.read()
        with pytest.raises(RuntimeError, match="render or observe"):
            depth.blocks
        with pytest.raises(RuntimeError, match="render and probe"):
            probes.read()
        with pytest.raises(RuntimeError, match="render the camera poses"):
            env.probe()
        acquire()
        captured = depth.read()
        assert np.any(np.abs(env.outputs[Flow()].read()) > 1e-4)
        with pytest.raises(RuntimeError, match="render and probe"):
            probes.read()
        env.probe()
        for _ in range(env.frame_stride):
            env.step()
        with pytest.raises(RuntimeError, match="render or observe"):
            depth.read()
        env.step()
        assert np.any(depth.read() != captured)
        with pytest.raises(RuntimeError, match="render and probe"):
            probes.read()
        acquire()
        with pytest.raises(RuntimeError, match="render and probe"):
            probes.blocks
        env.probe()
        probes.read()
        env.reset()
        with pytest.raises(RuntimeError, match="render or observe"):
            depth.read()
        with pytest.raises(RuntimeError, match="render the camera poses"):
            env.probe()
        acquire()
        np.testing.assert_allclose(env.outputs[Flow()].read(), 0, atol=2e-4)
        with pytest.raises(RuntimeError, match="render and probe"):
            probes.read()
        env.probe()
        probes.read()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_cuda_batched_transport_without_tensor_framework(scene_directory, device):
    if render._backend() != "optix":
        pytest.skip("CUDA output transport requires OptiX")
    import ctypes
    from hyperdrone import _cuda

    runtime = ctypes.CDLL("libcudart.so")
    runtime.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    runtime.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
    runtime.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
    runtime.cudaMemcpyAsync.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
    runtime.cudaMemcpy2DAsync.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
    capsule_pointer = ctypes.pythonapi.PyCapsule_GetPointer
    capsule_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
    capsule_pointer.restype = ctypes.c_void_p
    consumer = ctypes.c_void_p()
    assert runtime.cudaStreamCreate(ctypes.byref(consumer)) == 0
    cameras = (CameraSpec("front", outputs=(RGB("float32"), Depth())), CameraSpec("aux", outputs=(RGB(format="rgba8"), Probes())))
    config = replace(CONFIG, num_worlds=2, cameras=cameras, device=device)
    env = HyperDrone([scene_directory / "ProcTHOR-Train-1.glb"] * 2, config=config, seed=11)
    try:
        env.reset()
        env.render()
        env.probe()
        for handle in (env.outputs[RGB("float32")], env.outputs[Depth()], env.outputs[RGB(format="rgba8")], env.outputs[Probes()].distance, env.outputs[Probes()].hit):
            snapshot = handle.read()
            for member, block in enumerate(handle.blocks):
                capsule = block.__dlpack__(stream=consumer.value)
                pointer = ctypes.c_void_p.from_address(capsule_pointer(capsule, b"dltensor")).value
                copied = np.empty_like(snapshot[:2])
                if handle.descriptor == Probes():
                    assert runtime.cudaMemcpy2DAsync(copied.ctypes.data, 4, pointer, 8, 4, copied.size, 2, consumer) == 0
                else:
                    assert runtime.cudaMemcpyAsync(copied.ctypes.data, pointer, copied.nbytes, 2, consumer) == 0
                assert runtime.cudaStreamSynchronize(consumer) == 0
                np.testing.assert_array_equal(copied, snapshot[member * 2:(member + 1) * 2])
            if snapshot.dtype == np.float32:
                destination = _cuda.upload(np.zeros_like(snapshot)).view()
                handle.read(out=destination, stream=consumer.value)
                destination_capsule = destination.__dlpack__()
                pointer = ctypes.c_void_p.from_address(capsule_pointer(destination_capsule, b"dltensor")).value
                copied = np.empty_like(snapshot)
                assert runtime.cudaMemcpyAsync(copied.ctypes.data, pointer, copied.nbytes, 2, consumer) == 0
                assert runtime.cudaStreamSynchronize(consumer) == 0
                np.testing.assert_array_equal(copied, snapshot)
        handle = env.outputs[Depth()]
        expected = handle.read()
        retained = handle.blocks
        env.rotate_scene()
        env.close()
        for member, block in enumerate(retained):
            capsule = block.__dlpack__(stream=consumer.value)
            pointer = ctypes.c_void_p.from_address(capsule_pointer(capsule, b"dltensor")).value
            copied = np.empty_like(expected[:2])
            assert runtime.cudaMemcpyAsync(copied.ctypes.data, pointer, copied.nbytes, 2, consumer) == 0
            assert runtime.cudaStreamSynchronize(consumer) == 0
            np.testing.assert_array_equal(copied, expected[member * 2:(member + 1) * 2])
    finally:
        env.close()
        assert runtime.cudaStreamDestroy(consumer) == 0
