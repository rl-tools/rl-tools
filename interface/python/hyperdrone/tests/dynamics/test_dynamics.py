import ctypes
import json
import math
import subprocess
import sys

import numpy as np
import pytest

from hyperdrone import dynamics

N = 8


@pytest.fixture(scope="module")
def sim():
    return dynamics.Sim(num_drones=N, device="cpu")


def rollout_positions(sim, steps=50):
    for _ in range(steps):
        sim.step(np.full((sim.num_drones, sim.action_dim), 0.1, dtype=np.float32))
    return sim.state["position"]


def test_reset_determinism():
    first = dynamics.Sim(N, seed=0)
    second = dynamics.Sim(N, seed=0)
    for _ in range(2):
        first.reset(sample_states=True)
        second.reset(sample_states=True)
        assert np.array_equal(rollout_positions(first), rollout_positions(second))


def test_seed_variation():
    first = dynamics.Sim(N, seed=0)
    second = dynamics.Sim(N, seed=1)
    first.reset(sample_states=True)
    second.reset(sample_states=True)
    assert not np.array_equal(rollout_positions(first), rollout_positions(second))


def test_min_throttle_descends(sim):
    sim.reset()
    actions = -np.ones((N, sim.action_dim), dtype=np.float32)
    for _ in range(int(1.0 / sim.dt)):
        sim.step(actions)
    z = sim.state["position"][:, 2]
    # rotors spin down over their time constants, so this approaches but does not reach
    # pure freefall (-g/2 ~ -4.9m after 1s)
    assert (z < -2.0).all() and (z > -6.0).all()


def test_max_throttle_climbs(sim):
    sim.reset()
    actions = np.ones((N, sim.action_dim), dtype=np.float32)
    for _ in range(int(0.5 / sim.dt)):
        sim.step(actions)
    z = sim.state["position"][:, 2]
    velocity_z = sim.state["linear_velocity"][:, 2]
    assert (z > 0.1).all() and (velocity_z > 0.5).all()  # thrust-to-weight > 1, accelerating up


def test_quaternion_stays_normalized(sim):
    rollout_positions(sim)
    orientation = sim.state["orientation"]
    norms = np.linalg.norm(orientation, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)


def test_state_write_roundtrip(sim):
    sim.reset()
    positions = np.arange(N * 3, dtype=np.float32).reshape(N, 3)
    sim.state["position"] = positions
    assert np.array_equal(sim.state["position"], positions)
    # zero-copy view matches on the cpu variant
    assert np.array_equal(np.from_dlpack(sim.state_dlpack("position")), positions)


def test_observation_shape(sim):
    sim.reset(sample_states=True)
    observations = sim.observe()
    assert observations.shape == (N, sim.observation_dim)
    assert np.isfinite(observations).all()


def test_camera_bases_identity_pose(sim):
    sim.reset()
    position = np.tile(np.array([1.0, 2.0, 3.0], dtype=np.float32), (N, 1))
    orientation = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (N, 1))
    sim.state["position"] = position
    sim.state["orientation"] = orientation
    fov_degrees, aspect = 90.0, 2.0
    bases = sim.camera_bases(fov=fov_degrees, aspect=aspect)
    scale = 2.0 * math.tan(math.radians(fov_degrees) / 2.0)
    # identity attitude + identity mount: forward +X, right -Y, up +Z (FLU)
    expected_du = np.array([0.0, -scale, 0.0])
    expected_dv = np.array([0.0, 0.0, scale / aspect])
    expected_dir_00 = np.array([1.0, 0.0, 0.0]) - 0.5 * expected_du + 0.5 * expected_dv
    assert np.allclose(bases[:, 0:3], position, atol=1e-5)
    assert np.allclose(bases[:, 3:6], expected_dir_00, atol=1e-5)
    assert np.allclose(bases[:, 6:9], expected_du, atol=1e-5)
    assert np.allclose(bases[:, 9:12], -expected_dv, atol=1e-5)


def test_camera_bases_mount_offset(sim):
    sim.reset()
    position = np.zeros((N, 3), dtype=np.float32)
    # yaw 90 degrees: body +X now points along world +Y
    orientation = np.tile(np.array([math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)], dtype=np.float32), (N, 1))
    sim.state["position"] = position
    sim.state["orientation"] = orientation
    mount = np.array([[1, 0, 0, 0.1], [0, 1, 0, 0.0], [0, 0, 1, 0.0]], dtype=np.float32)
    bases = sim.camera_bases(mount=mount, aspect=1.0)
    assert np.allclose(bases[:, 0:3], [0.0, 0.1, 0.0], atol=1e-5)   # offset rotated into world +Y
    forward = bases[:, 3:6] - 0.5 * (-bases[:, 9:12]) + 0.5 * bases[:, 6:9]
    forward = forward / np.linalg.norm(forward, axis=1, keepdims=True)
    assert np.allclose(forward, [0.0, 1.0, 0.0], atol=1e-5)


def test_parameter_roundtrip_and_physics(sim):
    sim.reset()
    nominal_mass = sim.parameters["dynamics"]["dynamics"]["mass"]
    mask = np.arange(N) >= N // 2
    sim.set_instance_parameters({"dynamics": {"dynamics": {"mass": nominal_mass * 3}}}, mask)
    masses = np.array([entry["dynamics"]["dynamics"]["mass"] for entry in sim.instance_parameters()])
    np.testing.assert_allclose(masses, np.where(mask, nominal_mass * 3, nominal_mass))
    for _ in range(int(0.5 / sim.dt)):
        sim.step(np.ones((N, sim.action_dim), dtype=np.float32))
    z = sim.state["position"][:, 2]
    assert (z[: N // 2] > z[N // 2:] + 0.05).all()
    sim.reset(mask)
    np.testing.assert_allclose([entry["dynamics"]["dynamics"]["mass"] for entry in sim.instance_parameters()], nominal_mass)


def test_rotor_rpm_calibration(sim):
    sim.reset()
    dynamics_parameters = sim.parameters["dynamics"]["dynamics"]
    assert dynamics_parameters["max_rpm"] > 0
    scale = dynamics_parameters["max_rpm"] / dynamics_parameters["action_limit"]["max"]
    np.testing.assert_allclose(sim.rotor_rpm(), sim.state["motor_command"] * scale, rtol=1e-6)
    mask = np.arange(N) < N // 2
    sim.set_instance_parameters({"dynamics": {"dynamics": {"max_rpm": 2 * dynamics_parameters["max_rpm"]}}}, mask)
    expected = sim.state["motor_command"] * scale * np.where(mask, 2.0, 1.0)[:, None]
    np.testing.assert_allclose(sim.rotor_rpm(), expected, rtol=1e-6)
    sim.reset(mask)
    np.testing.assert_allclose(sim.rotor_rpm(), sim.state["motor_command"] * scale, rtol=1e-6)


def test_all_model_presets():
    for model in dynamics.MODELS:
        sim = dynamics.Sim(N, parameters={"dynamics": {"dynamics": {"model": model}}})
        sim.step(np.zeros((N, sim.action_dim), dtype=np.float32))
        assert np.isfinite(sim.state["position"]).all(), model


CUDA_ROLLOUT = """
import json, sys
import numpy as np
from hyperdrone import dynamics
sim = dynamics.Sim(num_drones={n}, seed=3, device="{device}")
sim.reset(sample_states=True)
actions = np.linspace(-0.2, 0.2, {n} * sim.action_dim, dtype=np.float32).reshape({n}, sim.action_dim)
for _ in range(100):
    sim.step(actions)
sim.synchronize()
print(json.dumps({{
    "position": sim.state["position"].tolist(),
    "orientation": sim.state["orientation"].tolist(),
}}))
"""


def cuda_available():
    try:
        subprocess.run(["nvidia-smi"], capture_output=True, check=True)
        return True
    except (OSError, subprocess.CalledProcessError):
        return False


@pytest.mark.skipif(not cuda_available(), reason="CUDA driver not available")
def test_cpu_cuda_consistency():
    # one process can hold one dynamics variant, so each rollout runs in its own
    # interpreter; resets are host-sampled, so both variants start bitwise-identical and
    # only the step arithmetic differs
    results = {}
    for device in ("cpu", "cuda"):
        run = subprocess.run(
            [sys.executable, "-c", CUDA_ROLLOUT.format(n=4, device=device)],
            capture_output=True, text=True,
        )
        assert run.returncode == 0, run.stdout + run.stderr
        results[device] = json.loads(run.stdout.splitlines()[-1])
    cpu_position = np.array(results["cpu"]["position"])
    cuda_position = np.array(results["cuda"]["position"])
    assert np.allclose(cpu_position, cuda_position, atol=1e-3), (cpu_position, cuda_position)
    assert np.allclose(np.array(results["cpu"]["orientation"]), np.array(results["cuda"]["orientation"]), atol=1e-3)


def test_model_parameters():
    sim = dynamics.Sim(N, parameters={"dynamics": {"dynamics": {"model": "x500"}}})
    x500 = sim.parameters["dynamics"]["dynamics"]
    arm = 0.176776695296636
    np.testing.assert_allclose(x500["rotor_positions"], [[arm, -arm, 0], [-arm, -arm, 0], [-arm, arm, 0], [arm, arm, 0]], atol=1e-6)
    np.testing.assert_allclose(np.asarray(x500["rotor_torque_directions"])[:, 2], [-1, 1, -1, 1])


NOISY_PARAMETERS = {"dynamics": {"mdp": {"action_noise": {"normalized_rpm": 0.05}, "observation_noise": {"position": 0.01, "orientation": 0.01, "linear_velocity": 0.01, "angular_velocity": 0.01}}}}


def test_consecutive_resets_continue_sampling():
    sim = dynamics.Sim(N, seed=7)
    sim.reset(sample_states=True)
    first = sim.state["position"]
    sim.reset(sample_states=True)
    assert not np.array_equal(first, sim.state["position"])


def test_masked_reset_isolates_unselected_drones():
    sim = dynamics.Sim(N, seed=7)
    sim.reset(sample_states=True)
    before = {name: sim.state[name] for name in sim.state.keys()}
    mask = np.arange(N) % 2 == 0
    sim.reset(mask, sample_states=True)
    for name in before:
        assert np.array_equal(before[name][~mask], sim.state[name][~mask])
    assert not np.array_equal(before["position"][mask], sim.state["position"][mask])


def test_observation_reads_do_not_perturb_dynamics():
    first = dynamics.Sim(N, seed=11, parameters=NOISY_PARAMETERS)
    second = dynamics.Sim(N, seed=11, parameters=NOISY_PARAMETERS)
    actions = np.full((N, 4), 0.1, dtype=np.float32)
    for _ in range(20):
        first.observe()
        first.observe()
        first.step(actions)
        second.step(actions)
    for name in first.state.keys():
        assert np.array_equal(first.state[name], second.state[name])


def test_seed_neighbors_do_not_alias_drone_streams():
    first = dynamics.Sim(N, seed=11, parameters=NOISY_PARAMETERS)
    second = dynamics.Sim(N, seed=12, parameters=NOISY_PARAMETERS)
    for _ in range(5):
        first.step(np.zeros((N, 4), dtype=np.float32))
        second.step(np.zeros((N, 4), dtype=np.float32))
    assert not np.array_equal(first.state["position"][1:], second.state["position"][:-1])


def test_nominal_reset_and_snapshot_ownership(sim):
    sim.reset()
    position = sim.state["position"]
    assert np.array_equal(position, np.zeros((N, 3), dtype=np.float32))
    position[:] = 10
    assert np.array_equal(sim.state["position"], np.zeros_like(position))
    assert not np.shares_memory(sim.state["position"], np.from_dlpack(sim.state_dlpack("position")))
    assert np.array_equal(sim.camera_bases(aspect=1.0), np.from_dlpack(sim.camera_bases_dlpack(aspect=1.0)))


def test_published_buffer_mutation_cannot_move_camera(sim):
    sim.reset()
    before = sim.camera_bases(aspect=1.0)
    view = np.from_dlpack(sim.state_dlpack("position"))
    writable = np.ctypeslib.as_array((ctypes.c_float * view.size).from_address(view.ctypes.data)).reshape(view.shape)
    writable[:] = 10
    assert np.array_equal(sim.camera_bases(aspect=1.0), before)
    sim.state["position"] = np.full((N, 3), 2, dtype=np.float32)
    assert np.array_equal(sim.camera_bases(aspect=1.0)[:, :3], np.full((N, 3), 2, dtype=np.float32))


def test_shared_parameter_changes_are_rejected_without_mutation(sim):
    before = sim.instance_parameters()
    with pytest.raises(ValueError, match="shared"):
        sim.set_instance_parameters({"dynamics": {"integration": {"dt": 0.02}}})
    assert sim.instance_parameters() == before


def test_device_default_and_validation(monkeypatch):
    monkeypatch.setenv("HYPERDRONE_DYNAMICS_DEVICE", "cuda")
    assert dynamics.Sim(1).device == "cpu"
    with pytest.raises(ValueError, match="device"):
        dynamics.Sim(1, device="gpu")
    for count in (0, -1, 2.5, True):
        with pytest.raises(ValueError, match="positive integer"):
            dynamics.Sim(count)


def test_import_does_not_register_cuda_provider():
    run = subprocess.run([sys.executable, "-c", "from hyperdrone import dynamics, _cuda; assert not _cuda._providers"], capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr


def test_consumer_stream_forwarding():
    from hyperdrone._dlpack import DLPackBuffer
    calls = []
    producer = DLPackBuffer(None, lambda: "capsule", 2, 3, calls.append)
    assert producer.__dlpack_device__() == (2, 3)
    assert producer.__dlpack__(stream=1) == "capsule"
    producer.__dlpack__(stream=29)
    producer.__dlpack__(stream=-1)
    assert calls == [1, 29]


RESET_ROLLOUT = """
import json
from hyperdrone import dynamics
sim = dynamics.Sim(4, device={device!r}, seed=71)
sim.reset(sample_states=True)
first = {{name: sim.state[name].tolist() for name in sim.state.keys()}}
sim.reset([True, False, True, False], sample_states=True)
print(json.dumps([first, {{name: sim.state[name].tolist() for name in sim.state.keys()}}]))
"""


@pytest.mark.skipif(not cuda_available(), reason="CUDA driver not available")
def test_cpu_cuda_sampled_resets_are_identical():
    results = []
    for device in ("cpu", "cuda"):
        run = subprocess.run([sys.executable, "-c", RESET_ROLLOUT.format(device=device)], capture_output=True, text=True)
        assert run.returncode == 0, run.stdout + run.stderr
        results.append(json.loads(run.stdout.splitlines()[-1]))
    assert results[0] == results[1]


CUDA_HANDOFF = r'''
import ctypes, gc, json, time
import numpy as np
from hyperdrone import dynamics, _cuda

runtime = ctypes.CDLL("libcudart.so")
runtime.cudaSetDevice.argtypes = [ctypes.c_int]
runtime.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
runtime.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
runtime.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
runtime.cudaMemcpyAsync.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
get_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
get_pointer.restype = ctypes.c_void_p

assert runtime.cudaSetDevice(1) == 0
sim = dynamics.Sim(4, device="cuda")
producer = sim.state_dlpack("position")
assert producer.__dlpack_device__() == (2, 1)
assert sim.camera_bases_dlpack(aspect=1.0).__dlpack_device__() == (2, 1)
assert sim.observe_dlpack().__dlpack_device__() == (2, 1)
consumer = ctypes.c_void_p()
assert runtime.cudaStreamCreate(ctypes.byref(consumer)) == 0
for _ in range(100):
    actions = np.full((4, 4), 0.1, dtype=np.float32)
    uploaded = _cuda.upload(actions).view()
    assert uploaded.__dlpack_device__() == (2, 1)
    sim.step(uploaded)
    del uploaded
assert len(sim._pending_inputs) == 100
gc.collect()
callback_type = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
delay = callback_type(lambda _: time.sleep(0.1))
runtime.cudaLaunchHostFunc.argtypes = [ctypes.c_void_p, callback_type, ctypes.c_void_p]
last_actions = _cuda.upload(actions).view()
assert runtime.cudaLaunchHostFunc(ctypes.c_void_p(sim.stream), delay, None) == 0
sim.step(last_actions, stream=sim.stream)
capsule = producer.__dlpack__(stream=consumer.value)
managed = get_pointer(capsule, b"dltensor")
data = ctypes.c_void_p.from_address(managed).value
copied = np.empty((4, 3), dtype=np.float32)
assert runtime.cudaMemcpyAsync(copied.ctypes.data, data, copied.nbytes, 2, consumer) == 0
assert runtime.cudaStreamSynchronize(consumer) == 0
assert np.array_equal(copied, sim.state["position"])
assert not sim._pending_inputs
assert runtime.cudaStreamDestroy(consumer) == 0
reference = dynamics.Sim(4, device="cuda")
for _ in range(101):
    reference.step(actions)
for name in sim.state.keys():
    assert np.array_equal(sim.state[name], reference.state[name]), name
print(json.dumps({"ordinal": producer.__dlpack_device__()[1], "steps": 101}))
'''


@pytest.mark.skipif(not cuda_available(), reason="CUDA driver not available")
def test_cuda_consumer_ordering_and_input_lifetime():
    count = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], capture_output=True, text=True, check=True)
    if len(count.stdout.splitlines()) < 2:
        pytest.skip("the ordinal test needs two CUDA devices")
    run = subprocess.run([sys.executable, "-c", CUDA_HANDOFF], capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr
    assert json.loads(run.stdout.splitlines()[-1]) == {"ordinal": 1, "steps": 101}
