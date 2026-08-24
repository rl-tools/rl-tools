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
    return dynamics.Sim(num_drones=N, model="crazyflie", device="cpu")


def rollout_positions(sim, seed, steps=50):
    sim.reset(seed=seed)
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        actions = rng.uniform(-0.2, 0.2, size=(sim.num_drones, sim.action_dim)).astype(np.float32)
        sim.step(actions)
    return sim.state.numpy("position").copy()


def test_reset_determinism(sim):
    first = rollout_positions(sim, seed=0)
    second = rollout_positions(sim, seed=0)
    assert np.array_equal(first, second)


def test_seed_variation(sim):
    first = rollout_positions(sim, seed=0)
    second = rollout_positions(sim, seed=1)
    assert not np.array_equal(first, second)


def test_min_throttle_descends(sim):
    sim.reset(seed=0, sample_states=False)
    actions = -np.ones((N, sim.action_dim), dtype=np.float32)
    for _ in range(int(1.0 / sim.dt)):
        sim.step(actions)
    z = sim.state.numpy("position")[:, 2]
    # rotors spin down over their time constants, so this approaches but does not reach
    # pure freefall (-g/2 ~ -4.9m after 1s)
    assert (z < -2.0).all() and (z > -6.0).all()


def test_max_throttle_climbs(sim):
    sim.reset(seed=0, sample_states=False)
    actions = np.ones((N, sim.action_dim), dtype=np.float32)
    for _ in range(int(0.5 / sim.dt)):
        sim.step(actions)
    z = sim.state.numpy("position")[:, 2]
    velocity_z = sim.state.numpy("linear_velocity")[:, 2]
    assert (z > 0.1).all() and (velocity_z > 0.5).all()  # thrust-to-weight > 1, accelerating up


def test_quaternion_stays_normalized(sim):
    rollout_positions(sim, seed=2)
    orientation = sim.state.numpy("orientation")
    norms = np.linalg.norm(orientation, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)


def test_state_write_roundtrip(sim):
    sim.reset(seed=0, sample_states=False)
    positions = np.arange(N * 3, dtype=np.float32).reshape(N, 3)
    sim.state["position"] = positions
    assert np.array_equal(sim.state.numpy("position"), positions)
    # zero-copy view matches on the cpu variant
    assert np.array_equal(np.from_dlpack(sim.state["position"]), positions)


def test_observation_shape(sim):
    sim.reset(seed=0)
    observations = sim.observe()
    assert observations.shape == (N, sim.observation_dim)
    assert np.isfinite(observations).all()


def test_camera_bases_identity_pose(sim):
    sim.reset(seed=0, sample_states=False)
    position = np.tile(np.array([1.0, 2.0, 3.0], dtype=np.float32), (N, 1))
    orientation = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (N, 1))
    sim.state["position"] = position
    sim.state["orientation"] = orientation
    fov, aspect = math.radians(90.0), 2.0
    bases = sim.camera_bases_numpy(fov=fov, aspect=aspect)
    scale = 2.0 * math.tan(fov / 2.0)
    # identity attitude + identity mount: forward +X, right -Y, up +Z (FLU)
    expected_du = np.array([0.0, -scale, 0.0])
    expected_dv = np.array([0.0, 0.0, scale / aspect])
    expected_dir_00 = np.array([1.0, 0.0, 0.0]) - 0.5 * expected_du + 0.5 * expected_dv
    assert np.allclose(bases[:, 0:3], position, atol=1e-5)
    assert np.allclose(bases[:, 3:6], expected_dir_00, atol=1e-5)
    assert np.allclose(bases[:, 6:9], expected_du, atol=1e-5)
    assert np.allclose(bases[:, 9:12], -expected_dv, atol=1e-5)


def test_camera_bases_mount_offset(sim):
    sim.reset(seed=0, sample_states=False)
    position = np.zeros((N, 3), dtype=np.float32)
    # yaw 90 degrees: body +X now points along world +Y
    orientation = np.tile(np.array([math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)], dtype=np.float32), (N, 1))
    sim.state["position"] = position
    sim.state["orientation"] = orientation
    mount = np.array([[1, 0, 0, 0.1], [0, 1, 0, 0.0], [0, 0, 1, 0.0]], dtype=np.float32)
    bases = sim.camera_bases_numpy(mount=mount)
    assert np.allclose(bases[:, 0:3], [0.0, 0.1, 0.0], atol=1e-5)   # offset rotated into world +Y
    forward = bases[:, 3:6] - 0.5 * (-bases[:, 9:12]) + 0.5 * bases[:, 6:9]
    forward = forward / np.linalg.norm(forward, axis=1, keepdims=True)
    assert np.allclose(forward, [0.0, 1.0, 0.0], atol=1e-5)


def test_parameter_roundtrip_and_physics(sim):
    sim.reset(seed=0, sample_states=False)
    nominal_mass = sim.parameters["mass"].copy()
    assert (nominal_mass > 0).all()
    # heavier drones climb slower under identical max throttle
    masses = nominal_mass.copy()
    masses[N // 2:] *= 3.0
    sim.parameters["mass"] = masses
    assert np.allclose(sim.parameters["mass"], masses)
    actions = np.ones((N, sim.action_dim), dtype=np.float32)
    for _ in range(int(0.5 / sim.dt)):
        sim.step(actions)
    z = sim.state.numpy("position")[:, 2]
    assert (z[: N // 2] > z[N // 2:] + 0.05).all()
    sim.reset(seed=0)  # reset restores the preset
    assert np.allclose(sim.parameters["mass"], nominal_mass)


def test_all_model_presets(sim):
    for model in dynamics.MODELS:
        sim.set_model(model)
        sim.reset(seed=0)
        sim.step(np.zeros((N, sim.action_dim), dtype=np.float32))
        assert np.isfinite(sim.state.numpy("position")).all(), model
    sim.set_model("crazyflie")


CUDA_ROLLOUT = """
import json, sys
import numpy as np
from hyperdrone import dynamics
sim = dynamics.Sim(num_drones={n}, model="crazyflie", device="{device}")
sim.reset(seed=3)
actions = np.linspace(-0.2, 0.2, {n} * sim.action_dim, dtype=np.float32).reshape({n}, sim.action_dim)
for _ in range(100):
    sim.step(actions)
sim.synchronize()
print(json.dumps({{
    "position": sim.state.numpy("position").tolist(),
    "orientation": sim.state.numpy("orientation").tolist(),
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
