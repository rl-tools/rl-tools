"""The device-resident environment on OptiX: CUDA arrays are read and written in place on the
environment's stream and agree bit-exactly with the host (numpy) path."""
import os
from pathlib import Path

import numpy as np
import pytest

from hyperdrone import _cuda as cuda, render
from hyperdrone.env import EnvConfig, HyperDrone
from hyperdrone._jit import source_root

pytestmark = pytest.mark.skipif(
    render._backend() != "optix",
    reason="the device-resident environment needs the OptiX backend",
)

CONFIG = EnvConfig(num_worlds=2, instances_per_world=2, width=16, height=16, fidelity="low", device="cuda")
HOST_WORLD_CONFIG = EnvConfig(num_worlds=1, instances_per_world=2, width=16, height=16, fidelity="low", device="cpu")


@pytest.fixture(scope="module")
def scenes():
    source = Path(os.environ.get("HYPERDRONE_TEST_SCENE_DIR", source_root() / "tests" / "data")) / "ProcTHOR-Train-1.glb"
    if not source.exists():
        pytest.skip(f"no ProcTHOR test scene at {source}")
    return [str(source), str(source)]


@pytest.fixture
def pair(scenes):
    device = HyperDrone(scenes, config=CONFIG, seed=11)
    host = HyperDrone(scenes, config=CONFIG, seed=11)
    try:
        yield device, host
    finally:
        device.close()
        host.close()


@pytest.fixture
def cupy():
    return pytest.importorskip("cupy")


def assert_same_state(device, host):
    np.testing.assert_array_equal(device.observe(), host.observe())
    np.testing.assert_array_equal(device.observe_privileged(), host.observe_privileged())
    np.testing.assert_array_equal(device.rewards(), host.rewards())
    np.testing.assert_array_equal(device.terminated(), host.terminated())


def test_environment_is_device_resident(pair):
    device, _ = pair
    assert device.device == "cuda"
    assert device.stream != 0
    device.reset()
    assert device.observe().shape == (device.total_instances, device.observation_dim)
    device.synchronize()


def test_uploaded_actions_match_host_actions(pair):
    device, host = pair
    for environment in (device, host):
        environment.reset()
    rng = np.random.default_rng(3)
    no_reset = np.zeros(device.total_instances, dtype=bool)
    for _ in range(3):
        actions = rng.uniform(-1, 1, size=(device.total_instances, device.action_dim)).astype(np.float32)
        device.step(cuda.upload(actions[None]).view(0))
        host.step(actions)
        assert_same_state(device, host)


def test_cuda_arrays_in_place_on_the_bound_stream(pair, cupy):
    device, host = pair
    assert device.device_id == cupy.cuda.Device().id
    total = device.total_instances
    stream = cupy.cuda.Stream(non_blocking=True)
    device.stream = stream.ptr
    assert device.stream == stream.ptr
    observations = cupy.empty((total, device.observation_dim), dtype=cupy.float32)
    privileged = cupy.empty((total, device.observation_dim_privileged), dtype=cupy.float32)
    rewards = cupy.empty(total, dtype=cupy.float32)
    terminated = cupy.empty(total, dtype=cupy.bool_)
    rng = np.random.default_rng(5)
    mask = np.ones(total, dtype=bool)
    for _ in range(4):
        with stream:
            # temporaries die right after each call; the allocator reuses them in stream order
            device.reset(cupy.asarray(mask))
            host.reset(mask)
            assert device.observe(out=observations) is observations
            device.observe_privileged(out=privileged)
            np.testing.assert_array_equal(observations.get(), host.observe())
            np.testing.assert_array_equal(privileged.get(), host.observe_privileged())
            actions = rng.uniform(-1, 1, size=(total, device.action_dim)).astype(np.float32)
            device.step(cupy.asarray(actions))
            host.step(actions)
            device.rewards(out=rewards)
            device.terminated(out=terminated)
            np.testing.assert_array_equal(rewards.get(), host.rewards())
            np.testing.assert_array_equal(terminated.get(), host.terminated())
        mask = rng.random(total) < 0.5


SPIN = r'''
extern "C" __global__ void spin(long long cycles){
    const long long start = clock64();
    while(clock64() - start < cycles){
    }
}
'''


def test_stream_switch_waits_for_pending_work(pair, cupy):
    device, host = pair
    old, new = cupy.cuda.Stream(non_blocking=True), cupy.cuda.Stream(non_blocking=True)
    device.stream = old.ptr
    for environment in (device, host):
        environment.reset()
    actions = np.random.default_rng(7).uniform(-1, 1, size=(device.total_instances, device.action_dim)).astype(np.float32)
    with old:
        # keeps the step pending on the old stream while the new one reads the stepped state
        cupy.RawKernel(SPIN, "spin")((1,), (1,), (cupy.int64(500_000_000),))
        device.step(cupy.asarray(actions))
    host.step(actions)
    device.stream = new.ptr
    privileged = cupy.empty((device.total_instances, device.observation_dim_privileged), dtype=cupy.float32)
    with new:
        device.observe_privileged(out=privileged)
        np.testing.assert_array_equal(privileged.get(), host.observe_privileged())


@pytest.fixture
def host_world(scenes):
    environment = HyperDrone(scenes[:1], config=HOST_WORLD_CONFIG, seed=0)
    try:
        yield environment
    finally:
        environment.close()


def test_host_world_has_no_stream(host_world):
    assert host_world.device == "cpu" and host_world.stream == 0
    with pytest.raises(ValueError, match="no stream"):
        host_world.stream = 1


def test_rejects_foreign_arrays(pair, host_world, cupy):
    device, _ = pair
    total = device.total_instances
    with pytest.raises(TypeError):
        device.step(cupy.zeros((total, device.action_dim), dtype=cupy.float64))
    with pytest.raises(ValueError, match="buffer size"):
        device.step(cupy.zeros(1, dtype=cupy.float32))
    with pytest.raises(ValueError, match="buffer size"):
        device.observe(out=cupy.empty(1, dtype=cupy.float32))
    with pytest.raises(ValueError, match="runs on the host"):
        host_world.reset(cupy.ones(host_world.total_instances, dtype=cupy.bool_))
