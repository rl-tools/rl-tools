import gc
import os
import threading
import time
import weakref

import pytest

from hyperdrone.camera import CameraSpec, RGB, ImageObservation
from hyperdrone.env import HyperDrone


class NativeEnvironment:
    def __init__(self):
        self.calls = []
        self.open = False
        self.available = True

    def spy_open(self):
        self.calls.append("open")
        if not self.available:
            raise RuntimeError("spy was not compiled")
        self.open = True

    def spy_update(self):
        self.calls.append("update")
        return self.open

    def spy_close(self):
        self.calls.append("close")
        self.open = False


@pytest.fixture
def environment():
    env = HyperDrone.__new__(HyperDrone)
    env._environment = NativeEnvironment()
    env._spy = None
    yield env
    env.close()


def test_context_closes_on_exception_and_old_handle_cannot_close_reopened_spy(environment):
    native = environment._environment
    with pytest.raises(ValueError):
        with environment.spy() as first:
            assert first.update()
            with pytest.raises(RuntimeError, match="already has an open spy"):
                environment.spy()
            raise ValueError("training failed")
    with environment.spy() as second:
        first.close()
        assert not first.update()
        assert second.update()
    assert native.calls == ["open", "update", "close", "open", "update", "close"]


def test_window_close_leaves_environment_alive(environment):
    native = environment._environment
    spy = environment.spy()
    native.open = False
    assert not spy.update()
    assert not spy.update()
    assert environment._environment is native
    with environment.spy() as reopened:
        assert reopened.update()


def test_environment_close_closes_spy_and_rejects_reopen(environment):
    native = environment._environment
    spy = environment.spy()
    environment.close()
    environment.close()
    assert not spy.update()
    assert native.calls == ["open", "close"]
    with pytest.raises(RuntimeError, match="closed environment"):
        environment.spy()


def test_spy_keeps_native_environment_alive_without_reference_cycle():
    env = HyperDrone.__new__(HyperDrone)
    env._environment = NativeEnvironment()
    env._spy = None
    native = weakref.ref(env._environment)
    owner = weakref.ref(env)
    spy = env.spy()
    del env
    gc.collect()
    assert owner() is None
    assert native() is not None
    assert spy.update()
    spy.close()
    assert native() is None


def test_missing_native_dependency_does_not_leave_open_wrapper(environment):
    environment._environment.available = False
    with pytest.raises(RuntimeError, match="not compiled"):
        environment.spy()
    assert environment._spy is None
    environment._environment.available = True
    with environment.spy() as spy:
        assert spy.update()


def test_main_thread_owns_all_window_operations(environment):
    errors = []

    def worker(operation):
        try:
            operation()
        except RuntimeError as error:
            errors.append(str(error))

    thread = threading.Thread(target=worker, args=(environment.spy,))
    thread.start()
    thread.join()
    with environment.spy() as spy:
        for operation in (spy.update, spy.close):
            thread = threading.Thread(target=worker, args=(operation,))
            thread.start()
            thread.join()
        assert spy.update()
    assert len(errors) == 3
    assert all("main thread" in error for error in errors)


@pytest.mark.skipif(os.environ.get("HYPERDRONE_TEST_SPY") != "1", reason="set HYPERDRONE_TEST_SPY=1 with a display to test the native GUI")
@pytest.mark.parametrize("num_cameras_per_agent", [1, 2])
def test_native_spy_preserves_rollout_and_environment_lifecycle(tmp_path, num_cameras_per_agent):
    import numpy as np

    from hyperdrone.camera import CameraSpec, RGB, ImageObservation
    from hyperdrone.env import EnvConfig
    from test_taxonomy import room_glb

    scenes = [room_glb(tmp_path / "first.glb", taxonomy=None), room_glb(tmp_path / "second.glb", taxonomy=None, extent=6)]
    config = EnvConfig(instances_per_world=2, width=32, height=32, fidelity="low", cameras=tuple(CameraSpec(name, outputs=(RGB("float32"),)) for name in ("front", "rear")[:num_cameras_per_agent]), observation=ImageObservation(cameras=("front", "rear")[:num_cameras_per_agent]))
    control = HyperDrone(scenes, config=config, seed=1337)
    observed = HyperDrone(scenes, config=config, seed=1337)
    try:
        control.reset()
        observed.reset()
        actions = np.zeros((2, observed.action_dim), dtype=np.float32)
        with observed.spy() as spy:
            for step in range(45):
                np.testing.assert_array_equal(control.observe(), observed.observe())
                np.testing.assert_array_equal(control.observe_privileged(), observed.observe_privileged())
                assert spy.update()
                for environment in (control, observed):
                    environment.step(actions)
                np.testing.assert_array_equal(control.rewards(), observed.rewards())
                np.testing.assert_array_equal(control.terminated(), observed.terminated())
                if step == 20:
                    for environment in (control, observed):
                        environment.rotate_scene()
                        environment.reset()
                time.sleep(0.025)
        assert not spy.update()
        with observed.spy() as reopened:
            for _ in range(15):
                observed.observe()
                assert reopened.update()
                time.sleep(0.025)
        observed.step(actions)
        assert np.isfinite(observed.observe()).all()
        active = observed.spy()
        observed.close()
        assert not active.update()
    finally:
        control.close()
        observed.close()
