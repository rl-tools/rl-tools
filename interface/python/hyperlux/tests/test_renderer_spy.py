import gc
import os
import sys
import threading
import time
import weakref
from types import SimpleNamespace

import pytest

from hyperlux import HyperLux
from hyperlux import _spy_ipython


class NativeRenderer:
    def __init__(self):
        self.calls = []
        self.open = False
        self.available = True

    def spy_open(self, camera):
        self.calls.append(("open", camera))
        if not self.available:
            raise RuntimeError("spy was not compiled")
        self.open = True

    def spy_update(self):
        self.calls.append("poll")
        return self.open

    def spy_close(self):
        self.calls.append("close")
        self.open = False

    def update(self):
        self.calls.append("publish")

    def synchronize(self):
        pass

    def release_inputs(self):
        return 0


@pytest.fixture
def renderer():
    renderer = HyperLux.__new__(HyperLux)
    renderer._renderer = NativeRenderer()
    renderer.config = SimpleNamespace(num_overlays=1)
    renderer._spy = None
    yield renderer
    if renderer._spy is not None:
        renderer._spy.close()


@pytest.fixture
def shell(monkeypatch):
    pytest.importorskip("IPython")
    from IPython.core.events import EventManager
    from IPython.terminal import pt_inputhooks

    class Shell:
        active_eventloop = None
        _inputhook = None

        def __init__(self):
            self.events = EventManager(self, ["post_run_cell"])

        def enable_gui(self, name=None):
            if name is None:
                self.active_eventloop = self._inputhook = None
            else:
                self.active_eventloop, self._inputhook = pt_inputhooks.get_inputhook_name_and_func(name)

    shell = Shell()
    monkeypatch.setattr(_spy_ipython, "_interactive_shell", lambda: shell)
    callbacks = []
    monkeypatch.setattr(_spy_ipython.atexit, "register", callbacks.append)
    monkeypatch.setattr(_spy_ipython.atexit, "unregister", callbacks.remove)
    shell.exit_callbacks = callbacks
    return shell


def test_manual_lifecycle_and_context_cleanup(renderer):
    native = renderer._renderer
    with pytest.raises(ValueError):
        with renderer.spy(camera=1) as first:
            renderer.update()
            assert first.update()
            with pytest.raises(RuntimeError, match="already has an open spy"):
                renderer.spy()
            raise ValueError("script failed")
    with renderer.spy() as second:
        first.close()
        assert not first.update()
        assert second.update()
    assert native.calls == [("open", 1), "publish", "poll", "close", ("open", 0), "poll", "close"]


def test_window_close_leaves_renderer_alive(renderer):
    native = renderer._renderer
    spy = renderer.spy()
    native.open = False
    assert not spy.update()
    assert not spy.update()
    assert renderer._renderer is native
    with renderer.spy() as reopened:
        assert reopened.update()
    with pytest.raises(RuntimeError, match="closed"):
        spy.__enter__()


def test_spy_keeps_native_renderer_alive_without_reference_cycle():
    renderer = HyperLux.__new__(HyperLux)
    renderer._renderer = NativeRenderer()
    renderer.config = SimpleNamespace(num_overlays=0)
    renderer._spy = None
    native = weakref.ref(renderer._renderer)
    owner = weakref.ref(renderer)
    spy = renderer.spy()
    del renderer
    gc.collect()
    assert owner() is None
    assert native() is not None
    assert spy.update()
    spy.close()
    assert native() is None


def test_missing_native_dependency_leaves_no_open_wrapper(renderer):
    renderer._renderer.available = False
    with pytest.raises(RuntimeError, match="not compiled"):
        renderer.spy()
    assert renderer._spy is None
    renderer._renderer.available = True
    with renderer.spy() as spy:
        assert spy.update()


def test_missing_ipython_only_prevents_interactive_mode(renderer, monkeypatch):
    monkeypatch.setitem(sys.modules, "IPython", None)
    with pytest.raises(RuntimeError, match="requires terminal IPython"):
        renderer.spy(interactive=True)
    assert renderer._renderer.calls == []
    with renderer.spy() as spy:
        assert spy.update()


def test_main_thread_owns_window_operations(renderer):
    errors = []

    def worker(operation):
        try:
            operation()
        except RuntimeError as error:
            errors.append(str(error))

    thread = threading.Thread(target=worker, args=(renderer.spy,))
    thread.start()
    thread.join()
    with renderer.spy() as spy:
        for operation in (spy.update, spy.close):
            thread = threading.Thread(target=worker, args=(operation,))
            thread.start()
            thread.join()
        assert spy.update()
    assert len(errors) == 3
    assert all("main thread" in error for error in errors)


@pytest.mark.parametrize("overlays", [0, 1])
def test_interactive_publishes_after_commands_and_polls_only_while_idle(renderer, shell, monkeypatch, overlays):
    from IPython.terminal import pt_inputhooks

    renderer.config.num_overlays = overlays
    native = renderer._renderer
    spy = renderer.spy(interactive=True)
    assert shell.active_eventloop == "hyperlux"
    native.calls.clear()
    shell.events.trigger("post_run_cell", None)
    assert native.calls == (["publish"] if overlays else []) + ["poll"]
    native.calls.clear()
    readiness = iter([False, False, True])
    sleeps = []
    monkeypatch.setattr(_spy_ipython.time, "sleep", sleeps.append)
    shell._inputhook(SimpleNamespace(input_is_ready=lambda: next(readiness)))
    assert native.calls == ["poll", "poll"]
    assert len(sleeps) == 2
    spy.close()
    assert shell.active_eventloop is None
    assert shell._inputhook is None
    assert not shell.events.callbacks["post_run_cell"]
    assert not shell.exit_callbacks
    assert "hyperlux" not in pt_inputhooks.registered


def test_interactive_window_close_removes_callbacks(renderer, shell):
    renderer.spy(interactive=True)
    renderer._renderer.open = False
    shell._inputhook(SimpleNamespace(input_is_ready=lambda: False))
    assert shell._inputhook is None
    assert not shell.events.callbacks["post_run_cell"]
    assert not shell.exit_callbacks
    with renderer.spy(interactive=True) as reopened:
        assert reopened.update()


def test_close_does_not_disable_replacement_gui(renderer, shell):
    spy = renderer.spy(interactive=True)
    replacement = lambda context: None
    shell.active_eventloop = "other"
    shell._inputhook = replacement
    spy.close()
    assert shell.active_eventloop == "other"
    assert shell._inputhook is replacement
    assert not shell.events.callbacks["post_run_cell"]


def test_interactive_setup_failure_closes_native_window(renderer, shell, monkeypatch):
    def fail(*args):
        raise RuntimeError("hook failed")

    monkeypatch.setattr(shell, "enable_gui", fail)
    monkeypatch.setattr(_spy_ipython.atexit, "unregister", lambda callback: None)
    with pytest.raises(RuntimeError, match="hook failed"):
        renderer.spy(interactive=True)
    assert renderer._spy is None
    assert not renderer._renderer.open
    assert not shell.events.callbacks["post_run_cell"]


@pytest.mark.parametrize("kind", ["plain", "notebook", "simple", "dumb", "other_gui"])
def test_unsupported_interactive_shell_fails_before_native_open(renderer, monkeypatch, kind):
    IPython = pytest.importorskip("IPython")
    from IPython.terminal import interactiveshell
    from prompt_toolkit import utils

    class Terminal:
        simple_prompt = kind == "simple"
        active_eventloop = "qt" if kind == "other_gui" else None
        _inputhook = None

    shell = None if kind == "plain" else (object() if kind == "notebook" else Terminal())
    monkeypatch.setattr(IPython, "get_ipython", lambda: shell)
    monkeypatch.setattr(interactiveshell, "TerminalInteractiveShell", Terminal)
    monkeypatch.setattr(utils, "is_dumb_terminal", lambda: kind == "dumb")
    with pytest.raises(RuntimeError, match="IPython"):
        renderer.spy(interactive=True)
    assert renderer._renderer.calls == []


@pytest.mark.skipif(os.environ.get("HYPERLUX_TEST_SPY") != "1", reason="set HYPERLUX_TEST_SPY=1 with a display to test the native GUI")
def test_native_spy_preserves_outputs_and_supports_multiple_overlays():
    import numpy as np

    import hyperlux
    from hyperlux.camera import RGB

    scene = hyperlux.Scene()
    asset = hyperlux.ObjectAssembly()
    part = asset.add_part(hyperlux.primitives.box(), hyperlux.make_transform())
    asset.set_joint(part, "revolute", axis=(1, 0, 0), coordinate=0)
    pool = hyperlux.AssetPool()
    handle = pool.add_assembly(asset)
    renderer = hyperlux.HyperLux(width=32, height=32, fidelity="low", num_probes=0,
                               num_overlays=2, max_overlay_instances=2, max_overlays_per_camera=2)
    renderer.init(scene, asset_pool=pool)
    renderer.set_cameras(renderer.camera(position=(0, 0, 0), look_at=(1, 0, 0)))
    placements = []
    for overlay in range(2):
        renderer.attach(0, overlay)
        placements.append(renderer.spawn(overlay, handle, hyperlux.Pose(hyperlux.make_transform(position=(3, overlay - 0.5, 0)))))
    renderer.update()
    renderer.render()
    before = renderer.outputs[RGB("rgba8")].read().copy()
    with renderer.spy() as spy:
        for frame in range(15):
            renderer.set_pose(placements[0], hyperlux.Pose(hyperlux.make_transform(position=(3, -0.5, 0))), coordinates=[frame * 0.1])
            if frame == 5:
                renderer.despawn(placements[1])
            renderer.update()
            assert spy.update()
            np.testing.assert_array_equal(renderer.outputs[RGB("rgba8")].read(), before)
            time.sleep(0.025)
    assert not spy.update()
    with renderer.spy() as reopened:
        assert reopened.update()
    renderer.render()
    assert np.any(renderer.outputs[RGB("rgba8")].read() != before)


@pytest.mark.skipif(os.environ.get("HYPERLUX_TEST_SPY") != "1", reason="set HYPERLUX_TEST_SPY=1 with a display to test the native GUI")
def test_native_spy_without_overlays_or_source_rgb():
    import numpy as np

    import hyperlux
    from hyperlux.camera import CameraSpec, Depth

    scene = hyperlux.Scene()
    scene.add_object(hyperlux.primitives.box(), hyperlux.make_transform(position=(2, 0, 0)))
    renderer = hyperlux.HyperLux(width=16, height=16, fidelity="low", num_probes=0,
                               cameras=(CameraSpec("depth", (Depth(),)),))
    with pytest.raises(RuntimeError, match="not initialized"):
        renderer.spy()
    renderer.init(scene)
    renderer.set_cameras(renderer.camera(position=(0, 0, 0), look_at=(1, 0, 0)))
    renderer.render()
    before = renderer.outputs[Depth()].read().copy()
    with pytest.raises(IndexError, match="camera index out of range"):
        renderer.spy(camera=1)
    with renderer.spy() as spy:
        for _ in range(10):
            assert spy.update()
            time.sleep(0.025)
    np.testing.assert_array_equal(renderer.outputs[Depth()].read(), before)
