import os
import sys
import time

import pytest


@pytest.mark.skipif(os.name != "posix", reason="terminal integration needs a POSIX pseudoterminal")
def test_real_ipython_publishes_polls_and_removes_hooks(tmp_path):
    pytest.importorskip("IPython")
    pexpect = pytest.importorskip("pexpect")
    setup = tmp_path / "setup.py"
    setup.write_text("""
from types import SimpleNamespace
from pathlib import Path
from hyperdrone.render import Renderer
from IPython.terminal import pt_inputhooks

idle = Path(__file__).with_name("idle")

class NativeRenderer:
    open = False
    published = 0
    polled = 0
    signal_idle = 0

    def spy_open(self, camera):
        self.open = True

    def spy_update(self):
        self.polled += 1
        if self.signal_idle and self.polled >= self.signal_idle:
            self.signal_idle = 0
            idle.write_text(str(self.polled))
        return self.open

    def spy_close(self):
        self.open = False

    def update(self):
        self.published += 1

    def synchronize(self):
        pass

    def release_inputs(self):
        return 0

native = NativeRenderer()
renderer = Renderer.__new__(Renderer)
renderer._renderer = native
renderer.config = SimpleNamespace(num_overlays=1)
renderer._spy = None
shell = get_ipython()
callbacks_before = list(shell.events.callbacks["post_run_cell"])
""")
    env = dict(os.environ, IPYTHONDIR=str(tmp_path / "ipython"),
               PROMPT_TOOLKIT_NO_CPR="1", TERM="xterm-256color")
    with pexpect.spawn(
        sys.executable,
        ["-m", "IPython", "--no-banner", "--no-confirm-exit", "--colors=NoColor",
         "--TerminalInteractiveShell.simple_prompt=False", "-i", str(setup)],
        env=env, encoding="utf-8", timeout=20, echo=False,
    ) as terminal:
        prompt_number = 1
        terminal.expect(r"In \[1\]:")

        def run(command):
            nonlocal prompt_number
            terminal.sendline(command)
            prompt_number += 1
            terminal.expect(rf"In \[{prompt_number}\]:")
            assert "Traceback" not in terminal.before, terminal.before

        run("spy = renderer.spy(interactive=True)")
        run("assert native.open and shell.active_eventloop == 'hyperdrone'; before = native.published")
        run("assert native.published == before + 1")
        run("native.signal_idle = native.polled + 2")
        deadline = time.monotonic() + 10
        while not (tmp_path / "idle").exists():
            assert time.monotonic() < deadline, "IPython did not poll the spy while idle"
            time.sleep(0.01)
        run("assert native.polled > 2")
        run("spy.close(); assert not native.open and shell.active_eventloop is None")
        run("assert shell.events.callbacks['post_run_cell'] == callbacks_before")
        run("assert 'hyperdrone' not in pt_inputhooks.registered")
        run("spy = renderer.spy(interactive=True)")
        run("native.open = False")
        run("assert spy._renderer is None and shell.active_eventloop is None")
        run("assert shell.events.callbacks['post_run_cell'] == callbacks_before")
        run("assert 'hyperdrone' not in pt_inputhooks.registered")
        run("spy = renderer.spy(); assert spy.update(); spy.close()")
        terminal.sendline("exit()")
        terminal.expect(pexpect.EOF)
        terminal.close()
        assert terminal.exitstatus == 0
