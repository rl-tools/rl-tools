import atexit
import time

from ._spy import Spy, _main_thread


def _interactive_shell():
    try:
        from IPython import get_ipython
        from IPython.terminal.interactiveshell import TerminalInteractiveShell
        from prompt_toolkit.utils import is_dumb_terminal
    except ImportError as error:
        raise RuntimeError("hyperlux: interactive=True requires terminal IPython") from error
    shell = get_ipython()
    if not isinstance(shell, TerminalInteractiveShell) or shell.simple_prompt or is_dumb_terminal():
        raise RuntimeError("hyperlux: interactive=True requires terminal IPython with its full prompt")
    if shell.active_eventloop is not None or shell._inputhook is not None:
        raise RuntimeError("hyperlux: another IPython GUI event loop is active; disable it with %gui before opening the spy")
    return shell


class InteractiveSpy(Spy):
    """Optional terminal IPython polling for the native spectator."""

    def __init__(self, renderer, *, camera, has_overlays):
        _main_thread()
        shell = _interactive_shell()
        from IPython.terminal import pt_inputhooks

        super().__init__(renderer, camera=camera)
        self._shell = shell
        self._has_overlays = has_overlays
        try:
            pt_inputhooks.register("hyperlux", self._inputhook)
            shell.events.register("post_run_cell", self._post_run_cell)
            shell.enable_gui("hyperlux")
            atexit.register(self.close)
            self._post_run_cell(None)
        except BaseException:
            self.close()
            raise

    def _post_run_cell(self, result):
        if self._renderer is not None:
            if self._has_overlays:
                self._renderer.update()
            self.update()

    def _inputhook(self, context):
        try:
            while not context.input_is_ready() and self.update():
                time.sleep(1 / 60)
        except KeyboardInterrupt:
            pass
        except Exception:
            self.close()
            raise

    def close(self):
        if self._renderer is None:
            return
        _main_thread()
        try:
            if self._shell is not None:
                from IPython.terminal import pt_inputhooks

                shell, self._shell = self._shell, None
                if shell._inputhook == self._inputhook:
                    shell.enable_gui(None)
                if pt_inputhooks.registered.get("hyperlux") == self._inputhook:
                    del pt_inputhooks.registered["hyperlux"]
                if self._post_run_cell in shell.events.callbacks["post_run_cell"]:
                    shell.events.unregister("post_run_cell", self._post_run_cell)
                atexit.unregister(self.close)
        finally:
            super().close()
