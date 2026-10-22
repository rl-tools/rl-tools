import threading


def _main_thread():
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("hyperlux: create, update and close the spy on the main thread")


class Spy:
    """An independent native spectator, created by HyperLux.spy()."""

    def __init__(self, renderer, *, camera):
        _main_thread()
        renderer.spy_open(camera)
        self._renderer = renderer

    def update(self):
        """Poll and present the published scene; return False when the window closes."""
        _main_thread()
        if self._renderer is None:
            return False
        if not self._renderer.spy_update():
            self.close()
            return False
        return True

    def close(self):
        if self._renderer is None:
            return
        _main_thread()
        self._renderer.spy_close()
        self._renderer = None

    def __enter__(self):
        if self._renderer is None:
            raise RuntimeError("hyperlux: this spy is closed; call renderer.spy() to open another")
        return self

    def __exit__(self, *exception):
        self.close()
