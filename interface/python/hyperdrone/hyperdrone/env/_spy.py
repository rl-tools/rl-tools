import threading


def _main_thread():
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("hyperdrone: create, update and close the spy on the main thread")


class Spy:
    """An explicitly polled native viewer, created by HyperDrone.spy()."""

    def __init__(self, environment):
        _main_thread()
        environment.spy_open()
        self._environment = environment

    def update(self):
        """Poll and present after observe/render, before step/reset; return False when closed."""
        _main_thread()
        if self._environment is None:
            return False
        if not self._environment.spy_update():
            self.close()
            return False
        return True

    def close(self):
        if self._environment is not None:
            _main_thread()
            self._environment.spy_close()
            self._environment = None

    def __enter__(self):
        if self._environment is None:
            raise RuntimeError("hyperdrone: this spy is closed; call env.spy() to open another")
        return self

    def __exit__(self, *exception):
        self.close()
