"""Advisory file lock serializing configure/build on one build tree across processes
(vectorized RL workloads routinely spawn many workers that would otherwise race the same
CMake tree). POSIX flock; a Windows port replaces this module with an msvcrt equivalent."""
import fcntl
import os


class FileLock:
    def __init__(self, path):
        self._path = path
        self._handle = None

    def __enter__(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = os.open(self._path, os.O_CREAT | os.O_RDWR, 0o644)
        fcntl.flock(self._handle, fcntl.LOCK_EX)
        return self

    def __exit__(self, *exception):
        fcntl.flock(self._handle, fcntl.LOCK_UN)
        os.close(self._handle)
        self._handle = None
        return False
