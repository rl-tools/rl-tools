"""hyperdrone.cuda — CUDA staging helpers shared by render and dynamics.

The native CudaBuffer class is compiled into every CUDA-capable component core
(render on OptiX, dynamics on CUDA). This package defines the Python-side views and
resolves a provider from whichever domain package registered one at import time, keeping
the dependency direction render/dynamics -> cuda.
"""
import numpy as np

_providers = []


def register_provider(load_core):
    """Called by domain packages (render, dynamics) at import time; load_core is a
    zero-argument callable returning a core module that may expose CudaBuffer."""
    if load_core not in _providers:
        _providers.append(load_core)


def _core():
    errors = []
    for load_core in _providers:
        try:
            core = load_core()
        except Exception as error:  # provider may fail to build on this machine
            errors.append(str(error))
            continue
        if getattr(core, "HAS_CUDA", False):
            return core
    detail = ("; ".join(errors)) if errors else "no CUDA-capable component core is available"
    raise RuntimeError(
        "hyperdrone: CUDA staging needs a CUDA-capable component (render on the OptiX "
        f"backend, or dynamics on CUDA); import one first ({detail})"
    )


class CudaView:
    """DLPack producer over one slice of an uploaded CUDA buffer. Fresh capsule per
    __dlpack__ call; holds the parent buffer alive."""

    def __init__(self, buffer, index):
        self._buffer = buffer
        self._index = index

    def __dlpack__(self, stream=None, **_kwargs):
        if self._index is None:
            return self._buffer.dlpack()
        return self._buffer.slice_dlpack(self._index)

    def __dlpack_device__(self):
        return (2, 0)  # kDLCUDA


class TensorSet:
    """A host float32 array uploaded to CUDA device memory once; view(i) exposes slice i
    along axis 0 as a zero-copy DLPack CUDA tensor (e.g. one camera set per slice)."""

    def __init__(self, array):
        self._buffer = _core().CudaBuffer(np.ascontiguousarray(array, dtype=np.float32))

    @property
    def shape(self):
        return tuple(self._buffer.shape)

    @property
    def num_sets(self):
        return self.shape[0]

    @property
    def raw(self):
        return self._buffer

    def view(self, index=None):
        return CudaView(self._buffer, index)


def upload(array):
    return TensorSet(array)
