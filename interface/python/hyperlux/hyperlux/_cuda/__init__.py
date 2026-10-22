"""hyperlux._cuda — CUDA staging helpers shared by every native component.

The native CudaBuffer class is compiled into every CUDA-capable module (the renderer core on
OptiX, hyperdrone's CUDA dynamics core and OptiX environment); nanobind shares one binding across
them. The first loaded module that exposes it hands the class to this package through the
extension-loading hook, so staging works in renderer-only, dynamics-only, environment-only and
mixed processes alike, and never builds a component on its own.
"""
import numpy as np

_buffer_type = None


def register_extension(extension):
    global _buffer_type
    buffer_type = getattr(extension, "CudaBuffer", None)
    if buffer_type is not None:
        _buffer_type = buffer_type


def _buffer():
    if _buffer_type is None:
        raise RuntimeError("hyperlux: CUDA staging needs a loaded CUDA-capable native module (a renderer on the OptiX backend, a CUDA Sim or an OptiX environment); initialize one first")
    return _buffer_type


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
        return (2, self._buffer.device_id)


class TensorSet:
    """A host float32 array uploaded to CUDA device memory once; view(i) exposes slice i
    along axis 0 as a zero-copy DLPack CUDA tensor (e.g. one camera set per slice)."""

    def __init__(self, array):
        self._buffer = _buffer()(np.ascontiguousarray(array, dtype=np.float32))

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
