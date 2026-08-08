import numpy as np

from ._build import load_core


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


class CudaTensorSet:
    """A host float32 array uploaded to CUDA device memory once; view(i) exposes slice i
    along axis 0 as a zero-copy DLPack CUDA tensor (e.g. one camera set per slice)."""

    def __init__(self, array):
        core = load_core()
        if not getattr(core, "HAS_CUDA", False):
            raise RuntimeError("hypert: CUDA staging requires the OptiX backend (HYPERT_BACKEND=OPTIX)")
        self._buffer = core.CudaBuffer(np.ascontiguousarray(array, dtype=np.float32))

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


def cuda_upload(array):
    return CudaTensorSet(array)
