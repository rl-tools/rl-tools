"""DLPack plumbing shared by the render, dynamics, and env packages."""

KDL_CPU = 1
KDL_CUDA = 2


class DLPackBuffer:
    """DLPack producer over a live native buffer. Capsules are single-use, so each
    __dlpack__ call mints a fresh one; the owner reference keeps the memory alive."""

    def __init__(self, owner, make_capsule, device_type):
        self._owner = owner
        self._make_capsule = make_capsule
        self._device_type = device_type

    def __dlpack__(self, stream=None, **_kwargs):
        return self._make_capsule()

    def __dlpack_device__(self):
        return (self._device_type, 0)


def is_cuda(value):
    """True for DLPack producers that report CUDA device memory."""
    device = getattr(value, "__dlpack_device__", None)
    return device is not None and device()[0] == KDL_CUDA
