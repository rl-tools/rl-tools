"""DLPack plumbing shared by render, dynamics, and the environment."""

KDL_CPU = 1
KDL_CUDA = 2


class DLPackBuffer:
    """DLPack producer over a live native buffer. Capsules are single-use, so each
    __dlpack__ call mints a fresh one; the owner reference keeps the memory alive."""

    def __init__(self, owner, make_capsule, device_type, device_id=0, order_consumer=None):
        self._owner = owner
        self._make_capsule = make_capsule
        self._device_type = device_type
        self._device_id = device_id
        self._order_consumer = order_consumer

    def __dlpack__(self, stream=None, **_kwargs):
        if self._order_consumer is not None and stream not in (None, -1):
            self._order_consumer(int(stream))
        return self._make_capsule()

    def __dlpack_device__(self):
        return (self._device_type, self._device_id)


def is_cuda(value):
    """True for DLPack producers that report CUDA device memory."""
    device = getattr(value, "__dlpack_device__", None)
    return device is not None and device()[0] == KDL_CUDA
