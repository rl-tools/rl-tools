"""Metadata-driven access to compiled renderer output buffers."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from .camera import descriptor_from_metadata


@dataclass(frozen=True)
class OutputLayout:
    shape: tuple[int, ...]
    dtype: str
    axes: tuple[str, ...]
    cameras: tuple[tuple[int, str] | int, ...]
    device: tuple[int, int]
    block_count: int


class OutputHandle:
    def __init__(self, owner, index, metadata, camera_names, field=""):
        self._owner = owner
        self._index = index
        self._field = field
        self.descriptor = descriptor_from_metadata(metadata["descriptor"])
        selected = metadata
        if field:
            selected = metadata["packed"] if field == "packed" else metadata["fields"][field]
        identities = []
        for physical_camera in metadata.get("cameras", ()):
            if camera_names:
                batch, mount = divmod(int(physical_camera), len(camera_names))
                identities.append((batch, camera_names[mount]))
            else:
                identities.append(int(physical_camera))
        self.layout = OutputLayout(shape=tuple(selected.get("shape", ())), dtype=selected.get("dtype", ""), axes=tuple(selected.get("axes", ())), cameras=tuple(identities), device=tuple(metadata["device"]), block_count=int(metadata["block_count"]))
        self.fields = MappingProxyType({name: OutputHandle(owner, index, metadata, camera_names, name) for name in metadata.get("fields", {})}) if not field else MappingProxyType({})
        self._packed = OutputHandle(owner, index, metadata, camera_names, "packed") if not field and "packed" in metadata else None

    @property
    def distance(self):
        try:
            return self.fields["distance"]
        except KeyError:
            raise AttributeError("this output has no distance field") from None

    @property
    def hit(self):
        try:
            return self.fields["hit"]
        except KeyError:
            raise AttributeError("this output has no hit field") from None

    @property
    def packed(self):
        if self._packed is None:
            raise AttributeError("this output has no packed representation")
        return self._packed

    def read(self, out=None, *, stream=0):
        """Copy the last acquisition; CUDA callers supply their destination stream."""
        if self.fields:
            if out is not None:
                if not isinstance(out, Mapping) or set(out) != set(self.fields):
                    raise ValueError(f"probe destination must map {tuple(self.fields)} to destination tensors")
            return {name: handle.read(None if out is None else out[name], stream=stream) for name, handle in self.fields.items()}
        if out is None:
            import numpy as np
            out = np.empty(self.layout.shape, dtype=self.layout.dtype)
        self._owner.output_read(self._index, out, self._field, stream)
        return out

    @property
    def blocks(self):
        """Live native allocations; captures and environment resets may overwrite them."""
        if self.fields:
            raise TypeError("access probe distance.blocks or hit.blocks separately")
        return tuple(self._owner.output_block(self._index, block, self._field) for block in range(self.layout.block_count))


class OutputCollection(Mapping):
    def __init__(self, owner, cameras=()):
        camera_names = tuple(camera if isinstance(camera, str) else camera.name for camera in cameras)
        self._handles = {}
        for index, metadata in enumerate(owner.output_metadata()):
            handle = OutputHandle(owner, index, metadata, camera_names)
            if handle.descriptor in self._handles:
                raise ValueError(f"native output metadata contains duplicate descriptor {handle.descriptor!r}")
            self._handles[handle.descriptor] = handle

    def __getitem__(self, descriptor):
        try:
            return self._handles[descriptor]
        except KeyError:
            raise KeyError(f"output {descriptor!r} is not configured; available outputs: {tuple(self._handles)}") from None

    def __iter__(self):
        return iter(self._handles)

    def __len__(self):
        return len(self._handles)
