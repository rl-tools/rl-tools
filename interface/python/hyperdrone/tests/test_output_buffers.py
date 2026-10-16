import numpy as np
import pytest

from hyperdrone._output_buffers import OutputCollection
from hyperdrone.camera import CameraSpec, Depth, Probes, RGB, descriptor_metadata


class NativeBuffers:
    def __init__(self):
        self.values = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
        self.ready = False
        self.calls = []

    def output_metadata(self):
        return [{"descriptor": descriptor_metadata(Depth()), "shape": self.values.shape, "dtype": "float32", "axes": ("camera", "height", "width"), "cameras": (0, 3), "device": (1, 0), "block_count": 1}]

    def output_read(self, index, out, field, stream):
        if not self.ready:
            raise RuntimeError("output has not been acquired")
        if tuple(out.shape) != self.values.shape or out.dtype != self.values.dtype:
            raise ValueError("destination shape or dtype does not match native output")
        self.calls.append((index, field))
        np.copyto(out, self.values)

    def output_block(self, index, block, field):
        if not self.ready:
            raise RuntimeError("output has not been acquired")
        return self.values


def test_snapshot_destination_and_native_camera_identity():
    owner = NativeBuffers()
    outputs = OutputCollection(owner, (CameraSpec("front", (Depth(),)), CameraSpec("rear", (RGB("float32"),))))
    output = outputs[Depth()]
    assert output.layout.cameras == ((0, "front"), (1, "rear"))
    assert output.layout.axes == ("camera", "height", "width")
    with pytest.raises(RuntimeError, match="not been acquired"):
        output.read()
    owner.ready = True
    snapshot = output.read()
    owner.values += 1
    assert not np.array_equal(snapshot, owner.values)
    destination = np.empty_like(owner.values)
    assert output.read(out=destination) is destination
    np.testing.assert_array_equal(destination, owner.values)
    with pytest.raises(ValueError, match="shape or dtype"):
        output.read(out=np.zeros((3,), dtype=np.float32))
    assert owner.calls == [(0, ""), (0, "")]


def test_live_blocks_and_unconfigured_outputs():
    owner = NativeBuffers()
    owner.ready = True
    outputs = OutputCollection(owner)
    assert list(outputs) == [Depth()]
    assert outputs[Depth()].layout.cameras == (0, 3)
    assert not hasattr(outputs[Depth()], "distance")
    view = np.from_dlpack(outputs[Depth()].blocks[0])
    owner.values += 3
    np.testing.assert_array_equal(view, owner.values)
    with pytest.raises(KeyError, match="not configured"):
        outputs[RGB("float32")]


def test_probe_fields_use_native_shapes_and_dtypes():
    class ProbeBuffers:
        def output_metadata(self):
            return [{"descriptor": descriptor_metadata(Probes()), "cameras": (2,), "device": (1, 0), "block_count": 1, "fields": {"distance": {"shape": (1, 2), "dtype": "float32", "axes": ("camera", "probe")}, "hit": {"shape": (1, 2), "dtype": "int32", "axes": ("camera", "probe")}}}]

        def output_read(self, index, out, field, stream):
            out[:] = [[4, float("inf")]] if field == "distance" else [[1, 0]]

    output = OutputCollection(ProbeBuffers())[Probes()]
    snapshot = output.read()
    assert snapshot["distance"].shape == (1, 2)
    assert snapshot["hit"].dtype == np.int32
    assert output.distance.layout.axes == ("camera", "probe")
    assert output.read(out=snapshot)["hit"] is snapshot["hit"]
    with pytest.raises(TypeError, match="distance.blocks"):
        output.blocks
    with pytest.raises(ValueError, match="probe destination"):
        output.read(out={"distance": snapshot["distance"]})


def test_rgba8_packed_view_metadata():
    class PackedBuffers:
        def output_metadata(self):
            return [{"descriptor": descriptor_metadata(RGB("rgba8")), "shape": (1, 2, 3, 4), "dtype": "uint8", "axes": ("camera", "height", "width", "channel"), "cameras": (0,), "device": (1, 0), "block_count": 1, "packed": {"shape": (1, 2, 3), "dtype": "uint32", "axes": ("camera", "height", "width")}}]

        def output_read(self, index, out, field, stream):
            assert field == "packed"
            out[:] = 0xFF010203

    output = OutputCollection(PackedBuffers())[RGB("rgba8")]
    assert output.packed.layout.shape == (1, 2, 3)
    assert output.packed.read().dtype == np.uint32
