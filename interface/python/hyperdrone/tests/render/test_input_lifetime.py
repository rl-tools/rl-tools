import ctypes
import gc
import threading
import time

import numpy as np
import pytest

from hyperdrone import _cuda, render
from hyperdrone.camera import Depth
from test_pose import two_placements


class DLDevice(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class DLDataType(ctypes.Structure):
    _fields_ = [("code", ctypes.c_uint8), ("bits", ctypes.c_uint8), ("lanes", ctypes.c_uint16)]


class DLTensor(ctypes.Structure):
    _fields_ = [("data", ctypes.c_void_p), ("device", DLDevice), ("ndim", ctypes.c_int), ("dtype", DLDataType), ("shape", ctypes.POINTER(ctypes.c_int64)), ("strides", ctypes.POINTER(ctypes.c_int64)), ("byte_offset", ctypes.c_uint64)]


class DLManagedTensor(ctypes.Structure):
    pass


DLDeleter = ctypes.CFUNCTYPE(None, ctypes.POINTER(DLManagedTensor))
DLManagedTensor._fields_ = [("tensor", DLTensor), ("context", ctypes.c_void_p), ("deleter", DLDeleter)]


class ExportOwnedCUDA:
    def __init__(self, owner, values):
        self.owner = owner
        values = np.ascontiguousarray(values, dtype=np.float32)
        self.allocation = ctypes.c_void_p()
        assert owner.runtime.cudaMalloc(ctypes.byref(self.allocation), values.nbytes) == 0
        owner.allocations.append(self.allocation)
        assert owner.runtime.cudaMemcpy(self.allocation, values.ctypes.data, values.nbytes, 1) == 0
        self.shape = values.shape

    def __dlpack_device__(self):
        return (2, self.owner.device)

    def __dlpack__(self, **kwargs):
        assert self.allocation is not None
        allocation, self.allocation = self.allocation, None
        owner = self.owner
        shape = (ctypes.c_int64 * len(self.shape))(*self.shape)

        @DLDeleter
        def release(managed):
            owner.released.append(owner.runtime.cudaEventQuery(owner.completed))

        managed = DLManagedTensor(DLTensor(allocation, DLDevice(2, owner.device), len(self.shape), DLDataType(2, 32, 1), shape, None, 0), None, release)
        owner.leases.append((managed, shape, release))
        return owner.capsule_new(ctypes.addressof(managed), b"dltensor", None)


class CUDAExports:
    def __init__(self):
        self.runtime = ctypes.CDLL("libcudart.so")
        self.driver = ctypes.CDLL("libcuda.so.1")
        for name, arguments in {
            "cudaGetDevice": [ctypes.POINTER(ctypes.c_int)],
            "cudaSetDevice": [ctypes.c_int],
            "cudaMalloc": [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t],
            "cudaFree": [ctypes.c_void_p],
            "cudaMemcpy": [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int],
            "cudaMemset": [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t],
            "cudaStreamCreateWithFlags": [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint],
            "cudaStreamDestroy": [ctypes.c_void_p],
            "cudaStreamQuery": [ctypes.c_void_p],
            "cudaEventCreateWithFlags": [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint],
            "cudaEventRecord": [ctypes.c_void_p, ctypes.c_void_p],
            "cudaEventQuery": [ctypes.c_void_p],
            "cudaEventDestroy": [ctypes.c_void_p],
        }.items():
            getattr(self.runtime, name).argtypes = arguments
        for name in ("cuStreamWaitValue32_v2", "cuStreamWriteValue32_v2"):
            getattr(self.driver, name).argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint32, ctypes.c_uint]
        self.capsule_new = ctypes.pythonapi.PyCapsule_New
        self.capsule_new.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
        self.capsule_new.restype = ctypes.py_object
        device = ctypes.c_int()
        assert self.runtime.cudaGetDevice(ctypes.byref(device)) == 0
        self.device = device.value
        self.word, self.control, self.completed = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
        assert self.runtime.cudaMalloc(ctypes.byref(self.word), 4) == 0
        assert self.runtime.cudaMemset(self.word, 0, 4) == 0
        assert self.runtime.cudaStreamCreateWithFlags(ctypes.byref(self.control), 1) == 0
        assert self.runtime.cudaEventCreateWithFlags(ctypes.byref(self.completed), 2) == 0
        self.allocations, self.leases, self.released = [], [], []

    def block(self, stream):
        assert self.driver.cuStreamWaitValue32_v2(stream, self.word.value, 1, 1) == 0
        assert self.runtime.cudaEventRecord(self.completed, stream) == 0

    def unblock(self):
        assert self.driver.cuStreamWriteValue32_v2(self.control, self.word.value, 1, 0) == 0

    def close(self):
        self.unblock()
        assert self.runtime.cudaDeviceSynchronize() == 0
        # Defer real frees so a lifetime regression reports an assertion instead of an invalid GPU read.
        for allocation in self.allocations:
            assert self.runtime.cudaFree(allocation) == 0
        assert self.runtime.cudaFree(self.word) == 0
        assert self.runtime.cudaEventDestroy(self.completed) == 0
        assert self.runtime.cudaStreamDestroy(self.control) == 0


@pytest.fixture
def cuda_exports():
    exports = CUDAExports()
    try:
        yield exports
    finally:
        exports.close()


def submit(renderer, batch, inputs, kind):
    if kind == "pose":
        renderer.set_pose(batch, render.Pose(inputs[0]))
    elif kind == "shutter_pose":
        renderer.set_pose(batch, render.Pose(inputs[0], scale=inputs[1]), render.Pose(inputs[3], scale=inputs[4]), coordinates=inputs[2], coordinates_close=inputs[5])
    elif kind == "cameras":
        renderer.set_cameras(inputs[0])
    else:
        renderer.set_motion_blur_cameras(*inputs)


@pytest.mark.skipif(render._backend() != "optix", reason="device input leases require OptiX")
@pytest.mark.parametrize("input_kind", ("pose", "shutter_pose", "cameras", "shutter_cameras"))
@pytest.mark.parametrize("completion", ("poll", "next_input", "synchronize", "teardown"))
def test_export_owned_inputs_live_until_their_cuda_read_finishes(cuda_exports, input_kind, completion):
    renderer, placements, bases, scales, coordinates = two_placements(motion=True)
    batch = renderer.placement_batch(placements)
    if input_kind == "pose":
        values = (bases,)
    elif input_kind == "shutter_pose":
        values = (bases, scales, coordinates, bases.copy(), scales.copy(), -coordinates)
    else:
        camera = np.asarray(renderer.camera((0, 0, 0), (1, 0, 0))).reshape(1, 12)
        values = (camera, camera.copy()) if input_kind == "shutter_cameras" else (camera,)
    warmup = [_cuda.upload(value).view() for value in values]
    submit(renderer, batch, warmup, input_kind)
    renderer.synchronize()
    inputs = [ExportOwnedCUDA(cuda_exports, value) for value in values]
    following = [[ExportOwnedCUDA(cuda_exports, value) for value in values] for _ in range(4)] if completion == "next_input" else []
    cuda_exports.block(renderer.render_stream)
    try:
        submit(renderer, batch, inputs, input_kind)
        assert all(value.allocation is None for value in inputs)
        assert cuda_exports.runtime.cudaEventQuery(cuda_exports.completed) == 600
        assert cuda_exports.released == []
        if completion == "teardown":
            def finish():
                assert cuda_exports.runtime.cudaSetDevice(cuda_exports.device) == 0
                time.sleep(0.02)
                cuda_exports.unblock()

            worker = threading.Thread(target=finish)
            worker.start()
            try:
                renderer = None
                gc.collect()
            finally:
                worker.join()
        else:
            cuda_exports.unblock()
            if completion == "poll":
                deadline = time.monotonic() + 10
                while len(cuda_exports.released) < len(inputs) and time.monotonic() < deadline:
                    renderer._renderer.release_inputs()
                    time.sleep(0.001)
            elif completion == "next_input":
                submitted = len(inputs)
                for next_inputs in following + [warmup]:
                    deadline = time.monotonic() + 10
                    while cuda_exports.runtime.cudaStreamQuery(renderer.render_stream) == 600 and time.monotonic() < deadline:
                        time.sleep(0.001)
                    assert cuda_exports.runtime.cudaStreamQuery(renderer.render_stream) == 0
                    submit(renderer, batch, next_inputs, input_kind)
                    assert len(cuda_exports.released) >= submitted
                    submitted += len(next_inputs)
            else:
                renderer.synchronize()
        assert cuda_exports.released == [0] * (len(inputs) * (1 + len(following)))
    finally:
        cuda_exports.unblock()
        if renderer is not None:
            renderer.synchronize()


@pytest.mark.skipif(render._backend() != "optix", reason="device input leases require OptiX")
def test_wrapper_teardown_releases_inputs_owned_by_its_native_renderer(cuda_exports):
    renderer, _, _, _, _ = two_placements()
    renderer.update()
    renderer.render()
    native = renderer._renderer
    block = renderer.outputs[Depth()].blocks[0]
    capsule_pointer = ctypes.pythonapi.PyCapsule_GetPointer
    capsule_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
    capsule_pointer.restype = ctypes.c_void_p

    class CameraView:
        def __dlpack_device__(self):
            return block.__dlpack_device__()

        def __dlpack__(self, **kwargs):
            capsule = block.__dlpack__(**kwargs)
            tensor = DLTensor.from_address(capsule_pointer(capsule, b"dltensor"))
            tensor.ndim = 2
            tensor.shape[0], tensor.shape[1] = 1, 12
            if tensor.strides:
                tensor.strides[0], tensor.strides[1] = 12, 1
            return capsule

    cuda_exports.block(renderer.render_stream)
    try:
        renderer.set_cameras(CameraView())
        assert cuda_exports.runtime.cudaEventQuery(cuda_exports.completed) == 600
        assert native.release_inputs() == 0
        cuda_exports.unblock()
        native.synchronize()
        renderer = None
        gc.collect()
        assert native.release_inputs() == 0
    finally:
        cuda_exports.unblock()
        native.synchronize()
        native.release_inputs()
