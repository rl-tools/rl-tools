import gc
from dataclasses import replace

import numpy as np
import pytest

import hyperlux
from hyperlux.camera import CameraSpec, RGB, Depth, Segmentation, Normals, Flow, FlowStatus, Probes
from test_render import make_scene, make_quad, look_forward


IMAGE_OUTPUTS = (RGB("float32"), RGB(format="rgba8"), Depth(), Depth(format="rgba8", scale=3), Segmentation(), Segmentation(format="rgba8"), Segmentation(labels="semantic"), Segmentation(labels="semantic", format="rgba8"), Normals(), Normals(format="rgba8"), Flow(), Flow(format="rgba8"), FlowStatus(), FlowStatus(format="rgba8"), Flow(direction="forward"), Flow(direction="forward", format="rgba8"), FlowStatus(direction="forward"), FlowStatus(direction="forward", format="rgba8"))


def test_camera_config_canonicalization():
    cameras = (CameraSpec("front", outputs=(RGB("float32"), Depth())), CameraSpec("rear", outputs=(Depth(),)))
    config = hyperlux.Config(width=8, height=8, cameras=cameras, batch_size=4)
    renamed = (CameraSpec("policy", outputs=(Depth(), RGB("float32"))), CameraSpec("display", outputs=(Depth(),)))
    assert config.key() == replace(config, cameras=renamed).key()
    assert config.num_cameras == 8
    assert config.key() != replace(config, cameras=tuple(reversed(cameras))).key()
    assert config.key() == replace(config, num_probes=32).key()
    assert [int(code) for code in hyperlux.FlowStatus] == list(range(5))
    with pytest.raises(ValueError, match="conflicting"):
        replace(config, cameras=(CameraSpec("front", outputs=(Depth(format="rgba8", scale=1),)), CameraSpec("rear", outputs=(Depth(format="rgba8", scale=2),))))


def test_all_native_formats_compact_views_and_snapshot_ownership():
    cameras = (CameraSpec("front", outputs=IMAGE_OUTPUTS), CameraSpec("gap", outputs=(Depth(),)), CameraSpec("rear", outputs=IMAGE_OUTPUTS))
    renderer = hyperlux.HyperLux(16, 16, cameras=cameras, batch_size=2, fidelity="low")
    renderer.init(make_scene(2))
    look_forward(renderer)
    with pytest.raises(RuntimeError, match="acquire"):
        renderer.outputs[Depth()].read()
    renderer.render_launch()
    renderer.render_sync()
    for descriptor in IMAGE_OUTPUTS:
        handle = renderer.outputs[descriptor]
        count = 6 if descriptor == Depth() else 4
        channels = 4 if descriptor.format == "rgba8" else 3 if isinstance(descriptor, (RGB, Normals)) else 2 if isinstance(descriptor, Flow) else 1
        shape = (count, 16, 16) + ((channels,) if channels > 1 else ())
        dtype = np.uint8 if descriptor.format == "rgba8" else np.dtype(descriptor.format)
        assert handle.layout.shape == shape
        assert handle.layout.dtype == np.dtype(dtype).name
        assert handle.layout.cameras == tuple((batch, mount.name) for batch in range(2) for mount in cameras if descriptor in mount.outputs)
        snapshot = handle.read()
        destination = np.empty_like(snapshot)
        assert handle.read(out=destination) is destination
        np.testing.assert_array_equal(snapshot, destination)
        assert not np.shares_memory(snapshot, destination)
        if renderer.backend != "optix":
            block = handle.blocks[0]
            first = np.from_dlpack(block)
            second = np.from_dlpack(block)
            assert np.shares_memory(first, second)
            np.testing.assert_array_equal(first, snapshot)
            if descriptor.format == "rgba8":
                packed = handle.packed.read()
                np.testing.assert_array_equal(packed.view(np.uint8).reshape(shape), snapshot)
                assert np.shares_memory(first, np.from_dlpack(handle.packed.blocks[0]))
    np.testing.assert_array_equal(renderer.outputs[Normals()].read()[:, 8, 8], [[-1, 0, 0]] * 4)
    with pytest.raises((TypeError, ValueError)):
        renderer.outputs[Depth()].read(out=np.empty((6, 16, 16), dtype=np.float64))
    with pytest.raises((TypeError, ValueError)):
        renderer.outputs[Depth()].read(out=np.empty((6, 16, 32), dtype=np.float32)[:, :, ::2])
    readonly = np.empty((6, 16, 16), dtype=np.float32)
    readonly.flags.writeable = False
    with pytest.raises((TypeError, ValueError)):
        renderer.outputs[Depth()].read(out=readonly)
    if renderer.backend != "optix":
        retained = renderer.outputs[Depth()].blocks[0]
        expected = renderer.outputs[Depth()].read()
        del renderer, handle, block, first, second
        gc.collect()
        np.testing.assert_array_equal(np.from_dlpack(retained), expected)


@pytest.mark.parametrize("descriptor", [Flow(), FlowStatus(), Flow(direction="forward"), FlowStatus(direction="forward")])
def test_flow_and_status_select_independently(descriptor):
    renderer = hyperlux.HyperLux(8, 8, cameras=(CameraSpec("front", outputs=(descriptor,)),), fidelity="low")
    renderer.init(make_scene(2))
    look_forward(renderer)
    renderer.render()
    assert tuple(renderer.outputs) == (descriptor,)
    assert renderer.outputs[descriptor].read().shape[:3] == (1, 8, 8)


def test_camera_flow_advances_only_explicitly():
    scene = hyperlux.Scene()
    scene.add_mesh(make_quad(2, half_size=20))
    renderer = hyperlux.HyperLux(32, 32, cameras=(CameraSpec("front", outputs=(Flow(), FlowStatus())),), fidelity="low")
    renderer.init(scene)
    look_forward(renderer)
    renderer.render()
    np.testing.assert_allclose(renderer.outputs[Flow()].read(), 0, atol=1e-4)
    renderer.advance_flow()
    camera = renderer.camera(position=(0, 0.2, 0), look_at=(1, 0.2, 0), fov=60)
    renderer.set_cameras(camera[None])
    renderer.render()
    flow = renderer.outputs[Flow()].read()
    focal = 32 / (2 * np.tan(np.deg2rad(30)))
    np.testing.assert_allclose(flow[0, 16, 16], [-focal * 0.2 / 2, 0], atol=1e-4)
    status = renderer.outputs[FlowStatus()].read()
    assert status[0, 16, 16] == hyperlux.FlowStatus.VALID
    assert np.any(status == hyperlux.FlowStatus.OUT_OF_FRAME)
    renderer.render_launch()
    renderer.render_sync()
    np.testing.assert_array_equal(renderer.outputs[Flow()].read(), flow)
    renderer.advance_flow()
    renderer.render()
    np.testing.assert_allclose(renderer.outputs[Flow()].read(), 0, atol=1e-4)


def test_probe_only_compact_fields_and_independent_acquisition():
    renderer = hyperlux.HyperLux(1, 1, cameras=(CameraSpec("probe", outputs=(Probes(),)), CameraSpec("other", outputs=(Depth(),))), batch_size=2, num_probes=4, fidelity="low")
    renderer.init(make_scene(3))
    look_forward(renderer)
    renderer.generate_probe_directions()
    with pytest.raises(RuntimeError, match="acquire"):
        renderer.outputs[Probes()].distance.read()
    renderer.probe()
    distances = renderer.outputs[Probes()].distance.read()
    hits = renderer.outputs[Probes()].hit.read()
    assert distances.shape == hits.shape == (2, 4)
    assert hits.dtype == np.int32
    np.testing.assert_allclose(distances[:, 0], 3, atol=1e-5)
    np.testing.assert_array_equal(hits[:, 0], 1)
    if renderer.backend != "optix":
        for field in (renderer.outputs[Probes()].distance, renderer.outputs[Probes()].hit):
            live = np.from_dlpack(field.blocks[0])
            assert live.strides[-1] == 8
            np.testing.assert_array_equal(live, field.read())
    with pytest.raises(RuntimeError, match="acquire"):
        renderer.outputs[Depth()].read()
    renderer.render()
    np.testing.assert_array_equal(renderer.outputs[Probes()].distance.read(), distances)
    sampler = hyperlux.ClearanceSampler(make_scene(3), batch=2, probes=4)
    assert tuple(sampler._renderer.outputs) == (Probes(),)
    np.testing.assert_allclose(sampler.clearance_distances([[0, 0, 0], [1, 0, 0]]), [3, 2], atol=1e-5)


def test_cuda_output_destinations_and_streams():
    if hyperlux.backend() != "optix":
        pytest.skip("CUDA output transport requires OptiX")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("PyTorch needs CUDA")
    renderer = hyperlux.HyperLux(16, 16, cameras=(CameraSpec("front", outputs=(RGB("float32"), Depth(), Probes())),), num_probes=4, fidelity="low")
    renderer.init(make_scene(2))
    look_forward(renderer)
    renderer.generate_probe_directions()
    renderer.render_launch()
    renderer.probe_launch()
    consumer = torch.cuda.Stream()
    with torch.cuda.stream(consumer):
        for handle in (renderer.outputs[RGB("float32")], renderer.outputs[Depth()], renderer.outputs[Probes()].distance, renderer.outputs[Probes()].hit):
            expected = handle.read()
            live = torch.from_dlpack(handle.blocks[0])
            destination = torch.empty_like(live)
            handle.read(out=destination, stream=consumer.cuda_stream)
            np.testing.assert_array_equal(destination.cpu().numpy(), expected)
            np.testing.assert_array_equal(live.cpu().numpy(), expected)


def test_cuda_transport_without_tensor_framework():
    if hyperlux.backend() != "optix":
        pytest.skip("CUDA output transport requires OptiX")
    import ctypes
    from hyperlux import _cuda

    runtime = ctypes.CDLL("libcudart.so")
    runtime.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    runtime.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
    runtime.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
    runtime.cudaStreamQuery.argtypes = [ctypes.c_void_p]
    runtime.cudaMemcpyAsync.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
    runtime.cudaMemcpy2DAsync.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
    capsule_pointer = ctypes.pythonapi.PyCapsule_GetPointer
    capsule_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
    capsule_pointer.restype = ctypes.c_void_p
    consumer = ctypes.c_void_p()
    assert runtime.cudaStreamCreate(ctypes.byref(consumer)) == 0
    renderer = hyperlux.HyperLux(16, 16, cameras=(CameraSpec("front", outputs=(RGB("float32"), Depth(), Probes())),), num_probes=4, fidelity="low")
    renderer.init(make_scene(2))
    look_forward(renderer)
    renderer.generate_probe_directions()
    renderer.render_launch()
    renderer.probe_launch()
    try:
        for handle in (renderer.outputs[RGB("float32")], renderer.outputs[Depth()], renderer.outputs[Probes()].distance, renderer.outputs[Probes()].hit):
            snapshot = handle.read()
            block = handle.blocks[0]
            capsule = block.__dlpack__(stream=consumer.value)
            pointer = ctypes.c_void_p.from_address(capsule_pointer(capsule, b"dltensor")).value
            copied = np.empty_like(snapshot)
            if handle.descriptor == Probes():
                assert runtime.cudaMemcpy2DAsync(copied.ctypes.data, 4, pointer, 8, 4, copied.size, 2, consumer) == 0
            else:
                assert runtime.cudaMemcpyAsync(copied.ctypes.data, pointer, copied.nbytes, 2, consumer) == 0
            assert runtime.cudaStreamSynchronize(consumer) == 0
            np.testing.assert_array_equal(copied, snapshot)
            if snapshot.dtype == np.float32:
                destination = _cuda.upload(np.zeros_like(snapshot)).view()
                handle.read(out=destination, stream=consumer.value)
                destination_capsule = destination.__dlpack__()
                pointer = ctypes.c_void_p.from_address(capsule_pointer(destination_capsule, b"dltensor")).value
                assert runtime.cudaMemcpyAsync(copied.ctypes.data, pointer, copied.nbytes, 2, consumer) == 0
                assert runtime.cudaStreamSynchronize(consumer) == 0
                np.testing.assert_array_equal(copied, snapshot)
        import time
        handle = renderer.outputs[Depth()]
        expected = handle.read()
        destination = _cuda.upload(np.zeros_like(expected)).view()
        copied_live = np.empty_like(expected)
        copied_destination = np.empty_like(expected)
        initial_capsule = handle.blocks[0].__dlpack__()
        initial_pointer = ctypes.c_void_p.from_address(capsule_pointer(initial_capsule, b"dltensor")).value
        callback_type = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
        delay = callback_type(lambda _: time.sleep(0.05))
        runtime.cudaLaunchHostFunc.argtypes = [ctypes.c_void_p, callback_type, ctypes.c_void_p]
        runtime.cudaMemsetAsync.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t, ctypes.c_void_p]
        producer = ctypes.c_void_p(renderer.render_stream)
        assert runtime.cudaMemsetAsync(initial_pointer, 0, expected.nbytes, producer) == 0
        assert runtime.cudaLaunchHostFunc(producer, delay, None) == 0
        renderer.render_launch()
        live_capsule = handle.blocks[0].__dlpack__(stream=consumer.value)
        live_pointer = ctypes.c_void_p.from_address(capsule_pointer(live_capsule, b"dltensor")).value
        assert runtime.cudaMemcpyAsync(copied_live.ctypes.data, live_pointer, copied_live.nbytes, 2, consumer) == 0
        handle.read(out=destination, stream=consumer.value)
        destination_capsule = destination.__dlpack__()
        destination_pointer = ctypes.c_void_p.from_address(capsule_pointer(destination_capsule, b"dltensor")).value
        assert runtime.cudaMemcpyAsync(copied_destination.ctypes.data, destination_pointer, copied_destination.nbytes, 2, consumer) == 0
        assert runtime.cudaStreamSynchronize(consumer) == 0
        np.testing.assert_array_equal(copied_live, expected)
        np.testing.assert_array_equal(copied_destination, expected)
        camera = renderer.camera(position=(0, 0, 0), look_at=(1, 0, 0))
        camera_input = _cuda.upload(camera[None]).view()
        assert runtime.cudaLaunchHostFunc(producer, delay, None) == 0
        renderer.set_cameras(camera_input)
        renderer.render_launch()
        retained = renderer.outputs[Depth()].blocks[0]
        assert runtime.cudaStreamQuery(producer) != 0
        del camera_input, renderer
        gc.collect()
        assert retained.__dlpack__(stream=consumer.value) is not None
        assert runtime.cudaStreamSynchronize(consumer) == 0
    finally:
        assert runtime.cudaStreamDestroy(consumer) == 0
