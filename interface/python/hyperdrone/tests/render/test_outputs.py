import gc
from dataclasses import replace

import numpy as np
import pytest

from hyperdrone import render
from test_render import make_scene, make_quad, look_forward


OUTPUTS = ("rgb", "depth", "instance_segmentation", "semantic_segmentation", "normals", "flow")


def test_output_selection():
    config = render.RendererConfig(width=16, height=16, num_cameras=1, num_probes=0, fidelity="low", output=OUTPUTS, motion_blur_samples=1, anti_aliasing_grid=1, num_overlays=0, max_overlay_instances=0, max_overlays_per_camera=0)
    assert config == replace(config, output=tuple(reversed(OUTPUTS)) + ("flow",))
    assert config.key() == replace(config, output=list(reversed(OUTPUTS))).key()
    assert replace(config, output=("rgb", "depth", "instance_segmentation")).output == OUTPUTS[:3]
    for invalid in ((), ("velocity",), ("flow", 1), 3):
        with pytest.raises(ValueError):
            replace(config, output=invalid)
    assert [int(code) for code in render.FlowStatus] == list(range(5))


@pytest.mark.parametrize("output", [OUTPUTS, ("normals",), ("flow",)])
def test_typed_outputs_and_views(output):
    renderer = render.Renderer(width=16, height=16, num_cameras=2, num_probes=0,
                               fidelity="low", output=output)
    renderer.init(make_scene(2))
    look_forward(renderer)
    renderer.render_launch()
    renderer.render_sync()
    for name, channels, dtype in (("depth", 1, np.float32), ("instance_segmentation", 1, np.uint32),
                                  ("semantic_segmentation", 1, np.uint32), ("normals", 3, np.float32), ("flow", 2, np.float32),
                                  ("flow_status", 1, np.uint32)):
        accessor = getattr(renderer, name)
        if ("flow" if name == "flow_status" else name) not in output:
            with pytest.raises(RuntimeError, match="output"):
                accessor()
            continue
        shape = (2, 16, 16) + ((channels,) if channels > 1 else ())
        snapshot = accessor()
        view = accessor(out=np.empty(shape, dtype=dtype))
        assert snapshot.shape == shape and snapshot.dtype == dtype
        np.testing.assert_array_equal(snapshot, view)
        assert not np.shares_memory(snapshot, view)
        assert np.shares_memory(view, accessor(out=view))
        if renderer.backend != "optix":
            dlpack = np.from_dlpack(getattr(renderer, name + "_dlpack")())
            np.testing.assert_array_equal(dlpack, snapshot)
            assert not np.shares_memory(view, dlpack)
    if "normals" in output:
        np.testing.assert_array_equal(renderer.normals()[:, 8, 8], [[-1, 0, 0]] * 2)
        np.testing.assert_array_equal(renderer.normals()[:, 0, 0], 0)
        retained = renderer.normals()
        expected = retained.copy()
        del renderer
        gc.collect()
        np.testing.assert_array_equal(retained, expected)


def test_camera_flow_advances_only_explicitly():
    scene = render.Scene()
    scene.add_mesh(make_quad(2, half_size=20))
    renderer = render.Renderer(width=32, height=32, num_probes=0, output="flow", fidelity="low")
    renderer.init(scene)
    look_forward(renderer)
    renderer.render()
    np.testing.assert_allclose(renderer.flow(), 0, atol=1e-4)
    renderer.advance_flow()
    camera = renderer.camera(position=(0, 0.2, 0), look_at=(1, 0.2, 0), fov=60)
    renderer.set_cameras(camera[None])
    renderer.render()
    flow = renderer.flow()
    focal = 32 / (2 * np.tan(np.deg2rad(30)))
    np.testing.assert_allclose(flow[0, 16, 16], [-focal * 0.2 / 2, 0], atol=1e-4)
    status = renderer.flow_status()
    assert status[0, 16, 16] == render.FlowStatus.VALID
    assert np.any(status == render.FlowStatus.OUT_OF_FRAME)
    renderer.render_launch()
    renderer.render_sync()
    np.testing.assert_array_equal(renderer.flow(), flow)
    renderer.advance_flow()
    renderer.render()
    np.testing.assert_allclose(renderer.flow(), 0, atol=1e-4)


def test_object_flow_and_semantic_labels():
    scene = render.Scene()
    scene.add_mesh(make_quad(4, half_size=20))
    pool = render.AssetPool()
    obj = render.Object(name="moving", segmentation_class=23)
    obj.add_mesh(make_quad(0, half_size=0.5))
    asset = pool.add_object(obj)
    renderer = render.Renderer(width=32, height=32, num_probes=0, output=OUTPUTS, fidelity="low",
                               num_overlays=1, max_overlay_instances=1, max_overlays_per_camera=1)
    renderer.init(scene, asset_pool=pool)
    look_forward(renderer)
    renderer.attach(0, 0)
    placement = renderer.spawn(0, asset, render.make_transform(position=(2, 0, 0)))
    renderer.update()
    renderer.render()
    np.testing.assert_allclose(renderer.flow(), 0, atol=1e-4)
    renderer.advance_flow()
    renderer.set_transform(placement, render.make_transform(position=(2, 0.2, 0)))
    renderer.update()
    renderer.render()
    assert renderer.semantic_segmentation()[0, 16, 16] == 23
    focal = 32 / (2 * np.tan(np.deg2rad(30)))
    np.testing.assert_allclose(renderer.flow()[0, 16, 16], [focal * 0.2 / 2, 0], atol=1e-4)
    assert np.any(renderer.flow_status() == render.FlowStatus.OCCLUDED)


def test_cuda_output_dlpack():
    if render._backend() != "optix":
        pytest.skip("CUDA DLPack requires OptiX")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("PyTorch needs CUDA")
    renderer = render.Renderer(width=16, height=16, num_probes=0, output=OUTPUTS, fidelity="low")
    renderer.init(make_scene())
    look_forward(renderer)
    renderer.render()
    for name in ("depth", "instance_segmentation", "semantic_segmentation", "normals", "flow", "flow_status"):
        output = torch.from_dlpack(getattr(renderer, name + "_dlpack")())
        np.testing.assert_array_equal(output.cpu().numpy(), getattr(renderer, name)())
