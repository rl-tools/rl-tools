from hyperlux.camera import CameraSpec, RGB, Depth, Segmentation, Normals, Flow, FlowStatus, Probes
import numpy as np
import pytest

import hyperlux
from test_render import look_forward, make_quad


def test_placement_class_labels_every_part_without_changing_the_asset():
    scene = hyperlux.Scene()
    wall = hyperlux.Object()
    wall.assign_segmentation_class(7, taxonomy="room", classes={0: "Unknown", 7: "Wall"})
    wall.add_mesh(make_quad(8, half_size=20))
    scene.add_object(wall)
    obj = hyperlux.Object()
    obj.add_mesh(make_quad(0, half_size=0.5))
    pool = hyperlux.AssetPool()
    asset = pool.add_object(obj)
    renderer = hyperlux.HyperLux(32, 32, cameras=(CameraSpec("camera", outputs=(Segmentation(), Segmentation(labels="semantic"), Probes(),)),),
                               fidelity="low", num_overlays=1, max_overlay_instances=2,
                               max_overlays_per_camera=1)
    renderer.init(scene, asset_pool=pool)
    look_forward(renderer)
    renderer.attach(0, 0)
    agent = hyperlux.AGENT_SEGMENTATION_CLASS
    left = renderer.spawn(0, asset, hyperlux.Pose(hyperlux.make_transform(position=(3, 0.7, 0))), segmentation_class=agent)
    right = renderer.spawn(0, asset, hyperlux.Pose(hyperlux.make_transform(position=(3, -0.7, 0))))
    renderer.update()
    renderer.render()
    instances = renderer.outputs[Segmentation()].read()
    labels = renderer.outputs[Segmentation(labels="semantic")].read()
    left_mask = instances == scene.num_instances + left.first_slot
    right_mask = instances == scene.num_instances + right.first_slot
    assert left_mask.any() and right_mask.any()
    np.testing.assert_array_equal(labels[left_mask], agent)
    np.testing.assert_array_equal(labels[right_mask], 0)
    np.testing.assert_array_equal(labels[instances == 0], 7)
    assert pool.object_segmentation_class(asset, 0) == 0
    with pytest.raises(ValueError, match="reserved"):
        renderer.spawn(0, asset, hyperlux.Pose(hyperlux.make_transform(position=(3, 0, 0))), segmentation_class=0xFFFFFFFF)
    with pytest.raises(ValueError):
        wall.assign_segmentation_class(agent, taxonomy="room", classes={0: "Unknown", agent: "Agent"})
