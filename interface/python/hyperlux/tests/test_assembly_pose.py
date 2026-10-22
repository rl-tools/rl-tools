import numpy as np

import hyperlux
from hyperlux.camera import Segmentation
from test_pose import frame, pose, renderer_for


def test_box_hinge_and_slider_change_orientation_and_position():
    asset = hyperlux.ObjectAssembly()
    rod = hyperlux.primitives.box(size=(0.12, 1.6, 0.12), color=(1, 0.2, 0))
    cube = hyperlux.primitives.box(size=(0.3, 0.3, 0.3), color=(0, 0.8, 1))
    hinge = asset.add_part(rod, pose())
    block = asset.add_part(cube, pose(y=0.5), parent=hinge)
    asset.set_joint(hinge, "revolute", axis=(1, 0, 0), coordinate=0)
    asset.set_joint(block, "prismatic", axis=(0, 1, 0), coordinate=1)
    renderer, (handle,) = renderer_for(asset)
    placement = renderer.spawn(0, handle, hyperlux.Pose(pose(2)))
    images, masks = [], []
    for coordinates in ([0, 0], [np.pi / 2, 0.3]):
        renderer.set_pose(placement, hyperlux.Pose(pose(2)), coordinates=coordinates)
        images.append(frame(renderer)[0, ..., :3])
        masks.append(renderer.outputs[Segmentation()].read()[0].copy())
    for image, labels in zip(images, masks):
        rod_pixels = image[labels == 1].astype(np.float32)
        cube_pixels = image[labels == 2].astype(np.float32)
        assert len(rod_pixels) > 10 and len(cube_pixels) > 10
        assert np.all(rod_pixels[:, 0] > rod_pixels[:, 1])
        assert np.all(cube_pixels[:, 1] > cube_pixels[:, 0])
        assert np.all(cube_pixels[:, 2] > cube_pixels[:, 0])
    horizontal = np.argwhere(masks[0] == 1)
    vertical = np.argwhere(masks[1] == 1)
    assert np.ptp(horizontal[:, 1]) > 3 * np.ptp(horizontal[:, 0])
    assert np.ptp(vertical[:, 0]) > 3 * np.ptp(vertical[:, 1])
    before = np.argwhere(masks[0] == 2).mean(axis=0)
    after = np.argwhere(masks[1] == 2).mean(axis=0)
    assert np.linalg.norm(after - before) > 8
