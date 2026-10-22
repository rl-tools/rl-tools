import numpy as np
import pytest

import hyperlux


def test_box_bounds_and_assembly():
    box = hyperlux.primitives.box(size=(0.25, 2.0, 0.5), color=(0.2, 0.4, 0.8))
    assert isinstance(box, hyperlux.Object)
    assert box.num_meshes == 1
    asset = hyperlux.ObjectAssembly()
    assert asset.add_part(box, hyperlux.make_transform()) == 0
    low, high = asset.object_bounds(0)
    np.testing.assert_array_equal(low, [-0.125, -1.0, -0.25])
    np.testing.assert_array_equal(high, [0.125, 1.0, 0.25])
    pool = hyperlux.AssetPool()
    assert pool.add_assembly(asset) == 0
    assert pool.num_assets == 1


def test_default_box():
    asset = hyperlux.ObjectAssembly()
    asset.add_part(hyperlux.primitives.box(), hyperlux.make_transform())
    low, high = asset.object_bounds(0)
    np.testing.assert_array_equal(low, [-0.5, -0.5, -0.5])
    np.testing.assert_array_equal(high, [0.5, 0.5, 0.5])


@pytest.mark.parametrize("value", [0.0, -1.0, float("inf"), -float("inf"), float("nan")])
@pytest.mark.parametrize("axis", range(3))
def test_box_rejects_invalid_size(value, axis):
    size = [1.0, 1.0, 1.0]
    size[axis] = value
    with pytest.raises(ValueError, match="size"):
        hyperlux.primitives.box(size=size)


@pytest.mark.parametrize("value", [float("inf"), -float("inf"), float("nan")])
@pytest.mark.parametrize("axis", range(3))
def test_box_rejects_invalid_color(value, axis):
    color = [0.8, 0.8, 0.8]
    color[axis] = value
    with pytest.raises(ValueError, match="color"):
        hyperlux.primitives.box(color=color)


@pytest.mark.parametrize("kwargs", [dict(size=(1, 2)), dict(size=(1, 2, 3, 4)), dict(color=(1, 2))])
def test_box_rejects_wrong_vector_lengths(kwargs):
    with pytest.raises(TypeError):
        hyperlux.primitives.box(**kwargs)
