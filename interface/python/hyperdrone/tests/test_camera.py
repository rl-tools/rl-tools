import pytest

from hyperlux.camera import CameraSpec, Depth, RGB
from hyperdrone.camera import ImageObservation, camera_config_header, validate_observation


def test_camera_labels_do_not_change_generated_types():
    first = (CameraSpec("front", (RGB("float32"), Depth())), CameraSpec("rear", (Depth(),)))
    renamed = (CameraSpec("policy", (Depth(), RGB("float32"))), CameraSpec("diagnostic", (Depth(),)))
    assert camera_config_header(first, ImageObservation(("front",))) == camera_config_header(renamed, ImageObservation(("policy",)))


def test_observation_selection_is_explicit_and_preserves_requested_order():
    cameras = (CameraSpec("left", (RGB("float32"),)), CameraSpec("depth", (Depth(),)), CameraSpec("right", (RGB("float32"),)))
    observation = ImageObservation(("right", "left"))
    assert validate_observation(cameras, observation) == (2, 0)
    header = camera_config_header(cameras, observation)
    assert "    namespace h = rl_tools::rl::environments::hyperdrone;" in header
    assert "using CAMERA_OUTPUTS = h::CameraOutputs<out::List<Output1>, out::List<Output0>, out::List<Output1>>;" in header
    assert header.endswith("    using OBSERVATION = h::ImageObservation<2, 0>;\n}\n")
    assert "using OBSERVATION = h::NoObservation;" in camera_config_header(cameras)
    with pytest.raises(ValueError, match="must declare RGB"):
        validate_observation(cameras, ImageObservation(("depth",)))
    with pytest.raises(ValueError, match="not present"):
        validate_observation(cameras, ImageObservation(("missing",)))


def test_invalid_observation_fails_before_compilation():
    with pytest.raises(ValueError):
        ImageObservation(cameras=())
    with pytest.raises(TypeError):
        ImageObservation(cameras="front")
