from dataclasses import FrozenInstanceError

import pytest

from hyperdrone.camera import CameraSpec, Depth, Flow, FlowStatus, ImageObservation, Normals, Probes, RGB, Segmentation, backend_requirement, camera_config_header, descriptor_from_metadata, descriptor_metadata, validate_cameras, validate_observation


def test_profiles_are_immutable_and_normalize_output_order():
    camera = CameraSpec("front", [Depth(), RGB(), Depth()])
    assert camera.outputs == CameraSpec("front", (RGB(), Depth())).outputs
    with pytest.raises(FrozenInstanceError):
        camera.name = "rear"


def test_camera_labels_and_output_declaration_order_do_not_change_generated_types():
    first = (CameraSpec("front", (RGB(), Depth())), CameraSpec("rear", (Depth(),)))
    renamed = (CameraSpec("policy", (Depth(), RGB())), CameraSpec("diagnostic", (Depth(),)))
    assert camera_config_header(first) == camera_config_header(renamed)
    assert camera_config_header(first, ImageObservation(("front",)), environment=True) == camera_config_header(renamed, ImageObservation(("policy",)), environment=True)
    assert camera_config_header(first) != camera_config_header(tuple(reversed(first)))


def test_sparse_profiles_lower_into_periodic_selections():
    cameras = (CameraSpec("front", (RGB(), Flow())), CameraSpec("rear", (Depth(),)), CameraSpec("down", (RGB(), FlowStatus())))
    header = camera_config_header(cameras)
    assert "out::cameras::Every<3, 0, 2>" in header
    assert "out::cameras::Every<3, 1>" in header
    assert "out::Flow<out::direction::Backward, out::format::Float32" in header
    assert "out::FlowStatus<out::direction::Backward, out::format::UInt32>" in header
    assert "hyperdrone" not in header.replace("hyperdrone_camera_config", "")


def test_observation_selection_is_explicit_and_preserves_requested_order():
    cameras = (CameraSpec("left", (RGB(),)), CameraSpec("depth", (Depth(),)), CameraSpec("right", (RGB(),)))
    observation = ImageObservation(("right", "left"))
    assert validate_observation(cameras, observation) == (2, 0)
    assert "using OBSERVATION = h::ImageObservation<2, 0>;" in camera_config_header(cameras, observation, environment=True)
    assert "using OBSERVATION = h::NoObservation;" in camera_config_header(cameras, environment=True)
    with pytest.raises(ValueError, match="must declare RGB"):
        validate_observation(cameras, ImageObservation(("depth",)))
    with pytest.raises(ValueError, match="not present"):
        validate_observation(cameras, ImageObservation(("missing",)))


def test_conflicting_native_slots_rejected_across_disjoint_selections():
    cameras = (CameraSpec("near", (Depth(format="rgba8", scale=1),)), CameraSpec("far", (Depth(format="rgba8", scale=100),)))
    with pytest.raises(ValueError, match="quantity/representation slot"):
        validate_cameras(cameras)
    validate_cameras((CameraSpec("front", (Depth(), Depth(format="rgba8", scale=100))),))


@pytest.mark.parametrize("descriptor_type", [Depth, Flow])
def test_raw_display_scales_normalize_profiles_cache_and_native_metadata(descriptor_type):
    from hyperdrone.render import RendererConfig

    default = descriptor_type()
    assert descriptor_type(scale=37.5) == default
    cameras = validate_cameras((CameraSpec("front", (default,)), CameraSpec("rear", (descriptor_type(scale=100),))))
    equivalent = (CameraSpec("front", (default,)), CameraSpec("rear", (default,)))
    assert RendererConfig(width=8, height=8, cameras=cameras).key() == RendererConfig(width=8, height=8, cameras=equivalent).key()
    assert descriptor_from_metadata({**descriptor_metadata(default), "scale": 37.5}) == default
    with pytest.raises(ValueError, match="quantity/representation slot"):
        validate_cameras((CameraSpec("front", (descriptor_type(format="rgba8", scale=1),)), CameraSpec("rear", (descriptor_type(format="rgba8", scale=100),))))


@pytest.mark.parametrize("descriptor", [RGB(), RGB("rgba8"), Depth(), Depth("rgba8", 0.1), Segmentation(), Segmentation("rgba8", "semantic"), Normals(), Normals("rgba8"), Flow(), Flow("forward", "rgba8", 0.1), FlowStatus(), FlowStatus("forward", "rgba8"), Probes()])
def test_native_metadata_roundtrips_descriptor_identity(descriptor):
    assert descriptor_from_metadata(descriptor_metadata(descriptor)) == descriptor
    assert hash(descriptor_from_metadata(descriptor_metadata(descriptor))) == hash(descriptor)


@pytest.mark.parametrize("constructor,arguments", [(RGB, {"format": "float64"}), (Depth, {"scale": 0}), (Depth, {"scale": float("nan")}), (Flow, {"scale": float("inf")}), (Flow, {"direction": "sideways"}), (FlowStatus, {"format": "float32"}), (Segmentation, {"labels": "class"}), (ImageObservation, {"cameras": ()})])
def test_invalid_descriptor_parameters_fail_before_compilation(constructor, arguments):
    with pytest.raises(ValueError):
        constructor(**arguments)


def test_profile_validation_and_backend_requirements():
    with pytest.raises(ValueError, match="names must be unique"):
        validate_cameras((CameraSpec("front", (RGB(),)), CameraSpec("front", (Depth(),))))
    with pytest.raises(ValueError, match="at least one camera"):
        validate_cameras((CameraSpec("empty", ()),))
    assert backend_requirement((CameraSpec("probe", (Probes(),)),)) == "base"
    assert backend_requirement((CameraSpec("sensor", (Depth(), Segmentation(), Normals())),)) == "normals"
