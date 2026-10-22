"""The environment's camera vocabulary on top of hyperlux's rigs: which mounts feed the policy."""
from dataclasses import dataclass

from hyperlux.camera import RGB, camera_config_lines, validate_cameras
from hyperlux.camera import camera_config_header as _rig_header


@dataclass(frozen=True)
class ImageObservation:
    cameras: tuple[str, ...]

    def __post_init__(self):
        if isinstance(self.cameras, str):
            raise TypeError("observation cameras must be a sequence of camera names")
        cameras = tuple(self.cameras)
        if not cameras or any(not isinstance(name, str) or not name for name in cameras):
            raise ValueError("an image observation must select one or more camera names")
        if len(set(cameras)) != len(cameras):
            raise ValueError("observation camera names must be unique")
        object.__setattr__(self, "cameras", cameras)


def validate_observation(cameras, observation):
    cameras = validate_cameras(cameras)
    if observation is None:
        return ()
    if not isinstance(observation, ImageObservation):
        raise TypeError("observation must be ImageObservation or None")
    names = {camera.name: index for index, camera in enumerate(cameras)}
    indices = []
    for name in observation.cameras:
        if name not in names:
            raise ValueError(f"observation camera {name!r} is not present in the camera profiles")
        index = names[name]
        if RGB(format="float32") not in cameras[index].outputs:
            raise ValueError(f"observation camera {name!r} must declare RGB(format='float32')")
        indices.append(index)
    return tuple(indices)


def camera_config_header(cameras, observation=None):
    """hyperlux's rig header plus the environment's per-mount output profiles and policy observation."""
    cameras = validate_cameras(cameras)
    observation_indices = validate_observation(cameras, observation)
    _, aliases = camera_config_lines(cameras)
    profiles = [f"out::List<{', '.join(aliases[output] for output in camera.outputs)}>" for camera in cameras]
    observation_type = f"h::ImageObservation<{', '.join(str(index) for index in observation_indices)}>" if observation is not None else "h::NoObservation"
    declarations = [
        "    namespace h = rl_tools::rl::environments::hyperdrone;",
        f"    using CAMERA_OUTPUTS = h::CameraOutputs<{', '.join(profiles)}>;",
        f"    using OBSERVATION = {observation_type};",
    ]
    return _rig_header(cameras, declarations)
