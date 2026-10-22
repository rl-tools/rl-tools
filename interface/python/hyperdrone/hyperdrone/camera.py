"""Camera profiles and typed outputs shared by rendering and environments."""

from dataclasses import asdict, dataclass
import math
import struct
from typing import Literal, TypeAlias


CAMERA_SCHEMA_VERSION = 1


def _choice(name, value, choices):
    if value not in choices:
        raise ValueError(f"{name} must be one of {choices}, got {value!r}")


def _display_scale(descriptor, default):
    value = descriptor.scale
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError("display scale must be a finite positive number")
    try:
        scale = struct.unpack("f", struct.pack("f", value))[0]
    except OverflowError:
        raise ValueError("display scale must be representable as float32") from None
    if not math.isfinite(scale) or scale == 0:
        raise ValueError("display scale must be representable as float32")
    if descriptor.format != "rgba8" and scale != default:
        raise ValueError(f"{type(descriptor).__name__} scale applies only to format='rgba8'")
    object.__setattr__(descriptor, "scale", scale)


@dataclass(frozen=True)
class RGB:
    format: Literal["float32", "rgba8"] = "rgba8"

    def __post_init__(self):
        _choice("RGB format", self.format, ("float32", "rgba8"))


@dataclass(frozen=True)
class Depth:
    format: Literal["float32", "rgba8"] = "float32"
    scale: float = 10.0

    def __post_init__(self):
        _choice("depth format", self.format, ("float32", "rgba8"))
        _display_scale(self, 10.0)


@dataclass(frozen=True)
class Segmentation:
    format: Literal["uint32", "rgba8"] = "uint32"
    labels: Literal["instance", "semantic"] = "instance"

    def __post_init__(self):
        _choice("segmentation format", self.format, ("uint32", "rgba8"))
        _choice("segmentation labels", self.labels, ("instance", "semantic"))


@dataclass(frozen=True)
class Normals:
    format: Literal["float32", "rgba8"] = "float32"

    def __post_init__(self):
        _choice("normals format", self.format, ("float32", "rgba8"))


@dataclass(frozen=True)
class Flow:
    direction: Literal["backward", "forward"] = "backward"
    format: Literal["float32", "rgba8"] = "float32"
    scale: float = 20.0

    def __post_init__(self):
        _choice("flow direction", self.direction, ("backward", "forward"))
        _choice("flow format", self.format, ("float32", "rgba8"))
        _display_scale(self, 20.0)


@dataclass(frozen=True)
class FlowStatus:
    direction: Literal["backward", "forward"] = "backward"
    format: Literal["uint32", "rgba8"] = "uint32"

    def __post_init__(self):
        _choice("flow direction", self.direction, ("backward", "forward"))
        _choice("flow status format", self.format, ("uint32", "rgba8"))


@dataclass(frozen=True)
class Probes:
    pass


OutputDescriptor: TypeAlias = RGB | Depth | Segmentation | Normals | Flow | FlowStatus | Probes


_DESCRIPTORS = {"rgb": RGB, "depth": Depth, "segmentation": Segmentation, "normals": Normals, "flow": Flow, "flow_status": FlowStatus, "probes": Probes}
_QUANTITIES = {descriptor: quantity for quantity, descriptor in _DESCRIPTORS.items()}


def descriptor_metadata(output):
    if type(output) not in _QUANTITIES:
        raise TypeError(f"expected a camera output descriptor, got {output!r}")
    return {"quantity": _QUANTITIES[type(output)], **asdict(output)}


def descriptor_from_metadata(metadata):
    values = dict(metadata)
    quantity = values.pop("quantity")
    try:
        descriptor = _DESCRIPTORS[quantity]
    except KeyError:
        raise ValueError(f"unsupported native output quantity {quantity!r}") from None
    return descriptor(**values)


def _slot(output):
    metadata = descriptor_metadata(output)
    return tuple(metadata.get(name, "") for name in ("quantity", "format", "labels", "direction"))


def _output_key(output):
    return (*_slot(output), getattr(output, "scale", 0.0))


@dataclass(frozen=True)
class CameraSpec:
    name: str
    outputs: tuple[OutputDescriptor, ...]

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("camera name must be a nonempty string")
        try:
            outputs = tuple(self.outputs)
        except TypeError:
            raise TypeError("camera outputs must be an iterable of output descriptors") from None
        for output in outputs:
            descriptor_metadata(output)
        object.__setattr__(self, "outputs", tuple(sorted(set(outputs), key=_output_key)))


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


def validate_cameras(cameras):
    cameras = tuple(cameras)
    if not cameras or any(not isinstance(camera, CameraSpec) for camera in cameras):
        raise ValueError("cameras must contain one or more CameraSpec profiles")
    if len({camera.name for camera in cameras}) != len(cameras):
        raise ValueError("camera names must be unique")
    slots = {}
    for camera in cameras:
        for output in camera.outputs:
            previous = slots.setdefault(_slot(output), output)
            if previous != output:
                raise ValueError(f"camera {camera.name!r} uses {output!r}, conflicting with {previous!r}; each native quantity/representation slot must use identical parameters across all cameras")
    if not slots:
        raise ValueError("at least one camera must declare an output")
    return cameras


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


def backend_requirement(cameras):
    outputs = tuple(output for camera in validate_cameras(cameras) for output in camera.outputs)
    for descriptor, requirement in ((Normals, "normals"), (Segmentation, "segmentation"), (Depth, "depth")):
        if any(isinstance(output, descriptor) for output in outputs):
            return requirement
    return "base"


def camera_config_header(cameras, observation=None, *, environment=False):
    cameras = validate_cameras(cameras)
    observation_indices = validate_observation(cameras, observation)
    outputs = sorted({output for camera in cameras for output in camera.outputs}, key=_output_key)
    lines = ["#pragma once", '#include <rl_tools/hyperlux/outputs/outputs.h>', "namespace hyperdrone_camera_config {", "    namespace out = rl_tools::hyperlux::outputs;", f"    constexpr unsigned int SCHEMA_VERSION = {CAMERA_SCHEMA_VERSION};"]
    aliases = {}
    formats = {"float32": "Float32", "rgba8": "RGBA8", "uint32": "UInt32"}
    for index, output in enumerate(outputs):
        alias = f"Output{index}"
        aliases[output] = alias
        arguments = []
        if isinstance(output, (Flow, FlowStatus)):
            arguments.append(f"out::direction::{output.direction.title()}")
        if hasattr(output, "format"):
            arguments.append(f"out::format::{formats[output.format]}")
        if isinstance(output, Segmentation):
            arguments.append(f"out::labels::{output.labels.title()}")
        if isinstance(output, (Depth, Flow)):
            parameter = f"Parameters{index}"
            scale = repr(output.scale)
            lines.append(f"    struct {parameter} {{ static constexpr float SCALE = {scale}f; }};")
            arguments.append(parameter)
        cpp_type = type(output).__name__
        suffix = f"<{', '.join(arguments)}>" if arguments else ""
        lines.append(f"    using {alias} = out::{cpp_type}{suffix};")
    entries = []
    for output in outputs:
        selected = [str(index) for index, camera in enumerate(cameras) if output in camera.outputs]
        entry = aliases[output]
        if len(selected) != len(cameras):
            entry = f"out::Cameras<out::cameras::Every<{len(cameras)}, {', '.join(selected)}>, {entry}>"
        entries.append(entry)
    lines.append(f"    using OUTPUTS = out::List<{', '.join(entries)}>;")
    if environment:
        lines.append("    namespace h = rl_tools::rl::environments::hyperdrone;")
        profiles = [f"out::List<{', '.join(aliases[output] for output in camera.outputs)}>" for camera in cameras]
        lines.append(f"    using CAMERA_OUTPUTS = h::CameraOutputs<{', '.join(profiles)}>;")
        observation_type = f"h::ImageObservation<{', '.join(str(index) for index in observation_indices)}>" if observation is not None else "h::NoObservation"
        lines.append(f"    using OBSERVATION = {observation_type};")
    lines.append("}")
    return "\n".join(lines) + "\n"


__all__ = ["CameraSpec", "RGB", "Depth", "Segmentation", "Normals", "Flow", "FlowStatus", "Probes", "ImageObservation", "OutputDescriptor"]
