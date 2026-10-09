from enum import IntEnum


CHANNELS = ("rgb", "depth", "segmentation", "normals", "flow")
VALID_MOTION_BLUR_SAMPLES = (1, 2, 4, 8, 16, 32)


class FlowStatus(IntEnum):
    VALID = 0
    MISS = 1
    BEHIND = 2
    OUT_OF_FRAME = 3
    OCCLUDED = 4


def normalize_outputs(output):
    if isinstance(output, str):
        output = output.lower()
        output = {"rgbd": ("rgb", "depth"),
                  "rgbd_segmentation": ("rgb", "depth", "segmentation")}.get(output, (output,))
    if not isinstance(output, (tuple, list)) or not output or any(not isinstance(name, str) for name in output):
        raise ValueError(f"output must be a channel name or a nonempty sequence of {CHANNELS}")
    names = {name.lower() for name in output}
    if names.difference(CHANNELS):
        raise ValueError(f"unknown outputs {sorted(names.difference(CHANNELS))}; choose from {CHANNELS}")
    return tuple(name for name in CHANNELS if name in names)


def output_defines(output, prefix):
    return {f"{prefix}_OUTPUT_{name.upper()}": int(name in output) for name in CHANNELS}
