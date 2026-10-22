from enum import IntEnum


VALID_MOTION_BLUR_SAMPLES = (1, 2, 4, 8, 16, 32)


class FlowStatus(IntEnum):
    VALID = 0
    MISS = 1
    BEHIND = 2
    OUT_OF_FRAME = 3
    OCCLUDED = 4
