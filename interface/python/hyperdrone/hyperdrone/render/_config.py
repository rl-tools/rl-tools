from dataclasses import dataclass

from .. import _jit as jit
from .._outputs import VALID_MOTION_BLUR_SAMPLES, normalize_outputs, output_defines

FIDELITY = ("low", "medium", "high", "veryhigh")

VALID_ANTI_ALIASING_GRIDS = (1, 2, 3, 4)


@dataclass(frozen=True, kw_only=True)
class RendererConfig:
    width: int
    height: int
    num_cameras: int
    num_probes: int
    fidelity: str
    output: tuple
    motion_blur_samples: int
    anti_aliasing_grid: int
    num_overlays: int
    max_overlay_instances: int
    max_overlays_per_camera: int
    dynamic_motion_blur: bool = False

    def __post_init__(self):
        object.__setattr__(self, "output", normalize_outputs(self.output))
        for name in ("width", "height", "num_cameras", "motion_blur_samples", "anti_aliasing_grid"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("num_probes", "num_overlays", "max_overlay_instances", "max_overlays_per_camera"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.fidelity not in FIDELITY:
            raise ValueError(f"fidelity must be one of {sorted(FIDELITY)}")
        if self.motion_blur_samples not in VALID_MOTION_BLUR_SAMPLES:
            raise ValueError(f"motion_blur_samples must be one of {VALID_MOTION_BLUR_SAMPLES}")
        if self.anti_aliasing_grid not in VALID_ANTI_ALIASING_GRIDS:
            raise ValueError(f"anti_aliasing_grid must be one of {VALID_ANTI_ALIASING_GRIDS}")
        overlay_values = (self.num_overlays, self.max_overlay_instances, self.max_overlays_per_camera)
        if any(value > 0 for value in overlay_values) and not all(value > 0 for value in overlay_values):
            raise ValueError("overlay parameters (num_overlays, max_overlay_instances, max_overlays_per_camera) must be all zero or all nonzero")
        if self.dynamic_motion_blur and (self.motion_blur_samples < 2 or self.num_overlays == 0):
            raise ValueError("dynamic_motion_blur requires motion_blur_samples >= 2 and overlays")

    @property
    def has_probes(self):
        return self.num_probes > 0

    def canonical(self):
        return ";".join(f"{name}={value}" for name, value in sorted(self.defines().items()))

    def key(self):
        return jit.canonical_key(self.canonical())

    def defines(self):
        return {
            "HYPERDRONE_RENDER_WIDTH": self.width,
            "HYPERDRONE_RENDER_HEIGHT": self.height,
            "HYPERDRONE_RENDER_NUM_CAMERAS": self.num_cameras,
            "HYPERDRONE_RENDER_NUM_PROBES": self.num_probes,
            "HYPERDRONE_RENDER_SHADING": FIDELITY.index(self.fidelity),
            **output_defines(self.output, "HYPERDRONE_RENDER"),
            "HYPERDRONE_RENDER_MB_SAMPLES": self.motion_blur_samples,
            "HYPERDRONE_RENDER_AA_GRID": self.anti_aliasing_grid,
            "HYPERDRONE_RENDER_NUM_OVERLAYS": self.num_overlays,
            "HYPERDRONE_RENDER_MAX_OVERLAY_INSTANCES": self.max_overlay_instances,
            "HYPERDRONE_RENDER_MAX_OVERLAYS_PER_CAMERA": self.max_overlays_per_camera,
            "HYPERDRONE_RENDER_DYNAMIC_MB": 1 if self.dynamic_motion_blur else 0,
        }
