from dataclasses import dataclass

from .. import jit
from .._outputs import VALID_MOTION_BLUR_SAMPLES, normalize_outputs, output_defines

FIDELITY = {"low": 0, "medium": 1, "high": 2, "veryhigh": 3}

VALID_ANTI_ALIASING_GRIDS = (1, 2, 3, 4)


@dataclass(frozen=True)
class RendererConfig:
    width: int
    height: int
    num_cameras: int
    num_probes: int
    fidelity: int
    output: tuple
    motion_blur_samples: int
    anti_aliasing_grid: int
    num_overlays: int
    max_overlay_instances: int
    max_overlays_per_camera: int
    semantic_segmentation: bool
    dynamic_motion_blur: bool = False

    def __post_init__(self):
        object.__setattr__(self, "output", normalize_outputs(self.output))
        if self.width < 1 or self.height < 1 or self.num_cameras < 1:
            raise ValueError("width, height, and num_cameras must be >= 1")
        if self.num_probes < 0:
            raise ValueError("num_probes must be >= 0 (0 compiles the collision-probe pass out)")
        if self.fidelity not in FIDELITY.values():
            raise ValueError(f"fidelity must be one of {sorted(FIDELITY)}")
        if self.motion_blur_samples not in VALID_MOTION_BLUR_SAMPLES:
            raise ValueError(f"motion_blur_samples must be one of {VALID_MOTION_BLUR_SAMPLES}")
        if self.anti_aliasing_grid not in VALID_ANTI_ALIASING_GRIDS:
            raise ValueError(f"anti_aliasing_grid must be one of {VALID_ANTI_ALIASING_GRIDS}")
        overlay_values = (self.num_overlays, self.max_overlay_instances, self.max_overlays_per_camera)
        if any(value > 0 for value in overlay_values) and not all(value > 0 for value in overlay_values):
            raise ValueError("overlay parameters (num_overlays, max_overlay_instances, max_overlays_per_camera) must be all zero or all nonzero")
        if self.semantic_segmentation and not self.has_segmentation:
            raise ValueError("semantic_segmentation requires a segmentation output mode")
        if self.dynamic_motion_blur and (self.motion_blur_samples < 2 or self.num_overlays == 0):
            raise ValueError("dynamic_motion_blur requires motion_blur_samples >= 2 and overlays")

    @property
    def has_rgb(self):
        return "rgb" in self.output

    @property
    def has_depth(self):
        return "depth" in self.output

    @property
    def has_segmentation(self):
        return "segmentation" in self.output

    @property
    def has_normals(self):
        return "normals" in self.output

    @property
    def has_flow(self):
        return "flow" in self.output

    @property
    def has_probes(self):
        return self.num_probes > 0

    def canonical(self):
        return (
            f"w={self.width};h={self.height};nc={self.num_cameras};np={self.num_probes};"
            f"sh={self.fidelity};out={','.join(self.output)};mb={self.motion_blur_samples};aa={self.anti_aliasing_grid};"
            f"no={self.num_overlays};moi={self.max_overlay_instances};mopc={self.max_overlays_per_camera};"
            f"ss={1 if self.semantic_segmentation else 0};dmb={1 if self.dynamic_motion_blur else 0}"
        )

    def key(self):
        return jit.canonical_key(self.canonical())

    def defines(self):
        return {
            "HYPERDRONE_RENDER_WIDTH": self.width,
            "HYPERDRONE_RENDER_HEIGHT": self.height,
            "HYPERDRONE_RENDER_NUM_CAMERAS": self.num_cameras,
            "HYPERDRONE_RENDER_NUM_PROBES": self.num_probes,
            "HYPERDRONE_RENDER_SHADING": self.fidelity,
            **output_defines(self.output, "HYPERDRONE_RENDER"),
            "HYPERDRONE_RENDER_MB_SAMPLES": self.motion_blur_samples,
            "HYPERDRONE_RENDER_AA_GRID": self.anti_aliasing_grid,
            "HYPERDRONE_RENDER_NUM_OVERLAYS": self.num_overlays,
            "HYPERDRONE_RENDER_MAX_OVERLAY_INSTANCES": self.max_overlay_instances,
            "HYPERDRONE_RENDER_MAX_OVERLAYS_PER_CAMERA": self.max_overlays_per_camera,
            "HYPERDRONE_RENDER_SEMANTIC_SEGMENTATION": 1 if self.semantic_segmentation else 0,
            "HYPERDRONE_RENDER_DYNAMIC_MB": 1 if self.dynamic_motion_blur else 0,
        }
