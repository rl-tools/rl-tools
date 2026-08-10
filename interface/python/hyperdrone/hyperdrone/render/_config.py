from dataclasses import dataclass

from .. import jit

SHADING = {"low": 0, "medium": 1, "high": 2, "veryhigh": 3}
OUTPUT_MODE = {"rgb": 0, "rgbd": 1, "depth": 2, "segmentation": 3, "rgbd_segmentation": 4}

RENDER_TARGET = {"all": 0, "rgb": 1, "depth": 2, "segmentation": 3, "rgb_depth": 4, "collision": 5}
RENDER_PHASE = {"launch": 0, "sync": 1, "full": 2}
SAVE_TARGET = {"image": 0, "depth_image": 1, "depth_raw": 2, "segmentation_image": 3, "probes": 4}

VALID_MOTION_BLUR_SAMPLES = (1, 2, 4, 8, 16, 32)
VALID_ANTI_ALIASING_GRIDS = (1, 2, 3, 4)


@dataclass(frozen=True)
class RendererConfig:
    width: int
    height: int
    num_cameras: int
    num_probes: int
    shading: int
    output_mode: int
    motion_blur_samples: int
    anti_aliasing_grid: int
    num_overlays: int
    max_overlay_instances: int
    max_overlays_per_camera: int
    semantic_segmentation: bool
    dynamic_motion_blur: bool = False

    def __post_init__(self):
        if self.width < 1 or self.height < 1 or self.num_cameras < 1 or self.num_probes < 1:
            raise ValueError("width, height, num_cameras, and num_probes must be >= 1")
        if self.shading not in SHADING.values():
            raise ValueError(f"shading must be one of {sorted(SHADING)}")
        if self.output_mode not in OUTPUT_MODE.values():
            raise ValueError(f"output must be one of {sorted(OUTPUT_MODE)}")
        if self.motion_blur_samples not in VALID_MOTION_BLUR_SAMPLES:
            raise ValueError(f"motion_blur_samples must be one of {VALID_MOTION_BLUR_SAMPLES}")
        if self.anti_aliasing_grid not in VALID_ANTI_ALIASING_GRIDS:
            raise ValueError(f"anti_aliasing_grid must be one of {VALID_ANTI_ALIASING_GRIDS}")
        overlay_values = (self.num_overlays, self.max_overlay_instances, self.max_overlays_per_camera)
        if any(value > 0 for value in overlay_values) and not all(value > 0 for value in overlay_values):
            raise ValueError("overlay parameters (num_overlays, max_overlay_instances, max_overlays_per_camera) must be all zero or all nonzero")
        if self.semantic_segmentation and self.output_mode not in (OUTPUT_MODE["segmentation"], OUTPUT_MODE["rgbd_segmentation"]):
            raise ValueError("semantic_segmentation requires a segmentation output mode")
        if self.dynamic_motion_blur and (self.motion_blur_samples < 2 or self.num_overlays == 0):
            raise ValueError("dynamic_motion_blur requires motion_blur_samples >= 2 and overlays")

    @property
    def has_rgb(self):
        return self.output_mode in (0, 1, 4)

    @property
    def has_depth(self):
        return self.output_mode in (1, 2, 4)

    @property
    def has_segmentation(self):
        return self.output_mode in (3, 4)

    def canonical(self):
        # must match build_config_string() in _native/render/impl.cpp exactly
        return (
            f"w={self.width};h={self.height};nc={self.num_cameras};np={self.num_probes};"
            f"sh={self.shading};om={self.output_mode};mb={self.motion_blur_samples};aa={self.anti_aliasing_grid};"
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
            "HYPERDRONE_RENDER_SHADING": self.shading,
            "HYPERDRONE_RENDER_OUTPUT_MODE": self.output_mode,
            "HYPERDRONE_RENDER_MB_SAMPLES": self.motion_blur_samples,
            "HYPERDRONE_RENDER_AA_GRID": self.anti_aliasing_grid,
            "HYPERDRONE_RENDER_NUM_OVERLAYS": self.num_overlays,
            "HYPERDRONE_RENDER_MAX_OVERLAY_INSTANCES": self.max_overlay_instances,
            "HYPERDRONE_RENDER_MAX_OVERLAYS_PER_CAMERA": self.max_overlays_per_camera,
            "HYPERDRONE_RENDER_SEMANTIC_SEGMENTATION": 1 if self.semantic_segmentation else 0,
            "HYPERDRONE_RENDER_DYNAMIC_MB": 1 if self.dynamic_motion_blur else 0,
        }
