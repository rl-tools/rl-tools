from dataclasses import dataclass

from . import _jit as jit
from ._outputs import VALID_MOTION_BLUR_SAMPLES
from .camera import CameraSpec, RGB, Probes, backend_requirement, camera_config_header, validate_cameras

FIDELITY = ("low", "medium", "high", "veryhigh")
VALID_ANTI_ALIASING_GRIDS = (1, 2, 3, 4)


@dataclass(frozen=True, kw_only=True)
class Config:
    width: int
    height: int
    cameras: tuple[CameraSpec, ...] = (CameraSpec("camera", outputs=(RGB(format="rgba8"),)),)
    batch_size: int = 1
    num_probes: int = 1
    fidelity: str = "high"
    motion_blur_samples: int = 1
    anti_aliasing_grid: int = 1
    num_overlays: int = 0
    max_overlay_instances: int = 0
    max_overlays_per_camera: int = 0
    dynamic_motion_blur: bool = False

    def __post_init__(self):
        if isinstance(self.fidelity, str):
            object.__setattr__(self, "fidelity", self.fidelity.lower())
        object.__setattr__(self, "dynamic_motion_blur", bool(self.dynamic_motion_blur))
        object.__setattr__(self, "cameras", validate_cameras(self.cameras))
        for name in ("width", "height", "batch_size", "motion_blur_samples", "anti_aliasing_grid"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("num_probes", "num_overlays", "max_overlay_instances", "max_overlays_per_camera"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.has_probes and self.num_probes == 0:
            raise ValueError("Probes outputs require num_probes > 0")
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
    def num_cameras(self):
        return len(self.cameras) * self.batch_size

    @property
    def has_probes(self):
        return any(Probes() in camera.outputs for camera in self.cameras)

    def camera_config(self):
        return camera_config_header(self.cameras)

    def canonical(self):
        return ";".join(f"{name}={value}" for name, value in sorted(self.defines().items())) + "\n" + self.camera_config()

    def key(self):
        return jit.canonical_key(self.canonical())

    def defines(self):
        return {
            "HYPERLUX_WIDTH": self.width,
            "HYPERLUX_HEIGHT": self.height,
            "HYPERLUX_NUM_CAMERAS": self.num_cameras,
            "HYPERLUX_NUM_PROBES": self.num_probes if self.has_probes else 0,
            "HYPERLUX_SHADING": FIDELITY.index(self.fidelity),
            "HYPERLUX_CAMERA_BACKEND_REQUIREMENT": backend_requirement(self.cameras),
            "HYPERLUX_MB_SAMPLES": self.motion_blur_samples,
            "HYPERLUX_AA_GRID": self.anti_aliasing_grid,
            "HYPERLUX_NUM_OVERLAYS": self.num_overlays,
            "HYPERLUX_MAX_OVERLAY_INSTANCES": self.max_overlay_instances,
            "HYPERLUX_MAX_OVERLAYS_PER_CAMERA": self.max_overlays_per_camera,
            "HYPERLUX_DYNAMIC_MB": 1 if self.dynamic_motion_blur else 0,
        }
