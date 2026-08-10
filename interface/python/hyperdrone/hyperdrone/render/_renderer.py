import math

import numpy as np

from ._component import ensure_renderer_library, load_core
from ._config import (
    OUTPUT_MODE,
    RENDER_PHASE,
    RENDER_TARGET,
    SAVE_TARGET,
    SHADING,
    RendererConfig,
)


def _from_dlpack(value):
    # accept any DLPack producer (torch/jax/... CPU tensors) without copying
    if not isinstance(value, np.ndarray) and hasattr(value, "__dlpack__"):
        try:
            return np.from_dlpack(value)
        except (RuntimeError, TypeError, BufferError) as error:
            raise ValueError(
                "hyperdrone: input tensor could not be viewed as CPU memory (move it to "
                "the CPU first, or use a CUDA-resident tensor on the OptiX backend)"
            ) from error
    return value


def _as_cameras(cameras, num_cameras):
    array = np.ascontiguousarray(_from_dlpack(cameras), dtype=np.float32)
    if array.shape == (4, 3) or array.shape == (12,):
        array = array.reshape(1, 12)
    elif array.ndim == 3 and array.shape[1:] == (4, 3):
        array = array.reshape(array.shape[0], 12)
    elif array.ndim == 2 and array.shape[1] == 12:
        pass
    else:
        raise ValueError("cameras must have shape (num_cameras, 4, 3), (num_cameras, 12), or a single (4, 3)")
    if array.shape[0] != num_cameras:
        raise ValueError(f"expected {num_cameras} cameras, got {array.shape[0]}")
    return array


def _as_transform(transform):
    array = np.ascontiguousarray(_from_dlpack(transform), dtype=np.float32)
    if array.shape not in ((3, 4), (12,)):
        raise ValueError("transform must have shape (3, 4) or (12,)")
    return array


class _DLPackBuffer:
    """DLPack producer over a live renderer buffer. Capsules are single-use, so each
    __dlpack__ call mints a fresh one; the renderer reference keeps the memory alive."""

    def __init__(self, renderer, make_capsule, device_type):
        self._renderer = renderer
        self._make_capsule = make_capsule
        self._device_type = device_type

    def __dlpack__(self, stream=None, **_kwargs):
        return self._make_capsule()

    def __dlpack_device__(self):
        return (self._device_type, 0)


class Renderer:
    """A JIT-compiled instantiation of the RLtools raytracer.

    Compile-time constants (resolution, camera count, output mode, shading tier, motion
    blur, anti-aliasing, overlay capacities) are passed here; the first use of a new
    combination compiles a renderer library into the hyperdrone cache, subsequent uses
    load the cached library directly.
    """

    def __init__(self, width, height, num_cameras=1, num_probes=1, output="rgb", shading="high",
                 motion_blur_samples=0, anti_aliasing_grid=0,
                 num_overlays=0, max_overlay_instances=0, max_overlays_per_camera=0,
                 semantic_segmentation=False, dynamic_motion_blur=False):
        if isinstance(shading, str):
            shading = SHADING[shading.lower()]
        if isinstance(output, str):
            output = OUTPUT_MODE[output.lower()]
        self.config = RendererConfig(
            width=int(width),
            height=int(height),
            num_cameras=int(num_cameras),
            num_probes=int(num_probes),
            shading=shading,
            output_mode=output,
            motion_blur_samples=max(1, int(motion_blur_samples)),
            anti_aliasing_grid=max(1, int(anti_aliasing_grid)),
            num_overlays=int(num_overlays),
            max_overlay_instances=int(max_overlay_instances),
            max_overlays_per_camera=int(max_overlays_per_camera),
            semantic_segmentation=bool(semantic_segmentation),
            dynamic_motion_blur=bool(dynamic_motion_blur),
        )
        library = ensure_renderer_library(self.config)
        core = load_core()
        self._renderer = core.JitRenderer(str(library), self.config.canonical())
        self._scene = None
        self._asset_pool = None

    @property
    def width(self):
        return self.config.width

    @property
    def height(self):
        return self.config.height

    @property
    def num_cameras(self):
        return self.config.num_cameras

    @property
    def num_probes(self):
        return self.config.num_probes

    @property
    def aspect(self):
        return self.config.width / self.config.height

    @property
    def backend(self):
        return self._renderer.backend

    @property
    def scene_bounds(self):
        center, half_extent, camera_radius = self._renderer.scene_bounds()
        return {"center": center, "half_extent": half_extent, "camera_radius": camera_radius}

    def init(self, scene, asset_pool=None):
        """Build acceleration structures for a scene. The scene (and asset pool) must stay
        alive as long as this renderer; references are held here to guarantee that."""
        self._scene = scene
        self._asset_pool = asset_pool
        self._renderer.init(scene, asset_pool)
        return self

    def camera(self, position, look_at, up=(0.0, 0.0, 1.0), fov=math.radians(60.0)):
        core = load_core()
        return core.make_camera(tuple(position), tuple(look_at), tuple(up), float(fov), self.aspect)

    def set_cameras(self, cameras, stream=0):
        """Upload camera ray-gen bases. Accepts host arrays/tensors (any DLPack producer)
        or CUDA-resident tensors, which are handed over device-to-device without touching
        the host; stream is the producer's cudaStream_t handle (0 = default stream)."""
        device = getattr(cameras, "__dlpack_device__", None)
        if device is not None and device()[0] == 2:  # kDLCUDA
            if not hasattr(self._renderer, "set_cameras_device"):
                raise RuntimeError("hyperdrone: device-resident camera input requires the OptiX backend")
            self._renderer.set_cameras_device(cameras, stream)
            return
        self._renderer.set_cameras(_as_cameras(cameras, self.num_cameras))

    def set_motion_blur_cameras(self, cameras_open, cameras_close):
        self._renderer.set_motion_blur_cameras(
            _as_cameras(cameras_open, self.num_cameras),
            _as_cameras(cameras_close, self.num_cameras),
        )

    def generate_cameras(self, center=None, radius=None, up=(0.0, 0.0, 1.0), fov=math.radians(60.0)):
        bounds = self.scene_bounds
        if center is None:
            center = bounds["center"]
        if radius is None:
            radius = bounds["camera_radius"]
        self._renderer.generate_cameras(tuple(center), float(radius), tuple(up), float(fov))

    def generate_probe_directions(self):
        self._renderer.generate_probe_directions()

    def render(self, target="all"):
        self._renderer.render(RENDER_TARGET[target], RENDER_PHASE["full"])
        return self

    def render_launch(self, target="all"):
        self._renderer.render(RENDER_TARGET[target], RENDER_PHASE["launch"])
        return self

    def render_sync(self, target="all"):
        self._renderer.render(RENDER_TARGET[target], RENDER_PHASE["sync"])
        return self

    def synchronize(self):
        self._renderer.synchronize()

    def update(self):
        """Publish pending overlay changes (attach/detach/spawn/despawn/set_transform)."""
        self._renderer.update()

    def frame_raw(self, copy=True):
        """Packed RGBA8 frame buffer as uint32, shape (num_cameras, height, width).

        copy=False returns a numpy view of the renderer's staging buffer instead of a
        snapshot: no allocation, but the contents change on the next read/render."""
        if not copy:
            return self._renderer.frame_view(True)
        out = np.empty((self.num_cameras, self.height, self.width), dtype=np.uint32)
        self._renderer.read_frame_buffer(out)
        return out

    def frame(self, copy=True):
        """RGBA8 frame buffer, shape (num_cameras, height, width, 4)."""
        return self.frame_raw(copy=copy).view(np.uint8).reshape(self.num_cameras, self.height, self.width, 4)

    def depth(self, copy=True):
        """Hit distances, shape (num_cameras, height, width); misses hold the scene max depth."""
        if not copy:
            return self._renderer.depth_view(True)
        out = np.empty((self.num_cameras, self.height, self.width), dtype=np.float32)
        self._renderer.read_depth_buffer(out)
        return out

    def segmentation(self, copy=True):
        """Per-pixel instance ids (or segmentation classes when semantic_segmentation=True),
        shape (num_cameras, height, width); misses hold 0xFFFFFFFF."""
        if not copy:
            return self._renderer.segmentation_view(True)
        out = np.empty((self.num_cameras, self.height, self.width), dtype=np.uint32)
        self._renderer.read_segmentation_buffer(out)
        return out

    def frame_dlpack(self):
        """DLPack producer over the live packed-RGBA8 buffer where rendering writes it:
        CUDA device memory on the OptiX backend, CPU-visible memory elsewhere. Zero-copy —
        consume with torch.from_dlpack / jax.numpy.from_dlpack / np.from_dlpack (CPU only).
        Valid after render()/render_sync(); overwritten by the next render."""
        return _DLPackBuffer(self, self._renderer.frame_dlpack, self._renderer.buffer_device_type())

    def depth_dlpack(self):
        """DLPack producer over the live depth buffer; see frame_dlpack for semantics."""
        return _DLPackBuffer(self, self._renderer.depth_dlpack, self._renderer.buffer_device_type())

    def collisions(self):
        """Probe results as (distances, hits), each shaped (num_cameras, num_probes)."""
        distances = np.empty((self.num_cameras, self.num_probes), dtype=np.float32)
        hits = np.empty((self.num_cameras, self.num_probes), dtype=np.int32)
        self._renderer.read_collision_results(distances, hits)
        return distances, hits

    def framebuffer_device_ptr(self):
        return self._renderer.framebuffer_device_ptr()

    def depthbuffer_device_ptr(self):
        return self._renderer.depthbuffer_device_ptr()

    def save_image(self, path):
        self._renderer.save(SAVE_TARGET["image"], str(path))

    def save_depth_image(self, path):
        self._renderer.save(SAVE_TARGET["depth_image"], str(path))

    def save_depth_raw(self, path):
        self._renderer.save(SAVE_TARGET["depth_raw"], str(path))

    def save_segmentation_image(self, path):
        self._renderer.save(SAVE_TARGET["segmentation_image"], str(path))

    def save_probes(self, path):
        self._renderer.save(SAVE_TARGET["probes"], str(path))

    def can_attach(self, camera, overlay):
        return self._renderer.can_attach(camera, overlay)

    def attach(self, camera, overlay):
        self._renderer.attach(camera, overlay)

    def detach(self, camera, overlay):
        self._renderer.detach(camera, overlay)

    def can_spawn(self, overlay, asset):
        return self._renderer.can_spawn(overlay, asset)

    def spawn(self, overlay, asset, transform):
        return self._renderer.spawn(overlay, asset, _as_transform(transform))

    def despawn(self, overlay, placement):
        self._renderer.despawn(overlay, tuple(placement))

    def set_transform(self, overlay, placement, transform):
        self._renderer.set_transform(overlay, tuple(placement), _as_transform(transform))

    def set_part_transform(self, overlay, placement, part, transform):
        self._renderer.set_part_transform(overlay, tuple(placement), part, _as_transform(transform))

    def set_transform_pair(self, overlay, placement, open_transform, close_transform):
        self._renderer.set_transform_pair(overlay, tuple(placement), _as_transform(open_transform), _as_transform(close_transform))

    def set_part_transform_pair(self, overlay, placement, part, open_transform, close_transform):
        self._renderer.set_part_transform_pair(overlay, tuple(placement), part, _as_transform(open_transform), _as_transform(close_transform))
