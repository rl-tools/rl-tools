import numpy as np

from ._component import load_core, load_renderer
from ._config import (
    OUTPUT_MODE,
    FIDELITY,
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


def _as_rows(value, shape):
    array = np.ascontiguousarray(_from_dlpack(value), dtype=np.float32)
    if array.size != int(np.prod(shape)):
        raise ValueError(f"transforms must have shape {shape}, got {array.shape}")
    return array.reshape(shape)


def _is_cuda(value):
    device = getattr(value, "__dlpack_device__", None)
    return device is not None and device()[0] == 2  # kDLCUDA


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
    """RLtools raytracer with a JIT-compiled library cached per configuration.

    All constructor parameters are compile-time settings, fixed for this instance.

    Args:
        width, height: Per-camera resolution in pixels; both must be >= 1.
        num_cameras: Fixed camera batch size (>= 1); camera uploads must match it.
        num_probes: Collision rays per camera, run separately through probe().
            0 compiles the probe pass out; probe methods then raise.
        output: "rgb", "rgbd" (RGB + depth), "depth", "segmentation", or
            "rgbd_segmentation" (RGB + depth + segmentation). Depth is ray hit
            distance, not camera-space Z. render() produces every selected output.
        fidelity: RGB shading preset: "low" (flat, untextured colour), "medium"
            (textures, normal-based shading, metallic reflections, checker background),
            "high" (textured, physically based scene lighting), or "veryhigh"
            ("high" plus point/spot-light shadows). Names are case-insensitive,
            as are output names.
        motion_blur_samples: 0 or 1 disables blur; 2, 4, 8, 16, or 32 samples
            interpolate the shutter endpoints supplied by set_motion_blur_cameras().
        anti_aliasing_grid: 0 or 1 disables anti-aliasing; 2, 3, or 4 uses a
            grid of 4, 9, or 16 samples per pixel, respectively, per shutter sample.
        num_overlays: Number of overlay groups; a group can be shared by cameras.
        max_overlay_instances: Instance slots per overlay. Multipart assets consume
            one slot per part.
        max_overlays_per_camera: Maximum overlay groups attached to one camera.
        semantic_segmentation: Write per-object segmentation_class values instead
            of instance IDs; requires "segmentation" or "rgbd_segmentation" output.
        dynamic_motion_blur: Blur overlay motion as well as camera motion, using
            shutter-open/close transforms supplied by set_transform_pair() or per-part
            samples supplied by set_motion_transforms(). Requires overlays and
            motion_blur_samples >= 2.

    The three overlay capacities must be all zero (disabled) or all positive.
    Blur and anti-aliasing affect RGB and depth; segmentation always uses a single
    sample from the shutter-close camera.
    """

    def __init__(self, width, height, num_cameras=1, num_probes=1, output="rgb", fidelity="high",
                 motion_blur_samples=0, anti_aliasing_grid=0,
                 num_overlays=0, max_overlay_instances=0, max_overlays_per_camera=0,
                 semantic_segmentation=False, dynamic_motion_blur=False):
        if isinstance(fidelity, str):
            fidelity = FIDELITY[fidelity.lower()]
        if isinstance(output, str):
            output = OUTPUT_MODE[output.lower()]
        self.config = RendererConfig(
            width=int(width),
            height=int(height),
            num_cameras=int(num_cameras),
            num_probes=int(num_probes),
            fidelity=fidelity,
            output_mode=output,
            motion_blur_samples=max(1, int(motion_blur_samples)),
            anti_aliasing_grid=max(1, int(anti_aliasing_grid)),
            num_overlays=int(num_overlays),
            max_overlay_instances=int(max_overlay_instances),
            max_overlays_per_camera=int(max_overlays_per_camera),
            semantic_segmentation=bool(semantic_segmentation),
            dynamic_motion_blur=bool(dynamic_motion_blur),
        )
        self._renderer = load_renderer(self.config).Renderer()
        self._scene = None
        self._asset_pool = None
        self._pending_inputs = []  # CUDA inputs handed over device-to-device stay alive until the render stream synchronizes

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
    def has_probes(self):
        return self.config.has_probes

    @property
    def aspect(self):
        return self.config.width / self.config.height

    @property
    def motion_blur_samples(self):
        return self.config.motion_blur_samples

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
        self._renderer.init(scene, asset_pool)
        self._scene = scene
        self._asset_pool = asset_pool
        return self

    def camera(self, position, look_at, up=(0.0, 0.0, 1.0), fov=60.0):
        """Camera ray-gen base for set_cameras; fov is in degrees."""
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

    def generate_cameras(self, center=None, radius=None, up=(0.0, 0.0, 1.0), fov=60.0):
        bounds = self.scene_bounds
        if center is None:
            center = bounds["center"]
        if radius is None:
            radius = bounds["camera_radius"]
        self._renderer.generate_cameras(tuple(center), float(radius), tuple(up), float(fov))

    def _require_probes(self):
        if not self.config.has_probes:
            raise RuntimeError("hyperdrone: this renderer was compiled without collision probes (num_probes=0)")

    def generate_probe_directions(self):
        self._require_probes()
        self._renderer.generate_probe_directions()

    def render(self):
        self._renderer.render()
        self._pending_inputs.clear()
        return self

    def render_launch(self):
        self._renderer.render_launch()
        return self

    def render_sync(self):
        self._renderer.render_sync()
        self._pending_inputs.clear()
        return self

    def probe(self):
        self._require_probes()
        self._renderer.probe()
        return self

    def probe_launch(self):
        self._require_probes()
        self._renderer.probe_launch()
        return self

    def probe_sync(self):
        self._require_probes()
        self._renderer.probe_sync()
        return self

    def synchronize(self):
        self._renderer.synchronize()
        self._pending_inputs.clear()

    def update(self):
        """Publish pending overlay changes (attach/detach/spawn/despawn and the transform
        verbs) and rebuild the overlay acceleration structures."""
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

    @property
    def segmentation_mode(self):
        return "semantic" if self.config.semantic_segmentation else "instance"

    def segmentation(self, copy=True):
        """Per-pixel instance ids (or segmentation classes when semantic_segmentation=True),
        shape (num_cameras, height, width); misses hold 0xFFFFFFFF."""
        if not copy:
            return self._renderer.segmentation_view(True)
        out = np.empty((self.num_cameras, self.height, self.width), dtype=np.uint32)
        self._renderer.read_segmentation_buffer(out)
        return out

    def frame_dlpack(self):
        """Live RGBA8 buffer as a zero-copy uint8 DLPack producer with shape
        (num_cameras, height, width, 4). CUDA device memory on OptiX, CPU-visible memory
        elsewhere. Valid after render()/render_sync(); overwritten by the next render."""
        return _DLPackBuffer(self, self._renderer.frame_dlpack, self._renderer.buffer_device_type())

    def frame_raw_dlpack(self):
        """The same live framebuffer in its native packed representation: a zero-copy
        uint32 DLPack producer with shape (num_cameras, height, width)."""
        return _DLPackBuffer(self, self._renderer.frame_raw_dlpack, self._renderer.buffer_device_type())

    def depth_dlpack(self):
        """DLPack producer over the live depth buffer; see frame_dlpack for semantics."""
        return _DLPackBuffer(self, self._renderer.depth_dlpack, self._renderer.buffer_device_type())

    def collisions(self):
        """Probe results as (distances, hits), each shaped (num_cameras, num_probes)."""
        self._require_probes()
        distances = np.empty((self.num_cameras, self.num_probes), dtype=np.float32)
        hits = np.empty((self.num_cameras, self.num_probes), dtype=np.int32)
        self._renderer.read_collision_results(distances, hits)
        return distances, hits

    def framebuffer_device_ptr(self):
        return self._renderer.framebuffer_device_ptr()

    def depthbuffer_device_ptr(self):
        return self._renderer.depthbuffer_device_ptr()

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
        """Move a placement rigidly: every part at transform composed with its asset rest frame."""
        self._renderer.set_transform(overlay, tuple(placement), _as_transform(transform))

    def set_transform_pair(self, overlay, placement, open_transform, close_transform):
        """Rigid shutter-open/close move, interpolated into the motion samples (dynamic_motion_blur=True)."""
        self._renderer.set_transform_pair(overlay, tuple(placement), _as_transform(open_transform), _as_transform(close_transform))

    def _require_device_inputs(self, name):
        if not hasattr(self._renderer, name):
            raise RuntimeError("hyperdrone: device-resident transform input requires the OptiX backend")

    def set_transforms(self, overlay, placement, poses, stream=0):
        """Per-part geometry-to-world poses, shape (num_parts, 3, 4): the output of Rig.evaluate
        for the placement's assembly. CUDA tensors (OptiX) are copied device-to-device on the
        render stream after awaiting stream (the producer's cudaStream_t handle, 0 = default)."""
        if _is_cuda(poses):
            self._require_device_inputs("set_transforms_device")
            self._renderer.set_transforms_device(overlay, tuple(placement), poses, int(stream))
            self._pending_inputs.append(poses)
            return
        self._renderer.set_transforms(overlay, tuple(placement), _as_rows(poses, (placement[1], 3, 4)))

    def set_motion_transforms(self, overlay, placement, samples, close, stream=0):
        """Per-part motion for dynamic motion blur: samples (motion_blur_samples, num_parts, 3, 4)
        at the shutter times (s + 0.5) / motion_blur_samples and the shutter-close poses
        (num_parts, 3, 4) that segmentation, probes and the next frame see: the output of
        Rig.sample. CUDA tensors are handed over like in set_transforms."""
        if _is_cuda(samples) != _is_cuda(close):
            raise ValueError("samples and close poses must live in the same memory")
        if _is_cuda(samples):
            self._require_device_inputs("set_motion_transforms_device")
            self._renderer.set_motion_transforms_device(overlay, tuple(placement), samples, close, int(stream))
            self._pending_inputs.extend((samples, close))
            return
        num_parts = placement[1]
        self._renderer.set_motion_transforms(overlay, tuple(placement), _as_rows(samples, (self.motion_blur_samples, num_parts, 3, 4)), _as_rows(close, (num_parts, 3, 4)))

    def set_all_transforms(self, poses, stream=0):
        """Every slot at once, shape (num_overlays, max_overlay_instances, 3, 4): a batch of
        identical assets spawned one per overlay is exactly this tensor. Rows of inactive slots
        are ignored."""
        shape = (self.config.num_overlays, self.config.max_overlay_instances, 3, 4)
        if _is_cuda(poses):
            self._require_device_inputs("set_all_transforms_device")
            self._renderer.set_all_transforms_device(poses, int(stream))
            self._pending_inputs.append(poses)
            return
        self._renderer.set_all_transforms(_as_rows(poses, shape))

    def set_all_motion_transforms(self, samples, close, stream=0):
        """Every slot's motion at once: samples (motion_blur_samples, num_overlays,
        max_overlay_instances, 3, 4) and close (num_overlays, max_overlay_instances, 3, 4)."""
        shape = (self.config.num_overlays, self.config.max_overlay_instances, 3, 4)
        if _is_cuda(samples) != _is_cuda(close):
            raise ValueError("samples and close poses must live in the same memory")
        if _is_cuda(samples):
            self._require_device_inputs("set_all_motion_transforms_device")
            self._renderer.set_all_motion_transforms_device(samples, close, int(stream))
            self._pending_inputs.extend((samples, close))
            return
        self._renderer.set_all_motion_transforms(_as_rows(samples, (self.motion_blur_samples,) + shape), _as_rows(close, shape))

    def transforms_dlpack(self):
        """Live geometry-to-world tensor (num_overlays, max_overlay_instances, 3, 4) as a
        zero-copy DLPack producer for in-place writers (CUDA device memory on OptiX, CPU memory
        elsewhere); update() consumes it. Call update() to publish pending host writes before
        writing it, and order a CUDA producer against render_stream."""
        return _DLPackBuffer(self, self._renderer.transforms_dlpack, self._renderer.buffer_device_type())

    def transforms_motion_dlpack(self):
        """Live per-sample tensor (motion_blur_samples, num_overlays, max_overlay_instances, 3, 4);
        see transforms_dlpack for the contract."""
        return _DLPackBuffer(self, self._renderer.transforms_motion_dlpack, self._renderer.buffer_device_type())

    @property
    def render_stream(self):
        """cudaStream_t handle of the render stream (0 on host backends)."""
        return self._renderer.render_stream()

    def set_motion_blur_cameras_device(self, cameras_open, cameras_close, stream=0):
        self._renderer.set_motion_blur_cameras_device(cameras_open, cameras_close, int(stream))
