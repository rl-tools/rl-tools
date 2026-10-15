from dataclasses import dataclass
from itertools import count

import numpy as np

from .._dlpack import DLPackBuffer as _DLPackBuffer
from .._dlpack import is_cuda as _is_cuda
from ._component import load_core, load_renderer
from ._rig import _output
from ._config import RendererConfig


_RENDERER_SERIALS = count()


@dataclass(frozen=True, kw_only=True)
class OverlayPlacement:
    overlay: int
    first_slot: int
    num_parts: int
    _generation: int
    _renderer: int

    def __reduce_ex__(self, protocol):
        raise TypeError("hyperdrone: OverlayPlacement handles belong to one renderer and cannot be pickled")


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


class Renderer:
    """RLtools raytracer with a JIT-compiled library cached per configuration.

    All constructor parameters are compile-time settings, fixed for this instance.

    Args:
        width, height: Per-camera resolution in pixels; both must be >= 1.
        num_cameras: Fixed camera batch size (>= 1); camera uploads must match it.
        num_probes: Collision rays per camera, run separately through probe().
            0 compiles the probe pass out; probe methods then raise.
        output: A channel name or tuple of "rgb", "depth", "instance_segmentation",
            "semantic_segmentation", "normals", and "flow". The two segmentations label each
            pixel with its instance ID or its object's segmentation_class and may be combined.
            Depth is camera-forward metric distance. render() produces every selected output.
        fidelity: RGB shading preset: "low" (flat, untextured colour), "medium"
            (textures, normal-based shading, metallic reflections, checker background),
            "high" (textured, physically based scene lighting), or "veryhigh"
            ("high" plus point/spot-light shadows). Names are case-insensitive,
            as are output names.
        motion_blur_samples: 1 disables blur; 2, 4, 8, 16, or 32 samples
            interpolate the shutter endpoints supplied by set_motion_blur_cameras().
        anti_aliasing_grid: 1 disables anti-aliasing; 2, 3, or 4 uses a
            grid of 4, 9, or 16 samples per pixel, respectively, per shutter sample.
        num_overlays: Number of overlay groups; a group can be shared by cameras.
        max_overlay_instances: Instance slots per overlay. Multipart assets consume
            one slot per part.
        max_overlays_per_camera: Maximum overlay groups attached to one camera.
        dynamic_motion_blur: Blur overlay motion as well as camera motion, using
            shutter-open/close transforms supplied by set_transform_pair() or per-part
            samples supplied by set_motion_transforms(). Requires overlays and
            motion_blur_samples >= 2.

    The three overlay capacities must be all zero (disabled) or all positive.
    Blur and anti-aliasing affect RGB; depth, segmentation, normals and flow
    use pixel centres at shutter close. Flow compares against the previous capture,
    independently of the motion-blur shutter interval.
    """

    def __init__(self, width, height, *, num_cameras=1, num_probes=1, output="rgb", fidelity="high",
                 motion_blur_samples=1, anti_aliasing_grid=1,
                 num_overlays=0, max_overlay_instances=0, max_overlays_per_camera=0,
                 dynamic_motion_blur=False):
        if isinstance(fidelity, str):
            fidelity = fidelity.lower()
        self.config = RendererConfig(
            width=width,
            height=height,
            num_cameras=num_cameras,
            num_probes=num_probes,
            fidelity=fidelity,
            output=output,
            motion_blur_samples=motion_blur_samples,
            anti_aliasing_grid=anti_aliasing_grid,
            num_overlays=num_overlays,
            max_overlay_instances=max_overlay_instances,
            max_overlays_per_camera=max_overlays_per_camera,
            dynamic_motion_blur=bool(dynamic_motion_blur),
        )
        self._renderer = load_renderer(self.config).Renderer()
        self._scene = None
        self._asset_pool = None
        self._pending_inputs = []
        self._pending_cameras = []
        self._live = {}
        self._generation = count()
        self._serial = next(_RENDERER_SERIALS)

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

    def init(self, scene, *, asset_pool=None):
        """Build acceleration structures for a scene. The scene (and asset pool) must stay
        alive as long as this renderer; references are held here to guarantee that."""
        self._renderer.init(scene, asset_pool)
        self._scene = scene
        self._asset_pool = asset_pool
        self._live.clear()

    def camera(self, position, look_at, *, up=(0.0, 0.0, 1.0), fov=60.0):
        """Camera ray-gen base for set_cameras; fov is in degrees."""
        core = load_core()
        return core.make_camera(tuple(position), tuple(look_at), tuple(up), float(fov), self.aspect)

    def set_cameras(self, cameras, *, stream=0):
        """Upload camera ray-gen bases. Accepts host arrays/tensors (any DLPack producer)
        or CUDA-resident tensors; stream is the producer's cudaStream_t handle (0 = default).
        Each float32 (N, 12) row concatenates position, dir_00, dir_du and dir_dv.
        Pixel centre (x, y) casts normalize(dir_00 + u * dir_du + v * dir_dv),
        with u=(x+0.5)/width and v=(y+0.5)/height, in world-FLU coordinates.
        Inputs stay retained until the
        render or probe operation consuming their copy has synchronized."""
        if _is_cuda(cameras):
            if not hasattr(self._renderer, "set_cameras_device"):
                raise RuntimeError("hyperdrone: device-resident camera input requires the OptiX backend")
            self._renderer.set_cameras_device(cameras, int(stream))
            self._pending_cameras.append(cameras)
            return
        self._renderer.set_cameras(_as_cameras(cameras, self.num_cameras))

    def set_motion_blur_cameras(self, cameras_open, cameras_close, *, stream=0):
        if _is_cuda(cameras_open) != _is_cuda(cameras_close):
            raise ValueError("shutter cameras must live in the same memory")
        if _is_cuda(cameras_open):
            self._require_device_inputs("set_motion_blur_cameras_device")
            self._renderer.set_motion_blur_cameras_device(cameras_open, cameras_close, int(stream))
            self._pending_cameras.extend((cameras_open, cameras_close))
            return
        self._renderer.set_motion_blur_cameras(
            _as_cameras(cameras_open, self.num_cameras),
            _as_cameras(cameras_close, self.num_cameras),
        )

    def generate_cameras(self, *, center=None, radius=None, up=(0.0, 0.0, 1.0), fov=60.0):
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

    def advance_flow(self):
        """Snapshot current camera/object poses before changing them for the next frame.

        The first render seeds previous=current automatically. Subsequent renders keep
        the snapshot until this verb is called; reads never advance it.
        """
        self._renderer.advance_flow()

    def render(self):
        self._renderer.render()
        self._pending_inputs.clear()
        self._pending_cameras.clear()

    def render_launch(self):
        self._renderer.render_launch()

    def render_sync(self):
        self._renderer.render_sync()
        self._pending_inputs.clear()
        self._pending_cameras.clear()

    def probe(self):
        self._require_probes()
        self._renderer.probe()
        self._pending_cameras.clear()

    def probe_launch(self):
        self._require_probes()
        self._renderer.probe_launch()

    def probe_sync(self):
        self._require_probes()
        self._renderer.probe_sync()
        self._pending_cameras.clear()

    def synchronize(self):
        self._renderer.synchronize()
        self._pending_inputs.clear()
        self._pending_cameras.clear()

    def update(self):
        """Publish pending overlay changes (attach/detach/spawn/despawn and the transform
        verbs) and rebuild the overlay acceleration structures."""
        self._renderer.update()

    def frame_raw(self, *, out=None):
        """Packed RGBA8 uint32 snapshot with shape (num_cameras, height, width)."""
        return self._image_output("frame", np.uint32, 1, out)

    def frame(self, *, out=None):
        """sRGB RGBA8 snapshot with shape (num_cameras, height, width, 4); alpha is 255, reserved."""
        shape = (self.num_cameras, self.height, self.width, 4)
        out = _output(out, shape, np.uint8)
        packed = out.view(np.uint32).reshape(shape[:3])
        self.frame_raw(out=packed)
        return out

    def depth(self, *, out=None):
        """Camera-forward metric depth at pixel centre and shutter close; misses are +inf."""
        return self._image_output("depth", np.float32, 1, out)

    def instance_segmentation(self, *, out=None):
        """uint32 instance IDs at shutter close; misses are 0xFFFFFFFF."""
        return self._image_output("instance_segmentation", np.uint32, 1, out)

    def semantic_segmentation(self, *, out=None):
        """uint32 semantic classes at shutter close; misses are 0xFFFFFFFF."""
        return self._image_output("semantic_segmentation", np.uint32, 1, out)

    def normals(self, *, out=None):
        """World-FLU geometric normals facing against the ray, float32 (N, H, W, 3); misses are zero."""
        return self._image_output("normals", np.float32, 3, out)

    def flow(self, *, out=None):
        """Current-to-previous pixel displacement (dx, dy), float32 (N, H, W, 2); see flow_status()."""
        return self._image_output("flow", np.float32, 2, out)

    def flow_status(self, *, out=None):
        """FlowStatus codes as uint32 (N, H, W)."""
        return self._image_output("flow_status", np.uint32, 1, out)

    def _image_output(self, name, dtype, channels, out):
        shape = (self.num_cameras, self.height, self.width)
        if channels > 1:
            shape += (channels,)
        out = _output(out, shape, dtype)
        getattr(self._renderer, "read_" + name + "_buffer")(out)
        return out

    def frame_dlpack(self):
        """Live RGBA8 buffer as a zero-copy uint8 DLPack producer with shape
        (num_cameras, height, width, 4). CUDA device memory on OptiX, CPU-visible memory
        elsewhere. Valid after render()/render_sync(); overwritten by the next render."""
        return _DLPackBuffer(self, self._renderer.frame_dlpack, self._renderer.buffer_device_type(), self._renderer.buffer_device_id())

    def frame_raw_dlpack(self):
        """The same live framebuffer in its native packed representation: a zero-copy
        uint32 DLPack producer with shape (num_cameras, height, width)."""
        return _DLPackBuffer(self, self._renderer.frame_raw_dlpack, self._renderer.buffer_device_type(), self._renderer.buffer_device_id())

    def depth_dlpack(self):
        """DLPack producer over the live depth buffer; see frame_dlpack for semantics."""
        return _DLPackBuffer(self, self._renderer.depth_dlpack, self._renderer.buffer_device_type(), self._renderer.buffer_device_id())

    def instance_segmentation_dlpack(self):
        """DLPack producer over the live instance segmentation buffer; see frame_dlpack."""
        return _DLPackBuffer(self, self._renderer.instance_segmentation_dlpack, self._renderer.buffer_device_type(), self._renderer.buffer_device_id())

    def semantic_segmentation_dlpack(self):
        """DLPack producer over the live semantic segmentation buffer; see frame_dlpack."""
        return _DLPackBuffer(self, self._renderer.semantic_segmentation_dlpack, self._renderer.buffer_device_type(), self._renderer.buffer_device_id())

    def normals_dlpack(self):
        """DLPack producer over the live normals buffer; see frame_dlpack."""
        return _DLPackBuffer(self, self._renderer.normals_dlpack, self._renderer.buffer_device_type(), self._renderer.buffer_device_id())

    def flow_dlpack(self):
        """DLPack producer over the live flow buffer; see frame_dlpack."""
        return _DLPackBuffer(self, self._renderer.flow_dlpack, self._renderer.buffer_device_type(), self._renderer.buffer_device_id())

    def flow_status_dlpack(self):
        """DLPack producer over the live flow status buffer; see frame_dlpack."""
        return _DLPackBuffer(self, self._renderer.flow_status_dlpack, self._renderer.buffer_device_type(), self._renderer.buffer_device_id())

    def collisions(self):
        """Probe results as (distances, hits), each shaped (num_cameras, num_probes)."""
        self._require_probes()
        distances = np.empty((self.num_cameras, self.num_probes), dtype=np.float32)
        hits = np.empty((self.num_cameras, self.num_probes), dtype=np.int32)
        self._renderer.read_collision_results(distances, hits)
        return distances, hits.astype(bool)

    def can_attach(self, camera, overlay):
        return self._renderer.can_attach(camera, overlay)

    def attach(self, camera, overlay):
        self._renderer.attach(camera, overlay)

    def detach(self, camera, overlay):
        self._renderer.detach(camera, overlay)

    def can_spawn(self, overlay, asset):
        return self._renderer.can_spawn(overlay, asset)

    def spawn(self, overlay, asset, transform, *, segmentation_class=None):
        """Place an asset; segmentation_class labels every part instead of the asset's authored classes."""
        native = self._renderer.spawn(overlay, asset, _as_transform(transform), segmentation_class)
        placement = OverlayPlacement(overlay=overlay, first_slot=native[0], num_parts=native[1], _generation=next(self._generation), _renderer=self._serial)
        self._live[(overlay, placement.first_slot)] = (placement, native)
        return placement

    def _native(self, placement):
        if not isinstance(placement, OverlayPlacement):
            raise TypeError("hyperdrone: expected an OverlayPlacement returned by spawn")
        if placement._renderer != self._serial:
            raise ValueError("hyperdrone: placement belongs to another renderer")
        entry = self._live.get((placement.overlay, placement.first_slot))
        if entry is None or entry[0] != placement:
            raise ValueError("hyperdrone: placement is stale or does not match its allocation")
        return entry[1]

    def despawn(self, placement):
        native = self._native(placement)
        self._renderer.despawn(placement.overlay, native)
        del self._live[(placement.overlay, placement.first_slot)]

    def set_transform(self, placement, transform, *, rig=None, coordinates=None):
        """Move a placement rigidly: every part at transform composed with its asset rest frame.
        With a rig, transform is its base pose and coordinates its joint state (default: the
        neutral state), and the parts land at the rig's geometry-to-world poses."""
        native = self._native(placement)
        if rig is None:
            if coordinates is not None:
                raise ValueError("coordinates articulate a rig: pass rig=")
            self._renderer.set_transform(placement.overlay, native, _as_transform(transform))
            return
        self.set_transforms(placement, rig.evaluate(transform, coordinates))

    def set_transform_pair(self, placement, open_transform, close_transform, *, rig=None, coordinates_open=None, coordinates_close=None):
        """Rigid shutter-open/close move, interpolated into the motion samples (dynamic_motion_blur=True).
        With a rig, the transforms are its base poses and the coordinates its joint states at
        the shutter ends (default: the neutral state): the base follows the constant-twist screw,
        the coordinates interpolate before the kinematics, so a revolute joint keeps every
        revolution in between."""
        native = self._native(placement)
        if rig is None:
            if coordinates_open is not None or coordinates_close is not None:
                raise ValueError("coordinates articulate a rig: pass rig=")
            self._renderer.set_transform_pair(placement.overlay, native, _as_transform(open_transform), _as_transform(close_transform))
            return
        if coordinates_open is None:
            coordinates_open = rig.default_coordinates
        if coordinates_close is None:
            coordinates_close = rig.default_coordinates
        samples, close = rig.sample(open_transform, close_transform, coordinates_open, coordinates_close, self.motion_blur_samples)
        self.set_motion_transforms(placement, samples, close)

    def _require_device_inputs(self, name):
        if not hasattr(self._renderer, name):
            raise RuntimeError("hyperdrone: device-resident transform input requires the OptiX backend")

    def set_transforms(self, placement, poses, *, stream=0):
        """Per-part geometry-to-world poses, shape (num_parts, 3, 4): the output of Rig.evaluate
        for the placement's assembly. CUDA tensors (OptiX) are copied device-to-device on the
        render stream after awaiting stream (the producer's cudaStream_t handle, 0 = default)."""
        native = self._native(placement)
        if _is_cuda(poses):
            self._require_device_inputs("set_transforms_device")
            self._renderer.set_transforms_device(placement.overlay, native, poses, int(stream))
            self._pending_inputs.append(poses)
            return
        self._renderer.set_transforms(placement.overlay, native, _as_rows(poses, (placement.num_parts, 3, 4)))

    def set_motion_transforms(self, placement, samples, close, *, stream=0):
        """Per-part motion for dynamic motion blur: samples (motion_blur_samples, num_parts, 3, 4)
        at the shutter times (s + 0.5) / motion_blur_samples and the shutter-close poses
        (num_parts, 3, 4) that segmentation, probes and the next frame see: the output of
        Rig.sample. CUDA tensors are handed over like in set_transforms."""
        native = self._native(placement)
        if _is_cuda(samples) != _is_cuda(close):
            raise ValueError("samples and close poses must live in the same memory")
        if _is_cuda(samples):
            self._require_device_inputs("set_motion_transforms_device")
            self._renderer.set_motion_transforms_device(placement.overlay, native, samples, close, int(stream))
            self._pending_inputs.extend((samples, close))
            return
        num_parts = placement.num_parts
        self._renderer.set_motion_transforms(placement.overlay, native, _as_rows(samples, (self.motion_blur_samples, num_parts, 3, 4)), _as_rows(close, (num_parts, 3, 4)))

    def set_all_transforms(self, poses, *, stream=0):
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

    def set_all_motion_transforms(self, samples, close, *, stream=0):
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
        return _DLPackBuffer(self, self._renderer.transforms_dlpack, self._renderer.buffer_device_type(), self._renderer.buffer_device_id())

    def transforms_motion_dlpack(self):
        """Live per-sample tensor (motion_blur_samples, num_overlays, max_overlay_instances, 3, 4);
        see transforms_dlpack for the contract."""
        return _DLPackBuffer(self, self._renderer.transforms_motion_dlpack, self._renderer.buffer_device_type(), self._renderer.buffer_device_id())

    @property
    def render_stream(self):
        """cudaStream_t handle of the render stream (0 on host backends)."""
        return self._renderer.render_stream()
