from dataclasses import KW_ONLY, dataclass
from itertools import count

import numpy as np

from .._dlpack import is_cuda as _is_cuda
from ._component import load_core, load_renderer
from ._config import RendererConfig
from ..camera import CameraSpec, RGB
from .._output_buffers import OutputCollection


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


class PlacementBatch:
    """Distinct placements of one renderer, checked once by Renderer.placement_batch, for set_pose
    to move together; it goes stale when one of them is despawned."""

    __slots__ = ("placements", "_native", "_renderer", "_despawns")

    def __init__(self, placements, native, renderer, despawns):
        self.placements = placements
        self._native = native
        self._renderer = renderer
        self._despawns = despawns

    def __len__(self):
        return len(self.placements)

    def __reduce_ex__(self, protocol):
        raise TypeError("hyperdrone: a PlacementBatch belongs to one renderer and cannot be pickled")


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


@dataclass(frozen=True, eq=False)
class Pose:
    """The rigid state of a placement: base, its (3, 4) pose (default identity), and scale, a
    uniform size factor applied after the base (motion blur interpolates the rigid base). The
    asset's joint coordinates travel beside it as a set_pose argument."""
    base: object = None
    _: KW_ONLY
    scale: float = 1.0

    def _native(self):
        base = np.eye(4, dtype=np.float32)[:3] if self.base is None else _as_transform(self.base)
        return base, float(self.scale)


def _as_coordinates(coordinates):
    return None if coordinates is None else np.ascontiguousarray(_from_dlpack(coordinates), dtype=np.float32).reshape(-1)


def _batch_state(base, scale, coordinates, count, device):
    if device:
        if base is None:
            raise ValueError("hyperdrone: a device batch needs CUDA bases")
        bases = base
    else:
        bases = np.broadcast_to(np.eye(4, dtype=np.float32)[:3], (count, 3, 4)) if base is None else _from_dlpack(base)
        bases = np.ascontiguousarray(bases, dtype=np.float32)
    if np.isscalar(scale):
        scales, scale = None, float(scale)
    else:
        scales, scale = (scale if device else np.ascontiguousarray(_from_dlpack(scale), dtype=np.float32)), 1.0
    if coordinates is not None and not device:
        coordinates = np.ascontiguousarray(_from_dlpack(coordinates), dtype=np.float32)
    return bases, scales, scale, coordinates


class Renderer:
    """Compile a named camera rig; repeat it with batch_size and acquire declared outputs explicitly."""

    def __init__(self, width, height, *, cameras=(CameraSpec("camera", outputs=(RGB(format="rgba8"),)),), batch_size=1, num_probes=1, fidelity="high", motion_blur_samples=1, anti_aliasing_grid=1, num_overlays=0, max_overlay_instances=0, max_overlays_per_camera=0, dynamic_motion_blur=False):
        if isinstance(fidelity, str):
            fidelity = fidelity.lower()
        self.config = RendererConfig(
            width=width,
            height=height,
            cameras=cameras,
            batch_size=batch_size,
            num_probes=num_probes,
            fidelity=fidelity,
            motion_blur_samples=motion_blur_samples,
            anti_aliasing_grid=anti_aliasing_grid,
            num_overlays=num_overlays,
            max_overlay_instances=max_overlay_instances,
            max_overlays_per_camera=max_overlays_per_camera,
            dynamic_motion_blur=bool(dynamic_motion_blur),
        )
        self._renderer = load_renderer(self.config).Renderer()
        self.outputs = OutputCollection(self._renderer, cameras=self.config.cameras)
        self._scene = None
        self._asset_pool = None
        self._live = {}
        self._generation = count()
        self._despawns = 0
        self._serial = next(_RENDERER_SERIALS)

    def __del__(self):
        native = getattr(self, "_renderer", None)
        if native is not None:
            # Imported leases can retain the native renderer through its own output buffers.
            native.synchronize()
            native.release_inputs()

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
        CUDA inputs stay referenced until the render stream has copied them."""
        if _is_cuda(cameras):
            if not hasattr(self._renderer, "set_cameras_device"):
                raise RuntimeError("hyperdrone: device-resident camera input requires the OptiX backend")
            self._renderer.set_cameras_device(cameras, int(stream))
            return
        self._renderer.set_cameras(_as_cameras(cameras, self.num_cameras))

    def set_motion_blur_cameras(self, cameras_open, cameras_close, *, stream=0):
        if _is_cuda(cameras_open) != _is_cuda(cameras_close):
            raise ValueError("shutter cameras must live in the same memory")
        if _is_cuda(cameras_open):
            self._require_device_inputs("set_motion_blur_cameras_device")
            self._renderer.set_motion_blur_cameras_device(cameras_open, cameras_close, int(stream))
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
            raise RuntimeError("hyperdrone: this renderer was compiled without a Probes output")

    def generate_probe_directions(self):
        self._require_probes()
        self._renderer.generate_probe_directions()

    def advance_flow(self):
        """Snapshot current camera/object poses before changing them for the next frame.

        The first render seeds previous=current automatically. Subsequent renders keep
        the snapshot until this verb is called; reads never advance it.
        """
        self._renderer.advance_flow()

    def advance_flow_launch(self):
        """advance_flow without waiting for it: queued on the render stream (OptiX)."""
        self._renderer.advance_flow_launch()

    def render(self):
        self._renderer.render()
        self._renderer.release_inputs()

    def render_launch(self):
        self._renderer.render_launch()

    def render_sync(self):
        self._renderer.render_sync()
        self._renderer.release_inputs()

    def probe(self):
        self._require_probes()
        self._renderer.probe()
        self._renderer.release_inputs()

    def probe_launch(self):
        self._require_probes()
        self._renderer.probe_launch()

    def probe_sync(self):
        self._require_probes()
        self._renderer.probe_sync()
        self._renderer.release_inputs()

    def synchronize(self):
        self._renderer.synchronize()
        self._renderer.release_inputs()

    def update(self):
        """Publish pending overlay changes (attach/detach/spawn/despawn and the transform
        verbs) and rebuild the overlay acceleration structures."""
        self._renderer.update()

    def update_launch(self):
        """update without waiting for it: queued on the render stream (OptiX)."""
        self._renderer.update_launch()

    def can_attach(self, camera, overlay):
        return self._renderer.can_attach(camera, overlay)

    def attach(self, camera, overlay):
        self._renderer.attach(camera, overlay)

    def detach(self, camera, overlay):
        self._renderer.detach(camera, overlay)

    def can_spawn(self, overlay, asset):
        return self._renderer.can_spawn(overlay, asset)

    def spawn(self, overlay, asset, pose=None, *, coordinates=None, segmentation_class=None):
        """Place an asset at pose (default: the neutral Pose) with its joints at coordinates
        (default the rest state), in both flow instants, so it has no motion of its own until
        set_pose moves it; segmentation_class labels every part instead of the asset's authored
        classes."""
        native = self._renderer.spawn(overlay, asset, *(pose or Pose())._native(), _as_coordinates(coordinates), segmentation_class)
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
        self._despawns += 1

    def placement_batch(self, placements):
        """Check distinct live placements once, for set_pose to move them together."""
        placements = tuple(placements)
        table = [(placement.overlay, *self._native(placement)) for placement in placements]
        return PlacementBatch(placements, self._renderer.placement_batch(table), self._serial, self._despawns)

    def set_pose(self, placement, pose, close=None, *, coordinates=None, coordinates_close=None, stream=0):
        """Move a placement: its parts land where the pose and coordinates (the asset's joint
        values in coordinate order, see ObjectAssembly.set_joint; default the rest state) drive
        the asset's joints. With close, pose and coordinates are the shutter-open state and close
        and coordinates_close the shutter-close state (dynamic_motion_blur=True): the base follows
        the constant-twist screw between them, scale and coordinates interpolate (a revolute joint
        keeps every revolution in between); the close state is what segmentation, probes and the
        next frame see, and all a renderer without dynamic motion blur uses.

        A PlacementBatch of B placements moves them all: pose.base is (B, 3, 4) or (B, 12),
        pose.scale a float or (B,), coordinates (B, C) with C at least every member's
        num_coordinates, and likewise for close. On OptiX, CUDA tensors run one kernel on the
        render stream after awaiting stream, the producer's cudaStream_t handle, and stay
        referenced until the kernel has read them."""
        if close is None and coordinates_close is not None:
            raise ValueError("hyperdrone: coordinates_close belongs to the shutter-close state, pass close")
        inputs = (pose.base, pose.scale, coordinates) + ((close.base, close.scale, coordinates_close) if close is not None else ())
        if isinstance(placement, PlacementBatch):
            self._set_pose_batch(placement, pose, close, coordinates, coordinates_close, inputs, stream)
            return
        native = self._native(placement)
        if any(_is_cuda(value) for value in inputs):
            raise ValueError("hyperdrone: a single placement takes host arrays; move CUDA poses through a placement_batch")
        if close is None:
            self._renderer.set_pose(placement.overlay, native, *pose._native(), _as_coordinates(coordinates))
            return
        self._renderer.set_pose_pair(placement.overlay, native, *pose._native(), *close._native(), _as_coordinates(coordinates), _as_coordinates(coordinates_close))

    def _set_pose_batch(self, batch, pose, close, coordinates, coordinates_close, inputs, stream):
        if batch._renderer != self._serial:
            raise ValueError("hyperdrone: the placement batch belongs to another renderer")
        if batch._despawns != self._despawns:
            for placement in batch.placements:
                entry = self._live.get((placement.overlay, placement.first_slot))
                if entry is None or entry[0] != placement:
                    raise ValueError("hyperdrone: a member of this placement batch was despawned; build a new batch")
            batch._despawns = self._despawns
        count = len(batch)
        device = any(_is_cuda(value) for value in inputs)
        if device:
            self._require_device_inputs("set_pose_batch_device")
            for value, what in zip(inputs, ("base", "scale", "coordinates", "close.base", "close.scale", "coordinates_close")):
                if value is not None and not np.isscalar(value) and not _is_cuda(value):
                    raise ValueError(f"hyperdrone: {what} must be a CUDA tensor like the other inputs of this device batch")
        opened = _batch_state(pose.base, pose.scale, coordinates, count, device)
        closed = _batch_state(close.base, close.scale, coordinates_close, count, device) if close is not None else (None, None, 1.0, None)
        if device:
            self._renderer.set_pose_batch_device(batch._native, *opened, *closed, int(stream))
            return
        self._renderer.set_pose_batch(batch._native, *opened, *closed)

    def _require_device_inputs(self, name):
        if not hasattr(self._renderer, name):
            raise RuntimeError("hyperdrone: device-resident input requires the OptiX backend")

    @property
    def render_stream(self):
        """cudaStream_t handle of the render stream (0 on host backends)."""
        return self._renderer.render_stream()
