# hypert — Hyper Ray Tracer

Python interface to the RLtools raytracing renderer (OptiX / Metal / Vulkan / generic CPU).

Scene assembly (GLB loading, procedural meshes, lights, instances, asset pools) is runtime
data and bound directly. Renderer constants (resolution, camera count, output mode, shading
tier, motion blur, anti-aliasing, overlay capacities) stay compile-time: the first use of a
new combination JIT-compiles a renderer library into a persistent cache; later uses load it
directly. Staleness against the RLtools headers is handled by CMake dependency tracking on
every `Renderer` construction (skip with `HYPERT_SKIP_BUILD=1`).

## Install

```bash
.venv/bin/pip install -e interface/python/hypert
```

Requirements: CMake >= 3.24, a C++17 compiler, assimp (system package), and the backend
prerequisites (CUDA + OptiX driver for OPTIX, Vulkan dev + glslang for VULKAN, nothing for
GENERIC). The build reuses the repository's `.dependencies` FetchContent sources when
present — including the locally patched OWL tree.

## Environment

| Variable | Meaning |
|---|---|
| `HYPERT_BACKEND` | `OPTIX` \| `METAL` \| `VULKAN` \| `GENERIC` \| `AUTO` (default: Metal on macOS, OptiX elsewhere) |
| `HYPERT_CACHE_DIR` | cache root (default `~/.cache/hypert`) |
| `HYPERT_RLTOOLS_ROOT` | repository root override (default: inferred from the package location) |
| `HYPERT_SKIP_BUILD` | skip the CMake staleness check when the artifacts already exist |

Each backend gets its own build directory; switching `HYPERT_BACKEND` never invalidates
another backend's cache.

## Usage

```python
import math, numpy as np, hypert

scene = hypert.load_scene("tests/data/ProcTHOR-Train-1.glb", shading="high")

renderer = hypert.Renderer(width=320, height=240, num_cameras=4, output="rgbd", shading="high")
renderer.init(scene)

camera = renderer.camera(position=(0, 0, 1.5), look_at=(1, 0, 1.5), fov=math.radians(80))
renderer.set_cameras(np.repeat(camera[None], 4, axis=0))
renderer.render()

rgb   = renderer.frame()    # (4, 240, 320, 4) uint8
depth = renderer.depth()    # (4, 240, 320) float32
```

The renderer operates in the L2F FLU frame: +X forward, +Y left, +Z up; GLB assets are
swizzled at load time.

Procedural scenes:

```python
scene = hypert.Scene()
obj = hypert.Object(name="wall")
obj.add_mesh(hypert.Mesh(vertices, indices, color=(1, 0, 0)))     # (V,3) f32, (F,3) i32
scene.add_object(obj, transform=hypert.make_transform(position=(2, 0, 0)))
scene.add_light(hypert.SceneLight.directional(direction=(1, 0, 0), color=(1, 1, 1)))
```

Segmentation: `output="segmentation"` (or `"rgbd_segmentation"`) yields per-pixel instance
ids (miss = `0xFFFFFFFF`); load GLBs with `hypert.load_assembly` + `scene.add_assembly` so
each glTF root node keeps its own instance id and name. With `semantic_segmentation=True`
the ids are `Object.segmentation_class` values instead.

Overlays (per-camera dynamic content on the static world):

```python
pool = hypert.AssetPool()
drone = pool.add_assembly(hypert.load_assembly("drone.glb"))
renderer = hypert.Renderer(..., num_overlays=2, max_overlay_instances=8, max_overlays_per_camera=1)
renderer.init(scene, asset_pool=pool)
placement = renderer.spawn(overlay=0, asset=drone, transform=hypert.make_transform(position=p, orientation_wxyz=q))
renderer.attach(camera=0, overlay=0)
renderer.update()                       # publish, then render()
renderer.set_transform(0, placement, hypert.make_transform(...))  # move it later
```

Async rendering: `renderer.render_launch(...)`, do CPU work, `renderer.render_sync(...)`.

## Zero-copy I/O (DLPack)

Outputs come in three flavors:

```python
renderer.frame()                  # snapshot copy (safe to keep)
renderer.frame(copy=False)        # numpy view of the staging buffer (refreshed in place)
renderer.frame_dlpack()           # DLPack producer over the LIVE buffer where rendering
                                  # writes: CUDA device memory on OptiX, CPU elsewhere
torch.from_dlpack(renderer.frame_dlpack())   # zero-copy GPU tensor on OptiX
np.from_dlpack(renderer.frame_dlpack())      # zero-copy on CPU-visible backends
```

`depth_dlpack()` works the same; segmentation and collisions are host-side
(`segmentation(copy=False)` avoids the snapshot copy). Live/dlpack views are valid after
`render()`/`render_sync()` and are overwritten by the next render — copy what you keep.

Inputs (`set_cameras`, transforms) accept any DLPack producer (torch/JAX CPU tensors,
numpy) — a C-contiguous float32 tensor crosses the boundary without copying. Device-resident
camera input is not wired yet (planned with the drone-dynamics coupling).

## Tests

```bash
HYPERT_BACKEND=GENERIC .venv/bin/python -m pytest interface/python/hypert/tests -v
```
