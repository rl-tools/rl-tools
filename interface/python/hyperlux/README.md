# hyperlux

HyperLux is the batch raytracer behind HyperDrone: N cameras over one static scene plus per-camera dynamic overlays, rendered by OptiX, Metal, Vulkan, WebGPU or a generic CPU backend behind one interface. The engine is the C++ `rl_tools::HyperLux` of RLtools (`include/rl_tools/hyperlux`); this package compiles it per configuration.

| Module | What it is |
|---|---|
| `hyperlux` | `HyperLux`, the renderer compiled per `hyperlux.Config`; the scene model (`Scene`, `Object`, `Mesh`, `ObjectAssembly`, `AssetPool`, `SceneLight`), the GLB loaders, `primitives`, `ClearanceSampler` and `datasets.procthor.Train200`. |
| `hyperlux.camera` | Camera rigs (`CameraSpec`) and the typed output descriptors, shared with packages built on hyperlux such as hyperdrone. |

Native components are JIT-compiled per set of compile-time constants (resolution, camera count, outputs, motion blur, ...) into a persistent cache; later uses load the cached library directly. The scene module is shared, and every renderer configuration is a directly bound Python extension; scene and asset-pool objects can be reused across configurations. Packages built on hyperlux compile their own components through the same JIT (`hyperlux._jit`) and share the cache, the CUDA staging buffers and the output-buffer protocol: hyperdrone's dynamics and environment are the first.

## Notebooks

| Notebook | Open in Colab |
|---|---|
| [Articulation](hyperlux/examples/Articulation.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rl-tools/rl-tools/blob/main/interface/python/hyperlux/hyperlux/examples/Articulation.ipynb) |
| [Benchmark](hyperlux/examples/Benchmark.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rl-tools/rl-tools/blob/main/interface/python/hyperlux/hyperlux/examples/Benchmark.ipynb) |

## Install

From an rl-tools checkout (development):

```bash
.venv/bin/pip install -e interface/python/hyperlux
```

From an sdist (self-contained — bundles the rl-tools headers and the HyperLux sources, no checkout needed):

```bash
pip install hyperlux-<version>.tar.gz
```

Install the dependencies used by the packaged examples with:

```bash
pip install "hyperlux[examples]"
```

Requirements: CMake >= 3.24, a C++17 compiler, and git for dependency downloads. GLB loading uses the Assimp 6.0.4 release archive with a pinned SHA-256, shared by every component built on this JIT. Cached archives are reused offline and verified by CMake before extraction. CUDA + OptiX driver is required for the OptiX backend, Vulkan dev + glslang for the VULKAN backend. The WEBGPU backend fetches a hash-pinned wgpu-native prebuilt at first build (no extra system packages). Nothing beyond the build toolchain for GENERIC.

## Environment

| Variable | Meaning |
|---|---|
| `HYPERLUX_BACKEND` | `OPTIX` \| `METAL` \| `VULKAN` \| `WEBGPU` \| `GENERIC` \| `AUTO` (AUTO preference: OPTIX > METAL > VULKAN > WEBGPU > GENERIC) |
| `HYPERLUX_CACHE_DIR` | root for CMake build trees and downloaded build dependencies (default `~/.cache/hyperlux`) |
| `HYPERLUX_RLTOOLS_ROOT` | rl-tools source root override (default: enclosing checkout, else the vendored tree) |
| `HYPERLUX_PROCTHOR_PATH` | examples only: local ProcTHOR scene override (otherwise downloaded and cached) |
| `HYPERLUX_SKIP_BUILD` | skip the CMake staleness check when the artifacts already exist |
| `HYPERLUX_OFFLINE` | forbid network during builds (requires seeded/vendored dependencies) |
| `HYPERLUX_BUILD_JOBS` | parallel build jobs (default 5) |

Each component, variant, Python interpreter path and ABI, and native CPU instruction set gets its own build tree (`renderer-optix-*`, and `environment-*`, `dynamics-*` for hyperdrone); switching never invalidates another's cache. All configure/build steps run under a per-tree file lock, so many worker processes can share one cache safely. CMake FetchContent sources and native build state live in `<cache>/.dependencies`; the native build does not write generated files into an editable checkout or installed package. CUDA trees also distinguish the native GPU architecture or explicit `CUDAARCHS` target.

At the default verbosity, each successful renderer allocation reports the selected backend on stderr, for example `#rl_tools::hyperlux: backend=metal`; `HyperLux.backend` provides the same lowercase name from the compiled renderer type and `hyperlux.backend()` the name the next build would select. `hyperlux.set_verbosity(level)` selects the native log level, `error`, `warning`, `info` (default, includes the backend line) or `debug`, for loaded and future native modules alike, those of hyperdrone included. Changes apply to subsequent diagnostic checks, including during native work on other threads; a diagnostic already checked may finish at its previous level. Application stdout is independent of diagnostic verbosity.

## Rendering

```python
import numpy as np
import hyperlux
from hyperlux.camera import CameraSpec, RGB, Depth
from hyperlux.examples.data import procthor_scene_path

scene = hyperlux.load_scene(procthor_scene_path(), fidelity="high")

renderer = hyperlux.HyperLux(width=320, height=240, batch_size=4, cameras=(CameraSpec("front", (RGB("rgba8"), Depth())),), fidelity="high")
renderer.init(scene)

camera = renderer.camera(position=(0, 0, 1.5), look_at=(1, 0, 1.5), fov=80)
renderer.set_cameras(np.repeat(camera[None], 4, axis=0))
renderer.render()

rgb = renderer.outputs[RGB("rgba8")].read()  # (4, 240, 320, 4) uint8
depth = renderer.outputs[Depth()].read()      # (4, 240, 320) float32
```

The constructor's keywords are the fields of `hyperlux.Config`, the renderer's JIT key; `hyperlux.HyperLux(config=hyperlux.Config(...))` takes a config directly and `renderer.config` reads it back.

The renderer operates in the L2F FLU frame: +X forward, +Y left, +Z up; GLB assets are swizzled at load time. Procedural scenes (`hyperlux.Scene()` + `hyperlux.Object` / `hyperlux.Mesh` / `hyperlux.SceneLight`), segmentation output, overlays (per-camera dynamic content via `AssetPool` + `spawn`/`attach`), motion blur, and anti-aliasing follow the scene and placement APIs below.

Converted ProcTHOR scenes carry ambient `(0.6, 0.6, 0.6)` and a black background in `scenes[].extras.rl_tools.environment` (`mode: "solid"`, `ambient` and `background` are nonnegative linear RGB triples). Scene loading imports these settings; importing objects or assemblies leaves the parent environment alone. Ambient affects `high` and `veryhigh`, approximating indirect illumination without light bounces. Override imported settings with `scene.set_environment(...)` before `renderer.init(scene)`.

Lights have a `radius`: metres for `SceneLight.point` / `spot`, an angular radius in radians for `SceneLight.directional`, default `0.1` (also for GLB lights, which glTF leaves sizeless). It widens `high` / `veryhigh` specular highlights by the light's apparent size, so smooth surfaces show the light's reflection instead of unboundedly bright sub-pixel glints; `radius=0` gives ideal point lights.

`render()` acquires the declared image outputs; `render_launch()` and `render_sync()` split submission from waiting. Reads never create a capture or advance temporal state. Declare `Probes()` on the cameras that need collision probes, set `num_probes` per selected camera, call `generate_probe_directions()`, and execute `probe()` separately. `renderer.outputs[Probes()].distance.read()` returns float32 distances; `.hit.read()` returns native int32 hit flags. Probe 0 follows the camera's principal axis; remaining directions form a fixed Fibonacci set in camera FLU coordinates.

### Interactive renderer spy

`renderer.spy()` opens an independent spectator using the selected camera's scene and attached overlays, without rendering source outputs or advancing flow history. Initialize the renderer and set its cameras first. Call `renderer.update()` after overlay edits and `spy.update()` regularly on the main thread; use `with renderer.spy(camera=0) as spy:` or call `spy.close()`. This explicitly polled viewer does not import or require IPython.

The optional `renderer.spy(interactive=True)` integration publishes edits after each command in terminal IPython and keeps the window responsive while the prompt is idle. Run the articulated mechanism example with:

```bash
.venv/bin/pip install ipython
.venv/bin/ipython -i -m hyperlux.examples.articulation_spy
```

```python
renderer.set_pose(placement, hyperlux.Pose(base), coordinates=[0.8, 0.3])
second = renderer.spawn(0, handle, hyperlux.Pose(hyperlux.make_transform(position=(2, 0, 1))))
renderer.despawn(second)
spy.close()
```

Long-running commands need explicit `spy.update()` calls to process GUI events. Interactive polling requires terminal IPython's full prompt and no other active `%gui` integration; notebooks and the plain Python shell use explicit polling. Scene geometry and the asset pool remain fixed after initialization. The optional native dependencies are GLFW and desktop OpenGL (on Ubuntu: `libglfw3-dev`, `libgl1-mesa-dev`); CMake fetches the pinned Dear ImGui release with git, and `RL_TOOLS_HYPERLUX_DISABLE_SPY=ON` disables the viewer. No display or window is opened until `spy()` is called.

C++ uses `hyperlux::Spy<hyperlux::spy::Config<RENDERER>>` with device-first `malloc`, `init(device, spy, renderer, scene, pool, metadata, camera)`, `update(device, spy, renderer)` and `free`, linking `RLtools::HyperLuxSpy`. Pass the scene and pool used to initialize the source renderer and keep all three alive until freeing the spy; camera poses and placements remain editable, while scene and pool geometry stay fixed.

### Output buffers and DLPack

```python
output = renderer.outputs[RGB("rgba8")]
snapshot = output.read()
output.read(out=rgba_array)
rgba = torch.from_dlpack(output.blocks[0])
packed = torch.from_dlpack(output.packed.blocks[0])
output.layout  # shape, dtype, axes, selected camera identities, device, block count
```

Snapshots own their data. `read(out=...)` requires the compiled shape and dtype and copies directly into supported CPU/CUDA buffers; CUDA callers supply `stream=torch.cuda.current_stream().cuda_stream` when using a nondefault stream. Live blocks retain their native allocation, keeping it alive after `close()`, and support DLPack consumer-stream ordering; subsequent captures may overwrite their contents. RGBA8 uses `uint8[..., 4]`, and `.packed` views the same words as `uint32[...]`. A standalone renderer has one block; an environment built on hyperlux may expose several and copy into contiguous caller-owned buffers for aggregate reads.

Inputs accept any DLPack producer. CUDA-resident camera input (OptiX): `set_cameras` dispatches on `__dlpack_device__`, so a CUDA tensor — a torch GPU tensor, a CuPy/JAX CUDA array, or hyperdrone's `Sim.camera_bases_dlpack(aspect=renderer.aspect)` — is handed over device-to-device on the render stream, fully async; `stream=` takes the producer's cudaStream_t handle for event-ordered hand-off. The renderer keeps CUDA inputs referenced until the render stream has read them, without waiting for a synchronization; do not mutate their contents before then.

### Outputs

Camera profiles declare immutable output descriptors from `hyperlux.camera`:

| Descriptor | Default dtype | Per-camera shape | Contract |
|---|---|---|---|
| `RGB()` | uint8 | `(H, W, 4)` | sRGB with alpha 255, like C++ `RGB<>`; `RGB("float32")` gives float32 `(H, W, 3)` |
| `Depth()` | float32 | `(H, W)` | Camera-forward depth at shutter close; miss `+inf` |
| `Segmentation(labels="instance")` | uint32 | `(H, W)` | Instance IDs; miss `0xFFFFFFFF` |
| `Segmentation(labels="semantic")` | uint32 | `(H, W)` | Semantic classes; miss `0xFFFFFFFF` |
| `Normals()` | float32 | `(H, W, 3)` | World-FLU geometric unit normal facing against the ray; miss zero |
| `Flow(direction="backward")` | float32 | `(H, W, 2)` | Current-to-previous `(dx, dy)` displacement in pixels |
| `FlowStatus(direction="backward")` | uint32 | `(H, W)` | VALID, MISS, BEHIND, OUT_OF_FRAME, OCCLUDED |

All image descriptors also support `format="rgba8"`. Depth and flow `scale=` applies only to RGBA8; raw descriptors reject a non-default scale. Flow and its status are independently selected, and both `direction="forward"` and `"backward"` can coexist. One native quantity/representation slot must use identical parameters across the rig, including cameras with disjoint selections. MISS and BEHIND flow vectors are zero; OUT_OF_FRAME and OCCLUDED retain geometric displacement. `hyperlux.FlowStatus` contains status values; `camera.FlowStatus` selects the status output.

Buffers contain only selected cameras, ordered by physical camera index; image shapes are `(selected_cameras, H, W[, channels])`, and layout camera identities pair each mount name with its repetition index. Native metadata supplies dimensions and dtypes. Names and runtime poses/FOVs do not change compiled specializations; camera topology and output semantics do. Depth, segmentation, normals, and flow use pixel centres at shutter close, independently of RGB anti-aliasing and motion blur. Flow advances explicitly:

```python
renderer = hyperlux.HyperLux(width=128, height=128, cameras=(CameraSpec("front", (Flow(), FlowStatus())),))
renderer.init(scene)
renderer.set_cameras(previous_cameras)
renderer.render()
renderer.advance_flow()
renderer.set_cameras(current_cameras)
renderer.render()
flow = renderer.outputs[Flow()].read()
status = renderer.outputs[FlowStatus()].read()
```

For labelled ProcTHOR rendering, preserve GLB root instances; instance and semantic labels can be rendered together:

```python
scene = hyperlux.load_scene(procthor_scene_path(), rgb=False, preserve_instances=True)
renderer = hyperlux.HyperLux(width=512, height=512, cameras=(CameraSpec("front", (Segmentation(), Segmentation(labels="semantic"))),))
renderer.init(scene)
# Set cameras and render as usual.
instances = renderer.outputs[Segmentation()].read()
labels = renderer.outputs[Segmentation(labels="semantic")].read()
classes = scene.segmentation_classes  # {class_id: name}, taxonomy: scene.segmentation_taxonomy
```

Semantic IDs come from AI2THOR-Hab's taxonomy: unknown is `0` (including Ceiling, which has no dataset category); background is `0xFFFFFFFF` in both modes. Instance IDs identify scene instances: use `scene.instance_object(id)` then `scene.object_name(object_index)` for static hits. Overlay instance IDs are `scene.num_instances + placement.overlay * renderer.config.max_overlay_instances + placement.first_slot + part`. `spawn()` returns an `OverlayPlacement`; transform and despawn methods reject stale, foreign and forged handles. `spawn(..., segmentation_class=id)` labels every part of the placement `id` instead of its asset's authored classes, and `replace()` keeps that label. Semantic IDs must never be used as object indices. Loading remains welded by default for RGB; semantic rendering rejects a welded object containing multiple classes. Assemblies and asset-pool copies retain labels and the taxonomy. Composing conflicting taxonomies requires an explicit remapping before semantic rendering. The reserved class `hyperlux.AGENT_SEGMENTATION_CLASS` (`0xFFFFFFFD`) is what hyperdrone labels drone placements with; dataset taxonomies may not use IDs from `0xFFFFFF00` upwards.

Assign labels before `HyperLux.init()`: `object.segmentation_class = id`, `scene.set_object_segmentation_class(index, id)`, `assembly.set_segmentation_class(index, id)`, or `pool.set_object_segmentation_class(asset, object_index, id)` retains the current taxonomy and explicitly makes that object's class valid, including welded objects. Assignment rejects IDs absent from that taxonomy. `object.assign_segmentation_class(id, taxonomy="custom", classes={0: "Unknown", 7: "Furniture"})` replaces the taxonomy; omitting taxonomy/classes selects unnamed numeric IDs.

Objects, scenes, assemblies and pools expose `remap_segmentation_classes(mapping, taxonomy="custom", classes={...})`; pools also expose `remap_asset_segmentation_classes(asset, mapping, ...)`. Omitting taxonomy/classes selects unnamed numeric IDs. Mappings must cover every used class, and validation failures leave labels unchanged. Remapping preserves valid class boundaries and rejects mixed-class welded objects; explicitly assign one class first if collapsing their labels is intentional. Remap conflicting sources separately before composition. The C++ equivalents are device-first `assign_segmentation_class` and `remap_segmentation_classes` in `hyperlux/segmentation_cpu.h`, with a shared immutable target taxonomy (or `nullptr` for unnamed IDs).

`hyperlux.datasets.procthor.Train200()` exposes the header-pinned 200-scene ProcTHOR training subset (scene IDs 0–199). Catalog access loads the native core without resolving assets; use `dataset["ProcTHOR-Train-1"].reference` with `hyperlux.load_scene`. The RLtools commit identifies the catalog's membership, order, and hashes. This collection's Train-1 artifact differs from the scene pinned by `examples.data.procthor_scene_path()` and the rendering goldens.

```python
dataset = hyperlux.datasets.procthor.Train200()
scene = hyperlux.load_scene(dataset["ProcTHOR-Train-1"].reference)
```

### Articulated assets

[`Articulation.ipynb`](hyperlux/examples/Articulation.ipynb) builds a hinge and slider with `hyperlux.primitives.box(size=(x, y, z), color=(r, g, b))` and displays a three-second MP4. Boxes are ordinary `Object`s centered at the origin, with full dimensions in FLU coordinates.

Every part of an `ObjectAssembly` moves relative to its parent part (`add_part(object, transform, parent=...)`, or the enclosing node that `load_assembly(path, part_nodes=[...])` records; an enclosing node comes first) through its joint, by default `fixed`. `assembly.set_joint(part, "revolute", axis=..., pivot=..., coordinate=i)` turns the part (index or object name) and its children `coordinates[i]` radians about `axis` through `pivot`, both in the part frame; `"prismatic"` slides them `coordinates[i]` metres along `axis`. At zero coordinates every part sits at its rest frame. Joints are bound when the assembly enters the pool, which requires every coordinate below `assembly.num_coordinates` to drive a joint (several joints may share one).

A placement's rigid state is a `hyperlux.Pose(base, scale=1.0)`: a `(3, 4)` base and a uniform scale applied after it; the asset's joint coordinates (default the rest state) travel beside it as a `spawn` or `set_pose` argument. `spawn(overlay, asset, pose, coordinates=...)` places an asset in both flow instants; `set_pose` is the one way to move it:

```python
asset = pool.add_assembly(assembly)
placement = renderer.spawn(0, asset)
renderer.set_pose(placement, hyperlux.Pose(body, scale=scale), coordinates=coordinates)
renderer.set_pose(placement, hyperlux.Pose(body_open, scale=scale), hyperlux.Pose(body_close, scale=scale),
                  coordinates=coordinates_open, coordinates_close=coordinates_close)
renderer.update()
renderer.render()
```

The two-pose form is the shutter (`dynamic_motion_blur=True`): the base follows the constant-twist screw between the states and the scale and coordinates interpolate before the kinematics, so a revolute joint keeps every revolution; the close state is what segmentation, probes and the next frame see, and all a renderer without dynamic motion blur uses.

`renderer.placement_batch(placements)` checks distinct placements once and returns a `PlacementBatch` that `set_pose` moves as a whole: `pose.base` of shape `(B, 3, 4)` or `(B, 12)`, `pose.scale` a float or `(B,)`, `coordinates` `(B, C)` with `C` at least every member's `num_coordinates` (each reads its own first columns), and likewise for the close state. Despawning a member makes the batch stale. Host arrays run the single-placement verb per member in one native call; see below for CUDA tensors. `python -m hyperlux.examples.articulation` renders a spinning hinge and a sliding block with motion blur; hyperdrone's `articulate` and `Rotorcraft` rig body/prop_* drone assets on top of these joints.

### Bulk posing on the GPU

On the OptiX backend a placement batch takes CUDA tensors (any DLPack producer): `set_pose` runs one kernel on the render stream after awaiting `stream=`, the producer's `cudaStream_t` handle, with constant host work per call. `update_launch()` and `advance_flow_launch()` queue their work without waiting, so a simulation that keeps its poses on the device drives the renderer with no host data path and no synchronization per frame:

```python
batch = renderer.placement_batch(placements)
for step in range(steps):
    # the producer writes bases (B, 12) and coordinates (B, C) on producer_stream
    renderer.set_pose(batch, hyperlux.Pose(bases, scale=scale), coordinates=coordinates, stream=producer_stream)
    renderer.update_launch()
    renderer.render_launch()
    renderer.outputs[RGB(format="rgba8")].read(out=frames, stream=consumer_stream)
```

Read outputs in such a loop with `read(out=..., stream=...)`, which copies on the render stream: an exported block makes its consumer wait for the render, but the next `render_launch` does not wait for the consumer.

## Tests

Use the repository `.venv` with editable `interface/python/hyperlux` installed.

```bash
HYPERLUX_BACKEND=GENERIC .venv/bin/python -m pytest interface/python/hyperlux/tests -v
```

With the spy dependencies installed, run `HYPERLUX_TEST_SPY=1 xvfb-run -a .venv/bin/python -m pytest interface/python/hyperlux/tests/test_renderer_spy.py` to check the native GUI lifecycle and unchanged renderer outputs; omit `xvfb-run -a` when using a desktop display. `tests/test_spy_ipython_terminal.py` exercises the optional integration in a real terminal when IPython and pexpect are installed; it requires no display.

`tests/jit` exercises the build infrastructure against a toy component in seconds; `tests/test_architecture.py` enforces the import DAG (the infrastructure never imports the renderer modules, hyperlux never imports hyperdrone); `interface/python/scripts/test_sdist.sh` is the self-containment rot guard of the hyperlux and hyperdrone sdists (build → clean venv → vendored-tree render).

`scripts/benchmark_camera_outputs.py` compares fixed full and compact rigs on a procedural scene, checks identical shared outputs and renamed-config cache reuse, and reports output bytes, render/read timings, and Python allocation peaks; `scripts/benchmark_bulk_pose.py` measures the GPU posing path. Both log to metra when `METRA_URL` is set.

## Publishing

hyperlux and hyperdrone are released as a pair with one version number: `interface/python/scripts/set_version.py <version>` writes it into both packages and into hyperdrone's exact `hyperlux==` pin, and `interface/python/scripts/publish.sh` builds both, rehearses the install, and uploads hyperlux before hyperdrone.

## Stability

The JIT key hashes the configuration and the top-level header only; changes to included headers rebuild that tree in place. Reproducibility is per configuration and call sequence on one host build (`-O3 -march=native -ffast-math`). The `Train200` catalog is immutable; cite its RLtools commit for paper datasets and identify the examples and golden scenes separately as fixtures.

All RGB fidelity tiers store sRGB; RGBA8 alpha is always 255 and reserved. Medium uses a checker background. Probe misses return `+inf` with int32 hit flags; `ClearanceSampler` accepts candidates whose sampled rays have no hit closer than the clearance radius; it does not compute hyperdrone's free-space annotation table.
