# hyperdrone

Drone simulation stack for RLtools: raytracing renderer, L2F multirotor dynamics, and
environment setup under one Python package.

| Subpackage | What it is |
|---|---|
| `hyperdrone.render` | Raytracing renderer (OptiX / Metal / Vulkan / generic CPU). Self-contained — no dependency on drone dynamics. |
| `hyperdrone.dynamics` | Vectorized, stateful L2F multirotor simulator (`Sim`); cpu and cuda variants. |
| `hyperdrone.env` | Environment setup: free-space sampling, spawning, and the `World` wiring of sim + renderer. |
| `hyperdrone.jit` | Shared compile-and-cache infrastructure both domain packages build on. |
| `hyperdrone.cuda` | CUDA staging helpers (`upload` → DLPack tensor sets). |

Native components are JIT-compiled per set of compile-time constants (resolution, camera
count, drone count, ...) into a persistent cache; later uses load the cached library
directly. `render` and `dynamics` are peers that never import each other — the coupling
surface is a plain DLPack tensor of packed camera bases, device-resident on OptiX + CUDA.

## Install

From an rl-tools checkout (development):

```bash
.venv/bin/pip install -e interface/python/hyperdrone
```

From an sdist (self-contained — bundles the rl-tools headers and raytracing sources, no
checkout needed):

```bash
pip install hyperdrone-<version>.tar.gz
```

Install the dependencies used by the packaged examples with:

```bash
pip install "hyperdrone[examples]"
```

Requirements: CMake >= 3.24, a C++17 compiler, and per feature: assimp (system package)
for GLB loading, CUDA + OptiX driver for the OptiX render backend and cuda dynamics,
Vulkan dev + glslang for the VULKAN backend. Nothing beyond the compiler for
GENERIC + cpu.

## Environment

| Variable | Meaning |
|---|---|
| `HYPERDRONE_RENDER_BACKEND` | `OPTIX` \| `METAL` \| `VULKAN` \| `GENERIC` \| `AUTO` (default: Metal on macOS, OptiX elsewhere) |
| `HYPERDRONE_DYNAMICS_DEVICE` | `CPU` \| `CUDA` \| `AUTO` (default: cuda when available) |
| `HYPERDRONE_CACHE_DIR` | root for CMake build trees and downloaded build dependencies (default `~/.cache/hyperdrone`) |
| `HYPERDRONE_RLTOOLS_ROOT` | rl-tools source root override (default: enclosing checkout, else the vendored tree) |
| `HYPERDRONE_PROCTHOR_PATH` | local ProcTHOR example scene override (otherwise downloaded and cached) |
| `HYPERDRONE_SKIP_BUILD` | skip the CMake staleness check when the artifacts already exist |
| `HYPERDRONE_OFFLINE` | forbid network during builds (requires seeded/vendored dependencies) |
| `HYPERDRONE_BUILD_JOBS` | parallel build jobs (default 5) |

Each component and variant gets its own build tree (`render-optix-*`, `dynamics-cuda-*`,
...); switching never invalidates another's cache. All configure/build steps run under a
per-tree file lock, so many worker processes can share one cache safely. CMake
FetchContent sources and native build state live in `<cache>/.dependencies`; the native
build does not write generated files into an editable checkout or installed package.

Whenever a renderer is successfully allocated, RLtools reports the selected backend on
stderr, for example `#rl_tools::rendering::raytracing: backend=metal`. The
`Renderer.backend` property provides the same lowercase name programmatically.

## Rendering

```python
import math, numpy as np
from hyperdrone import render
from hyperdrone.examples.data import procthor_scene_path

scene = render.load_scene(procthor_scene_path(), fidelity="high")

renderer = render.Renderer(width=320, height=240, num_cameras=4, output="rgbd", fidelity="high")
renderer.init(scene)

camera = renderer.camera(position=(0, 0, 1.5), look_at=(1, 0, 1.5), fov=math.radians(80))
renderer.set_cameras(np.repeat(camera[None], 4, axis=0))
renderer.render()

rgb   = renderer.frame()    # (4, 240, 320, 4) uint8
depth = renderer.depth()    # (4, 240, 320) float32
```

The renderer operates in the L2F FLU frame: +X forward, +Y left, +Z up; GLB assets are
swizzled at load time. Procedural scenes (`render.Scene()` + `render.Object` /
`render.Mesh` / `render.SceneLight`), segmentation output, overlays (per-camera dynamic
content via `AssetPool` + `spawn`/`attach`), motion blur, and anti-aliasing follow the
same API as before under `hyperdrone.render.*`.

### Zero-copy I/O (DLPack)

```python
renderer.frame()                  # snapshot copy (safe to keep)
renderer.frame(copy=False)        # numpy view of the staging buffer (refreshed in place)
rgba = torch.from_dlpack(renderer.frame_dlpack())
# live uint8 (num_cameras, height, width, 4), zero-copy GPU tensor on OptiX
packed = torch.from_dlpack(renderer.frame_raw_dlpack())
# same memory as packed uint32 (num_cameras, height, width)
```

Inputs accept any DLPack producer. CUDA-resident camera input (OptiX):
`set_cameras` dispatches on `__dlpack_device__`, so a CUDA tensor — a torch GPU tensor,
a set pre-uploaded with `hyperdrone.cuda.upload`, or `Sim.camera_bases()` — is handed
over device-to-device on the render stream, fully async; `stream=` takes the producer's
cudaStream_t handle for event-ordered hand-off.

## Dynamics

```python
from hyperdrone import dynamics

sim = dynamics.Sim(num_drones=4096, model="x500_sim", device="cuda")
sim.reset(seed=0)                     # deterministic; host-sampled, identical on cpu/cuda
sim.step(actions)                     # (N, 4) float32 in [-1, 1], host or CUDA tensor
sim.state["position"]                 # zero-copy DLPack views, CUDA-resident on cuda
sim.state.numpy("position")           # host copy
sim.observe()                         # (N, 18): position, rotation matrix, velocities
sim.parameters["mass"] = masses       # per-drone runtime parameters (until the next reset)

cameras = sim.camera_bases(fov=math.radians(100), aspect=renderer.aspect)
renderer.set_cameras(cameras, stream=sim.stream)   # device-resident hand-off
```

`num_drones` and domain randomization are compile-time (the JIT key); the model preset
(`crazyflie`, `x500_real`, `x500_sim`, `mrs`, ...), integration `dt`, and all physical
parameters are runtime. `set_compute_mdp(True)` adds reward/termination computation to
each step (`sim.rewards()`, `sim.terminated()`). One process can use one dynamics
variant (cpu or cuda).

## Environment setup

```python
from hyperdrone import dynamics, env, render

scene = render.load_scene("warehouse.glb", fidelity="medium")
positions = env.FreeSpaceSampler(scene, clearance=0.5).sample(4096, seed=0)

sim = dynamics.Sim(num_drones=4096, model="crazyflie")
renderer = render.Renderer(width=64, height=64, num_cameras=4096, output="rgb", fidelity="medium")
world = env.World(scene, sim, renderer)
world.spawn(positions)
frames = world.step(actions).frame()
```

`FreeSpaceSampler` rides the renderer's collision probes (works on every backend,
including GENERIC on CPU-only machines) and is deterministic given a seed. `World` is
sugar, not load-bearing: it wires `sim.step → camera_bases → set_cameras → render` with
the right streams and nothing else.

End-to-end example: `python -m hyperdrone.examples.drone_flythrough`; renderer benchmark:
`python -m hyperdrone.examples.benchmark` (flag-compatible with the C++ benchmark
counterpart).

## Tests

```bash
HYPERDRONE_RENDER_BACKEND=GENERIC .venv/bin/python -m pytest interface/python/hyperdrone/tests -v
```

`tests/jit` exercises the build infrastructure against a toy component in seconds;
`tests/test_architecture.py` enforces the import DAG (render and dynamics never see each
other); `scripts/test_sdist.sh` is the self-containment rot guard (build sdist → clean
venv → vendored-tree render).
