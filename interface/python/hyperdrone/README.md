# hyperdrone

Drone simulation stack for RLtools: raytracing renderer, L2F multirotor dynamics, and
environment setup under one Python package.

| Subpackage | What it is |
|---|---|
| `hyperdrone.render` | Raytracing renderer (OptiX / Metal / Vulkan / generic CPU). Self-contained — no dependency on drone dynamics. |
| `hyperdrone.dynamics` | Vectorized, stateful L2F multirotor simulator (`Sim`); cpu and cuda variants. |
| `hyperdrone.env` | The RL environment: the C++ `MultiEnvironment<hyperdrone::World>` driven through the exact rl_tools batch verbs. |
| `hyperdrone.gym` | Optional Gymnasium `VectorEnv` adapter over `hyperdrone.env` (`pip install "hyperdrone[gym]"`). |
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
Vulkan dev + glslang for the VULKAN backend. The WEBGPU backend fetches a hash-pinned
wgpu-native prebuilt at first build (no extra system packages). Nothing beyond the
compiler for GENERIC + cpu.

## Environment

| Variable | Meaning |
|---|---|
| `HYPERDRONE_RENDER_BACKEND` | `OPTIX` \| `METAL` \| `VULKAN` \| `WEBGPU` \| `GENERIC` \| `AUTO` (default: Metal on macOS, OptiX elsewhere) |
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
parameters are runtime. Reward and termination live in the
environment (`hyperdrone.env`), not the simulator. One process can use one dynamics
variant (cpu or cuda).

## RL environment

```python
import numpy as np
from hyperdrone.env import EnvConfig, MultiEnvironment

config = EnvConfig(num_environments=1, instances=1024, cam_width=32, cam_height=32, task="target_frame")
env = MultiEnvironment("scenes/", config=config, seed=0)

mask = np.ones(env.total_instances, dtype=np.uint8)
env.reset(mask)                    # resample parameters + states where mask is set
env.render(mask)                   # render the FPV cameras (mask marks fresh episodes)
observations = env.observe()       # (total, observation_dim) float32
env.step(actions)                  # (total, action_dim) float32 in [-1, 1]
rewards, terminated = env.rewards(), env.terminated()
critic_input = env.observe_privileged()
env.rotate_scene()                 # deterministic scene rotation; reset all instances after
env.observation_layout             # named blocks: which channels/values mean what
```

All environment semantics — reset, reward, termination, scene scheduling, observation
composition — live on the C++ side (`rl_tools::rl::environments::hyperdrone::MultiEnvironment<World>`);
the binding marshals tensors and nothing else, and a seeded rollout is pinned bit-exact
against the C++ verbs by a golden test. The scene argument is a directory of `.glb`
scenes, partitioned across environments. Configuration follows the C++ extension ladder:
`preset=` names the platform (`"crazyflie"`, `"x500_fpv"` — SELF_VISIBLE, pass a
body/prop_* GLB via `drone_asset=`), `task=` names the wrapper (`"target_frame"`,
`"moving_gate"`), `n_agents=` enables multi-agent, and `EnvConfig(spec_header=...)` pins
an arbitrary C++ specification (a header defining `hyperdrone_env_user::WORLD`, hashed
into the JIT key). A spec header can go beyond constants to a full user-authored task
wrapper — verb overloads registering and moving entities, compiled reward, termination:
`hyperdrone/examples/Orbiter.ipynb` walks through one, and its contract is pinned by
`tests/env/user_task_header.h`. The render backend for the environment follows
`HYPERDRONE_ENV_BACKEND` (default: the render backend selection).

Manual composition of `Sim` and `Renderer` (no MDP — rendering research, data
generation) remains first-class: see `python -m hyperdrone.examples.drone_flythrough`
for the wiring. `FreeSpaceSampler` lives in `hyperdrone.render` — it rides the
renderer's collision probes and is deterministic given a seed.

### Gymnasium

```python
from hyperdrone.gym import VectorEnv   # pip install "hyperdrone[gym]"
env = VectorEnv("scenes/", config=EnvConfig(instances=1024), seed=0)
observations, infos = env.reset()
observations, rewards, terminations, truncations, infos = env.step(actions)
```

Same-step autoreset over the env verbs; the core packages never import gymnasium.

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
