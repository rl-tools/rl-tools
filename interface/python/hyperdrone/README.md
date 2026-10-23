# hyperdrone

Drone simulation stack for RLtools: L2F multirotor dynamics and the HyperDrone RL environments on the HyperLux raytracer. Rendering is the [hyperlux](../hyperlux/README.md) package, which hyperdrone pins to its own version; hyperdrone's native components are built by hyperlux's JIT and share its cache.

| Module | What it is |
|---|---|
| `hyperdrone` | The RL environment `HyperDrone` and its compile-time `Config`: the C++ `HyperDrone<hyperdrone::World>` driven through the exact rl_tools batch verbs. `ImageObservation` selects the policy's cameras. `articulate` and `Rotorcraft` rig body/prop_* drone assets for the renderer. |
| `hyperdrone.dynamics` | Vectorized, stateful L2F multirotor simulator (`Sim`); cpu and cuda variants. |
| `hyperdrone.gym` | Optional Gymnasium `VectorEnv` adapter over `HyperDrone` (`pip install "hyperdrone[gym]"`). |

Native components are JIT-compiled per set of compile-time constants (resolution, camera count, drone count, ...) into hyperlux's persistent cache; later uses load the cached library directly. `hyperlux` and `dynamics` are peers that never import each other — the coupling surface is a plain DLPack tensor of packed camera bases, device-resident on OptiX + CUDA. Environment configurations are directly bound extensions, sharing their C++ implementation with the golden-rollout generator.

## Notebooks

| Notebook | Open in Colab |
|---|---|
| [Overview](hyperdrone/examples/Overview.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rl-tools/rl-tools/blob/main/interface/python/hyperdrone/hyperdrone/examples/Overview.ipynb) |
| [HyperDrone](hyperdrone/examples/HyperDrone.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rl-tools/rl-tools/blob/main/interface/python/hyperdrone/hyperdrone/examples/HyperDrone.ipynb#scrollTo=g-cDv8bdN7j9) |
| [Renderer](hyperdrone/examples/Renderer.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rl-tools/rl-tools/blob/main/interface/python/hyperdrone/hyperdrone/examples/Renderer.ipynb) |
| [Orbiter](hyperdrone/examples/Orbiter.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rl-tools/rl-tools/blob/main/interface/python/hyperdrone/hyperdrone/examples/Orbiter.ipynb) |

## Install

From an rl-tools checkout (development):

```bash
.venv/bin/pip install -e interface/python/hyperlux -e interface/python/hyperdrone
```

From PyPI or an sdist (`hyperdrone` depends on `hyperlux` of the same version, whose sdist bundles the rl-tools sources):

```bash
pip install hyperdrone
```

Install the dependencies used by the packaged examples with:

```bash
pip install "hyperdrone[examples]"
```

Requirements: CMake >= 3.24, a C++17 compiler, and git for dependency downloads.
CUDA + OptiX driver is required for the OptiX render backend and cuda dynamics,
Vulkan dev + glslang for the VULKAN backend. The WEBGPU backend fetches a hash-pinned
wgpu-native prebuilt at first build (no extra system packages). Nothing beyond the
build toolchain for GENERIC + cpu.

## Environment

Build settings are hyperlux's (`HYPERLUX_BACKEND`, `HYPERLUX_CACHE_DIR`, `HYPERLUX_RLTOOLS_ROOT`, `HYPERLUX_SKIP_BUILD`, `HYPERLUX_OFFLINE`, `HYPERLUX_BUILD_JOBS`; see its README); the environment's raytracing backend follows `HYPERLUX_BACKEND` like the renderer. hyperdrone adds `HYPERDRONE_<NAME>_PATH` for the rigged example assets, and the test suite reads `HYPERDRONE_TEST_SCENE_DIR` and `HYPERDRONE_TEST_SPY`.

At the default verbosity, each successful renderer allocation reports the selected backend on stderr, for example `#rl_tools::hyperlux: backend=metal`; `HyperDrone.render_backend` provides the same lowercase name from the compiled renderer type. `hyperlux.set_verbosity(level)` selects the native log level of every native module, hyperdrone's included.

## Dynamics

```python
from hyperdrone import dynamics

sim = dynamics.Sim(4096, parameters={"dynamics": {"dynamics": {"model": "x500_sim"}}}, seed=0, device="cuda")
sim.reset()                          # nominal state; reset(sample_states=True) samples the configured distribution
sim.step(actions)                    # (N, 4) float32 in [-1, 1], host or CUDA tensor
sim.state["position"]                # independent host copy
sim.state_dlpack("position")         # live backend buffer
sim.rotor_rpm()                      # (N, 4) rotor speeds in RPM, motor_command through max_rpm
sim.observe()                        # (N, 18): position, row-major body-to-world rotation, velocities
sim.set_instance_parameters({"dynamics": {"dynamics": {"mass": 1.2}}}, mask=selected)
sim.instance_parameters()            # per-drone values; overrides last until each drone's next reset
sim.parameters                       # fresh copy of the constructor's resolved shared parameters

cameras = sim.camera_bases_dlpack(fov=100, aspect=renderer.aspect)
renderer.set_cameras(cameras, stream=sim.stream)
```

`Sim` defaults to `device="cpu"`; choose `"cuda"` or `"auto"` explicitly. Seed at construction, then `reset(mask=None, *, sample_states=False)` advances the sampling sequence only for selected drones. Observation noise uses a separate stream from stepping. The shared parameter schema matches the environment's `"dynamics"` tree; integration `dt` is fixed at construction and available as read-only `sim.dt`. Per-drone overrides affect `dynamics.dynamics` and, on ideal actuation, `dynamics.ideal_rates`; a reset restores that drone's nominal parameters. `motor_command` is the rotor command in the model's action-limit units; `sim.rotor_rpm()` converts it to rotor speed in RPM through the model's `max_rpm`, the speed at `action_limit.max` (linear through zero), which every registry entry declares and `set_instance_parameters` can override. A process can load one dynamics variant.

`Sim` host accessors return independent snapshots; its methods ending in `_dlpack()` expose live storage. Drive each object on its construction CUDA device, and finish consuming a borrowed buffer before the producer overwrites it. `Sim` orders DLPack exports onto the consumer's stream. Camera bases have shape `(N, 12)` and concatenate `position`, `dir_00`, `dir_du`, and `dir_dv` world-FLU vectors; the pixel-centre ray is `normalize(dir_00 + (x+0.5)/width * dir_du + (y+0.5)/height * dir_dv)`; `aspect` is required when constructing them.

Choose the command signal with `control=` and the vehicle model with `actuation=` at construction, on `Sim` and `hyperdrone.Config` alike; `"motor"` on `"rotors"` is the default. Commands keep the usual batch shape, with each agent's values concatenated in agent order, and every command is normalized to `[-1, 1]`:

| `control` | Command per agent | `actuation` |
| --- | --- | --- |
| `"motor"` | Four motor commands | `"rotors"`: motor response and rigid-body dynamics. |
| `"ctbr"` | `[collective, roll_rate, pitch_rate, yaw_rate]` | `"ideal"`: the collective maps onto `[0, thrust_to_weight * g]` of acceleration along body `+Z`, the rates onto `±rate_limit` body FLU rad/s, and both hold exactly throughout the current integration interval. Both limits form the `ideal_rates` parameter group, a per-drone vehicle property: its default thrust-to-weight is derived from the preset's rotors at build time and does not follow later rotor overrides. On `"rotors"` the same signal would need the PD rate controller's gains, which the presets do not carry. |
| `"position"` | `[x, y, z, yaw]`, m and rad, unbounded | `"rotors"`: RAPTOR produces motor commands once per step, targeting zero linear velocity. `Sim` uses its simulation frame; `HyperDrone` accepts scene coordinates and converts internally. |

`action_low` and `action_high` expose the controller bounds, also used by Gymnasium. `sim.num_rotors` describes physical rotors independently of the command dimension.

Masked resets clear recurrent controller memory for selected drones (every agent of a selected environment instance). Successful instance parameter updates clear those rows; assigning `sim.state[...]` clears all rows. Custom `spec_header` Worlds choose their own controller. The autonomous localization task retains its empty action space and synchronized episode resets.

Registry quadrotors use Crazyflie/RAPTOR motor order: **front-right, rear-right, rear-left, front-left**, body FLU, yaw reaction-torque signs `−,+,−,+`. Commands, speeds, histories and per-rotor parameters share this order. Old policies or rotor-indexed data for formerly noncanonical presets require conversion when used with updated presets; explicit legacy motor-mode parameters remain supported, while RAPTOR validates canonical allocation.

C++ configurations select the command signal through `CONTROLLER` and the actuation model through the dynamics static parameters; `World` binds the controller's model, `State`, `Buffer` and the native dynamics interface:

```cpp
#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/rl/environments/hyperdrone/controllers/direct/operations_cpu.h>
#include <rl_tools/rl/environments/hyperdrone/operations_cpu.h>

namespace l2f = rl_tools::rl::environments::l2f;
struct Spec: ExistingSpec {
    using CONTROLLER = rl_tools::rl::environments::hyperdrone::controllers::direct::Config<l2f::parameters::ActionInterface::CTBR>;
    using DYNAMICS_STATIC_PARAMETERS = l2f::WithIdealRates<ExistingSpec::DYNAMICS_STATIC_PARAMETERS>;
};
```

`l2f::WithIdealRates` sets `ACTUATION_MODEL = l2f::actuation::IdealRates` (the default is `Rotors`) and appends the `ParametersIdealRates` component with the thrust-to-weight ratio derived from the base vehicle's rotors. l2f derives the adapter between signal and model: identity when they agree, the PD rate controller for CTBR commands on rotors (which needs `ParametersRateController` and `StateRateController` in the chains), and a compile-time error for motor commands on ideal rates.

Include controller operations before World operations so static overload lookup sees them. The shared `control(device, controller, state, buffer, parameters, physical_states, commands, rng)` returns the borrowed dynamics actions; it aliases commands for direct control and controller output for RAPTOR. Custom batched state-sampling overrides finish with `complete_reset(device, world, mask, rng)` to clear selected controller rows and request rendering.

The contract is structural: `Config` supplies `Bind` and `DynamicsParameters`; the bound controller exposes `COMMAND_DIM`, `ACTION_INTERFACE`, `NAME`, `SUPPORTS_SCALAR`, `State` and `Buffer`. Device-first overloads in `rl_tools` provide allocation/free, initialization, reset, control and dynamics validation; constexpr `command_bounds` supplies API metadata. Persistent memory belongs in `State`, scratch in `Buffer`; `transition_action` selects the native reward input from the command or resulting physical state, so deferred rewards do not depend on mutable controller output.

The C++ controller layer follows these consumers and constraints:

| Consumers | Design constraint |
| --- | --- |
| CPU/CUDA Worlds, multi-agent batches and dynamics-only `Sim` | One static controller selection binds dynamics actions and command metadata; borrowed tensors avoid command copies and RAPTOR batches inference. |
| TargetFrame, MovingGate and scene placement | Tasks retain sampling and rewards; shared reset completion clears controller memory, and World adapts scene targets before inference. |
| Motor training, PD CTBR, ideal rates and existing checkpoints | One normalized action convention and one state and observation layout across signals and actuation models; canonical motor indexing is shared. |
| Web, localization and trajectory callers | Reusable controller parameters, recurrent state and scratch; localization retains its existing observation preparation. |
| Python/Gym, custom specifications, rendering and IMU | Compiled bounds remain authoritative; physical rotor count and geometric mesh-hub matching remain independent of command shape. |

## RL environment

```python
import numpy as np
from hyperdrone import Config, HyperDrone

config = Config(num_worlds=1, instances_per_world=1024, width=32, height=32, task="target_frame")
env = HyperDrone("scenes/", config=config, seed=0)

mask = np.ones(env.total_instances, dtype=bool)
env.reset(mask)                    # resample parameters + states where mask is set
observations = env.observe()       # (total, observation_dim) float32
env.step(actions)                  # (total, action_dim) float32 in [-1, 1]
rewards, terminated = env.rewards(), env.terminated()
critic_input = env.observe_privileged()
env.rotate_scene()                 # deterministic scene rotation; reset all instances after
env.observation_layout             # named blocks: which channels/values mean what
```

A `HyperDrone` holds `num_worlds` Worlds; each World shows one active scene to its `instances_per_world` instances, and each instance flies `num_agents` agents. Instance tensors are World-major, so `env.total_instances == num_worlds * instances_per_world`.

`reset(mask)` samples parameters and states immediately. `observe()` renders as needed,
and each observation is acquired at most once per simulation step. Frame history advances once per simulation step, and repeated reads reuse that step's frame; skipped reads do not stretch camera exposure beyond `shutter_fraction * dt`. The training loop owns episode counters and
chooses when to reset; the Gymnasium adapter below supplies same-step autoreset and time limits.

On the OptiX backend the World runs on the GPU (`env.device == "cuda"`): its RNG and every
instance tensor live there, and every verb is enqueued on one CUDA stream, `env.stream`, which
you bind to the stream your tensors live on. CUDA arrays (any DLPack producer on
`env.device_id`) are then read in place and outputs written into `out=`, in stream order,
with no copies or host synchronization:

```python
import torch
env = HyperDrone("scenes/", config=config, seed=0, stream=torch.cuda.current_stream().cuda_stream)
observations = torch.empty((T, env.total_instances, env.observation_dim), device="cuda")
rewards = torch.empty((T, env.total_instances), device="cuda")
terminated = torch.empty((T, env.total_instances), dtype=torch.bool, device="cuda")
reset_mask = torch.ones(env.total_instances, dtype=torch.bool, device="cuda")
for t in range(T):
    env.reset(reset_mask)                     # CUDA bool mask
    env.observe(out=observations[t])          # written into the rollout buffer
    env.step(policy(observations[t]))         # CUDA float32 actions
    env.rewards(out=rewards[t])
    env.terminated(out=terminated[t])
    reset_mask = terminated[t]                # plus your episode limits
```

Order data produced on another stream yourself (`env_stream.wait_stream(producer)`), as
between any two torch streams; setting `env.stream` orders the new stream after the old
one's work. numpy arrays work with every World: they are staged through the environment
and the call synchronizes. `Config(device="cpu")` keeps the World on the host; the
visual-inertial localization task (no CUDA verbs yet) and `spec_header` Worlds (whose verbs
may be host loops) default to it.

`Config` is the compile-time part (the JIT key); the runtime part is `HyperDrone(..., parameters={"dynamics": {"dynamics": {"model": "x500", "mass": 2.1}}, "cameras": {"front": {"fov": 90}}})`, a partial dict merged over the nominal parameters as the base of every drone (the specification's randomization ranges apply on top at reset). `env.parameters` returns drone zero's resolved nominal parameters, including its entry-specific overrides: `"dynamics"` is the l2f parameter set, where a `"model"` key loads an l2f registry entry (`hyperdrone.dynamics.MODELS`) before the rest is merged, and each entry in `"cameras"` holds `mount.pose` (`position` and `orientation`, a (w, x, y, z) quaternion mapping the camera's FLU axes into the body frame) and the horizontal `fov` in degrees. `env.instance_parameters()` returns what each instance is using in the current episode (after randomization), and `env.set_instance_parameters(overrides, mask)` changes it until that instance's next reset (one dict for all selected instances or one per instance); only the drone itself varies (`dynamics.dynamics`, `dynamics.imu`, `cameras`, and `"drone"`, which starts from another table entry's nominal and swaps the rendered body).

Several drones fly in one environment through the drone table, `HyperDrone(..., drones=[{"asset": x500, "parameters": {...}}, {"asset": crazyflie, "parameters": {"dynamics": {"dynamics": {"model": "crazyflie"}}}}, ...])`: every agent samples one entry uniformly at each group reset, independently of the scene its World shows, so every scene meets every drone. An entry pairs the body/prop_* GLB rendered into the instance's cameras with a partial parameter dict merged over the shared constructor base (the per-drone keys `dynamics.dynamics`, `dynamics.imu`, `dynamics.ideal_rates`, `cameras`); the same GLB may back several dynamics entries, distinct GLBs load once and all of them need the same part count (an agent's body swaps in place at reset, on the GPU for a CUDA World). The table holds up to `env.max_drone_types` entries (8, a `spec_header` World sets its own), `env.drones` returns the resolved native table, and each agent's `"drone"` value in `env.instance_parameters()` indexes it. `env.self_visible` and `env.max_drone_types` report the compiled world's capabilities, including custom `spec_header` worlds. `drone_asset=X` is the one-entry table. The six rigged assemblies in `hyperdrone.examples.data.RIGGED_MODELS` use matching l2f registry entries; each prop spins with the dynamics rotor at its position whatever rotor order the entry uses.

With `Config(preset="x500_fpv", num_agents=7, num_camera_agents=7)`, each instance is a group of seven mutually visible drones; groups share scene geometry but remain isolated episodes. `instance_parameters()` returns `{"agents": [...]}` per group, and `set_instance_parameters({"agents": [{"drone": i} for i in choices]}, mask)` assigns models independently (`{}` leaves an agent unchanged). Single-agent entries keep the flat schema. Scene placement, exposure, shutter fraction, reset, reward and termination remain group-wide; all agents share the integration timestep.

`num_camera_agents` defaults to 1 and selects agents `[0, num_camera_agents)` to carry the declared rig. Every physical camera shares resolution and capture timing; all agents still receive actions and appear in state, privileged observations, and rendered geometry. `cameras=(CameraSpec("front", ...), CameraSpec("chase", ...))` declares two mounts per camera-equipped agent. Runtime `parameters={"cameras": {"front": fpv, "chase": chase}}` sets named mount/FOV overrides; omitted names retain their values. Custom C++ specifications expose names `camera0`, `camera1`, and so on.

`observation=ImageObservation(cameras=("front",))` selects policy inputs independently of physical cameras. Every selected mount must explicitly declare float32 RGB; auxiliary cameras do not change policy dimensions or history allocation. `observation=None` omits image history and policy observations while retaining rendering, dynamics, privileged state, and applicable IMU access. `observe()` and Gym construction reject a missing policy observation. Frame retention and history-stack composition remain defined by the native task. `frames()` reshapes the processed policy observation, including task composition; `env.outputs[RGB("float32")].read()` returns raw renderer pixels.

For declared `Probes()` outputs, call `env.render()` before the first `env.probe()`. Probing uses the last rendered camera and geometry poses; it does not refresh those poses or advance the task's capture cadence. Read the results through `env.outputs[Probes()].distance` and `.hit`.

Every `env.reset()`, including an all-false mask, invalidates image and probe reads because task reset work may overwrite raw buffers, and so does any `step()` that makes a frame due. Call `render()` or `observe()` before reading images again, then `probe()` before reading probe results. A `step()` that makes no frame due, such as an intermediate IMU tick, preserves valid image and probe results; a capture performed internally by `step()` invalidates earlier probe results.

All environment semantics — reset, reward, termination, scene scheduling, observation
composition — live on the C++ side (`rl_tools::rl::environments::HyperDrone<World>`);
the binding marshals tensors and nothing else, and a seeded host-World rollout is pinned
bit-exact against the C++ verbs by a golden test. The scenes argument is a directory of `.glb`
scenes (sorted corpus) or a list of references — `.glb` paths, `conta:HASH` strings, or
conta store manifest entries `{"description": ..., "hash": ...}` — used as the corpus in
list order and partitioned across Worlds; conta references (scenes, `drone_asset=`,
`assets={"gate": reference}`) resolve through the `conta` package into the shared content-addressed
cache, so a config pins the exact corpus on any machine. Configuration follows the C++
extension ladder:
`preset=` names the platform (`"crazyflie"`, `"x500_fpv"` — SELF_VISIBLE, pass a
body/prop_* GLB via `drone_asset=`, articulated like `hyperdrone.Rotorcraft` (see Articulated assets) — and `"x500_fpv_imu"`, the same platform stepping at
IMU rate), `task=` names the wrapper (`"target_frame"`, `"moving_gate"`,
`"visual_inertial_localization"`), `num_agents=` enables multi-agent, `num_camera_agents=` selects the prefix with cameras, `cameras=` declares their ordered rig, and `Config(spec_header=...)` pins
an arbitrary C++ specification (a header defining `hyperdrone_user::WORLD`, hashed
into the JIT key); its final C++ specification must define `N_CAMERA_AGENTS` explicitly. A spec header can go beyond constants to a full user-authored task
wrapper — verb overloads registering and moving entities, compiled reward, termination:
`hyperdrone/examples/Orbiter.ipynb` walks through one, and `tests/environment/test_orbiter.py`
compiles and flies its header. The environment's raytracing backend follows
`HYPERLUX_BACKEND` like the renderer.

Custom C++ World specifications declare one output list per mount in `CAMERA_OUTPUTS`; the mount count is derived from that list. For example, inside a specification:

```cpp
using OUTPUT_PROFILE = rl_tools::hyperlux::outputs::List<
    rl_tools::hyperlux::outputs::RGB<rl_tools::hyperlux::outputs::format::Float32>,
    rl_tools::hyperlux::outputs::Depth<>,
    rl_tools::hyperlux::outputs::Probes>;
using DIAGNOSTIC_PROFILE = rl_tools::hyperlux::outputs::List<
    rl_tools::hyperlux::outputs::Depth<>>;
using CAMERA_OUTPUTS = rl_tools::rl::environments::hyperdrone::CameraOutputs<OUTPUT_PROFILE, DIAGNOSTIC_PROFILE>;
using OBSERVATION = rl_tools::rl::environments::hyperdrone::ImageObservation<0>;
```

This two-mount rig is shared by every camera-equipped agent and instance. Only the observation-selected mount requires float32 RGB; the second mount produces depth alone. Native metadata exposes each output's compact rows through the same Python handles as generated configurations. Use `NoObservation` instead of `ImageObservation<...>` for a World without policy images.

Entity wrappers share their own pose writes in `prepare_entity_poses`: `render` calls it before forwarding acquisition, and `step` calls it before forwarding when a frame is pending and history or flow requires its poses. The base World separates input production (`prepare_render_inputs`), temporal reset (`reset_render_state`, using prepared poses for flow) and advancement (`advance_flow`), and acquisition (`render`, which publishes transforms, updates acceleration, and scatters images into history). Pose helpers neither render nor call `update`; CUDA may fuse temporal bookkeeping into existing kernels without changing these semantics.

The target-frame task requires one agent and one selected observation camera; auxiliary physical cameras remain available. It pre-renders a bank of episode setups (drone, scene placement, cameras, brightness and target perturbations), one per instance, in one pass. A reset refreshes this bank when its age reaches `TARGET_BANK_REFRESH_STEPS` simulation steps (a compile-time task constant, default 32) or the scene changes. Otherwise, resets adopt setups in turn and copy the selected camera's target frame without rendering; resets beyond the instance count within one refresh reuse setups. Each episode's drone and cameras come from its setup, so `set_instance_parameters` rejects drone and camera overrides for this task; a replayed spawn re-renders its target frame.

`Config` defaults to `fidelity="veryhigh"` and `anti_aliasing_grid=2` (2×2 AA); fidelity accepts `low`, `medium`, `high`, and `veryhigh`, and AA grids are 1 (disabled), 2, 3, or 4. Custom spec headers select `SHADING`, `ENABLE_ANTI_ALIASING`, and `ANTI_ALIASING_GRID_SIZE` directly.

`Config(motion_blur_samples=8)` enables camera and propeller motion blur for presets and tasks; valid counts are 1 (default, disabled), 2, 4, 8, 16, and 32. Custom spec headers control `ENABLE_MOTION_BLUR` and `MOTION_BLUR_SAMPLES` directly.

For repeatable shots, capture `points = env.spawn_points()` after a reset and save the returned dictionaries, then use `env.reset(mask, spawn_points=points)` with one entry per selected instance (a host mask). Each entry pins the active scene by content hash (the bare hash as recorded, or the constructor's reference form, `conta:HASH` or a store manifest entry), one drone-table index per agent in `"drone"`, scene translation/yaw, per-agent pose (`xyz, wxyz`), and shutter fraction. The reset uses constructor-supplied drone dynamics and camera mounts; velocity/motor initialization and task-specific randomization still follow their configured ranges. Scene changes remain explicit through `rotate_scene()`, which requires resetting every instance before further use; a shot for a different active scene is rejected.

Manual composition of `Sim` and `HyperLux` (no MDP — rendering research, data
generation) remains first-class: see `python -m hyperdrone.examples.drone_flythrough`
for the wiring. `hyperlux.FreeSpace` provides spawn positions from the same
free-space annotation table the environment uses, deterministic given a seed.

### Native spy

`env.spy()` opens a live camera mosaic and an independent spectator camera. Call `update()` on the main thread after your usual `observe()` or `render()` and before changing the environment; the mosaic reads existing raw RGB outputs without acquiring another environment capture. It requires a camera with RGB output, independently of policy observation selection or image history. GUI events are processed only by `update()`, so call it regularly even when training is paused. Closing its window leaves the environment running; `update()` then returns `False`.

```python
env.reset()
with env.spy() as spy:
    for _ in range(1000):
        observations = env.observe()
        spy.update()
        env.step(policy(observations))
        env.reset(env.terminated())
env.close()
```

Click a camera tile to select its member, instance, agent and camera, then click the spectator for mouse look. WASD moves, Q/E moves down/up, Shift accelerates, and Esc releases the mouse; the spectator pose is retained for each world independently. The spectator has its own renderer; camera capture and spectator refresh are capped at 20 Hz, and GUI work at 60 Hz. They still consume GPU bandwidth and rendering capacity. Use `with` or explicitly close the spy on the main thread before environment teardown. The Gymnasium adapter also exposes `env.spy()`; call `update()` after `reset()` or `step()`.

The optional native dependencies are GLFW and desktop OpenGL (on Ubuntu: `libglfw3-dev`, `libgl1-mesa-dev`); CMake fetches the pinned Dear ImGui release with git. After adding dependencies to an existing build, run `cmake /path/to/cache/environment-<backend>-<tag>`, unset `HYPERLUX_SKIP_BUILD`, and restart Python; constructing the next environment rebuilds its native module. The build directory lives under `HYPERLUX_CACHE_DIR` (default `~/.cache/hyperdrone`); CMake reports `Raytracing spy` availability. Set `RL_TOOLS_HYPERLUX_DISABLE_SPY=ON` to disable both native viewers. Ordinary environments remain usable when these dependencies are absent; `env.spy()` then explains the missing build feature. No display or window is opened until `env.spy()` is called.

Run `python -m hyperdrone.examples.spy` to watch four X500 drones with rigged bodies and spinning propellers hold position with RAPTOR in the shared ProcTHOR scene; optionally pass a local GLB path as the positional argument.

C++ uses `hyperdrone::Spy<hyperdrone::spy::Config<ENVIRONMENT>>` with device-first `malloc`, `init`, `update` and `free`; link `RLtools::HyperLuxSpy` and close the spy before freeing its environment. The [C++ example](../../../src/rl/environments/hyperdrone/spy/example.cpp) builds as `example_hyperdrone_spy` when the GUI dependencies are available. Both interfaces share the same capture, spectator and window implementation.

### Visual-inertial localization

`task="visual_inertial_localization"` on `preset="x500_fpv_imu"` is the localization
benchmark: fixed-length synchronized episodes stepping at IMU rate (`env.dt`), a camera
frame every `env.frame_stride` steps, and the IMU sample of each step from
`env.observe_imu()` (named by `env.observation_layout_imu`: accelerometer, gyroscope,
frame age, new-frame flag). The task is autonomous: the RAPTOR autopilot built into it
flies a waypoint route (its checkpoint is pinned by content hash and fetched through
conta on first use), so `action_dim` is 0 and `step()` takes no actions. The privileged
observation carries the current `waypoint` next to the dynamics state; the drone's own
body is not rendered, matching the C++ benchmark harness. Resets must be all-or-none per
World.

```python
config = Config(instances_per_world=2, width=160, height=120, preset="x500_fpv_imu", task="visual_inertial_localization")
env = HyperDrone(scenes, config=config, seed=0)
env.reset()
for step in range(env.episode_step_limit):
    if step % env.frame_stride == 0:
        frames = env.frames()          # the estimator's camera frame
    env.step()
    imu = env.observe_imu()            # the estimator's IMU sample for this step
    truth = env.observe_privileged()   # position / rotation matrix / velocities / waypoint
```

`python -m hyperdrone.examples.visual_inertial_localization` runs an episode and scores
IMU dead reckoning against the ground truth, the same protocol as the C++ demo.

### Gymnasium

```python
from hyperdrone import Config
from hyperdrone.gym import VectorEnv   # pip install "hyperdrone[gym]"
env = VectorEnv("scenes/", config=Config(instances_per_world=1024), seed=0)
observations, infos = env.reset()
observations, rewards, terminations, truncations, infos = env.step(actions)
```

The adapter resets instances that terminate or reach `episode_step_limit` within the same
step. Returned observations start the new episode for those instances; no final observation
is exposed. `truncations` flags time limits that are not terminations. The core packages
never import gymnasium.

End-to-end example: `python -m hyperdrone.examples.drone_flythrough`; renderer benchmark:
`python -m hyperdrone.examples.benchmark` (flag-compatible with the C++ benchmark
counterpart).

## Publishing

hyperlux and hyperdrone are released as a pair with one version number: `interface/python/scripts/set_version.py <version>` writes it into both packages and into hyperdrone's exact `hyperlux==` pin (a `.postN` suffix bumps hyperdrone alone), and `interface/python/scripts/publish.sh` builds both, rehearses the install from the built artifacts, uploads hyperlux and then hyperdrone. Native components remain JIT-compiled on first use.

## Tests

Use the repository `.venv` with editable `interface/python/hyperlux` and `interface/python/hyperdrone[gym]` installed; point `CONTA_ROOT` at the local test assets.

```bash
HYPERLUX_BACKEND=GENERIC .venv/bin/python -m pytest interface/python/hyperdrone/tests -v
```

With the spy dependencies installed, run `HYPERDRONE_TEST_SPY=1 xvfb-run -a .venv/bin/python -m pytest interface/python/hyperdrone/tests/environment/test_spy.py` to check the native GUI lifecycle and exact simulation equivalence; omit `xvfb-run -a` when using a desktop display.

`tests/test_architecture.py` enforces the import DAG (dynamics never imports the environment); `tests/test_versions.py` checks the hyperlux pin against both package versions; `interface/python/scripts/test_sdist.sh` is the self-containment rot guard of the package pair.

`hyperlux.datasets.procthor.Train200()`, `hyperlux.datasets.hssd.Train125()` and `hyperlux.datasets.hssd.Val43()` list the header-pinned scene catalogs (see the hyperlux README); `dataset.references[:4]` is a valid `scenes` argument.

Camera profiles declare immutable output descriptors from `hyperlux.camera`; the hyperlux README lists them with dtypes, shapes and contracts, and `ImageObservation` from `hyperdrone` selects the policy's mounts among them.

```python
from hyperdrone import Config, HyperDrone
from hyperlux.camera import CameraSpec, RGB, Depth, Flow, FlowStatus
from hyperdrone.camera import ImageObservation

profiles = (
    CameraSpec("front", (RGB("float32"), Depth(), Flow(), FlowStatus())),
    CameraSpec("chase", (RGB("rgba8"),)),
    CameraSpec("down", (Depth(),)),
)
config = Config(cameras=profiles, observation=ImageObservation(cameras=("front",)))
env = HyperDrone(scenes, config=config)
env.reset()
env.render()
depth = env.outputs[Depth()].read()
print(depth.shape, env.outputs[Depth()].layout.cameras)
env.close()
```

Buffers contain only selected cameras, ordered by physical camera index; environment shapes are `(total_instances, selected_views, H, W[, channels])`, and layout camera identities pair each mount name with its agent index. Native metadata supplies dimensions and dtypes, including custom C++ specifications. Names and runtime poses/FOVs do not change compiled specializations; camera topology, observation selections, and output semantics do.

Depth, segmentation, normals, and flow use pixel centres at shutter close, independently of RGB anti-aliasing and motion blur. `env.render()` honors task frame cadence. Repeated reads share a capture, and flow follows due captures and resets per instance or scene change; the first/reset capture has zero motion.

`Config(dataset="glb")` (the default) uses the taxonomy authored in the GLBs;
`dataset="procthor"` requires each scene to declare `ai2thor-hab` when semantic output is enabled.
Dataset selection is compiled into the JIT module and also applies with `spec_header`.
In semantic mode, initialization validates all scenes and pooled drone/gate assets against one
taxonomy, then shares its immutable dictionary across them. Conflicting names or complete class
mappings fail initialization; IDs are preserved. Scene rotation does not change the dictionary.
Self-visible worlds label every drone placement with the reserved class
`hyperlux.AGENT_SEGMENTATION_CLASS` (`0xFFFFFFFD`, `"Agent"` in `segmentation_classes`); dataset
taxonomies may not use IDs from `0xFFFFFF00` upwards.

```python
config = Config(dataset="procthor", cameras=(CameraSpec("front", (RGB("float32"), Segmentation(labels="semantic"))),))
env = HyperDrone(scenes, config=config)
classes = env.segmentation_classes  # dict[int, str], an independent copy
taxonomy = env.segmentation_taxonomy  # "ai2thor-hab"
env.reset()
env.render()
labels = env.outputs[Segmentation(labels="semantic")].read()
agent_pixels = labels == hyperlux.AGENT_SEGMENTATION_CLASS
env.close()
```

These properties return `{}` and `""` when semantic output is disabled or all classes are
unnamed. Unnamed objects in a named taxonomy must use IDs in that dictionary. The reserved
background ID `0xFFFFFFFF` is excluded from the dictionary.

### Articulated assets

Joints, poses, placement batches and the shutter form of `set_pose` are hyperlux's (see its README, Articulated assets); hyperdrone rigs drone assets on top of them:

```python
drone = hyperlux.load_assembly(path)
hyperdrone.articulate(drone)
asset = pool.add_assembly(drone)
rotorcraft = hyperdrone.Rotorcraft(drone, rotor_positions=rotors, rotor_torque_directions=torques)
...
placement = renderer.spawn(0, asset)
renderer.set_pose(placement, hyperlux.Pose(body, scale=rotorcraft.scale), coordinates=rotorcraft.coordinates(phases))
renderer.set_pose(placement, hyperlux.Pose(body_open, scale=rotorcraft.scale), hyperlux.Pose(body_close, scale=rotorcraft.scale),
                  coordinates=rotorcraft.coordinates(phases_open), coordinates_close=rotorcraft.coordinates(phases_close))
renderer.update()
renderer.render()
```

`hyperdrone.articulate(assembly)` writes the joints of a body/prop_* drone asset, as the environment does for `drone_asset=`: prop p, the p-th `prop_*` part, spins about +z through its hub (the centre of its geometry) as coordinate p, every other part is fixed, and the asset carries nothing else. Articulate before `add_assembly`, which binds the joints. `hyperdrone.Rotorcraft(assembly, rotor_positions=None, rotor_torque_directions=None)` reads them and keeps the convention that drives the props: `prop_rotor[p]` is the rotor driving prop p, `prop_spin[p]` its direction, and `rotorcraft.coordinates(rotor_phases)` maps a `(num_rotors,)` or batched `(..., num_rotors)` array of rotor phases (radians) onto the joints. Without rotors prop p follows rotor p spinning per Quad-X (front-right CCW) and `scale` is 1. Given a dynamics model's rotor layout (`sim.parameters["dynamics"]["dynamics"]`), each hub pairs with its nearest rotor and spins against its reaction torque, so `sim.rotor_rpm()` supplies the speeds whose integral over the step advances the phases; `scale` fits the asset onto the rotors' wheelbase and `rotor_position_deviation` is the largest hub error as a fraction of the wheelbase (warned above 10%). `hyperdrone/examples/Renderer.ipynb` flies the x500 with spinning props.

## Stability

The release contract covers documented names, single-agent shapes and the documented environment variables. Multi-agent parameter/spawn shapes, observation layouts (including `agent<i>/` block prefixes), team-mean rewards and any-agent termination remain experimental. The shipped tasks are single-agent. `spec_header` constants and their shadowing form the stable extension tier; custom verb-overload wrappers are experimental and should pin the package version. The JIT key hashes the top-level header only; changes to included headers rebuild that tree in place.

Reproducibility is per seed, configuration, reset masks and call sequence on one host build (`-O3 -march=native -ffast-math`). RNG streams may change in minor releases. Environment IMU noise and shutter sampling still share the physics RNG. The `Train200`, `Train125` and `Val43` catalogs are immutable; cite their RLtools commit for paper datasets and identify the examples and golden scenes separately as fixtures.

Presets name platforms and default to a neutral MDP: zero reward and disabled termination. Configure the task through `parameters["dynamics"]["mdp"]`; `moving_gate` adds +10 for a pass and terminates on gate crashes. Preset and task names are stable. Privileged observations are ground truth, with row-major body-to-world rotation matrices. Raw views follow physical `(instance, agent, camera)` order; selected policy views are stacked along image height, and all cameras share resolution.
