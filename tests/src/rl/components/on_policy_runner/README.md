# On-policy runner dispatch and ownership

The runner exposes unqualified, device-first operations in `rl_tools`. Generic
operations accept `DEVICE&` and require the operations they call, without a
`DEVICE_ID`, backend predicate, or backend registry. Concrete CUDA overloads
specialize complete phases for `batch::Independent`; native batches retain
composed environment operations.

## Include entry points

After device, model, and scalar/native environment operations, include
`on_policy_runner/operations_cpu_mux.h`. The existing
`RL_TOOLS_OPERATIONS_CPU_MUX_INCLUDE_CUDA` option selects the CUDA implementation.
`operations_cpu.h` directly includes generic operations; it does not select a
backend. Direct CUDA consumers can include `operations_cuda.h` instead.

CUDA definitions precede generic orchestration within the CUDA header. Small
freestanding helpers are defined first in `operations_generic_common.h` and
`batch/operations_generic_common.h`. This makes the overload set visible when
`collect`, generic `epilogue`, and generic `reset` are defined. Including generic
orchestration first and adding a backend later is not a supported entry order.

PPO's `operations_collection.h` is a convenience entry point which loads the
runner mux before collection. `operations_generic_collection.h` and generic loop
headers remain freestanding; callers selecting a backend include the runner mux
or the loop's CUDA entry header first. Environment persistence overloads must
likewise precede batch/loop persistence templates.

## Consumers and decisions

| Consumers/use cases considered | Constraint and resulting decision |
| --- | --- |
| Generic CPU, MKL, Accelerate, OpenBLAS, and custom/freestanding devices | Plain generic overloads and freestanding helper headers; no required device identifier. Backend selection stays in mux headers. |
| CUDA independent Pendulum/L2F, scalar and recurrent policies, portable and cuRAND engines | Preserve fused phases, one RNG stream per instance, deterministic reset order, and memory residence. Share scalar transition/reset/observation logic without changing phase ordering. |
| CPU batches using a shared scalar RNG | Preserve all-parameters, all-states, and observation-channel ordering across instances. Do not replace composition with a per-instance fused loop. |
| Native HyperDrone World/MultiEnvironment, target-frame, moving-gate, localization, self-visible and multi-agent tasks | Preserve composed batch verbs, renderer caches, scene sharing, and stream ownership within environments. Accumulate pending reset masks until rendering consumes them. |
| Generic, Metal, OptiX, Vulkan, WebGPU rendering | Compute devices can reference a separately owned rendering device. Backend lifecycle remains with the caller and renderer library. |
| Hybrid MuJoCo and shaped visual observations | Share transfer, canonical input reshaping, rollout-mode evaluation, and action sampling. Transfer observations in `SPEC::OBSERVATION_T`; retain separate PPO critic orchestration. Hybrid collection remains feedforward. |
| Feedforward/recurrent PPO, all bootstrap/ignore-termination combinations | Model-level sequence reset modes preserve the old episode's critic state for terminal observations. Legacy GRU mode names are aliases to the same types. |
| Manual phases, external actions, visual PPO and RAPTOR imitation | Keep public prologue/interlude/epilogue operations independently callable; no callback framework or extra collector. |
| Loop checkpoints, curriculum updates, evaluation, Python/Gym consumers | Environments live outside runners. Independent-batch persistence delegates mutable instances to their environment serializers; empty instances have no payload. No implicit serialization of arbitrary environment objects. |
| Training diagnostics, timing and benchmarks | Reporting stays in training consumers; runner storage is limited to collection. Compare eager and graph execution separately with matching trajectory and RNG hashes before interpreting timings. |

## Migration repairs

These changes are separate from overload dispatch:

- Loop checkpoints save `environment` beside `on_policy_runner`. L2F serializers
  save parameter values; visual L2F also saves its dynamics and target-mode
  setting, preserving the caller's renderer, annotations and staging allocation.
  MuJoCo Ant uses MuJoCo's model file representation and named simulation-state
  fields, including warm-start state and the cached position used by reward.
  Simulator/renderer addresses are never checkpoint payloads. Both tar and HDF5
  use the same environment schema. Binary parameter payloads follow the existing
  checkpoint requirement of matching parameter types/build layout.
- Mutable third-party environments provide `save`/`load` operations before the
  loop persistence header. Missing serializers fail to compile. Loop-state
  checkpoints from before environment ownership moved must migrate their old
  environment payload to the new schema; policy checkpoints are unaffected.
- Visual L2F training configures environment default scene translations before
  loop initialization, so initial sampling and later resets use those defaults.
- A CUDA rendering extension holds a non-owning host-device pointer. Initialize
  the host device and attach it with `compute.rendering = &host` before world or
  multi-environment lifecycle calls. Keep it alive until environments and their
  shared renderer library are freed. Worlds no longer initialize hidden devices.
- Pending render masks accumulate across partial resets and ordinary step
  invalidations. Explicit renders merge their reset mask with pending resets.
  After consumption, masks clear; repeated observations reuse the cached frame.
  CUDA consumption clears masks after the renderer stream finishes using them.
- HyperDrone training accumulates episode length and return from dataset
  transitions in local consumer storage across rollouts. Administrative scene
  resets clear partial accumulations and are logged separately from completed
  episodes. Reporting adds no storage or work to the runner.

Two behavioral changes in `c2d04a51..HEAD` are retained independently of this
refactor: hybrid PPO evaluates its critic on the environment device, enabling
terminal-value capture with the current critic state; continuous observation
normalization updates occur after training on each rollout, keeping the
rollout's policy/value normalization fixed through that update. These affect
execution placement and training semantics, so collection throughput results
must not be read as end-to-end training comparisons.

See [CUDA validation and benchmark details](cuda/README.md). Regression tests
cover devices without identifiers, direct/mux/PPO collection entry headers,
mixed observation precision, shaped hybrid evaluation, checkpoint restoration,
partial reset accumulation, and native rendering.
