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

The runner's `operations_generic.h` defines value evaluation and the single
collection loop after its phases. It remains freestanding. CPU, CUDA mux/direct,
PPO loop, and manual-phase consumers use the same definitions; CUDA overloads
must precede generic orchestration, which the CUDA entry header ensures.
Model and environment operations must precede collection templates. Environment
persistence overloads must likewise precede batch/loop persistence templates.

## Collection ownership

The PPO loop owns the models, environment, runner, dataset, and allocation of
collection state and scratch. The runner module owns collection scheduling and
generic actor/critic evaluation; it has no dependency on PPO. PPO supplies
`PPO_SPEC::COLLECTION_MODE` and retains GAE, bootstrap eligibility, and training.

```cpp
collect(device, dataset, runner, runner_buffer, environment,
        ppo.actor, actor_buffer, ppo.critic, value_state, value_buffer,
        rng, typename PPO_SPEC::COLLECTION_MODE{});
```

`on_policy_runner::ValueState<CRITIC, DATASET_SPEC>` contains the live critic
state. `ValueBuffer<CRITIC, DATASET_SPEC>` contains its bootstrap copy and model
scratch. Allocate/free both explicitly. Critic state is reset at the beginning
of every rollout, matching sequence replay during PPO training; it is not a
new checkpoint payload. Actor/environment state and their persistence are
unchanged. The old PPO `CollectionBuffer` and collection headers are removed.

Actor-only and actor-critic `collect` overloads delegate to the same loop
through `Mode<ActorOnly>` or `Mode<ActorCritic>`. `Sequential` preserves
time/environment axes in bulk value evaluation. Evaluation argument structures
hold references to caller-owned models, state, and buffers; they own no
storage. The individual phases and value operations remain independently callable.

With next-observation capture, each iteration evaluates the current observation
into the live critic state, evaluates the actor, steps and autoresets the
environment, then evaluates the captured pre-reset observation into a copy of
the critic state. The copy includes hidden states and step counters; it never
advances the live state. Critic evaluations retain `NoAutoResetMode`, including
the bootstrap at a sequence-length boundary. Without capture, value evaluation
retains the bulk path and `bootstrap_values` aliases the next value row.

## Consumers and decisions

| Consumers/use cases considered | Constraint and resulting decision |
| --- | --- |
| Generic CPU, MKL, Accelerate, OpenBLAS, and custom/freestanding devices | Plain generic overloads and freestanding helper headers; no required device identifier. Backend selection stays in mux headers. |
| CUDA independent Pendulum/L2F, scalar and recurrent policies, portable and cuRAND engines | Preserve fused phases, one RNG stream per instance, deterministic reset order, and memory residence. Share scalar transition/reset/observation logic without changing phase ordering. |
| CPU batches using a shared scalar RNG | Preserve all-parameters, all-states, and observation-channel ordering across instances. Do not replace composition with a per-instance fused loop. |
| Native HyperDrone World/MultiEnvironment, target-frame, moving-gate, localization, self-visible and multi-agent tasks | Preserve composed batch verbs, renderer caches, scene sharing, and stream ownership within environments. Accumulate pending reset masks until rendering consumes them. |
| Generic, Metal, OptiX, Vulkan, WebGPU rendering | Compute devices own their rendering device through the extension. Callers initialize the owned member explicitly; renderer libraries retain their backend lifecycle. |
| CUDA Adam/SGD, TD3, and renderer producer/physics kernels | Runtime devices may contain noncopyable extensions. Host launchers retain stream ownership; kernels receive CUDA tags by value and their data separately. |
| Legacy Pendulum training inside a CUDA kernel | The complete training loop needs a device-side logging context. Construct it inside the kernel and pass only the training-state pointer across the launch boundary. |
| MuJoCo Ant CPU training | Uses ordinary CPU collection. The former CPU-environment/CUDA-policy training, collection/throughput benchmarks, transfer buffers, and hybrid operations are removed. Their dedicated tests are removed with the API. |
| Shaped visual observations | Native collection retains the observation shape and precision through actor and critic evaluation. |
| Feedforward PPO, recurrent actors with feedforward/recurrent critics, asymmetric observations, all bootstrap/ignore-termination combinations | PPO selects collection capabilities. Preserve current/next observation ordering, separate actor/critic states, whole-state bootstrap copies, and the existing streaming/bulk evaluation paths. |
| Normalization warmup, actor-only collection benchmarks and parity tests | Use actor-only mode in the same collection loop, preserving observation capture and RNG order. |
| Manual phases, external actions, visual PPO and RAPTOR imitation | Keep public prologue/interlude/epilogue and value operations independently callable. |
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
- A CUDA rendering extension owns its host device by value. Use
  `auto& host = compute.rendering` and initialize both devices before world or
  multi-environment lifecycle calls. Free environments and their shared renderer
  library before releasing host resources. Worlds do not initialize devices.
  Host operations take devices by reference; CUDA kernels receive
  `devices::cuda::TAG<DEVICE, true>` by value, without copying runtime extensions.
  `tests/src/nn/cuda/device_extension.cu` checks owned noncopyable host storage
  and CPU/CUDA Adam/SGD parity through updates and resets, with and without
  master parameters.
- Pending render masks accumulate across partial resets and ordinary step
  invalidations. Explicit renders merge their reset mask with pending resets.
  After consumption, masks clear; repeated observations reuse the cached frame.
  CUDA consumption clears masks after the renderer stream finishes using them.
- HyperDrone training accumulates episode length and return from dataset
  transitions in local consumer storage across rollouts. Administrative scene
  resets clear partial accumulations and are logged separately from completed
  episodes. Reporting adds no storage or work to the runner.

Continuous observation normalization updates occur after training on each
rollout, keeping the rollout's policy/value normalization fixed through that
update. This training behavior predates collection consolidation; collection
throughput results must not be read as end-to-end training comparisons.

See [CUDA validation and benchmark details](cuda/README.md). Regression tests
cover devices without identifiers, direct/mux/PPO loop entry headers,
shaped observations, checkpoint restoration,
partial reset accumulation, and native rendering.

`tests/src/rl/algorithms/ppo/collection.{h,cpp}` and `cuda/collection.cu` exercise
the complete collector with real GRU actors, recurrent/feedforward critics, and analytical values. They
cover staggered termination/time limits, repeated rollouts, all bootstrap flag
combinations, final-step boundaries, one-step rollouts, actor/critic sequence
replay, and preservation of live hidden states and counters. GTest XML records
rollout/RNG digests; the recurrent loop tests also record serialized training
state digests after updates. CPU/CUDA rollout digests and CPU training-state
digests were compared with the implementation before consolidation. Training
comparisons compile identical test source with the same compiler flags against
both header versions; all eight serialized snapshots (two updates for each
critic/bootstrap combination) matched byte for byte.
