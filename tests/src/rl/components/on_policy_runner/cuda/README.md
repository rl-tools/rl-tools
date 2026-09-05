# CUDA rollout phase dispatch

`collect` has one implementation. Its unqualified `prologue`, `interlude`, and
`epilogue` calls select device/environment overloads in `namespace rl_tools`.
Include the runner's `operations_cpu.h` with the CUDA CPU-mux option, or include
`operations_cuda.h` before its generic definitions. CUDA overload declarations
must be visible when those definitions are parsed.

The specialization boundary is a complete runner phase, not a batch verb:

| Consumer | Execution and preserved contract |
| --- | --- |
| CUDA `batch::Independent` | One initial-observation kernel per rollout; one action-sampling and one complete transition kernel per step, excluding policy evaluation/reset. Full and masked external resets each use one kernel. |
| Native batches, including HyperDrone | Composed batch operations; scene sharing, rendering, stream coordination, and scene diversity remain environment responsibilities. |
| CPU and hybrid CPU-environment/CUDA-policy collection | Original composed operation ordering, including shared-RNG consumption order. |
| Recurrent actors | Reset masks apply before policy evaluation; CUDA-resident hidden state and step counters stay device-resident. |
| Manual collection and imitation | The same public phases accept externally produced actions without a second collector implementation. |
| PPO, persistence, reward diagnostics | The dataset stores training transitions. The runner keeps policy state, environment parameters/states, a reset mask, and the compile-time time-limit counter. `Buffer::next_states` remains the transition state before autoreset. |

Both execution paths share transition recording. The fused transition performs
`step`, reward conversion, termination, time-limit checking, conditional reset,
and next-row observation in that order. Each independent instance uses its own
RNG stream. Episode reports and training progress belong to consumers; the
runner and dataset do not accumulate returns or store completion events.

This does not add a runtime fast-path flag, a second `collect`, hidden state
buffer aliasing, or production CUDA Graph ownership. Callers of the former
`rl::components::on_policy_runner::{prologue,interlude,epilogue,reset,sample_actions}`
operations should import their `rl_tools` counterparts and call them unqualified.

## Correctness tests

Build the following targets, then run the focused CTest selection:

```sh
cmake --build build --target \
  test_rl_components_on_policy_runner_fused_cuda \
  test_rl_components_on_policy_runner_fused_cuda_direct \
  test_rl_components_on_policy_runner_fused_recurrent_cuda \
  test_nn_layers_gru_reset_cuda \
  test_nn_layers_gru_reset_cuda_helper_first -j5
ctest --test-dir build \
  -R 'RL_TOOLS_ON_POLICY_RUNNER_FUSED|RL_TOOLS_NN_LAYERS_GRU_RESET_CUDA' \
  --output-on-failure --timeout 20 -j1
```

`fused.cu` compares the selected CUDA phases against explicitly selected generic
phases using the same CUDA batch verbs. It checks every step's observations,
actions, rewards, termination/reset flags, time-limit counters, parameters, current and
pre-reset states, mutable environment data, and RNG state. Coverage includes
1/37 instances, asymmetric and mixed-precision observation storage, multi-agent
actions, portable/cuRAND RNGs, compile-time limits of 0/1/3, strided no/some/all reset masks,
and full external resets. Both supported header entry paths are compiled.

`fused_recurrent.cu` compares complete collection against composed phases with
GRU policies on Pendulum and L2F, with continuing and forced-rollout boundaries.
The separate GRU tests verify matrix masks and views of tensor masks, partial blocks, strided
masks, and default/no-auto-reset counter boundaries against CPU and explicit
expected values. Run these executables with Compute Sanitizer's `memcheck` and
`--leak-check full` when changing the kernels.

## Throughput benchmark

```sh
cmake --build build --target \
  benchmark_rl_components_on_policy_runner_cuda_64 \
  benchmark_rl_components_on_policy_runner_cuda_1024 -j5
./build/tests/src/rl/components/on_policy_runner/cuda/benchmark_rl_components_on_policy_runner_cuda_64
./build/tests/src/rl/components/on_policy_runner/cuda/benchmark_rl_components_on_policy_runner_cuda_1024
```

The benchmark uses Pendulum, a float MLP with two width-32 hidden layers,
64 steps per rollout, fixed weight/RNG seeds, one warmup rollout, and five
synchronized samples of ten rollouts. It reports the median, range, reward sum,
trajectory hash, and metra metrics. Compilation, initialization, and result
copies are outside the timed region. Benchmarks are excluded from the default
build and CTest; run them serially after builds/tests finish.

Optional `--graph` captures collection solely as a diagnostic for launch
overhead. Compare graph execution against graph execution, never against an
eager baseline. For revision comparisons preserve the exact initialization,
workload, compiler flags, backend, and device. Require matching hashes for
this nonterminating workload; use the semantic tests above for episode
boundaries, whose representation changed from the common base.

The CUDA profiler capture range covers 50 rollouts / 3,200 batched steps.
For the width-32, 64-instance workload it should contain 3,200 action-sampling
kernels, 3,200 fused transition kernels, and 50 prologue kernels. Policy kernels
are additional. No device-to-device copies or intermediate reset/observation
kernels should occur in this range.

## Recovery measurements (2026-09-04)

RTX 5060 Ti, CUDA 13.2.78, GCC 15.2, C++17, `-O2 -DNDEBUG`, `sm_120`;
no fast-math options. The common base is `c2d04a5147db0fab18b3fbb38452df242d34e80b`.
The unfused comparison is `ed86ee319` with only the CUDA build/reset-header fix.
Each entry below is the median of five separate process runs, each reporting
its own five-sample median. Runs were serial and interleaved, without concurrent
builds or tests. Rates are millions of transitions per second.

| Workload | Common base | Unfused opr1 | Fused opr1 | Fused / base |
| --- | ---: | ---: | ---: | ---: |
| CUDA eager, 64 instances, width 32 | 1.995 | 1.223 | 2.067 | 103.6% |
| CUDA eager, 1,024 instances, width 32 | 31.691 | 19.826 | 30.478 | 96.2% |
| CUDA eager, 64 instances, width 128 | 2.028 | — | 1.947 | 96.0% |
| CUDA Graph, 64 instances, width 32 | 3.984 | 3.060 | 4.071 | 102.2% |
| CPU generic, 64 instances, width 32 | 2.153 | 2.150 | 2.186 | 101.5% |

All compared runs have identical trajectory hashes within each workload:
CUDA width-32/64 `3389130740161743012`, width-32/1,024
`1863133734530264491`, width-128/64 `13694792184588688955`, and CPU
`4609462277508597651`. CPU and CUDA are not required to be bitwise identical
to each other. Eager timings fluctuate more than graph timings; these results
establish recovery to within 4% of the base, not a universal speedup claim.

Nsight confirms 41,650 kernel launches in the captured region, versus 60,900
for unfused opr1 and 41,700 for the base. The 35,200 actor kernels are unchanged;
the fused runner has no intermediate device copies. Rollout kernels per step
fall from eight to two, plus the one-per-rollout prologue.

Validation: all 28 focused cases passed through the real project CMake build;
the additional CPU/CUDA runner, PPO, PPO persistence, end-to-end PPO, and CPU
HyperDrone regression executables also passed. Compute Sanitizer reported zero
errors and leaks for phase equivalence, Pendulum/L2F recurrent collection, and
the GRU reset/counter cases. The HyperDrone determinism test compares actual
state fields rather than uninitialized struct padding.

Scope: these are collection-throughput results, not end-to-end training or
rendering throughput. L2F is covered for correctness, not throughput. Native
CUDA HyperDrone/OptiX training was not validated: the available generic-renderer
harness cannot supply its CUDA renderer stream, and configuring all project
targets encountered the existing OpenVINS/OpenCV prerequisite. The focused
project build disabled executable targets and raytracing; the CPU HyperDrone
tests used a separate generic-renderer harness. This work does not claim a
successful complete build or complete test suite.
