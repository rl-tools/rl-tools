# RLtools Development Guide

Before starting any work, read `/infra/AGENTS.md` and follow its instructions. The following project-specific instructions below take precedence if they conflict.


## Building & Testing

### Python Packages

Always use the `.venv` virtual environment. If it does not exist, create it with `python3 -m venv .venv`.

### CMake Configuration

The build fully auto-configures: every feature is enabled by default when its prerequisites are available, and a configure summary ("RLtools configure summary:") reports each feature as ON/OFF with the reason.

Ubuntu (x86)
```bash
CUDACXX=/usr/local/cuda-13.1/bin/nvcc cmake -B build -DCMAKE_BUILD_TYPE=Release
```

macOS
```
cmake -B build -DCMAKE_BUILD_TYPE=Release
```

Opt-outs follow the `RL_TOOLS_XXX_DISABLE_YYY=ON` pattern: `RL_TOOLS_DISABLE_{TARGETS,TESTS,EXPERIMENTAL,TAR,GIT_DIFF,FAST_MATH,JSON,HDF5,ZLIB,TENSORBOARD,CLI11}`, `RL_TOOLS_BACKEND_DISABLE_{BLAS,MKL,DNNL,CUDA,CUDNN}`, `RL_TOOLS_NUMERIC_TYPES_DISABLE_BF16`, `RL_TOOLS_RL_ENVIRONMENTS_DISABLE_MUJOCO`, `RL_TOOLS_TESTS_DISABLE_EIGEN`, `RL_TOOLS_RENDERING_DISABLE_RAYTRACING`, `RL_TOOLS_RENDERING_RAYTRACING_DISABLE_{OPTIX,METAL,VULKAN,WEBGPU}`. The pre-auto-configure `*_ENABLE_*` flags are gone: a legacy cache entry that agrees with the new defaults is scrubbed with a STATUS note, one that would change behavior fails with the replacement flag (`cmake/legacy_flags.cmake`). The `RL_TOOLS_ENABLE_*` / `RL_TOOLS_BACKEND_ENABLE_*` spellings are detection *outputs* (plain variables + compile definitions), not inputs — passing them as `-D` flags warns and is ignored.

**Environment variables:**
- `CUDACXX=/usr/local/cuda-13.1/bin/nvcc` — required for CUDA test targets

**Raytracing backend selection:** `-DRL_TOOLS_RENDERING_RAYTRACING_BACKEND=AUTO|OPTIX|METAL|VULKAN|WEBGPU|GENERIC`. `AUTO` (the default) probes availability in preference order `OPTIX > METAL > VULKAN > WEBGPU > GENERIC` (OptiX needs a working CUDA toolchain and git for the OWL fetch, Metal needs macOS, Vulkan needs headers + `glslangValidator`, WebGPU needs linux-x86_64 or macOS arm64 and network access; all backends use Assimp v6.0.4 via FetchContent with a SHA-256-pinned release archive (CMake 3.22+) — offline builds need the cached archive or explicitly supplied sources). An explicitly selected backend is hard-required and fails the configure if its prerequisites are missing. The `VULKAN` backend uses compute shaders with `VK_KHR_ray_query` (structural mirror of the Metal backend, GLSL compiled to SPIR-V at build time via `glslangValidator`); it needs the Vulkan dev headers and `glslang-tools` installed, runs headless, and works on Mesa's lavapipe CPU driver (`VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.json`) for GPU-less machines/CI. `RL_TOOLS_VULKAN_DEVICE_INDEX` overrides device selection. The `WEBGPU` backend codes strictly against the standard `webgpu.h`/WGSL (browser/WASM is the eventual target): ray tracing is not in the WebGPU standard, so it reuses the generic backend's deterministic CPU BVH build and traverses it in WGSL compute; the runtime is a hash-pinned wgpu-native prebuilt fetched via FetchContent (no Rust toolchain, Linux-x86_64 and macOS-arm64 wired up so far), which sits on Vulkan (Metal on macOS) and therefore also runs headless and on lavapipe. `RL_TOOLS_WEBGPU_DEVICE_INDEX` overrides adapter selection. `RL_TOOLS_WEBGPU_BVH=median` switches the BLAS/TLAS build back to the shared generic median-split baseline (default: deterministic binned SAH, traversed near-first).

### Running Tests

```bash
make -j5
ctest --timeout 20 -j5
```
Use no more than 5 parallel jobs for builds or tests.
Note: the `--timeout` is essential for debugging because some tests require many minutes to finish. We want to debug efficiently, hence we only fully run the long running tests after all the short running tests pass.

Training/RL tests (SAC, TD3, PPO, MLP full training) legitimately take longer than 20 seconds; use a higher timeout (e.g. `--timeout 300`) to run them to completion.

### Auto-Generated Test Data Headers

Several tests use a two-step workflow: a **generation** test produces a header file under `tests/data/`, and a **compile** test includes that header to verify round-trip correctness. If the internal model API changes (e.g. `nn_models::sequential` module structure, persist code format), the checked-in headers become stale and the compile tests fail.

**Stale-header symptoms:** errors referencing `tuple_element`, `get<0>(m.content)`, missing `CONTENT`/`NEXT_CARRIER_MODULE` types, or wrong `Module<>` nesting in files under `tests/data/`.

**Regeneration procedure** (run from the repo root):

| Generated header | Generator target |
|---|---|
| `tests/data/test_nn_layers_gru_persist_code.h` | `test_nn_layers_gru_persist_code` |
| `tests/data/nn_models_multi_agent_wrapper_persist_code.h` | `test_nn_models_multi_agent_wrapper_persist_code` |
| `tests/data/nn_models_multi_agent_wrapper_persist_code_backward.h` | (same target, BACKWARD test) |
| `tests/data/nn_models_multi_agent_wrapper_persist_code_forward.h` | (same target, FORWARD test) |
| `tests/data/test_inference_executor_policy.h` | `test_inference_l2f_c_interface_write_policy` |

```bash
# Build and run all generators (from repo root so output paths resolve correctly)
make -j5 test_nn_layers_gru_persist_code \
         test_nn_models_multi_agent_wrapper_persist_code \
         test_inference_l2f_c_interface_write_policy

cd /path/to/repo   # generators write relative to cwd
./build/tests/src/nn/layers/gru/test_nn_layers_gru_persist_code
./build/tests/src/nn_models/multi_agent_wrapper/test_nn_models_multi_agent_wrapper_persist_code
./build/tests/src/inference/executor/test_inference_l2f_c_interface_write_policy
```

After regeneration, rebuild the compile tests before re-running `ctest`.

### Raytracing Golden Renderings

The corpus version and scene are defined in `tests/src/rendering/raytracing/golden_manifest.h`; comparators skip a missing corpus unless `RL_TOOLS_REQUIRE_RAYTRACING_GOLDENS=ON`, and reject other versions. `GOLDEN_LEDGER.md` in that directory records the generating RLtools commit, test-data commit and reason for each generation. The source commit pins the entire recipe, including the SDK/runtime archives in `cmake/dependencies/wasm.cmake`. Regenerate only for deliberate output changes or new cases, batch pending changes into one version, and trace unexplained deviations before regenerating.

The reference is the generic renderer and Assimp loader compiled as WASI (`tests/src/rendering/raytracing/wasm/`). Its build and runtime profile must reproduce every stored bit, including signed zero, miss distances, non-VALID flow and raw normals. Native backends retain measured tolerances in `golden_comparison.cpp`; comparisons log metrics to metra. `golden-wasm.yml` qualifies relocated builds and cross-host module/output identity; it does not replace comparisons against the published corpus. A toolchain or runtime change that preserves all output bits does not require a corpus version bump.

Configure downloads the pinned SDK and runtime into `.dependencies` on supported hosts and reuses the native build's Assimp, stb and json sources. Opt out with `RL_TOOLS_RENDERING_RAYTRACING_DISABLE_WASM_GOLDENS`; `RL_TOOLS_OFFLINE_BUILD` reuses cached dependencies.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target rendering_raytracing_golden_wasm -j5
./build/tests/src/rendering/raytracing/wasm/generate_goldens
```

Building compiles the module and configures the `generate_goldens` script, which resolves the pinned scene with the `conta` tool and runs Wasmtime. For a direct Wasmtime command, see [wasm/README.md](tests/src/rendering/raytracing/wasm/README.md).

The standalone build (`cmake -S tests/src/rendering/raytracing/wasm -B build-wasm -DCMAKE_BUILD_TYPE=Release`) fetches and selects the same SDK and generates `build-wasm/generate_goldens`; put `conta` on `PATH` or set `CONTA` to its executable, and `RL_TOOLS_DEPENDENCY_SOURCES` to a native build's dependency directory to reuse its sources. `CONTA_ROOT` selects a local conta checkout; `SCENE` supplies a local copy of the pinned scene. `--output-dir DIR` selects a corpus destination; `--dump-dir DIR` writes raw outputs for the native WASM comparator. Native generators produce candidates with `--output-dir` or `--dump-dir`.

The corpus lives under `tests/data/rendering/raytracing/goldens/procthor-train-1/`; `golden_cases.h` defines its cases and `golden_layout.h` its paths. Each generation adds `golden-vNN.*` files. Binary outputs, including raw normals, live in `data/` beneath each output directory; review PNGs remain alongside it. Comparators write ignored `render-<backend>.png` and `diff-*.png` review artifacts. Download the corpus with `./tests/src/rendering/raytracing/download_data.sh`, which has its own test-data revision pin.

Run comparisons with `ctest --test-dir build -L '^procthor-golden$' --output-on-failure --timeout 300 -j5`. Backend sub-labels are `procthor-golden-<backend>` (`generic` for CPU); the WASM fixture renders its dumps once per run. Synthetic overlay contract tests use the `overlay` label and require no corpus.

Publication spans two repositories: generate from the exact RLtools source commit recorded in the ledger, review the goldens and comparisons, and require successful cross-host qualification. A maintainer commits and publishes the test-data generation, then records its commit in the ledger and the raytracing downloader; update the general downloader only if its consumers should move too. The corpus pin enables the CI comparison gate. Without a published pin, CI runs qualification and explicitly defers corpus comparisons.

### HDF5 Test Data

Some tests (`NN_LAYERS_RESNET_CUDA`, sequential persist tests) load `.h5` files from `tests/data/`. The path is set via `RL_TOOLS_TEST_DATA_PATH` (auto-detected by CMake). If these tests fail with "Object not found" HDF5 errors, check that dataset paths in the test source match the actual HDF5 group structure (inspect with `h5dump -H` or `python3 -c "import h5py; ..."`).


## Idiomatic RLtools Architecture Principles

Before making code changes, every coding agent must study RLtools' philosophies, principles, and best practices, primarily through relevant existing implementations in `./include`, and follow the RLtools way when choosing among C++ approaches. If the existing code does not establish a clear best practice for a proposed approach, explain the ambiguity and alternatives to the user and let the user decide before implementing that part.

1. Use free-function, device-first APIs (`malloc/init/step/free/copy/evaluate/train/...`) in `namespace rl_tools`; integrate new components by providing these operations.
2. Keep the architecture split strict: `Config` (compile-time wiring/params), `State` (runtime storage), operations headers (lifecycle and algorithm behavior).
3. Compose models at compile time (`nn_models::sequential::Module<...>`, `Build<...>`, capabilities), and enforce shape/type contracts with `static_assert`.
4. Route math/container/nn behavior through RLtools device-dispatched operations; select backend through mux headers, not ad-hoc backend-specific code in targets.
5. Keep memory lifecycle explicit and symmetric: allocation in `malloc`, initialization in `init`, teardown in `free`, with recursive handling for nested members.
6. Separate concerns in data layout: parameters in models, transient tensors/scratch in `Buffer`, recurrent/rollout runtime data in `State`.
7. Control precision and numeric semantics through `numeric_types::Policy` and category overrides, not hardcoded scalar assumptions in operations.
8. Use `Mode<...>` tag dispatch for rollout/evaluation/variant behavior instead of boolean-driven control flow.
9. Extend training via composable loop-step wrappers (`timing`, `evaluation`, `checkpoint`, `save_trajectories`, `nn_analytics`) rather than bloating core loop logic.
10. Keep target wiring explicit near the top (`DEVICE`, `TYPE_POLICY`, `TI`, `RNG`, `DYNAMIC_ALLOCATION`, loop aliases); avoid scattering macro-driven type decisions.
11. Treat include order as architectural: respect group/mux layering and prefer high-level mux includes in targets.
12. Keep reusable algorithmic logic in `include/`; keep `src/` focused on build-matrix wiring, target selection, and executable glue.
13. `operations_generic.h` must be strictly freestanding: no C++ standard library includes; depend only on RLtools abstractions for maximal platform/compiler portability. Code that depends on non-freestanding types (std::vector, std::string, std::mutex, File I/O, etc.) belongs in `operations_cpu.h` — never use `#ifdef` guards to conditionally include non-freestanding code in `operations_generic.h`.
14. Never redefine or provide fallbacks for `RL_TOOLS_NAMESPACE_WRAPPER_START/END` or forward-declare `Tensor`/`Matrix` — fix missing symbols by including `rl_tools.h`, `tensor.h`, `matrix.h` etc. in the correct order instead.
15. C++17 is the maximum compatibility standard; post-C++17 language/library features are not allowed in shared RLtools code.
16. For environment operation headers, prefer backend-specific naming such as `operations_cpu.h` (with optional thin compatibility wrappers like `operations.h`), and keep environment types/operations under `rl_tools::...` namespaces instead of top-level project-specific namespaces.
16. Keep public `include/` headers focused on reusable API/types/dispatch; place build artifacts and backend implementation translation units (e.g. `.cu` device programs, heavy third-party integration code) under `src/` and wire them via CMake.
17. When backend code requires generated symbols (e.g. embedded PTX blobs), use project-unique symbol names and expose them through namespaced C++ accessors instead of depending on ambiguous global names at call sites.
18. Preserve benchmarking comparability during refactors: keep initialization and workload semantics deterministic when measuring performance across revisions, and avoid mixing behavioral changes with structural/API migrations in the same benchmark comparison.
19. In hot paths, keep RLtools API boundaries explicit but minimize per-element abstraction overhead (prefer contiguous-buffer iteration patterns where appropriate) so architectural cleanup does not unintentionally regress throughput.
20. As an agent NEVER stash or commit anything. Git is read-only for you. 
21. We don't endorse comment noise. Comments signal two things: 1) complex/misleading (first thought) code and 2) important code. Only if both are the case should you add a comment. To calibrate this: There should be a comment for every few hundred lines of code.
22. RLtools is deterministic given a fixed seed. NEVER use atomic operations (e.g. when using CUDA)
23. The tar and hdf5 representations of a model should have a lossless bijective mapping
24. The raytracing renderer operates in the same FLU frame as L2F: camera principal axis = +X (forward), image horizontal axis = Y (left-right), image vertical axis = Z (up-down). GLB meshes (Y-up) are swizzled to FLU at load time so the entire pipeline uses a single coordinate frame.
25. L2F state/observation components are composed innermost-first following their declaration order in `multirotor.h`; operation overloads (`json`, `from_json`, `post_integration`, etc.) in `operations_cpu.h` must follow the same order so that each overload is visible when the next outer component calls it.
26. Extrack-style checkpoints store examples at `example/inputs/0` + `example/outputs/0`; parallel models use `example/inputs/{0,1,…}` one per branch.
  - Every saved example tensor starts with `[TIME_STEP, BATCH, …features]` — axis 0 is T (1 for non-recurrent), axis 1 is B; producers (e.g. `rl/loop/steps/checkpoint/operations_cpu.h`) already assume this via `Replace<ACTOR::INPUT_SHAPE, BATCH_SIZE, 1>`.
  - Parallel `branch_N/input_shape` attribute and the saved `example/inputs/N` dataset must agree in rank and axis order; producers that build example tensors for extrack-compatible checkpoints must reshape to the canonical rank (e.g. `reshape_row_major` with a `[1, N_EXAMPLES, …]` shape) before save.
  - Dyn leaf shape propagation (`dyn/operations_generic.h::compute_leaf_output_shape`) preserves leading batch/time dims to mirror the real nn_models layers: Conv2D/MaxPool2D replace only the last 3 dims; Flatten/AvgPool2D collapse last 3 to 1 (rank decreases by 2); never hardcode output rank in new leaf ops.
27. The raytracing global instance-id layout (scene instances `[0,S)`, overlay `o` slot `s` at `S + o*MAX_OVERLAY_INSTANCES + s`, pool objects counted after scene objects in registration order) is deterministic, cross-backend API surface consumed by segmentation output: identical scene + spec + verb-call sequence must yield identical ids across runs and across generic/Metal/OptiX/Vulkan/WebGPU. `segmentation_object` in `operations_cpu_common.h` is the single owner of the layout; slot allocation is deterministic first-fit (`detail::first_fit_slot`) — never replace with order-dependent or non-deterministic allocation, and never reorder the all_objects/instance registration sequences without treating it as a breaking change.
28. Extrack experiment paths follow `{TIME}/{COMMIT}_{NAME}_{POPULATION}/{CONFIG}/{SEED}` where `TIME` is a lexicographically-sortable timestamp like `2024-05-26_06-26-52` (e.g. from `date '+%Y-%m-%d_%H-%M-%S'`) — never substitute human-readable labels for `TIME` at the experiment root; `POPULATION` is the underscore-separated list of varied keys and `CONFIG` is the underscore-separated list of their values in the same order and arity (e.g. `algorithm_environment` ↔ `sac_pendulum-v1` decodes as `{algorithm: sac, environment: pendulum-v1}`), so individual values must avoid `_` (use `-` instead) to keep the positional zip unambiguous.
29. metra (`include/metra/metra.h`, server in `tools/metra/`) is the regression-metrics store: log any quantitative measure with the one-liner `metra::log("target/metric", value)` (scalar, `std::vector<double>`, or `log_raw` for arbitrary JSON; commit, commit time, and a per-process run id are attached automatically; no-op unless `METRA_URL` is set, so calls stay in the code) — almost every target should log something in one way or another, especially timings, learning curves, and final average returns.
30. Keep the batch-first `OnPolicyRunner<on_policy_runner::Specification<...>>` limited to rollout collection. It owns policy state, environment parameters/states, a reset mask, and the counter needed to enforce the compile-time `STEP_LIMIT`; `Buffer` holds transient next states/actions/rewards/termination flags, and `Dataset` holds observations and training transitions. Do not add episode returns, completion records/reasons, reporting summaries, or training progress counters to the runner. `collect` runs `prologue` once, then `interlude` and `epilogue` per step, with same-step autoreset; when `Specification::COLLECT_NEXT_OBSERVATIONS` is enabled, capture the actual next privileged observation in `Buffer::next_observations_privileged` before resetting. PPO enables capture when `PPO_PARAMETERS::BOOTSTRAP_TRUNCATIONS` (default true) or `IGNORE_TERMINATION` is enabled, consumes it on every step using the old episode’s critic state, and stores `Dataset::bootstrap_values` per transition. With both flags false, capture and its storage are omitted, PPO evaluates rollout values in bulk, and `bootstrap_values` aliases the next row of `all_values`; GAE treats reset boundaries as terminal while retaining bootstrapping at continuing rollout cutoffs. GAE cuts its trace at resets; true termination suppresses bootstrapping unless explicitly ignored; `terminated` implies `truncated`. Between rollouts use immediate `reset(device, runner, environment[, mask], rng)`. Rendering stays inside visual environment operations. Scalar environments use `rl/environments/batch/`; recurrent policies consume `runner.reset` through `Mode<mode::sequential::ResetMask>`. Training loops own progress and reporting.
31. For every architecture proposal, enumerate all consumers and use cases considered and explicitly identify which subset constrains each architecture decision. Put this accounting in the proposal plan, or in the implementation summary when no plan preceded the work.
32. Define RLtools operation overloads in `namespace rl_tools` and call them unqualified so ordinary overload resolution and argument-type dispatch select the implementation. Namespace-qualify types, not operation calls, unless qualification is required to resolve a genuine ambiguity.
33. Always use braces for control-flow bodies (`if`, `if constexpr`, `else`, `for`, `while`, and `do`), even for a single statement; put the body on separate, indented lines.
34. In most cases, err on the side of hardcoded (compile-time) defaults instead of a slurry of CLI args: CLI args shine for long-term stable targets that can rely on a user learning curve and whose code the user does not change often, but most RLtools targets are configured at compile-time, so we expect almost all of their parameters to be compile-time specified. CLI args also tend to hurt reproducibility because they are not sustained in the code history, so prefer setting parameters by modifying the defaults such that the hill-climbed config lives in git — short-term hyperparameter sweeps are a reasonable exception, but they should generally end in either discarding the changes or adopting them as the new defaults depending on the quality of the results. Genuinely dynamic parameters are a different matter and a CLI arg is the right tool for them: the seed, or the number of GPUs when we can't or don't want to autodetect it, because we want to run the same target across hardware platforms without touching the code.
35. **Required acknowledgment notification:** When another rule invokes this mechanism, promptly give a standalone notice identifying the exception, affected files, and justification. Continue authorized work while acknowledgment is pending; this notification does not itself require pausing or waiting for a user decision. Repeat each pending item in a clearly labeled **Acknowledgment pending** paragraph in every subsequent final response until the user explicitly acknowledges that specific item. Silence, an unrelated reply, task completion, or a change of topic does not count as acknowledgment. Preserve pending items in context-compaction summaries and agent handoffs. If the exception is removed, report its removal and clear the item. This mechanism does not grant exceptions to rules that do not allow them.
36. **No C/C++ macros by default:** Single-line `#define` directives are the only default exception. Prefer C++ constants, types, templates, and free functions. A macro outside that exception may be added only when it is completely unavoidable or the alternative would be very complex and hard to maintain. In that case, proceed with the work and issue a **required acknowledgment notification** (rule 35), explicitly stating that a macro was added, its name and location, and why the alternatives are unsuitable. Keep that notice pending until explicitly acknowledged or the macro is removed.
37. Aim for lossless compression in everything you write (code, comments, documents, messages): every sentence must add information the reader cannot get from what is already in front of them, and everything that meets that bar stays. Example: a golden ledger is one sentence of purpose plus a table of version, date, commit hashes and reason; a paragraph restating the table's columns, host specs or measurement tables that belong next to the tolerance constants they justify, are loss-free to remove and therefore must not be there.
38. Do not carry backward compatibility that has no consumer: dual formats, legacy paths, deprecated aliases and fallback branches are complexity that outlives its reason (unpublished data, an internal API, a single caller you can update) — change the one consumer and delete the old path; keep compatibility only for a published interface or dataset that someone actually depends on, and name that dependency. Likewise do not document defensively: prefer one sentence next to the code over a section elsewhere, and prefer no comment over a comment that restates the code (rule 21).
39. Add complexity only for a demonstrated need, weighed against its context: a branch, cache, special case or option that saves work needs a measurement showing the saving matters relative to the enclosing operation, just as compatibility needs a consumer (rule 38); otherwise keep the straight-line code (e.g. skipping, when `debug` messages are filtered, the loop that counts a GLB's textured and metallic meshes for a debug log line would save ~6 µs of the ~400 ms load of a 667-mesh scene, under 0.002%, so the loop runs unconditionally).
