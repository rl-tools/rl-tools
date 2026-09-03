# RL-Tools Development Guide


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

**Raytracing backend selection:** `-DRL_TOOLS_RENDERING_RAYTRACING_BACKEND=AUTO|OPTIX|METAL|VULKAN|WEBGPU|GENERIC`. `AUTO` (the default) probes availability in preference order `OPTIX > METAL > VULKAN > WEBGPU > GENERIC` (OptiX needs a working CUDA toolchain and git for the OWL fetch, Metal needs macOS, Vulkan needs headers + `glslangValidator`, WebGPU needs linux-x86_64 and network access; all backends need assimp — without it raytracing auto-disables). An explicitly selected backend is hard-required and fails the configure if its prerequisites are missing. The `VULKAN` backend uses compute shaders with `VK_KHR_ray_query` (structural mirror of the Metal backend, GLSL compiled to SPIR-V at build time via `glslangValidator`); it needs the Vulkan dev headers and `glslang-tools` installed, runs headless, and works on Mesa's lavapipe CPU driver (`VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.json`) for GPU-less machines/CI. `RL_TOOLS_VULKAN_DEVICE_INDEX` overrides device selection. The `WEBGPU` backend codes strictly against the standard `webgpu.h`/WGSL (browser/WASM is the eventual target): ray tracing is not in the WebGPU standard, so it reuses the generic backend's deterministic CPU BVH build and traverses it in WGSL compute; the runtime is a hash-pinned wgpu-native prebuilt fetched via FetchContent (no Rust toolchain, Linux-x86_64 wired up so far), which sits on Vulkan and therefore also runs headless and on lavapipe. `RL_TOOLS_WEBGPU_DEVICE_INDEX` overrides adapter selection. `RL_TOOLS_WEBGPU_BVH=median` switches the BLAS/TLAS build back to the shared generic median-split baseline (default: deterministic binned SAH, traversed near-first).

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

Canonical renderings live in the separate test-data repository under `tests/data/rendering_raytracing_golden/`: the existing ProcTHOR suite uses `procthor_static_scene/` (per-pose `<case>.png`/`<case>_depth.bin`/`probes.bin` plus per-pose `normals.png`, `segmentation.bin`/`segmentation.png`, and `flow.bin`/`flow.png`/`flow_dynamic.bin`/`flow_dynamic.png` — normals, segmentation, and flow are single-sample and shading/AA/MB-independent, so one golden per pose covers all shading cases; the geometry case loads the scene as one instance per GLB root node so segmentation ids are pinned by node order, while the RGB and flow cases keep the welded single-object scene; the flow camera pair reuses `MOTION_BLUR_DELTA`, flow_dynamic the overlay shutter poses), and synthetic overlay cases use `overlay/<scenario>/<state>/<view>/{rgb.png,depth.bin,depth.png,segmentation.bin,segmentation.png,normals.png,flow.bin,flow.png}` (all overlay captures render with the shutter-open camera offset by the fixed `FLOW_PAIR_OFFSET`). Overlay PNGs are 2x2 grids in camera order `0,1 / 2,3`; versioned depth and segmentation binaries contain `[4,64,64]` (overlay) or `[1,64,64]` per pose (ProcTHOR), and `flow.bin` uses the version-2 (channels) header with 2 float channels per pixel — single-channel files keep the byte-identical version-1 header. `rgb.png`, the `.bin` files, and `normals.png` are the machine-compared targets (`normals.png` stores the float normals through the pinned `round((clamp(n,-1,1)*0.5+0.5)*255)` encoding — miss pixels are exactly `(128,128,128)`, which no unit normal can produce); `segmentation.png` stores ids through the pinned false-color encoding in `golden_io.h` and is validated against `segmentation.bin` (by the corpus test for overlay, by the backend comparator for ProcTHOR), while `depth.png` and `flow.png` (direction→hue, magnitude→saturation over the per-file maximum) are advisory review images only. On a fresh checkout, `./tests/download_data.sh --raytracing-goldens` downloads only this corpus. Configure `RL_TOOLS_REQUIRE_RAYTRACING_GOLDENS=ON` when missing manifests or LFS objects must fail instead of skip.

The generators are excluded from the default build and must be run deliberately from the repository root. The ProcTHOR generator builds per backend (`test_rendering_raytracing_generate_golden_generic` plus `test_rendering_raytracing_generate_golden_<active-backend>`); candidates default to `rendering_raytracing_golden/backend/<backend>` and are promoted to the canonical `procthor_static_scene/` via `--output-dir`. The overlay generator remains OptiX-only:

```bash
cmake --build build --target test_rendering_raytracing_generate_golden_generic \
                           test_rendering_raytracing_generate_overlay_golden -j5
./build/tests/src/rendering/raytracing/test_rendering_raytracing_generate_golden_generic
./build/tests/src/rendering/raytracing/test_rendering_raytracing_generate_overlay_golden
```

Run overlay comparisons with `ctest --test-dir build -L '^overlay-golden$' --output-on-failure --timeout 300 -j5` (this selects the backend comparators plus the backend-independent corpus validation suite; per-backend sub-labels `overlay-golden-<backend>` and `overlay-golden-corpus` exist too). Comparators write target/current/diff review images below `build/raytracing_golden_artifacts/<backend>/`; override `RL_TOOLS_RENDERING_RAYTRACING_GOLDEN_ARTIFACT_ROOT` at configure time if needed.

Publishing is a two-repository operation after reviewing the generated canonical files and comparison artifacts: first, a maintainer creates a branch from the detached revision in `tests/data`, commits and pushes the test-data changes, and records that commit hash; second, update `DATA_REVISION` in `tests/download_data.sh` to that hash and enable the CI `RL_TOOLS_REQUIRE_RAYTRACING_GOLDENS` publication gate in the RL-Tools repository. Do not enable the gate before both steps are complete.

### HDF5 Test Data

Some tests (`NN_LAYERS_RESNET_CUDA`, sequential persist tests) load `.h5` files from `tests/data/`. The path is set via `RL_TOOLS_TEST_DATA_PATH` (auto-detected by CMake). If these tests fail with "Object not found" HDF5 errors, check that dataset paths in the test source match the actual HDF5 group structure (inspect with `h5dump -H` or `python3 -c "import h5py; ..."`).


## Idiomatic RLtools Architecture Principles

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
30. HyperDrone episode bookkeeping lives in `rl/components/episodes/` (`Episodes<Specification<ENVIRONMENT>>`, CPU + CUDA): drivers call `begin_step` (reset decisions + `sample_initial_*` for the due instances; then `render`/`observe` with `episodes.reset`) and `end_step` (`terminated`, counters, truncation, end reason) instead of re-implementing step counters, reset masks or truncation. Semantics are fixed: same-step autoreset (the observation after a reset is the first of the new episode, no final observation), `terminated` implies `truncated`, `reset` is truncation delayed by one step (the on-policy dataset's `reset` column), `force_reset` marks external resets (scene rotation, epoch boundaries) with reason `FORCED`, `init` discards in-progress episodes (epoch boundaries), `SYNCHRONIZED` gives all-or-none resets for fixed-length tasks. The component knows no consumer: the per-step outputs (`finished*`, `reset`, `terminated`, `truncated`) are tensors, `record(log, episodes, step_i)` appends them to a per-rollout `Log`, and the on-policy runner ingests them through its own batched `record_step(dataset, step_i, rewards, terminated, truncated)` / `record_reset(dataset, reset)` verbs. Batched environments (tensor verbs over `ENVIRONMENT::INSTANCES`, optional `render(device, env, parameters, states, reset_mask)` for framebuffer state) are collected by `OnPolicyRunnerBatched<BatchedSpecification<...>>` in `rl/components/on_policy_runner/`: `init` applies the initial reset (+render), `collect` runs `prologue` (observe row 0, record the reset column) then per step `interlude` (recurrent reset from `episodes.reset`, batched policy evaluation, Gaussian `sample_actions`) and `epilogue` (step, reward, commit, episode bookkeeping, `record_step`, autoreset of the due instances, render, observe row t+1); drivers with custom policies call the phases individually and log with `record(log, runner.episodes, step_i)` after each epilogue. The scalar `OnPolicyRunner` (per-environment verbs) is untouched. Recurrent policies reset from `episodes.reset` through `Mode<mode::sequential::ResetMask>` (rank-1 bool tensor or 1×N matrix, device-generic).
