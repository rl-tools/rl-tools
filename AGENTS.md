# RL-Tools Development Guide


## Building & Testing

### Python Packages

Always use the `.venv` virtual environment. If it does not exist, create it with `python3 -m venv .venv`.

### CMake Configuration


Ubuntu (x86)
```bash
CUDACXX=/usr/local/cuda-13.1/bin/nvcc cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DRL_TOOLS_ENABLE_TESTS=ON \
  -DRL_TOOLS_EXPERIMENTAL=ON \
  -DRL_TOOLS_RL_ENVIRONMENTS_ENABLE_MUJOCO=ON \
  -DRL_TOOLS_NUMERIC_TYPES_ENABLE_BF16=ON \
  -DRL_TOOLS_ENABLE_TAR=ON \
  -DRL_TOOLS_RENDERING_ENABLE_RAYTRACING=ON
```

macOS
```
cmake -B build -DCMAKE_BUILD_TYPE=Release -DRL_TOOLS_EXPERIMENTAL=ON -DRL_TOOLS_ENABLE_TAR=ON -DRL_TOOLS_ENABLE_TESTS=ON -DRL_TOOLS_RL_ENVIRONMENTS_ENABLE_MUJOCO=ON
```

**Environment variables:**
- `CUDACXX=/usr/local/cuda-13.1/bin/nvcc` — required for CUDA test targets

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

`tests/data/rendering_raytracing_golden/` holds reference renderings of `tests/data/ProcTHOR-Train-1.glb` produced by the OptiX backend, for consistency testing over time and across backends. The cases (shading tiers, AA, motion blur, RGBD) and camera poses are defined in `tests/src/rendering/raytracing/golden_cases.h`. The generator is `EXCLUDE_FROM_ALL` and not registered with ctest — regenerate deliberately (requires CUDA + OptiX):

```bash
cmake --build build --target test_rendering_raytracing_generate_golden -j5
./build/tests/src/rendering/raytracing/test_rendering_raytracing_generate_golden
```

Outputs: `<case>.png` (2x2 grid of the 4 poses), `<case>_depth.bin` (`[num_cameras, height, width]` int header + float32 data, RGBD cases), `probes.bin` (`[num_cameras, num_probes]` int header + `CollisionResult` data). Same binary + same GPU/driver produce bit-identical output; across machines expect small silhouette differences, so downstream comparisons must be tolerance-based.

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
27. Extrack experiment paths follow `{TIME}/{COMMIT}_{NAME}_{POPULATION}/{CONFIG}/{SEED}` where `TIME` is a lexicographically-sortable timestamp like `2024-05-26_06-26-52` (e.g. from `date '+%Y-%m-%d_%H-%M-%S'`) — never substitute human-readable labels for `TIME` at the experiment root; `POPULATION` is the underscore-separated list of varied keys and `CONFIG` is the underscore-separated list of their values in the same order and arity (e.g. `algorithm_environment` ↔ `sac_pendulum-v1` decodes as `{algorithm: sac, environment: pendulum-v1}`), so individual values must avoid `_` (use `-` instead) to keep the positional zip unambiguous.
