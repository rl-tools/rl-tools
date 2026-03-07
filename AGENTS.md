# RL-Tools Development Guide


## Building & Testing

### Python Packages

Always use the `.venv` virtual environment. If it does not exist, create it with `python3 -m venv .venv`.

### CMake Configuration

```bash
CUDACXX=/usr/local/cuda-13.1/bin/nvcc cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DRL_TOOLS_ENABLE_TESTS=ON \
  -DRL_TOOLS_EXPERIMENTAL=ON \
  -DRL_TOOLS_RL_ENVIRONMENTS_ENABLE_MUJOCO=ON \
  -DRL_TOOLS_NUMERIC_TYPES_ENABLE_BF16=ON \
  -DRL_TOOLS_ENABLE_TAR=ON
```

**Environment variables:**
- `CUDACXX=/usr/local/cuda-13.1/bin/nvcc` — required for CUDA test targets

### Running Tests

```bash
make -j8
ctest --timeout 20 -j16
```
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
make -j8 test_nn_layers_gru_persist_code \
         test_nn_models_multi_agent_wrapper_persist_code \
         test_inference_l2f_c_interface_write_policy

cd /path/to/repo   # generators write relative to cwd
./build/tests/src/nn/layers/gru/test_nn_layers_gru_persist_code
./build/tests/src/nn_models/multi_agent_wrapper/test_nn_models_multi_agent_wrapper_persist_code
./build/tests/src/inference/executor/test_inference_l2f_c_interface_write_policy
```

After regeneration, rebuild the compile tests before re-running `ctest`.

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
13. `operations_generic.h` must be strictly freestanding: no C++ standard library includes; depend only on RLtools abstractions for maximal platform/compiler portability.
14. C++17 is the maximum compatibility standard; post-C++17 language/library features are not allowed in shared RLtools code.
15. For environment operation headers, prefer backend-specific naming such as `operations_cpu.h` (with optional thin compatibility wrappers like `operations.h`), and keep environment types/operations under `rl_tools::...` namespaces instead of top-level project-specific namespaces.
16. Keep public `include/` headers focused on reusable API/types/dispatch; place build artifacts and backend implementation translation units (e.g. `.cu` device programs, heavy third-party integration code) under `src/` and wire them via CMake.
17. When backend code requires generated symbols (e.g. embedded PTX blobs), use project-unique symbol names and expose them through namespaced C++ accessors instead of depending on ambiguous global names at call sites.
18. Preserve benchmarking comparability during refactors: keep initialization and workload semantics deterministic when measuring performance across revisions, and avoid mixing behavioral changes with structural/API migrations in the same benchmark comparison.
19. In hot paths, keep RLtools API boundaries explicit but minimize per-element abstraction overhead (prefer contiguous-buffer iteration patterns where appropriate) so architectural cleanup does not unintentionally regress throughput.
