# RL-Tools Development Guide

## Building & Testing

### CMake Configuration

```bash
mkdir -p build && cd build
CUDACXX=/usr/local/cuda-13.1/bin/nvcc cmake .. \
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
