#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
TEST_DATA_DIR="${REPO_ROOT}/tests/data"
CHECKPOINT="${TEST_DATA_DIR}/test_dyn_wasm_checkpoint.h5"

echo "=== Generating test data ==="
cmake -B "${REPO_ROOT}/build" -S "${REPO_ROOT}" -DRL_TOOLS_ENABLE_HDF5=ON -DRL_TOOLS_ENABLE_TESTS=ON -DRL_TOOLS_ENABLE_TAR=ON
cmake --build "${REPO_ROOT}/build" --target test_dyn_wasm_generate -j$(nproc)
"${REPO_ROOT}/build/tests/src/dyn/test_dyn_wasm_generate"

echo ""
echo "=== Building with Emscripten ==="
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"
emcmake cmake "${SCRIPT_DIR}" -DCMAKE_BUILD_TYPE=Release
emmake make -j$(nproc) VERBOSE=1

echo ""
echo "=== Running WASM test (direct) ==="
node test_dyn_h5.js "${CHECKPOINT}"

echo ""
echo "=== Running WASM test (emscripten bindings) ==="
cd "${SCRIPT_DIR}" && npm install jsfive 2>/dev/null
cd "${BUILD_DIR}"
node "${SCRIPT_DIR}/test_bind.mjs" "${CHECKPOINT}" "./test_dyn_h5_bind.js"
