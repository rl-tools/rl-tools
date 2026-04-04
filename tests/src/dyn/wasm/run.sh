#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"

echo "=== Generating test data ==="
python3 "${SCRIPT_DIR}/generate_test_data.py" "${SCRIPT_DIR}/test_checkpoint.h5"

echo ""
echo "=== Building with Emscripten ==="
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"
emcmake cmake "${SCRIPT_DIR}" -DCMAKE_BUILD_TYPE=Release
emmake make -j$(nproc) VERBOSE=1

echo ""
echo "=== Running WASM test ==="
node test_dyn_h5.js "${SCRIPT_DIR}/test_checkpoint.h5"
