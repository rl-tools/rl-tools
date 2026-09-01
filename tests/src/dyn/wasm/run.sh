#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
TEST_DATA_DIR="${REPO_ROOT}/tests/data"
CHECKPOINT_DEFAULT="${TEST_DATA_DIR}/test_dyn_wasm_checkpoint.h5"
CHECKPOINT_L2F="${TEST_DATA_DIR}/test_dyn_wasm_checkpoint_l2f_visual_imitation.h5"

echo "=== Generating test data ==="
cmake -B "${REPO_ROOT}/build" -S "${REPO_ROOT}"
cmake --build "${REPO_ROOT}/build" --target test_dyn_wasm_generate -j$(nproc)
"${REPO_ROOT}/build/tests/src/dyn/test_dyn_wasm_generate"

echo ""
echo "=== Building with Emscripten ==="
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"
emcmake cmake "${SCRIPT_DIR}" -DCMAKE_BUILD_TYPE=Release
emmake make -j$(nproc) VERBOSE=1

cd "${SCRIPT_DIR}" && npm install jsfive 2>/dev/null
cd "${BUILD_DIR}"

FAILED=0
for CP in "${CHECKPOINT_DEFAULT}" "${CHECKPOINT_L2F}"; do
    if [ ! -f "${CP}" ]; then
        echo ""
        echo "=== SKIP (not found): ${CP} ==="
        continue
    fi
    NAME="$(basename "${CP}")"

    echo ""
    echo "=== [${NAME}] WASM test (direct) ==="
    node test_dyn_h5.js "${CP}" || FAILED=1

    echo ""
    echo "=== [${NAME}] WASM test (emscripten bindings) ==="
    node "${SCRIPT_DIR}/test_bind.mjs" "${CP}" "./test_dyn_h5_bind.js" || FAILED=1
done

echo ""
if [ "${FAILED}" -eq 0 ]; then
    echo "ALL CHECKPOINTS PASSED"
else
    echo "SOME CHECKPOINTS FAILED"
    exit 1
fi
