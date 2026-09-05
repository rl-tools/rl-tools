#!/usr/bin/env bash
# Cross-compiles training.cpp to a plain WASI command module with a wasm32-wasip1 clang (wasi-sdk or the host toolchain from tools/ide/toolchain) and runs it with wasmtime when available.
# The browser IDE (static/ide) compiles the same file with the same flags using clang running inside the browser.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../../.." && pwd)"
WASI_SDK_VERSION="${WASI_SDK_VERSION:-34}"
WASI_SDK_PATH="${WASI_SDK_PATH:-$REPO_ROOT/.dependencies/ide/wasi-sdk-${WASI_SDK_VERSION}.0-x86_64-linux}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/build/ide}"

if [ ! -x "$WASI_SDK_PATH/bin/clang++" ]; then
    echo "no wasm32-wasip1 cross compiler at $WASI_SDK_PATH/bin/clang++" >&2
    echo "point WASI_SDK_PATH at a wasi-sdk installation or at the host toolchain built by tools/ide/toolchain; nothing is downloaded here" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
"$WASI_SDK_PATH/bin/clang++" -std=c++17 -O2 -fno-exceptions -I"$REPO_ROOT/include" "$SCRIPT_DIR/training.cpp" -o "$OUTPUT_DIR/training.wasm"
echo "wrote $OUTPUT_DIR/training.wasm"

if command -v wasmtime >/dev/null 2>&1; then
    wasmtime run "$OUTPUT_DIR/training.wasm" "${1:-0}"
else
    echo "wasmtime not found, skipping run (install from https://wasmtime.dev or run under node: tests/src/ide/pipeline.mjs)"
fi
