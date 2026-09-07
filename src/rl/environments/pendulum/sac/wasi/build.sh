#!/usr/bin/env bash
set -euo pipefail
BUILD_DIR="${1:?usage: build.sh <IDE CMake build directory> [seed]}"
shift
cmake --build "$BUILD_DIR" --target parity --parallel 5
if [ $# -gt 0 ]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    node "$SCRIPT_DIR/../../../../../../tools/ide/run.mjs" "$BUILD_DIR/parity/training.wasm" "$@"
fi
