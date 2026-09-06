#!/usr/bin/env bash
# Cross-compiles training.cpp to a plain WASI command module with the host clang against the sysroot built by the
# tools/ide/toolchain superbuild: the same libc and libc++ the browser IDE links against, so this is the parity
# reference for tests/src/ide/pipeline.mjs --reference. Pass a seed to also run the result under node.
# Environment: RL_TOOLS_IDE_TOOLCHAIN_DIR (default /vm/data/rl-tools/ide-toolchain), RL_TOOLS_IDE_HOST_LLVM_BIN
# (default /usr/lib/llvm-22/bin), OUTPUT_DIR (default build/ide in the repository).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../../.." && pwd)"
TOOLCHAIN_DIR="${RL_TOOLS_IDE_TOOLCHAIN_DIR:-/vm/data/rl-tools/ide-toolchain}"
HOST_LLVM_BIN="${RL_TOOLS_IDE_HOST_LLVM_BIN:-/usr/lib/llvm-22/bin}"
SYSROOT="${SYSROOT:-$TOOLCHAIN_DIR/prefix/usr}"
RESOURCE_DIR="${RESOURCE_DIR:-$TOOLCHAIN_DIR/resource}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/build/ide}"

if [ ! -x "$HOST_LLVM_BIN/clang++" ] || [ ! -x "$HOST_LLVM_BIN/wasm-ld" ]; then
    echo "no clang++/wasm-ld in $HOST_LLVM_BIN (Ubuntu: sudo apt install clang-22 lld-22, or set RL_TOOLS_IDE_HOST_LLVM_BIN)" >&2
    exit 1
fi
if [ ! -f "$SYSROOT/lib/wasm32-wasip1/libc.a" ] || [ ! -f "$RESOURCE_DIR/lib/wasm32-unknown-wasip1/libclang_rt.builtins.a" ]; then
    echo "no wasm32-wasip1 sysroot at $SYSROOT with resource dir $RESOURCE_DIR: build the tools/ide/toolchain superbuild (--target sysroot is enough)" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
"$HOST_LLVM_BIN/clang++" --target=wasm32-wasip1 --sysroot="$SYSROOT" -resource-dir="$RESOURCE_DIR" \
    -std=c++17 -O2 -fno-exceptions -I"$REPO_ROOT/include" "$SCRIPT_DIR/training.cpp" -o "$OUTPUT_DIR/training.wasm"
echo "wrote $OUTPUT_DIR/training.wasm ($(stat -c %s "$OUTPUT_DIR/training.wasm") bytes) with $("$HOST_LLVM_BIN/clang++" --version | head -n 1)"

if [ $# -gt 0 ]; then
    if ! command -v node >/dev/null 2>&1; then
        echo "node not found, cannot run (Ubuntu: sudo apt install nodejs)" >&2
        exit 1
    fi
    node --no-warnings "$REPO_ROOT/tools/ide/run.mjs" "$OUTPUT_DIR/training.wasm" "$@"
fi
