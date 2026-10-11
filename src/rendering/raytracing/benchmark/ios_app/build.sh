#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
BUILD_DIR="$REPO_ROOT/build/ios-paper-benchmark"

if [ ! -x "$REPO_ROOT/.venv/bin/python" ]; then
    python3 -m venv "$REPO_ROOT/.venv"
fi
"$REPO_ROOT/.venv/bin/python" "$SCRIPT_DIR/../download_paper_scene.py"
cmake -S "$SCRIPT_DIR" -B "$BUILD_DIR" -G Xcode \
    -DCMAKE_SYSTEM_NAME=iOS -DCMAKE_OSX_SYSROOT=iphoneos \
    -DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_OSX_DEPLOYMENT_TARGET=16.0 \
    -DCMAKE_BUILD_TYPE=Release "$@"
cmake --build "$BUILD_DIR" --config Release --target RLtoolsBenchmark -j5 -- CODE_SIGNING_ALLOWED=NO
printf '\nOpen %s/RLtoolsBenchmark.xcodeproj, select your signing team and iPhone, and run the RLtoolsBenchmark scheme in Release.\n' "$BUILD_DIR"
