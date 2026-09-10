#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RL_TOOLS_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DEBUG=${DEBUG:-0}
BUILD_TYPE="Release"
if [[ $DEBUG -eq 1 ]]; then
    BUILD_TYPE="Debug"
fi

docker run --rm --user "$(id -u):$(id -g)" \
    --mount "type=bind,source=${SCRIPT_DIR}/wasm,target=/mnt,readonly" \
    --mount "type=bind,source=${SCRIPT_DIR}/external/blob,target=/blob" \
    --mount "type=bind,source=${RL_TOOLS_DIR},target=/rl_tools,readonly" \
    emscripten/emsdk:4.0.17 \
    bash -c "
        mkdir -p /tmp/src /tmp/build && \
        cp /mnt/CMakeLists_dyn_inference.txt /tmp/src/CMakeLists.txt && \
        ln -s /mnt/dyn_inference.cpp /tmp/src/dyn_inference.cpp && \
        ln -s /mnt/dyn_inference.h /tmp/src/dyn_inference.h && \
        cd /tmp/build && \
        emcmake cmake /tmp/src \
            -DRL_TOOLS_INCLUDE_DIR=/rl_tools/include \
            -DCMAKE_BUILD_TYPE=$BUILD_TYPE && \
        cmake --build . --parallel 5 && \
        cp dyn-inference.* /blob/
    "

echo "Built: ${SCRIPT_DIR}/external/blob/dyn-inference.js + dyn-inference.wasm"
