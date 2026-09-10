#!/bin/bash
set -e

DEBUG=${DEBUG:-0}
BUILD_TYPE="Release"
if [[ $DEBUG -eq 1 ]]; then
    BUILD_TYPE="Debug"
fi

docker run --rm \
    --mount type=bind,source="$(pwd)/wasm",target=/mnt,readonly \
    --mount type=bind,source="$(pwd)/blob",target=/blob \
    --mount type=bind,source="$(cd ../../rl-tools/ && pwd)",target=/rl_tools,readonly \
    emscripten/emsdk:4.0.17 \
    bash -c "
        mkdir -p /src /build && \
        cp /mnt/CMakeLists_dyn_inference.txt /src/CMakeLists.txt && \
        ln -s /mnt/dyn_inference.cpp /src/dyn_inference.cpp && \
        ln -s /mnt/dyn_inference.h /src/dyn_inference.h && \
        cd /build && \
        emcmake cmake /src \
            -DRL_TOOLS_INCLUDE_DIR=/rl_tools/include \
            -DCMAKE_BUILD_TYPE=$BUILD_TYPE && \
        emmake make -j\$(nproc) && \
        cp dyn-inference.js dyn-inference.wasm /blob/
    "

echo "Built: blob/dyn-inference.js + blob/dyn-inference.wasm"
