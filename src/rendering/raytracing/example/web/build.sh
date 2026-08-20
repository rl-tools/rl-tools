#!/usr/bin/env bash
# Browser (WASM/WebGPU) build of the drone motion-blur demo. Requires an activated emsdk
# (source ~/git/emsdk/emsdk_env.sh); emcc 4.0.17 is the pinned toolchain (see
# src/rl/environments/pendulum/sac/wasm/README.MD). The WebGPU raytracing backend codes
# against the standard webgpu.h, provided in the browser by the emdawnwebgpu port.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
DEPS_DIR="${RL_TOOLS_WASM_DEPS_DIR:-$REPO_ROOT/.dependencies/wasm}"
BUILD_DIR="${RL_TOOLS_WASM_BUILD_DIR:-$REPO_ROOT/static/raytracing/build}"

command -v em++ >/dev/null || { echo "em++ not found: source ~/git/emsdk/emsdk_env.sh" >&2; exit 1; }
mkdir -p "$DEPS_DIR" "$BUILD_DIR"

ASSIMP_TAG=v6.0.4
ASSIMP_SRC="$DEPS_DIR/assimp"
ASSIMP_BUILD="$DEPS_DIR/assimp-build"
ASSIMP_INSTALL="$DEPS_DIR/assimp-install"
if [ ! -f "$ASSIMP_INSTALL/lib/libassimp.a" ]; then
    [ -d "$ASSIMP_SRC" ] || git clone --depth 1 --branch "$ASSIMP_TAG" https://github.com/assimp/assimp.git "$ASSIMP_SRC"
    emcmake cmake -S "$ASSIMP_SRC" -B "$ASSIMP_BUILD" -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$ASSIMP_INSTALL" \
        -DCMAKE_CXX_FLAGS=-fexceptions -DCMAKE_C_FLAGS=-fexceptions \
        -DBUILD_SHARED_LIBS=OFF \
        -DASSIMP_BUILD_TESTS=OFF \
        -DASSIMP_BUILD_ASSIMP_TOOLS=OFF \
        -DASSIMP_NO_EXPORT=ON \
        -DASSIMP_BUILD_ALL_IMPORTERS_BY_DEFAULT=OFF \
        -DASSIMP_BUILD_GLTF_IMPORTER=ON \
        -DASSIMP_BUILD_ZLIB=ON \
        -DASSIMP_WARNINGS_AS_ERRORS=OFF
    cmake --build "$ASSIMP_BUILD" -j5
    cmake --install "$ASSIMP_BUILD"
fi
ASSIMP_ZLIB="$(ls "$ASSIMP_INSTALL"/lib/libzlibstatic*.a 2>/dev/null | head -1 || true)"

STB_DIR="$DEPS_DIR/stb"
STB_REVISION=f1c79c02822848a9bed4315b12c8c8f3761e1296 # same pin as src/rendering/raytracing/CMakeLists.txt
if [ ! -f "$STB_DIR/stb_image.h" ]; then
    git clone https://github.com/nothings/stb.git "$STB_DIR"
    git -C "$STB_DIR" checkout "$STB_REVISION"
fi

JSON_DIR="$DEPS_DIR/json"
JSON_REVISION=a0c1318830519eac027a31edec1a99ce1ae5670e # same pin as cmake/autodetect/tier2-json-hdf5-zlib.cmake
if [ ! -f "$JSON_DIR/include/nlohmann/json.hpp" ]; then
    git clone https://github.com/nlohmann/json.git "$JSON_DIR"
    git -C "$JSON_DIR" checkout "$JSON_REVISION"
fi

cmake -DINPUT_FILE="$REPO_ROOT/src/rendering/raytracing/backends/webgpu/device.wgsl" \
      -DOUTPUT_FILE="$DEPS_DIR/device_wgsl.cpp" \
      -DSYMBOL_NAME=rl_tools_rendering_raytracing_webgpu_device_source \
      -P "$REPO_ROOT/cmake/scripts/embed_text.cmake"

em++ --std=c++17 -O3 -fexceptions \
    -I "$REPO_ROOT/include" \
    -I "$STB_DIR" \
    -I "$JSON_DIR/include" \
    -I "$ASSIMP_INSTALL/include" \
    -DRL_TOOLS_RENDERING_ENABLE_RAYTRACING \
    -DRL_TOOLS_RENDERING_RAYTRACING_BACKEND_WEBGPU \
    --use-port=emdawnwebgpu \
    -sASYNCIFY=1 \
    -sASYNCIFY_STACK_SIZE=131072 \
    -sSTACK_SIZE=8MB \
    -sALLOW_MEMORY_GROWTH=1 \
    -sINITIAL_MEMORY=512MB \
    -sMAXIMUM_MEMORY=4GB \
    -sMODULARIZE=1 \
    -sEXPORT_ES6=1 \
    -sENVIRONMENT=web \
    -sEXPORTED_RUNTIME_METHODS=ccall,FS,HEAPU8 \
    -sEXPORTED_FUNCTIONS=_demo_run,_main \
    "$SCRIPT_DIR/drone_web.cpp" \
    "$DEPS_DIR/device_wgsl.cpp" \
    "$ASSIMP_INSTALL/lib/libassimp.a" \
    ${ASSIMP_ZLIB:+"$ASSIMP_ZLIB"} \
    -o "$BUILD_DIR/drone_web.js"

echo "Built $BUILD_DIR/drone_web.js"
