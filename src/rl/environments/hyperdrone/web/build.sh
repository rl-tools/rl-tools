#!/usr/bin/env bash
# Browser (WASM/WebGPU) build of HyperDrone. Requires an activated emsdk (source
# emsdk_env.sh from your SDK installation); emcc 4.0.17 is the pinned toolchain (see
# src/rl/environments/pendulum/sac/wasm/README.MD). The WebGPU raytracing backend codes
# against the standard webgpu.h, provided in the browser by the emdawnwebgpu port.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
DEPS_DIR="${RL_TOOLS_WASM_DEPS_DIR:-$REPO_ROOT/.dependencies/wasm}"
BUILD_DIR="${RL_TOOLS_WASM_BUILD_DIR:-$REPO_ROOT/static/hyperdrone/blob/build}"

ALL_CONFIGURATIONS=()
for fidelity in low medium high very-high; do
    for motion_samples in 1 4; do
        for aa_grid in 1 2; do
            for resolution in 480x270 960x540; do
                for agents in 1 8; do
                    ALL_CONFIGURATIONS+=("$fidelity-mb$motion_samples-aa$aa_grid-$resolution-agents$agents")
                done
            done
        done
    done
done

JOBS=5
CONFIGURATIONS=()
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --help)
            cat <<'USAGE'
Usage: build.sh [-jN | -j N] [CONFIGURATION ...]

Build all 64 rendering/drone configurations, or only the named configurations:
  build.sh -j5
  build.sh high-mb4-aa1-480x270-agents1
  build.sh -j5 low-mb1-aa1-480x270-agents1 very-high-mb4-aa2-960x540-agents8

Parallel jobs default to 5 and apply to CMake, Emscripten and Binaryen.
Use --list-configurations to list the supported IDs. Each build replaces the
bundle's configuration catalog; the frontend discovers it from the loaded WASM.
USAGE
            exit 0
            ;;
        --list-configurations)
            printf '%s\n' "${ALL_CONFIGURATIONS[@]}"
            exit 0
            ;;
        -j*)
            if [[ "$1" == -j ]]; then
                shift
                JOBS="${1:-}"
            else
                JOBS="${1#-j}"
            fi
            if [[ ! "$JOBS" =~ ^[1-9][0-9]*$ ]]; then
                echo "Job count must be a positive integer: use -jN or -j N." >&2
                exit 1
            fi
            ;;
        *)
            requested="$1"
            found=false
            for supported in "${ALL_CONFIGURATIONS[@]}"; do
                if [[ "$requested" == "$supported" ]]; then
                    found=true
                    break
                fi
            done
            if [[ "$found" == false ]]; then
                echo "Unknown configuration: $requested. Use --list-configurations for supported IDs." >&2
                exit 1
            fi
            for selected in ${CONFIGURATIONS[@]+"${CONFIGURATIONS[@]}"}; do
                if [[ "$requested" == "$selected" ]]; then
                    echo "Duplicate configuration: $requested" >&2
                    exit 1
                fi
            done
            CONFIGURATIONS+=("$requested")
            ;;
    esac
    shift
done
if [[ "${#CONFIGURATIONS[@]}" == 0 ]]; then
    CONFIGURATIONS=("${ALL_CONFIGURATIONS[@]}")
fi

export EMCC_CORES="$JOBS" BINARYEN_CORES="$JOBS"
command -v em++ >/dev/null || { echo "em++ not found: source emsdk_env.sh from your Emscripten SDK installation" >&2; exit 1; }
mkdir -p "$DEPS_DIR" "$BUILD_DIR"

{
    printf '%s\n' 'namespace rl_tools::rl::environments::hyperdrone::web {' '    using CompiledConfigurations = Configurations<'
    separator=''
    for configuration in "${CONFIGURATIONS[@]}"; do
        [[ "$configuration" =~ ^(.+)-mb([14])-aa([12])-(480x270|960x540)-agents([18])$ ]]
        fidelity="${BASH_REMATCH[1]}"
        motion_samples="${BASH_REMATCH[2]}"
        aa_grid="${BASH_REMATCH[3]}"
        resolution="${BASH_REMATCH[4]}"
        agents="${BASH_REMATCH[5]}"
        case "$fidelity" in
            low) shading=Low ;;
            medium) shading=Medium ;;
            high) shading=High ;;
            very-high) shading=VeryHigh ;;
        esac
        scale=1
        if [[ "$resolution" == 960x540 ]]; then
            scale=2
        fi
        printf '%s        Config<hyperlux::%s, %s, %s, %s, %s>' "$separator" "$shading" "$motion_samples" "$aa_grid" "$scale" "$agents"
        separator=$',\n'
    done
    printf '\n%s\n' '    >;' '}'
} > "$BUILD_DIR/hyperdrone_web_configurations.h"

ASSIMP_BUILD="$DEPS_DIR/assimp-archive-build"
ASSIMP_INSTALL="$DEPS_DIR/assimp-install"
emcmake cmake -S "$SCRIPT_DIR/assimp" -B "$ASSIMP_BUILD" -DCMAKE_BUILD_TYPE=Release \
        -DRLTOOLS_ROOT="$REPO_ROOT" \
        -DFETCHCONTENT_BASE_DIR="$ASSIMP_BUILD/_deps" \
        -DCMAKE_INSTALL_PREFIX="$ASSIMP_INSTALL" \
        -DCMAKE_CXX_FLAGS=-fexceptions -DCMAKE_C_FLAGS=-fexceptions \
        -DBUILD_SHARED_LIBS=OFF \
        -DASSIMP_BUILD_TESTS=OFF \
        -DASSIMP_BUILD_ASSIMP_TOOLS=OFF \
        -DASSIMP_BUILD_USE_CCACHE=OFF \
        -DASSIMP_NO_EXPORT=ON \
        -DASSIMP_BUILD_ALL_IMPORTERS_BY_DEFAULT=OFF \
        -DASSIMP_BUILD_GLTF_IMPORTER=ON \
        -DASSIMP_BUILD_ZLIB=ON \
        -DASSIMP_WARNINGS_AS_ERRORS=OFF
cmake --build "$ASSIMP_BUILD" --target assimp -j "$JOBS"
cmake --install "$ASSIMP_BUILD/_deps/assimp-build"
ASSIMP_ZLIB="$(ls "$ASSIMP_INSTALL"/lib/libzlibstatic*.a 2>/dev/null | head -1 || true)"

STB_DIR="$DEPS_DIR/stb"
STB_REVISION=f1c79c02822848a9bed4315b12c8c8f3761e1296 # same pin as src/hyperlux/CMakeLists.txt
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

cmake -DINPUT_FILE="$REPO_ROOT/src/hyperlux/backends/webgpu/device.wgsl" \
      -DOUTPUT_FILE="$DEPS_DIR/device_wgsl.cpp" \
      -DSYMBOL_NAME=rl_tools_hyperlux_webgpu_device_source \
      -P "$REPO_ROOT/cmake/scripts/embed_text.cmake"

em++ --std=c++17 -O3 -fexceptions \
    -I "$REPO_ROOT/include" \
    -I "$BUILD_DIR" \
    -I "$STB_DIR" \
    -I "$JSON_DIR/include" \
    -I "$ASSIMP_INSTALL/include" \
    -DRL_TOOLS_ENABLE_HYPERLUX \
    -DRL_TOOLS_HYPERLUX_BACKEND_WEBGPU \
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
    -sEXPORTED_FUNCTIONS=_demo_run,_demo_scenes,_demo_configurations,_demo_api_version,_main \
    "$SCRIPT_DIR/hyperdrone_web.cpp" \
    "$DEPS_DIR/device_wgsl.cpp" \
    "$ASSIMP_INSTALL/lib/libassimp.a" \
    ${ASSIMP_ZLIB:+"$ASSIMP_ZLIB"} \
    -o "$BUILD_DIR/hyperdrone_web.js"

echo "Built $BUILD_DIR/hyperdrone_web.js"
