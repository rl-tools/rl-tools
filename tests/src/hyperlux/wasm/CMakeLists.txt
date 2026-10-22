cmake_minimum_required(VERSION 3.25)
get_filename_component(RL_TOOLS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../.." ABSOLUTE)
get_filename_component(GOLDEN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
set(RL_TOOLS_DEPENDENCY_SOURCES "${RL_TOOLS_ROOT}/.dependencies/build" CACHE PATH "FetchContent base directory of a native build whose dependency sources are reused")
if(NOT DEFINED FETCHCONTENT_BASE_DIR)
    get_filename_component(BUILD_NAME "${CMAKE_BINARY_DIR}" NAME)
    set(FETCHCONTENT_BASE_DIR "${RL_TOOLS_ROOT}/.dependencies/${BUILD_NAME}/wasm" CACHE PATH "Dependency download directory")
endif()
include("${RL_TOOLS_ROOT}/cmake/dependencies/wasm.cmake")
rl_tools_fetch_wasm("${FETCHCONTENT_BASE_DIR}")
if(RL_TOOLS_WASM_UNAVAILABLE_REASON)
    message(FATAL_ERROR "WASM goldens: ${RL_TOOLS_WASM_UNAVAILABLE_REASON}")
endif()
set(CMAKE_TOOLCHAIN_FILE "${WASI_SDK_PREFIX}/share/cmake/wasi-sdk-p1.cmake")
project(RLtoolsHyperLuxGoldenWasm LANGUAGES C CXX)
set(WASM_C_COMPILER clang)
set(WASM_CXX_COMPILER clang++)
foreach(LANGUAGE C CXX)
    file(REAL_PATH "${WASI_SDK_PREFIX}/bin/${WASM_${LANGUAGE}_COMPILER}" EXPECTED_COMPILER)
    file(REAL_PATH "${CMAKE_${LANGUAGE}_COMPILER}" ACTUAL_COMPILER)
    if(NOT ACTUAL_COMPILER STREQUAL EXPECTED_COMPILER)
        message(FATAL_ERROR "The cached ${LANGUAGE} compiler does not use the pinned SDK; configure a fresh build directory")
    endif()
endforeach()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE STREQUAL "Release")
    message(FATAL_ERROR "Golden builds require Release")
endif()
foreach(FLAGS CMAKE_C_FLAGS CMAKE_CXX_FLAGS CMAKE_EXE_LINKER_FLAGS)
    if(NOT "${${FLAGS}}" STREQUAL "")
        message(FATAL_ERROR "Golden builds require an empty ${FLAGS}; the profile is fixed here")
    endif()
endforeach()
set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# The module is the numerical definition of the corpus: strict IEEE evaluation (no fast-math, no
# contraction, saturating float-to-int). Exceptions use the standardized proposal (engines need it
# enabled, wasmtime: -W exceptions); the legacy-eh switch is an LLVM-internal option pinned with the sdk.
set(GOLDEN_COMPILE_OPTIONS -fno-fast-math -ffp-contract=off -mnontrapping-fptoint -mno-simd128 -mno-relaxed-simd -fwasm-exceptions -mllvm -wasm-use-legacy-eh=false)
add_compile_options(${GOLDEN_COMPILE_OPTIONS}
    "-ffile-prefix-map=${RL_TOOLS_ROOT}=/rl-tools"
    "-ffile-prefix-map=${CMAKE_BINARY_DIR}=/build"
    "-ffile-prefix-map=${FETCHCONTENT_BASE_DIR}=/dependencies"
    "-ffile-prefix-map=${RL_TOOLS_DEPENDENCY_SOURCES}=/dependencies"
    "-ffile-prefix-map=${WASI_SDK_PREFIX}=/wasi-sdk"
    "$<$<COMPILE_LANGUAGE:CXX>:SHELL:-isystem ${WASI_SDK_PREFIX}/share/wasi-sysroot/include/wasm32-wasip1/eh/c++/v1>")
add_link_options(-fwasm-exceptions -lunwind -lwasi-emulated-signal -lwasi-emulated-process-clocks -lwasi-emulated-getpid -Wl,-z,stack-size=8388608 -Wl,--strip-debug
    -Wl,--wrap=__wasi_clock_time_get -Wl,--wrap=__wasi_random_get)
# Assimp includes signal.h and uses clock(), conta uses getpid(); WASI provides these only as opt-in emulations (logging and temporary names only)
add_compile_definitions(_WASI_EMULATED_SIGNAL _WASI_EMULATED_PROCESS_CLOCKS _WASI_EMULATED_GETPID)

include(FetchContent)
if(RL_TOOLS_OFFLINE_BUILD)
    set(FETCHCONTENT_FULLY_DISCONNECTED ON)
endif()
foreach(DEPENDENCY assimp stb nlohmann_json)
    string(TOUPPER "${DEPENDENCY}" DEPENDENCY_UPPER)
    if(EXISTS "${RL_TOOLS_DEPENDENCY_SOURCES}/${DEPENDENCY}-src")
        set(FETCHCONTENT_SOURCE_DIR_${DEPENDENCY_UPPER} "${RL_TOOLS_DEPENDENCY_SOURCES}/${DEPENDENCY}-src")
    endif()
endforeach()
set(ASSIMP_IGNORE_GIT_HASH ON)
set(BUILD_SHARED_LIBS OFF)
set(ASSIMP_BUILD_TESTS OFF)
set(ASSIMP_BUILD_ASSIMP_TOOLS OFF)
set(ASSIMP_BUILD_SAMPLES OFF)
set(ASSIMP_INSTALL OFF)
set(ASSIMP_WARNINGS_AS_ERRORS OFF)
set(ASSIMP_NO_EXPORT ON)
set(ASSIMP_BUILD_ALL_IMPORTERS_BY_DEFAULT OFF)
set(ASSIMP_BUILD_GLTF_IMPORTER ON)
set(ASSIMP_BUILD_ZLIB ON)
include("${RL_TOOLS_ROOT}/cmake/dependencies/assimp.cmake")
FetchContent_Declare(stb
    URL https://codeload.github.com/nothings/stb/tar.gz/f1c79c02822848a9bed4315b12c8c8f3761e1296
    URL_HASH SHA256=a39b0f2d96d9b64171ac1fd73295b7c5866f904e653b379da886d050a6900db9)
FetchContent_Declare(nlohmann_json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
    URL_HASH SHA256=d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d)
set(JSON_BuildTests OFF)
FetchContent_MakeAvailable(assimp stb nlohmann_json)
# Assimp's bundled poly2tri uses std::ostream without including <ostream>; this libc++ does not provide it transitively
target_compile_options(assimp PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:SHELL:-include ostream>")

add_library(hyperlux_stb INTERFACE)
target_include_directories(hyperlux_stb SYSTEM INTERFACE "${stb_SOURCE_DIR}")
add_library(rl_tools INTERFACE)
target_include_directories(rl_tools INTERFACE "${RL_TOOLS_ROOT}/include")
set(RL_TOOLS_TEST_DATA_PATH /corpus)

include("${GOLDEN_DIR}/golden_build.cmake")
add_executable(rl_tools_hyperlux_generate_golden "${GOLDEN_DIR}/generate_golden.cpp" host.cpp)
target_link_libraries(rl_tools_hyperlux_generate_golden PRIVATE rl_tools assimp::assimp hyperlux_stb nlohmann_json::nlohmann_json)
rl_tools_configure_golden_generator(rl_tools_hyperlux_generate_golden wasm)
set_target_properties(rl_tools_hyperlux_generate_golden PROPERTIES OUTPUT_NAME generate_golden SUFFIX .wasm)

if(NOT DEFINED CONTA)
    set(CONTA conta)
endif()
file(STRINGS "${GOLDEN_DIR}/golden_manifest.h" RL_TOOLS_GOLDEN_SCENE_LINE REGEX "PROCTHOR_TRAIN_1_SCENE = \"")
string(REGEX REPLACE ".*\"([^\"]+)\".*" "\\1" RL_TOOLS_GOLDEN_SCENE "${RL_TOOLS_GOLDEN_SCENE_LINE}")
if(NOT DEFINED OUTPUT_DIR)
    set(OUTPUT_DIR "${RL_TOOLS_ROOT}/tests/data/hyperlux/goldens/procthor-train-1")
endif()
foreach(PATH_VARIABLE RL_TOOLS_WASMTIME CONTA OUTPUT_DIR CMAKE_CURRENT_BINARY_DIR)
    string(REPLACE "'" "'\\''" ${PATH_VARIABLE}_SHELL "${${PATH_VARIABLE}}")
endforeach()
configure_file(render.sh.in generate_goldens @ONLY
    FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
