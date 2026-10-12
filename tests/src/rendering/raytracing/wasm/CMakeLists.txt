cmake_minimum_required(VERSION 3.25)
project(RLtoolsRaytracingGoldenWasm LANGUAGES C CXX)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "WASI")
    message(FATAL_ERROR "Configure with -DCMAKE_TOOLCHAIN_FILE=<wasi-sdk>/share/cmake/wasi-sdk-p1.cmake -DWASI_SDK_PREFIX=<wasi-sdk>")
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
get_filename_component(RL_TOOLS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../../.." ABSOLUTE)
get_filename_component(GOLDEN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)

# The module is the numerical definition of the corpus: strict IEEE evaluation (no fast-math, no
# contraction, saturating float-to-int). Exceptions use the standardized proposal (engines need it
# enabled, wasmtime: -W exceptions); the legacy-eh switch is an LLVM-internal option pinned with the sdk.
add_compile_options(-fno-fast-math -ffp-contract=off -mnontrapping-fptoint -fwasm-exceptions -mllvm -wasm-use-legacy-eh=false)
add_link_options(-fwasm-exceptions -lunwind -lwasi-emulated-signal -lwasi-emulated-process-clocks -lwasi-emulated-getpid -Wl,-z,stack-size=8388608)
# Assimp includes signal.h and uses clock(), conta uses getpid(); WASI provides these only as opt-in emulations (logging and temporary names only)
add_compile_definitions(_WASI_EMULATED_SIGNAL _WASI_EMULATED_PROCESS_CLOCKS _WASI_EMULATED_GETPID)

include(FetchContent)
set(RL_TOOLS_DEPENDENCY_SOURCES "${RL_TOOLS_ROOT}/.dependencies/build" CACHE PATH "FetchContent base directory of a native build whose dependency sources are reused")
foreach(DEPENDENCY assimp stb nlohmann_json)
    string(TOUPPER "${DEPENDENCY}" DEPENDENCY_UPPER)
    if(EXISTS "${RL_TOOLS_DEPENDENCY_SOURCES}/${DEPENDENCY}-src")
        set(FETCHCONTENT_SOURCE_DIR_${DEPENDENCY_UPPER} "${RL_TOOLS_DEPENDENCY_SOURCES}/${DEPENDENCY}-src" CACHE PATH "Existing dependency sources")
    endif()
endforeach()
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
    GIT_REPOSITORY https://github.com/nothings/stb.git
    GIT_TAG f1c79c02822848a9bed4315b12c8c8f3761e1296)
FetchContent_Declare(nlohmann_json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
    URL_HASH SHA256=d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d)
set(JSON_BuildTests OFF)
FetchContent_MakeAvailable(assimp stb nlohmann_json)
# Assimp's bundled poly2tri uses std::ostream without including <ostream>; this libc++ does not provide it transitively
target_compile_options(assimp PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:SHELL:-include ostream>")

add_library(rendering_raytracing_stb INTERFACE)
target_include_directories(rendering_raytracing_stb SYSTEM INTERFACE "${stb_SOURCE_DIR}")
add_library(rl_tools INTERFACE)
target_include_directories(rl_tools INTERFACE "${RL_TOOLS_ROOT}/include")
target_compile_definitions(rl_tools INTERFACE RL_TOOLS_TEST_DATA_PATH=${RL_TOOLS_ROOT}/tests/data)

file(STRINGS "${WASI_SDK_PREFIX}/VERSION" RL_TOOLS_WASI_SDK_VERSION LIMIT_COUNT 1)
set(RL_TOOLS_GOLDEN_TOOLCHAIN "wasi-sdk ${RL_TOOLS_WASI_SDK_VERSION}")
include("${GOLDEN_DIR}/golden_build.cmake")
add_executable(rl_tools_rendering_raytracing_generate_golden "${GOLDEN_DIR}/generate_golden.cpp")
target_link_libraries(rl_tools_rendering_raytracing_generate_golden PRIVATE rl_tools assimp::assimp rendering_raytracing_stb nlohmann_json::nlohmann_json)
rl_tools_configure_golden_generator(rl_tools_rendering_raytracing_generate_golden wasm)
set_target_properties(rl_tools_rendering_raytracing_generate_golden PROPERTIES OUTPUT_NAME generate_golden SUFFIX .wasm)
