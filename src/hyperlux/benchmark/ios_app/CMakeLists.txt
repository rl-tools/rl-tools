cmake_minimum_required(VERSION 3.25)
project(RLtoolsBenchmark LANGUAGES C CXX OBJCXX Swift)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "iOS")
    message(FATAL_ERROR "Configure with -DCMAKE_SYSTEM_NAME=iOS -DCMAKE_OSX_SYSROOT=iphoneos")
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_STANDARD 17)
get_filename_component(RL_TOOLS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../.." ABSOLUTE)
get_filename_component(BENCHMARK_DIR "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
set(PAPER_SCENE "${RL_TOOLS_ROOT}/build/paper-benchmark/ProcTHOR-Train-1.glb" CACHE FILEPATH "Pinned paper scene")
if(NOT EXISTS "${PAPER_SCENE}")
    message(FATAL_ERROR "Run ../download_paper_scene.py with the repository .venv first")
endif()
file(SHA256 "${PAPER_SCENE}" SCENE_HASH)
if(NOT SCENE_HASH STREQUAL "76c463017f2a25683abaea3b29786294602bd2629b2f0725b7215dabf58f35aa")
    message(FATAL_ERROR "PAPER_SCENE does not match the pinned paper asset")
endif()

include(FetchContent)
foreach(DEPENDENCY assimp metal_cpp stb)
    string(TOUPPER "${DEPENDENCY}" DEPENDENCY_UPPER)
    if(EXISTS "${RL_TOOLS_ROOT}/.dependencies/build/${DEPENDENCY}-src")
        set(FETCHCONTENT_SOURCE_DIR_${DEPENDENCY_UPPER} "${RL_TOOLS_ROOT}/.dependencies/build/${DEPENDENCY}-src" CACHE PATH "Existing dependency sources")
    endif()
endforeach()
set(BUILD_SHARED_LIBS OFF)
set(ASSIMP_BUILD_TESTS OFF)
set(ASSIMP_BUILD_ASSIMP_TOOLS OFF)
set(ASSIMP_INSTALL OFF)
set(ASSIMP_WARNINGS_AS_ERRORS OFF)
set(ASSIMP_NO_EXPORT ON)
set(ASSIMP_BUILD_ALL_IMPORTERS_BY_DEFAULT OFF)
set(ASSIMP_BUILD_GLTF_IMPORTER ON)
set(ASSIMP_BUILD_ZLIB OFF)
include("${RL_TOOLS_ROOT}/cmake/dependencies/assimp.cmake")
FetchContent_Declare(metal_cpp
    URL https://developer.apple.com/metal/cpp/files/metal-cpp_macOS15.2_iOS18.2.zip
    URL_HASH SHA256=3437e4abfbd3d45217f34772ef3502f31ba3358e5fb6ac9d0ca952a047bcfe25)
FetchContent_Declare(stb
    GIT_REPOSITORY https://github.com/nothings/stb.git
    GIT_TAG f1c79c02822848a9bed4315b12c8c8f3761e1296)
FetchContent_MakeAvailable(assimp metal_cpp stb)
find_package(nlohmann_json 3.11 QUIET)
if(NOT nlohmann_json_FOUND)
    FetchContent_Declare(nlohmann_json
        URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
        URL_HASH SHA256=d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d)
    FetchContent_MakeAvailable(nlohmann_json)
endif()

add_library(hyperlux_metal_cpp INTERFACE)
target_include_directories(hyperlux_metal_cpp SYSTEM INTERFACE "${metal_cpp_SOURCE_DIR}")
target_link_libraries(hyperlux_metal_cpp INTERFACE "-framework Metal" "-framework Foundation" "-framework QuartzCore")
add_subdirectory("${BENCHMARK_DIR}/../backends/metal" metal)

set(SIM_SHADING Medium)
set(SIM_MODE benchmark)
set(SIM_PAPER true)
set(SIM_PHYSICS_BACKEND cpu)
set(SIM_NUM_CAMERAS 4096)
set(SIM_RESOLUTION 64)
include("${BENCHMARK_DIR}/simulator_matrix_identity.cmake")
configure_file("${BENCHMARK_DIR}/simulator_matrix_config.h.in" simulator_matrix_config.h @ONLY)
execute_process(COMMAND git rev-parse HEAD WORKING_DIRECTORY "${RL_TOOLS_ROOT}"
    OUTPUT_VARIABLE BUILD_COMMIT OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
execute_process(COMMAND git diff --quiet HEAD WORKING_DIRECTORY "${RL_TOOLS_ROOT}" RESULT_VARIABLE BUILD_DIRTY)
set(APP_SOURCE_IDENTITY "")
foreach(SOURCE CMakeLists.txt benchmark.h benchmark.mm Info.plist Sources/BenchmarkApp.swift)
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE}")
    file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE}" SOURCE_HASH)
    string(APPEND APP_SOURCE_IDENTITY "${SOURCE}:${SOURCE_HASH}\n")
endforeach()
string(SHA256 APP_SOURCE_HASH "${APP_SOURCE_IDENTITY}")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/build.json"
    "{\"rltools_commit\":\"${BUILD_COMMIT}\",\"tracked_source_dirty\":${BUILD_DIRTY},\"app_source_sha256\":\"${APP_SOURCE_HASH}\",\"workload_source_sha256\":\"${SIM_SOURCE_HASH}\",\"compiler\":\"${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}\",\"num_cameras\":4096}\n")

add_executable(RLtoolsBenchmark MACOSX_BUNDLE
    Sources/BenchmarkApp.swift benchmark.mm
    "${BENCHMARK_DIR}/simulator_matrix.cpp" "${BENCHMARK_DIR}/simulator_matrix_physics_cpu.cpp"
    "${PAPER_SCENE}" "${CMAKE_CURRENT_BINARY_DIR}/build.json")
set_source_files_properties("${PAPER_SCENE}" "${CMAKE_CURRENT_BINARY_DIR}/build.json"
    PROPERTIES MACOSX_PACKAGE_LOCATION Resources)
target_include_directories(RLtoolsBenchmark PRIVATE "${RL_TOOLS_ROOT}/include" "${CMAKE_CURRENT_BINARY_DIR}")
target_include_directories(RLtoolsBenchmark SYSTEM PRIVATE "${stb_SOURCE_DIR}")
target_link_libraries(RLtoolsBenchmark PRIVATE hyperlux_backend nlohmann_json::nlohmann_json "-framework UIKit")
target_compile_definitions(RLtoolsBenchmark PRIVATE
    "$<$<COMPILE_LANGUAGE:CXX,OBJCXX>:RL_TOOLS_ENABLE_JSON;RL_TOOLS_ENABLE_HYPERLUX;RL_TOOLS_HYPERLUX_BACKEND_METAL;RL_TOOLS_COMMIT_HASH_EXTERNAL=${BUILD_COMMIT}>")
target_compile_options(RLtoolsBenchmark PRIVATE "$<$<COMPILE_LANGUAGE:CXX,OBJCXX>:-O3;-ffast-math>")
set_target_properties(RLtoolsBenchmark PROPERTIES
    MACOSX_BUNDLE_INFO_PLIST "${CMAKE_CURRENT_SOURCE_DIR}/Info.plist"
    XCODE_GENERATE_SCHEME TRUE
    XCODE_SCHEME_LAUNCH_CONFIGURATION Release
    XCODE_SCHEME_ENABLE_GPU_API_VALIDATION OFF
    XCODE_SCHEME_ENABLE_GPU_SHADER_VALIDATION OFF
    XCODE_SCHEME_ENABLE_GPU_FRAME_CAPTURE_MODE Disabled
    XCODE_ATTRIBUTE_PRODUCT_BUNDLE_IDENTIFIER "com.rltools.paper-benchmark"
    XCODE_ATTRIBUTE_CODE_SIGN_STYLE Automatic
    XCODE_ATTRIBUTE_CLANG_ENABLE_OBJC_ARC YES
    XCODE_ATTRIBUTE_SWIFT_VERSION 5.0
    XCODE_ATTRIBUTE_SWIFT_OBJC_BRIDGING_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/benchmark.h"
    XCODE_ATTRIBUTE_TARGETED_DEVICE_FAMILY 1
    XCODE_ATTRIBUTE_ENABLE_USER_SCRIPT_SANDBOXING NO)
