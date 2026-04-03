cmake_minimum_required(VERSION 3.18)
project(rl_tools_dyn_h5_wasm_test)

set(CMAKE_CXX_STANDARD 17)

include(FetchContent)
FetchContent_Declare(
    libhdf5-wasm
    URL https://github.com/usnistgov/libhdf5-wasm/releases/download/v0.6.0_4.0.23/HDF5-2.0.0-Emscripten.tar.gz
    URL_HASH SHA256=d4dc6719ef164679728d9a50d6a4d97c826f563e22a40c4dec63e485b29781be
)
FetchContent_MakeAvailable(libhdf5-wasm)
set(HDF5_DIR "${libhdf5-wasm_SOURCE_DIR}/cmake" CACHE PATH "HDF5 cmake config" FORCE)
find_package(HDF5 REQUIRED CONFIG)

# Resolve rl_tools include path (four levels up from this CMakeLists.txt)
get_filename_component(RL_TOOLS_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../include" ABSOLUTE)
message(STATUS "RL_TOOLS include dir: ${RL_TOOLS_INCLUDE_DIR}")

add_executable(test_dyn_h5 test.cpp)
target_include_directories(test_dyn_h5 PRIVATE ${RL_TOOLS_INCLUDE_DIR})
target_link_libraries(test_dyn_h5 hdf5::hdf5-static)

if(EMSCRIPTEN)
    target_link_options(test_dyn_h5 PRIVATE
        "SHELL:-s NODERAWFS=1"
        "SHELL:-s ENVIRONMENT=node"
        "SHELL:-s ALLOW_MEMORY_GROWTH=1"
        "SHELL:-s STACK_SIZE=2097152"
    )
endif()
