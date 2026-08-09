cmake_minimum_required(VERSION 3.24)
project(hyperdrone_dynamics LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(RLTOOLS_ROOT "" CACHE PATH "Path to the rl-tools source root")
set(HYPERDRONE_VARIANT "CPU" CACHE STRING "Dynamics device (CPU|CUDA)")

if(NOT RLTOOLS_ROOT)
    message(FATAL_ERROR "RLTOOLS_ROOT must point to the rl-tools source root")
endif()

set(RL_TOOLS_ENABLE_TARGETS OFF CACHE BOOL "" FORCE)
set(RL_TOOLS_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(RL_TOOLS_ENABLE_GIT_DIFF OFF CACHE BOOL "" FORCE)
set(RL_TOOLS_WARNINGS_AS_ERRORS OFF CACHE BOOL "" FORCE)

if(HYPERDRONE_VARIANT STREQUAL "CUDA")
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES native)
    endif()
endif()

add_subdirectory(${RLTOOLS_ROOT} rl_tools EXCLUDE_FROM_ALL)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(
    COMMAND ${Python_EXECUTABLE} -m nanobind --cmake_dir
    OUTPUT_VARIABLE nanobind_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(hyperdrone_dynamics_core host.cpp)
target_include_directories(hyperdrone_dynamics_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../common)
target_link_libraries(hyperdrone_dynamics_core PRIVATE ${CMAKE_DL_LIBS})
set_target_properties(hyperdrone_dynamics_core PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

set(HYPERDRONE_DYNAMICS_EXTRA_LINK "")
if(HYPERDRONE_VARIANT STREQUAL "CUDA")
    find_package(CUDAToolkit REQUIRED)
    target_compile_definitions(hyperdrone_dynamics_core PRIVATE HYPERDRONE_DYNAMICS_CUDA=1)
    target_link_libraries(hyperdrone_dynamics_core PRIVATE CUDA::cudart)
    add_compile_definitions(HYPERDRONE_DYNAMICS_CUDA=1)
    set_source_files_properties(impl.cpp PROPERTIES LANGUAGE CUDA)
else()
    # per-drone parallelism on the CPU variant; drones are independent, so this stays
    # deterministic for a fixed seed
    find_package(OpenMP QUIET)
    if(OpenMP_CXX_FOUND)
        set(HYPERDRONE_DYNAMICS_EXTRA_LINK OpenMP::OpenMP_CXX)
    endif()
endif()

include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/hyperdrone_jit.cmake)
hyperdrone_jit_add_configs(
    COMPONENT dynamics
    SOURCES impl.cpp
    LINK RLtools::RLtools ${HYPERDRONE_DYNAMICS_EXTRA_LINK}
)
