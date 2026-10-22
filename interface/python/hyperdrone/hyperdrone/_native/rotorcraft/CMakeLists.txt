cmake_minimum_required(VERSION 3.24)
project(hyperdrone_rotorcraft LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(RLTOOLS_ROOT "" CACHE PATH "Path to the rl-tools source root")
if(NOT RLTOOLS_ROOT)
    message(FATAL_ERROR "RLTOOLS_ROOT must point to the rl-tools source root")
endif()

set(RL_TOOLS_DISABLE_TARGETS ON CACHE BOOL "" FORCE)
set(RL_TOOLS_DISABLE_TESTS ON CACHE BOOL "" FORCE)
set(RL_TOOLS_DISABLE_GIT_DIFF ON CACHE BOOL "" FORCE)
set(RL_TOOLS_RL_ENVIRONMENTS_DISABLE_MUJOCO ON CACHE BOOL "" FORCE)
set(RL_TOOLS_DISABLE_HYPERLUX ON CACHE BOOL "" FORCE)
set(RL_TOOLS_WARNINGS_AS_ERRORS OFF CACHE BOOL "" FORCE)

add_subdirectory(${RLTOOLS_ROOT} rl_tools EXCLUDE_FROM_ALL)

include(${HYPERLUX_NATIVE_ROOT}/cmake/hyperlux_jit.cmake)
hyperlux_jit_find_nanobind()

nanobind_add_module(hyperdrone_rotorcraft_core rotorcraft.cpp)
target_include_directories(hyperdrone_rotorcraft_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../common ${HYPERLUX_NATIVE_ROOT}/common)
target_link_libraries(hyperdrone_rotorcraft_core PRIVATE RLtools::RLtools)
set_target_properties(hyperdrone_rotorcraft_core PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
