# Minimal reference component for hyperdrone.jit's own test suite: no rl-tools
# dependency, no core module — just per-config JIT artifacts. Doubles as the smallest
# possible example of a component CMakeLists.
cmake_minimum_required(VERSION 3.24)
project(hyperdrone_toy LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(${HYPERDRONE_NATIVE_ROOT}/cmake/hyperdrone_jit.cmake)
hyperdrone_jit_add_configs(
    COMPONENT toy
    SOURCES impl.cpp
)
