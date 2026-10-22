# Minimal reference component for hyperlux._jit's own test suite: no rl-tools
# dependency, no core module — just per-config JIT artifacts. Doubles as the smallest
# possible example of a component CMakeLists.
cmake_minimum_required(VERSION 3.24)
project(hyperlux_toy LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(${HYPERLUX_NATIVE_ROOT}/cmake/hyperlux_jit.cmake)
hyperlux_jit_add_configs(
    COMPONENT toy
    SOURCES impl.cpp
)
