cmake_minimum_required(VERSION 3.16)
project(layer_in_c)

set(CMAKE_CXX_STANDARD 17)

option(LAYER_IN_C_BUILD_TESTS "Build the unit tests when enabled." off)

add_library(layer_in_c INTERFACE)
target_include_directories(layer_in_c INTERFACE include)

if (LAYER_IN_C_BUILD_TESTS)
    include(CTest)
    enable_testing()
    add_subdirectory(tests)
endif()


add_subdirectory(examples)