cmake_minimum_required(VERSION 3.19)
project(layer_in_c)

set(CMAKE_C_FLAGS "-O3")

set(CMAKE_CXX_STANDARD 17)
find_package(HDF5 REQUIRED COMPONENTS C HL)

add_executable(bare_metal src/main.cpp)


add_subdirectory(tests)