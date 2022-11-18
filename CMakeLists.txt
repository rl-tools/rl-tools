cmake_minimum_required(VERSION 3.19)
project(bare_metal)

set(CMAKE_C_FLAGS "-O3")

set(CMAKE_CXX_STANDARD 14)
find_package(HDF5 REQUIRED COMPONENTS C HL)

add_executable(bare_metal main.cpp)


add_subdirectory(tests)