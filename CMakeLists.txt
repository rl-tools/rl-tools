#cmake_minimum_required(VERSION 3.14)
#cmake_minimum_required(VERSION 3.18) # for BLAS::BLAS
cmake_minimum_required(VERSION 3.19) # for HDF5::HDF5
set(CMAKE_POLICY_VERSION_MINIMUM 3.5) # to prevent warnings about "Compatibility with CMake < 3.5 has been removed from CMake."

project(rl_tools_core)

option(RL_TOOLS_ENABLE_TARGETS "Enable building the main targets" ON)
option(RL_TOOLS_ENABLE_TESTS "Enable building the tests" ON)
option(RL_TOOLS_OFFLINE_BUILD "Use the dependencies in .dependencies without checking for updates (useful after nuking the build directory when offline)" OFF)
option(RL_TOOLS_WARNINGS_AS_ERRORS "Treat warnings as errors" ON)

add_library(rl_tools_core INTERFACE)
add_library(RLtools::Core ALIAS rl_tools_core)
target_compile_features(rl_tools_core INTERFACE cxx_std_17)
target_include_directories(rl_tools_core INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_library(rl_tools_full INTERFACE)
add_library(RLtools::RLtools ALIAS rl_tools_full)
target_link_libraries(rl_tools_full INTERFACE rl_tools_core)

# Compiler optimizations
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_compile_options(rl_tools_full INTERFACE -O3 -ffast-math -march=native)
    endif()
    if(RL_TOOLS_WARNINGS_AS_ERRORS)
        # Safe (just unused warnings)
        target_compile_options(rl_tools_full INTERFACE $<$<COMPILE_LANGUAGE:CXX>:-Werror -Wall -Wextra>) # the $<$<COMPILE_LANGUAGE:CXX>: is required for NVCC to not complain about -Wall
        target_compile_options(rl_tools_full INTERFACE $<$<COMPILE_LANGUAGE:CXX>:-Wno-unused-parameter -Wno-unused-local-typedefs -Wno-unused-variable -Wno-unused-but-set-variable>)
        if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            target_compile_options(rl_tools_full INTERFACE -Wno-unused-lambda-capture)
        endif()

        # todo (put exclusions here)
        if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            include(CheckCXXCompilerFlag)
            check_cxx_compiler_flag("-Wnan-infinity-disabled" HAS_NAN_INF_WARN)
            if (HAS_NAN_INF_WARN)
                target_compile_options(rl_tools_full INTERFACE $<$<COMPILE_LANGUAGE:CXX>:-Wno-error=nan-infinity-disabled>) # clang does not like some NaN handling in nlohmann_json when -ffast-math is on
            endif()
        endif()
    endif()
endif()


# Configure FetchContent
get_filename_component(CMAKE_BINARY_DIR_BASENAME ${CMAKE_BINARY_DIR} NAME)
if(NOT DEFINED FETCHCONTENT_BASE_DIR)
    set(FETCHCONTENT_BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/.dependencies/${CMAKE_BINARY_DIR_BASENAME}" CACHE PATH "Base dir for FetchContent downloads" FORCE)
else()
    set(FETCHCONTENT_BASE_DIR "${FETCHCONTENT_BASE_DIR}/${CMAKE_BINARY_DIR_BASENAME}" CACHE PATH "Base dir for FetchContent downloads")
endif()
if(RL_TOOLS_OFFLINE_BUILD)
    set(FETCHCONTENT_FULLY_DISCONNECTED ON)
endif()
include(FetchContent)

# tiers of optional dependencies
# tier 0: none
# tier 1: backend (MKL, OpenBLAS, Accelerate)
# tier 2: json, hdf5
# tier 3: tensorboard
# tier 4: cuda
include(cmake/tier1-blas.cmake)
include(cmake/tier2-json-hdf5-zlib.cmake)
include(cmake/tier3-tensorboard.cmake)
include(cmake/tier4-cuda.cmake)

include(cmake/git-diff.cmake)

if(RL_TOOLS_ENABLE_TESTS)
endif()

if(RL_TOOLS_ENABLE_TARGETS)
    add_subdirectory(src)
endif()

if (RL_TOOLS_ENABLE_TESTS)
    find_package(GTest)
    if(GTest_FOUND)
        include(CTest)
        add_subdirectory(tests)
    else()
        message(WARNING "GTest not found. Skipping tests. Disable RL_TOOLS_ENABLE_TESTS to suppress this warning.")
    endif()
endif()
