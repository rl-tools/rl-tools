#cmake_minimum_required(VERSION 3.14)
#cmake_minimum_required(VERSION 3.18) # for BLAS::BLAS
cmake_minimum_required(VERSION 3.19) # for HDF5::HDF5
set(CMAKE_POLICY_VERSION_MINIMUM 3.5) # to prevent warnings about "Compatibility with CMake < 3.5 has been removed from CMake."

project(rl_tools_core)

set(CMAKE_CXX_STANDARD 17)

# Everything auto-detects and auto-enables; the options below are opt-outs. Detection results
# are reported in the configure summary at the end and exposed as plain (non-cache) variables
# (RL_TOOLS_ENABLE_*, RL_TOOLS_BACKEND_ENABLE_*) plus compile definitions on rl_tools_full.
option(RL_TOOLS_OFFLINE_BUILD "Use the dependencies in .dependencies without checking for updates (useful after nuking the build directory when offline)" OFF)
option(RL_TOOLS_WARNINGS_AS_ERRORS "Treat warnings as errors" ON)
option(RL_TOOLS_DISABLE_FAST_MATH "Disable -ffast-math (/fp:fast on MSVC)" OFF)

option(RL_TOOLS_DISABLE_TARGETS "Disable building the main targets (src/)" OFF)
option(RL_TOOLS_DISABLE_TESTS "Disable building the tests" OFF)
option(RL_TOOLS_DISABLE_EXPERIMENTAL "Disable experimental targets" OFF)
option(RL_TOOLS_DISABLE_TAR "Disable tar checkpoint targets/tests" OFF)
option(RL_TOOLS_DISABLE_GIT_DIFF "Disable embedding git state into ExTrack runs" OFF)

option(RL_TOOLS_DISABLE_JSON "Disable nlohmann_json" OFF)
option(RL_TOOLS_DISABLE_HDF5 "Disable HDF5" OFF)
option(RL_TOOLS_DISABLE_ZLIB "Disable zlib" OFF)
option(RL_TOOLS_DISABLE_TENSORBOARD "Disable TensorBoard logging" OFF)
option(RL_TOOLS_DISABLE_CLI11 "Disable CLI11" OFF)

option(RL_TOOLS_BACKEND_DISABLE_BLAS "Disable all BLAS backends (naive fallback)" OFF)
option(RL_TOOLS_BACKEND_DISABLE_MKL "Disable the MKL BLAS backend" OFF)
option(RL_TOOLS_BACKEND_DISABLE_DNNL "Disable oneDNN (DNNL)" OFF)
option(RL_TOOLS_BACKEND_DISABLE_CUDA "Disable CUDA" OFF)
option(RL_TOOLS_BACKEND_DISABLE_CUDNN "Disable cuDNN" OFF)

option(RL_TOOLS_NUMERIC_TYPES_DISABLE_BF16 "Disable bf16 numeric type support" OFF)
option(RL_TOOLS_RL_ENVIRONMENTS_DISABLE_MUJOCO "Disable MuJoCo" OFF)
option(RL_TOOLS_TESTS_DISABLE_EIGEN "Disable Eigen-based benchmark tests" OFF)

option(RL_TOOLS_RENDERING_DISABLE_RAYTRACING "Disable raytracing" OFF)
option(RL_TOOLS_RENDERING_RAYTRACING_DISABLE_OPTIX "Exclude OptiX from raytracing backend auto-selection" OFF)
option(RL_TOOLS_RENDERING_RAYTRACING_DISABLE_METAL "Exclude Metal from raytracing backend auto-selection" OFF)
option(RL_TOOLS_RENDERING_RAYTRACING_DISABLE_VULKAN "Exclude Vulkan from raytracing backend auto-selection" OFF)
option(RL_TOOLS_RENDERING_RAYTRACING_DISABLE_WEBGPU "Exclude WebGPU from raytracing backend auto-selection" OFF)
set(RL_TOOLS_RENDERING_RAYTRACING_BACKEND "AUTO" CACHE STRING "Raytracing backend (AUTO|OPTIX|METAL|VULKAN|WEBGPU|GENERIC); AUTO probes availability in preference order OPTIX > METAL > VULKAN > WEBGPU > GENERIC, an explicit backend is hard-required")
set_property(CACHE RL_TOOLS_RENDERING_RAYTRACING_BACKEND PROPERTY STRINGS AUTO OPTIX METAL VULKAN WEBGPU GENERIC)

include(cmake/legacy_flags.cmake)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")
endif()

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
        target_compile_options(rl_tools_full INTERFACE -O3 -march=native)
        if(NOT RL_TOOLS_DISABLE_FAST_MATH)
            target_compile_options(rl_tools_full INTERFACE -ffast-math)
        endif()
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
elseif(MSVC)
    if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_compile_options(rl_tools_full INTERFACE /O2)
        if(NOT RL_TOOLS_DISABLE_FAST_MATH)
            target_compile_options(rl_tools_full INTERFACE /fp:fast)
        endif()
    endif()
    target_compile_options(rl_tools_full INTERFACE /bigobj)
    target_compile_definitions(rl_tools_full INTERFACE _CRT_SECURE_NO_WARNINGS)
    if(RL_TOOLS_WARNINGS_AS_ERRORS)
        target_compile_options(rl_tools_full INTERFACE /W3 /WX)
        target_compile_options(rl_tools_full INTERFACE
            /wd4100  # unreferenced formal parameter
            /wd4101  # unreferenced local variable
            /wd4146  # unary minus on unsigned type
            /wd4189  # local variable initialized but not referenced
            /wd4244  # conversion, possible loss of data
            /wd4267  # conversion from size_t, possible loss of data
            /wd4459  # declaration hides global
            /wd4996  # deprecated functions (getenv etc.)
            /wd4068  # unknown pragma (e.g. #pragma GCC)
            /wd4804  # unsafe use of type 'bool' in operation
        )
    endif()
endif()


# Configure FetchContent
get_filename_component(CMAKE_BINARY_DIR_BASENAME ${CMAKE_BINARY_DIR} NAME)
if(NOT DEFINED FETCHCONTENT_BASE_DIR)
    set(_test_file "${CMAKE_CURRENT_SOURCE_DIR}/.cmake_write_test")
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E touch "${_test_file}"
        RESULT_VARIABLE _write_test_result
        ERROR_QUIET
        OUTPUT_QUIET
    )
    if(_write_test_result EQUAL 0)
        file(REMOVE "${_test_file}")
        set(_fetchcontent_base_dir_root "${CMAKE_CURRENT_SOURCE_DIR}")
    else()
        set(_fetchcontent_base_dir_root "${CMAKE_CURRENT_BINARY_DIR}")
    endif()
    set(FETCHCONTENT_BASE_DIR "${_fetchcontent_base_dir_root}/.dependencies/${CMAKE_BINARY_DIR_BASENAME}" CACHE PATH "Base dir for FetchContent downloads" FORCE)
else()
    set(FETCHCONTENT_BASE_DIR "${FETCHCONTENT_BASE_DIR}/${CMAKE_BINARY_DIR_BASENAME}" CACHE PATH "Base dir for FetchContent downloads")
endif()
if(RL_TOOLS_OFFLINE_BUILD)
    set(FETCHCONTENT_FULLY_DISCONNECTED ON)
endif()
message(STATUS "RLtools dependencies: ${FETCHCONTENT_BASE_DIR}")
include(FetchContent)

# Suppress STATUS messages from FetchContent subprojects (arch details, feature
# summaries, proto generation lists, …) while still forwarding WARNING/ERROR.
# Requires CMake >= 3.25; silently ignored on older versions.
macro(rl_tools_fetchcontent_makeavailable_quiet)
    foreach(_rl_tools_fetch_name ${ARGN})
        string(TOLOWER "${_rl_tools_fetch_name}" _rl_tools_fetch_name_lower)
        string(TOUPPER "${_rl_tools_fetch_name}" _rl_tools_fetch_name_upper)
        if(NOT EXISTS "${FETCHCONTENT_BASE_DIR}/${_rl_tools_fetch_name_lower}-src" AND NOT DEFINED FETCHCONTENT_SOURCE_DIR_${_rl_tools_fetch_name_upper})
            message(STATUS "Fetching ${_rl_tools_fetch_name} (this can take a while)")
        endif()
    endforeach()
    set(_rl_tools_prev_log_level ${CMAKE_MESSAGE_LOG_LEVEL})
    set(CMAKE_MESSAGE_LOG_LEVEL ERROR)
    # the cmake_minimum_required compatibility deprecation only respects the CACHE variable
    if(DEFINED CACHE{CMAKE_WARN_DEPRECATED})
        set(_rl_tools_prev_warn_deprecated "$CACHE{CMAKE_WARN_DEPRECATED}")
    else()
        unset(_rl_tools_prev_warn_deprecated)
    endif()
    set(CMAKE_WARN_DEPRECATED OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(${ARGN})
    set(CMAKE_MESSAGE_LOG_LEVEL ${_rl_tools_prev_log_level})
    if(DEFINED _rl_tools_prev_warn_deprecated)
        set(CMAKE_WARN_DEPRECATED "${_rl_tools_prev_warn_deprecated}" CACHE BOOL "" FORCE)
    else()
        unset(CMAKE_WARN_DEPRECATED CACHE)
    endif()
endmacro()

include(cmake/autodetect/all.cmake)

include(cmake/optional/mujoco.cmake)
include(cmake/optional/raytracing.cmake)

if(NOT RL_TOOLS_DISABLE_TARGETS)
    add_subdirectory(src)
endif()

if(NOT RL_TOOLS_DISABLE_TESTS)
    if(RL_TOOLS_RENDERING_ENABLE_RAYTRACING AND RL_TOOLS_DISABLE_TARGETS)
        message(FATAL_ERROR "The raytracing tests link targets from src/; combine RL_TOOLS_DISABLE_TARGETS=ON with RL_TOOLS_DISABLE_TESTS=ON or RL_TOOLS_RENDERING_DISABLE_RAYTRACING=ON")
    endif()
    include(cmake/optional/googletest.cmake)
    if(RL_TOOLS_ENABLE_GTEST)
        if(NOT RL_TOOLS_TESTS_DISABLE_EIGEN)
            find_package(Eigen3 QUIET)
            if(Eigen3_FOUND)
                set(RL_TOOLS_TESTS_ENABLE_EIGEN ON)
                set(RL_TOOLS_SUMMARY_EIGEN_REASON "system ${Eigen3_VERSION}")
            else()
                set(RL_TOOLS_SUMMARY_EIGEN_REASON "not found (e.g. apt install libeigen3-dev)")
            endif()
        endif()
        if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/data)
            set(RL_TOOLS_TEST_DATA_PATH "${CMAKE_CURRENT_SOURCE_DIR}/tests/data" CACHE PATH "Path to test data")
            target_compile_definitions(rl_tools_full INTERFACE RL_TOOLS_TEST_DATA_PATH=${RL_TOOLS_TEST_DATA_PATH})
        endif()
        include(CTest)
        add_subdirectory(tests)
        set(RL_TOOLS_ENABLE_TESTS ON)
    else()
        set(RL_TOOLS_SUMMARY_TESTS_REASON "googletest unavailable (not found and no git)")
    endif()
endif()

include(cmake/autodetect/summary.cmake)
