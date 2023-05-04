cmake_minimum_required(VERSION 3.16)
project(backprop_tools)

set(CMAKE_CXX_STANDARD 17)

message("Build type: ${CMAKE_BUILD_TYPE}")
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
endif()
message("C++ Compiler: ${CMAKE_CXX_COMPILER}")

option(BACKPROP_TOOLS_ENABLE_TESTS "Build the unit tests when enabled." off)

# To obtain MKL: sudo apt install intel-oneapi-mkl-devel
option(BACKPROP_TOOLS_ENABLE_HDF5 "Enable HDF5 to allow persisting data to disk" OFF)
option(BACKPROP_TOOLS_ENABLE_TENSORBOARD "Enable the Tensorboard logging interface" OFF)
option(BACKPROP_TOOLS_BACKEND_ENABLE_MKL "Enable the MKL gemm library, only available on Intel CPUs" OFF)
option(BACKPROP_TOOLS_BACKEND_ENABLE_ACCELERATE "Enable the Accelerate gemm library, only available on Apple CPUs" OFF)
option(BACKPROP_TOOLS_BACKEND_ENABLE_CUDA "Enable Nvidia CUDA" OFF)
option(BACKPROP_TOOLS_RL_ENVIRONMENTS_ENABLE_MUJOCO "Enable MuJoCo" OFF)
option(BACKPROP_TOOLS_HDF5_HIGHFIVE_FROM_CONAN "Use HighFive from conan (don't use the submodule from /external)" OFF)


add_library(backprop_tools INTERFACE)
target_include_directories(backprop_tools INTERFACE include)


# dependencies
if(BACKPROP_TOOLS_BACKEND_ENABLE_MKL)
    find_package(MKL REQUIRED)
    add_compile_definitions(BACKPROP_TOOLS_BACKEND_ENABLE_MKL)
endif()

if(BACKPROP_TOOLS_BACKEND_ENABLE_ACCELERATE)
    find_package(BLAS REQUIRED)
    add_compile_definitions(BACKPROP_TOOLS_BACKEND_ENABLE_ACCELERATE)
endif()

function(BACKPROP_TOOLS_TARGET_LINK_BLAS target)
    if(BACKPROP_TOOLS_BACKEND_ENABLE_MKL)
        message(STATUS "Linking MKL to ${target}")
        target_link_libraries(${target} PRIVATE MKL::MKL)
    endif()
    if(BACKPROP_TOOLS_BACKEND_ENABLE_ACCELERATE)
        message(STATUS "Linking ACCELERATE to ${target}")
        target_link_libraries(${target} PRIVATE BLAS)
    endif()
endfunction()

if(BACKPROP_TOOLS_BACKEND_ENABLE_CUDA)
    set(BACKPROP_TOOLS_BACKEND_CUDA_ARCHITECTURES "native")
    find_package(CUDAToolkit REQUIRED)
    message(STATUS "CUDA version: ${CUDA_VERSION}")
    message(STATUS "CUDA include path: ${CUDA_INCLUDE_DIRS}")
    message(STATUS "CUDA libraries: ${CUDA_LIBRARIES}")
    message(STATUS "CUDA cublas library: ${CUDA_CUBLAS_LIBRARIES}")
endif()

if(BACKPROP_TOOLS_ENABLE_CLI11)
    add_subdirectory(external/cli11 EXCLUDE_FROM_ALL)
endif()

if(BACKPROP_TOOLS_ENABLE_HDF5)
    message(STATUS "HDF5 enabled")
    if(BACKPROP_TOOLS_HDF5_HIGHFIVE_FROM_CONAN)
        find_package(HighFive REQUIRED)
    else()
        add_subdirectory(external/highfive EXCLUDE_FROM_ALL)
    endif()
endif()

if(BACKPROP_TOOLS_ENABLE_TENSORBOARD)
	find_package(Protobuf REQUIRED)
    add_subdirectory(external/tensorboard EXCLUDE_FROM_ALL)
    get_target_property(library_type tensorboard_logger TYPE)
    message(STATUS "Tensorboard library type: ${library_type}")
endif()

if(BACKPROP_TOOLS_RL_ENVIRONMENTS_ENABLE_MUJOCO)
    if(BACKPROP_TOOLS_RL_ENVIRONMENTS_MUJOCO_ENABLE_UI)
        find_package(glfw3 REQUIRED)
    endif()
    set(MUJOCO_BUILD_EXAMPLES OFF)
    set(MUJOCO_BUILD_SIMULATE OFF)
    set(MUJOCO_BUILD_TESTS OFF)
    set(MUJOCO_TEST_PYTHON_UTIL OFF)
    add_subdirectory(external/mujoco)
    function (BACKPROP_TOOLS_TARGET_LINK_MUJOCO target)
        target_link_libraries(${target} PRIVATE mujoco)
        if(WIN32)
            add_custom_command(TARGET ${target} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    $<TARGET_FILE:mujoco>
                    $<TARGET_FILE_DIR:${target}>)
        endif()
    endfunction()
endif()

function(BACKPROP_TOOLS_ADD_DEFINITIONS TARGET)
    if(BACKPROP_TOOLS_BACKEND_ENABLE_ACCELERATE)
        add_compile_definitions(BACKPROP_TOOLS_BACKEND_ENABLE_ACCELERATE)
    endif()
    if(BACKPROP_TOOLS_BACKEND_ENABLE_MKL)
        add_compile_definitions(BACKPROP_TOOLS_BACKEND_ENABLE_MKL)
    endif()
    if(BACKPROP_TOOLS_BACKEND_ENABLE_CUDA)
        add_compile_definitions(BACKPROP_TOOLS_BACKEND_ENABLE_CUDA)
    endif()
    if(BACKPROP_TOOLS_ENABLE_TENSORBOARD)
        target_compile_definitions(${TARGET} PRIVATE BACKPROP_TOOLS_ENABLE_TENSORBOARD)
    endif()
    if(BACKPROP_TOOLS_ENABLE_HDF5)
        target_compile_definitions(${TARGET} PRIVATE BACKPROP_TOOLS_ENABLE_HDF5)
    endif()

    if(BACKPROP_TOOLS_TEST_MACHINE_LENOVO_P1)
        target_compile_definitions(${TARGET} PRIVATE BACKPROP_TOOLS_TEST_MACHINE_LENOVO_P1)
    endif()
    if(BACKPROP_TOOLS_TEST_MACHINE_MACBOOK_M1)
        target_compile_definitions(${TARGET} PRIVATE BACKPROP_TOOLS_TEST_MACHINE_MACBOOK_M1)
    endif()

    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        message (STATUS "Adding debug definitions to ${TARGET}")
        target_compile_definitions(${TARGET} PRIVATE -DBACKPROP_TOOLS_DEBUG_DEVICE_CUDA_CHECK_INIT)
        target_compile_definitions(${TARGET} PRIVATE -DBACKPROP_TOOLS_DEBUG_CONTAINER_CHECK_BOUNDS)
        target_compile_definitions(${TARGET} PRIVATE -DBACKPROP_TOOLS_DEBUG_CONTAINER_CHECK_MALLOC)
        target_compile_definitions(${TARGET} PRIVATE -DBACKPROP_TOOLS_DEBUG_CONTAINER_MALLOC_INIT_NAN)
        target_compile_definitions(${TARGET} PRIVATE -DBACKPROP_TOOLS_DEBUG_RL_COMPONENTS_OFF_POLICY_RUNNER_CHECK_INIT)
        target_compile_definitions(${TARGET} PRIVATE -DBACKPROP_TOOLS_DEBUG_RL_COMPONENTS_OFF_POLICY_RUNNER_GATHER_BATCH_CHECK_REPLAY_BUFFER_POSITION)
        target_compile_definitions(${TARGET} PRIVATE -DBACKPROP_TOOLS_DEBUG_RL_COMPONENTS_ON_POLICY_RUNNER_CHECK_INIT)
        target_compile_definitions(${TARGET} PRIVATE -DBACKPROP_TOOLS_DEBUG_RL_ALGORITHMS_PPO_GAE_CHECK_TERMINATED_TRUNCATED)
        target_compile_definitions(${TARGET} PRIVATE -DBACKPROP_TOOLS_DEBUG_RL_ALGORITHMS_PPO_CHECK_INIT)
        target_compile_definitions(${TARGET} PRIVATE -DBACKPROP_TOOLS_DEBUG_RL_ENVIRONMENTS_MUJOCO_CHECK_INIT)
    endif()
endfunction()


if (BACKPROP_TOOLS_ENABLE_TESTS)
    include(CTest)
    enable_testing()
    add_subdirectory(tests)
endif()

configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/backprop_tools.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/backprop_toolsConfig.cmake"
        @ONLY
)

add_subdirectory(src)