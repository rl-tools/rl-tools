cmake_minimum_required(VERSION 3.16)
# INTERPROCEDURAL_OPTIMIZATION is enforced when enabled.
set(CMAKE_POLICY_DEFAULT_CMP0069 NEW)
# Avoid BUILD_SHARED_LIBS getting overridden by an option() in ccd.
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
project(rl_tools)

set(CMAKE_CXX_STANDARD 17)

message("Build type: ${CMAKE_BUILD_TYPE}")
message("C++ Compiler: ${CMAKE_CXX_COMPILER}")


find_package(Git)
if(GIT_FOUND)
    execute_process(
            COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            OUTPUT_VARIABLE BACKPROP_TOOLS_COMMIT_HASH
            OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    message("Commit hash: ${BACKPROP_TOOLS_COMMIT_HASH}")
endif()

option(BACKPROP_TOOLS_ENABLE_TESTS "Build the unit tests when enabled." off)

# To obtain MKL: sudo apt install intel-oneapi-mkl-devel
option(BACKPROP_TOOLS_ENABLE_HDF5 "Enable HDF5 to allow persisting data to disk" OFF)
option(BACKPROP_TOOLS_ENABLE_TENSORBOARD "Enable the Tensorboard logging interface" OFF)
option(BACKPROP_TOOLS_ENABLE_LIBATTOPNG "Enable the libattopng library to log images to TensorBoard" OFF)
option(BACKPROP_TOOLS_BACKEND_ENABLE_MKL "Enable the MKL gemm library, only available on Intel CPUs" OFF)
option(BACKPROP_TOOLS_BACKEND_ENABLE_ACCELERATE "Enable the Accelerate gemm library, only available on Apple CPUs" OFF)
option(BACKPROP_TOOLS_BACKEND_ENABLE_CUDA "Enable Nvidia CUDA" OFF)
option(BACKPROP_TOOLS_RL_ENVIRONMENTS_ENABLE_MUJOCO "Enable MuJoCo" OFF)
option(BACKPROP_TOOLS_ENABLE_JSON "Enable JSON (used in the multirotor UI)" OFF)
option(BACKPROP_TOOLS_ENABLE_BOOST_BEAST "Enable BOOST beast (used in the multirotor UI)" OFF)
option(BACKPROP_TOOLS_HDF5_HIGHFIVE_FROM_CONAN "Use HighFive from conan (don't use the submodule from /external)" OFF)
option(BACKPROP_TOOLS_BUILD_TYPE "Use \"Release\" to e.g. build fat binaries with code for all CUDA architectures (not just for the GPU found in the machine used to build it)" OFF)
option(BACKPROP_TOOLS_INSTALL_INCLUDE_REDISTRIBUTABLES "Including redistributables from the redistributable submodule" OFF)
option(BACKPROP_TOOLS_DISABLE_LTO "Disable link-time optimization" OFF)

if(CMAKE_BUILD_TYPE STREQUAL "Release" AND NOT BACKPROP_TOOLS_DISABLE_LTO)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
endif()

set(BUILD_SHARED_LIBS OFF)

add_library(rl_tools_core INTERFACE) # This is the most "clean" target it just contains the include directories
target_include_directories(rl_tools_core INTERFACE include)
add_library(rl_tools_core_definitions INTERFACE) # This target additionally contains definitions
target_link_libraries(rl_tools_core_definitions INTERFACE rl_tools_core)
add_library(rl_tools_minimal INTERFACE) # This target additionally contains main and relatively harmless dependencies (BLAS, CLI11, TENSORBOARD, HDF5), the minimal config works with the CUDA comnpiler, when adding heavier dependencies in INTERFACE mode some cuda targets fail
add_library(rl_tools_optimizations INTERFACE) # This target contains optimization options
add_library(rl_tools_optimizations_cuda INTERFACE) # This target contains optimization options for CUDA
add_library(rl_tools INTERFACE) # This is the full target (containing, all configured dependencies including UI, Mujoco etc.)


set(BACKPROP_TOOLS_INSTALL_PREFIX rl_tools)


if(BACKPROP_TOOLS_RL_ENVIRONMENTS_ENABLE_MUJOCO)
    if(BACKPROP_TOOLS_RL_ENVIRONMENTS_MUJOCO_ENABLE_UI)
#        find_package(glfw3 REQUIRED)
        add_subdirectory(external/glfw EXCLUDE_FROM_ALL)
        target_link_libraries(rl_tools INTERFACE glfw)
        target_compile_definitions(rl_tools_core_definitions INTERFACE BACKPROP_TOOLS_RL_ENVIRONMENTS_MUJOCO_ENABLE_UI)
        install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/redistributable/licenses/glfw.txt" DESTINATION ${BACKPROP_TOOLS_INSTALL_PREFIX}/licenses)
    endif()
    set(MUJOCO_BUILD_EXAMPLES OFF)
    set(MUJOCO_BUILD_SIMULATE OFF)
    set(MUJOCO_BUILD_TESTS OFF)
    set(MUJOCO_TEST_PYTHON_UTIL OFF)
    add_subdirectory(external/mujoco EXCLUDE_FROM_ALL)
    #    set_property(TARGET mujoco PROPERTY EXCLUDE_FROM_ALL TRUE)
    target_link_libraries(rl_tools INTERFACE mujoco)
    if(WIN32)
        add_custom_command(TARGET rl_tools POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:mujoco> $<TARGET_FILE_DIR:${target}>)
    endif()

    # MuJoCo has its own install instructions which are hard to overwrite so we copy it manually:
    if (UNIX)
        set(MUJOCO_INSTALL_DIR ${CMAKE_INSTALL_PREFIX}/${BACKPROP_TOOLS_INSTALL_PREFIX}/lib/)
    else()
        set(MUJOCO_INSTALL_DIR ${CMAKE_INSTALL_PREFIX}/${BACKPROP_TOOLS_INSTALL_PREFIX}/bin/)
    endif()
    install(CODE "
        message(STATUS \"Creating directory if it does not exist: ${MUJOCO_INSTALL_DIR}\")
        file(MAKE_DIRECTORY ${MUJOCO_INSTALL_DIR})
        message(STATUS \"Copying Mujoco library to: ${MUJOCO_INSTALL_DIR}\")
        execute_process(COMMAND \${CMAKE_COMMAND} -E copy_if_different
            \"\$<TARGET_FILE:mujoco>\"
            \"${MUJOCO_INSTALL_DIR}\"
        )
    ")

    # MuJoCo is build as a shared library and bleeds the configuration to other targets
    set(BUILD_SHARED_LIBS OFF)
endif()

# dependencies
if(BACKPROP_TOOLS_BACKEND_ENABLE_MKL)
    find_package(MKL REQUIRED)
    target_compile_definitions(rl_tools_core_definitions INTERFACE BACKPROP_TOOLS_BACKEND_ENABLE_MKL)
    target_link_libraries(rl_tools_minimal INTERFACE MKL::MKL)
    if(BACKPROP_TOOLS_INSTALL_INCLUDE_REDISTRIBUTABLES)
        if(WIN32)
            file(GLOB MKL_REDISTRIBUTABLES "${CMAKE_CURRENT_SOURCE_DIR}/redistributable/windows/mkl/*.dll")
            install(FILES ${MKL_REDISTRIBUTABLES} DESTINATION ${BACKPROP_TOOLS_INSTALL_PREFIX}/bin)
        else()
            file(GLOB MKL_REDISTRIBUTABLES
                "${CMAKE_CURRENT_SOURCE_DIR}/redistributable/linux/mkl/*.so"
                "${CMAKE_CURRENT_SOURCE_DIR}/redistributable/linux/mkl/*.so.[0-9]*"
            )
            install(FILES ${MKL_REDISTRIBUTABLES} DESTINATION ${BACKPROP_TOOLS_INSTALL_PREFIX}/lib)
        endif()
        install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/redistributable/licenses/mkl.pdf" DESTINATION ${BACKPROP_TOOLS_INSTALL_PREFIX}/licenses)
    endif()
    set(BACKPROP_TOOLS_BACKEND_ENABLE_BLAS ON)
endif()

if(BACKPROP_TOOLS_BACKEND_ENABLE_ACCELERATE)
    find_package(BLAS REQUIRED)
    target_compile_definitions(rl_tools_core_definitions INTERFACE BACKPROP_TOOLS_BACKEND_ENABLE_ACCELERATE)
    target_link_libraries(rl_tools_minimal INTERFACE BLAS)
    set(BACKPROP_TOOLS_BACKEND_ENABLE_BLAS ON)
endif()

if(BACKPROP_TOOLS_BACKEND_ENABLE_OPENBLAS)
#    find_package(OpenBLAS REQUIRED)
    find_package(BLAS REQUIRED)
    target_compile_definitions(rl_tools_core_definitions INTERFACE BACKPROP_TOOLS_BACKEND_ENABLE_OPENBLAS)
    target_link_libraries(rl_tools_minimal INTERFACE ${BLAS_LIBRARIES})

#    message(STATUS "OpenBLAS_INCLUDE_DIRS: ${OpenBLAS_INCLUDE_DIRS}")
#    message(STATUS "OpenBLAS_LIBRARIES: ${OpenBLAS_LIBRARIES}")

    set(BACKPROP_TOOLS_BACKEND_ENABLE_BLAS ON)
endif()

if(BACKPROP_TOOLS_BACKEND_ENABLE_CUDA)
    if(BACKPROP_TOOLS_BUILD_TYPE STREQUAL "Release")
        set(CMAKE_CUDA_ARCHITECTURES "all")
    else()
        set(CMAKE_CUDA_ARCHITECTURES "native")
    endif()
    find_package(CUDAToolkit REQUIRED)
    message(STATUS "CUDA version: ${CUDA_VERSION}")
    message(STATUS "CUDA include path: ${CUDA_INCLUDE_DIRS}")
    message(STATUS "CUDA libraries: ${CUDA_LIBRARIES}")
    message(STATUS "CUDA cublas library: ${CUDA_CUBLAS_LIBRARIES}")
    target_compile_definitions(rl_tools_core_definitions INTERFACE BACKPROP_TOOLS_BACKEND_ENABLE_CUDA)
    target_link_libraries(rl_tools_minimal INTERFACE CUDA::cublas)
    if(BACKPROP_TOOLS_INSTALL_INCLUDE_REDISTRIBUTABLES)
        if(WIN32)
            file(GLOB CUDA_REDISTRIBUTABLES "${CMAKE_CURRENT_SOURCE_DIR}/redistributable/windows/cuda/*.dll")
            install(FILES ${CUDA_REDISTRIBUTABLES} DESTINATION ${BACKPROP_TOOLS_INSTALL_PREFIX}/bin)
        else()
            file(GLOB CUDA_REDISTRIBUTABLES
                "${CMAKE_CURRENT_SOURCE_DIR}/redistributable/linux/cuda/*.so"
                "${CMAKE_CURRENT_SOURCE_DIR}/redistributable/linux/cuda/*.so.[0-9]*"
            )
            message(STATUS "CUDA redistributables: ${CUDA_REDISTRIBUTABLES}")
            install(FILES ${CUDA_REDISTRIBUTABLES} DESTINATION ${BACKPROP_TOOLS_INSTALL_PREFIX}/lib)
        endif()
    endif()
endif()

if(BACKPROP_TOOLS_ENABLE_CLI11)
    add_subdirectory(external/cli11 EXCLUDE_FROM_ALL)

    target_link_libraries(rl_tools_minimal INTERFACE CLI11::CLI11)
    target_compile_definitions(rl_tools_core_definitions INTERFACE BACKPROP_TOOLS_ENABLE_CLI11)

    install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/redistributable/licenses/cli11.txt" DESTINATION ${BACKPROP_TOOLS_INSTALL_PREFIX}/licenses)
endif()

if(BACKPROP_TOOLS_ENABLE_HDF5)
    message(STATUS "HDF5 enabled")
    if(BACKPROP_TOOLS_HDF5_HIGHFIVE_FROM_CONAN)
        find_package(HighFive REQUIRED)
    else()
        add_subdirectory(external/highfive EXCLUDE_FROM_ALL)
    endif()

    target_link_libraries(rl_tools_minimal INTERFACE HighFive)
    target_compile_definitions(rl_tools_core_definitions INTERFACE BACKPROP_TOOLS_ENABLE_HDF5)

    install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/redistributable/licenses/highfive.txt" DESTINATION ${BACKPROP_TOOLS_INSTALL_PREFIX}/licenses)
endif()

if(BACKPROP_TOOLS_ENABLE_TENSORBOARD)
	find_package(Protobuf REQUIRED)
    add_subdirectory(external/tensorboard EXCLUDE_FROM_ALL)
    get_target_property(library_type tensorboard_logger TYPE)
    message(STATUS "Tensorboard library type: ${library_type}")
    target_link_libraries(rl_tools_minimal INTERFACE tensorboard_logger)
    target_compile_definitions(rl_tools_core_definitions INTERFACE BACKPROP_TOOLS_ENABLE_TENSORBOARD)
    install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/redistributable/licenses/protobuf.txt" DESTINATION ${BACKPROP_TOOLS_INSTALL_PREFIX}/licenses)
    install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/redistributable/licenses/tensorboard_logger.txt" DESTINATION ${BACKPROP_TOOLS_INSTALL_PREFIX}/licenses)
endif()
if(BACKPROP_TOOLS_ENABLE_LIBATTOPNG)
    add_library(libattopng
            external/libattopng/libattopng.c
    )
    target_include_directories(libattopng PUBLIC external/libattopng)
    target_compile_definitions(rl_tools_core_definitions INTERFACE BACKPROP_TOOLS_ENABLE_LIBATTOPNG)
    target_link_libraries(rl_tools_minimal INTERFACE libattopng)
endif()
if(BACKPROP_TOOLS_ENABLE_JSON)
    message(STATUS "JSON enabled")
    add_subdirectory(external/json EXCLUDE_FROM_ALL)
    target_link_libraries(rl_tools_minimal INTERFACE nlohmann_json::nlohmann_json)
    target_compile_definitions(rl_tools_core_definitions INTERFACE BACKPROP_TOOLS_ENABLE_JSON)
endif()

if(BACKPROP_TOOLS_ENABLE_GTK)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(GTK3 REQUIRED gtk+-3.0)
    target_link_libraries(rl_tools INTERFACE ${GTK3_LIBRARIES})
    target_include_directories(rl_tools INTERFACE ${GTK3_INCLUDE_DIRS})
    target_compile_options(rl_tools INTERFACE ${GTK3_CFLAGS_OTHER})
    target_link_directories(rl_tools INTERFACE ${GTK3_LIBRARY_DIRS})
endif()

if(BACKPROP_TOOLS_ENABLE_SDL2)
    find_package(SDL2 REQUIRED)
endif()

if(BACKPROP_TOOLS_COMMIT_HASH)
    target_compile_definitions(rl_tools_core_definitions INTERFACE BACKPROP_TOOLS_COMMIT_HASH=${BACKPROP_TOOLS_COMMIT_HASH})
endif()

if(BACKPROP_TOOLS_TEST_MACHINE_LENOVO_P1)
    target_compile_definitions(rl_tools_core_definitions INTERFACE BACKPROP_TOOLS_TEST_MACHINE_LENOVO_P1)
endif()
if(BACKPROP_TOOLS_TEST_MACHINE_MACBOOK_M1)
    target_compile_definitions(rl_tools_core_definitions INTERFACE BACKPROP_TOOLS_TEST_MACHINE_MACBOOK_M1)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(rl_tools_core_definitions INTERFACE -DBACKPROP_TOOLS_DEBUG_DEVICE_CUDA_CHECK_INIT)
    target_compile_definitions(rl_tools_core_definitions INTERFACE -DBACKPROP_TOOLS_DEBUG_CONTAINER_CHECK_BOUNDS)
    target_compile_definitions(rl_tools_core_definitions INTERFACE -DBACKPROP_TOOLS_DEBUG_CONTAINER_CHECK_MALLOC)
    target_compile_definitions(rl_tools_core_definitions INTERFACE -DBACKPROP_TOOLS_DEBUG_CONTAINER_MALLOC_INIT_NAN)
    target_compile_definitions(rl_tools_core_definitions INTERFACE -DBACKPROP_TOOLS_DEBUG_RL_COMPONENTS_OFF_POLICY_RUNNER_CHECK_INIT)
    target_compile_definitions(rl_tools_core_definitions INTERFACE -DBACKPROP_TOOLS_DEBUG_RL_COMPONENTS_OFF_POLICY_RUNNER_GATHER_BATCH_CHECK_REPLAY_BUFFER_POSITION)
    target_compile_definitions(rl_tools_core_definitions INTERFACE -DBACKPROP_TOOLS_DEBUG_RL_COMPONENTS_ON_POLICY_RUNNER_CHECK_INIT)
    target_compile_definitions(rl_tools_core_definitions INTERFACE -DBACKPROP_TOOLS_DEBUG_RL_ALGORITHMS_PPO_GAE_CHECK_TERMINATED_TRUNCATED)
    target_compile_definitions(rl_tools_core_definitions INTERFACE -DBACKPROP_TOOLS_DEBUG_RL_ALGORITHMS_PPO_CHECK_INIT)
    target_compile_definitions(rl_tools_core_definitions INTERFACE -DBACKPROP_TOOLS_DEBUG_RL_ENVIRONMENTS_MUJOCO_CHECK_INIT)
endif()
#if(CMAKE_BUILD_TYPE STREQUAL "Release")
#    #        target_compile_options(${TARGET} PRIVATE )
#    get_target_property(_is_cuda ${TARGET} IS_CUDA)
#    if(_is_cuda STREQUAL "YES" OR APPLE)
#    else()
#        get_target_property(_objs ${TARGET} SOURCES)
#        foreach(_obj ${_objs})
#            if(_obj MATCHES "\\.cu")
#                message(FATAL_ERROR "Target ${TARGET} is a CUDA target and the IS_CUDA property is not set!")
#                break()
#            endif()
#        endforeach()
#        target_compile_options(${TARGET} PRIVATE -march=native -mtune=native -ffast-math)
#    endif()
#endif()
if(BACKPROP_TOOLS_BUILD_TYPE STREQUAL "Release")
    if (UNIX)
        if (APPLE)
            set_target_properties(rl_tools PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE INSTALL_RPATH "@loader_path/../lib")
        else()
            set_target_properties(rl_tools PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE INSTALL_RPATH "$ORIGIN/../lib")
        endif()
    elseif (WIN32)
    endif()
    target_compile_definitions(rl_tools_core_definitions PRIVATE -DBACKPROP_TOOLS_RELEASE)

endif()
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(NOT APPLE AND NOT WIN32)
        target_compile_options(rl_tools_optimizations INTERFACE -ffast-math -march=native -mtune=native -mavx2)
        target_compile_options(rl_tools_optimizations_cuda INTERFACE -use_fast_math --optimize 3)# --dlink-time-opt --gen-opt-lto)
    elseif(APPLE)
        target_compile_options(rl_tools_optimizations INTERFACE -Ofast)
    endif()
endif()

function(BACKPROP_TOOLS_INSTALL TARGET)
    install(TARGETS ${TARGET} RUNTIME DESTINATION ${BACKPROP_TOOLS_INSTALL_PREFIX}/bin)
endfunction()

function(BACKPROP_TOOLS_TAG_IS_CUDA TARGET)
    set_target_properties(${TARGET} PROPERTIES IS_CUDA YES)
endfunction()


if (BACKPROP_TOOLS_ENABLE_TESTS)
    message(STATUS "Tests enabled")
    include(CTest)
    enable_testing()
    add_subdirectory(tests)
endif()

configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/rl_tools.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/rl_toolsConfig.cmake"
        @ONLY
)

if(BACKPROP_TOOLS_INSTALL_INCLUDE_REDISTRIBUTABLES)
    install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/release/readme.txt" DESTINATION ${BACKPROP_TOOLS_INSTALL_PREFIX}/)
    if(WIN32)
        file(GLOB MSVC_REDISTRIBUTABLES "${CMAKE_CURRENT_SOURCE_DIR}/redistributable/windows/msvc/*.dll")
        install(FILES ${MSVC_REDISTRIBUTABLES} DESTINATION ${BACKPROP_TOOLS_INSTALL_PREFIX}/bin)
        install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/release/windows/readme_windows.txt" DESTINATION ${BACKPROP_TOOLS_INSTALL_PREFIX})
    elseif(APPLE)
        install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/release/mac/readme_mac.txt" DESTINATION ${BACKPROP_TOOLS_INSTALL_PREFIX})
    elseif(UNIX)
        install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/release/linux/readme_linux.txt" DESTINATION ${BACKPROP_TOOLS_INSTALL_PREFIX})
    endif()
endif()

target_link_libraries(rl_tools_minimal INTERFACE rl_tools_core_definitions)
target_link_libraries(rl_tools_optimizations INTERFACE rl_tools_minimal)
target_link_libraries(rl_tools INTERFACE rl_tools_optimizations)

add_library(rl_tools::core ALIAS rl_tools_core)
add_library(rl_tools::core_definitions ALIAS rl_tools_core_definitions)
add_library(rl_tools::minimal ALIAS rl_tools_minimal)
add_library(rl_tools::optimizations ALIAS rl_tools_optimizations)
add_library(rl_tools::optimizations_cuda ALIAS rl_tools_optimizations_cuda)
add_library(rl_tools::rl_tools ALIAS rl_tools)

add_subdirectory(src)