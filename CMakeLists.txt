cmake_minimum_required(VERSION 3.16)
# INTERPROCEDURAL_OPTIMIZATION is enforced when enabled.
set(CMAKE_POLICY_DEFAULT_CMP0069 NEW)
# Avoid BUILD_SHARED_LIBS getting overridden by an option() in ccd.
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
project(RLtools VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

message("Build type: ${CMAKE_BUILD_TYPE}")
message("C++ Compiler: ${CMAKE_CXX_COMPILER}")


find_package(Git)
if(GIT_FOUND)
    execute_process(
            COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            OUTPUT_VARIABLE RL_TOOLS_COMMIT_HASH
            OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    message("Commit hash: ${RL_TOOLS_COMMIT_HASH}")
endif()

option(RL_TOOLS_ENABLE_TARGETS "Disable default targets" OFF)
option(RL_TOOLS_ENABLE_TESTS "Build the unit tests when enabled." off)

# To obtain MKL: sudo apt install intel-oneapi-mkl-devel
option(RL_TOOLS_ENABLE_HDF5 "Enable HDF5 to allow persisting data to disk" OFF)
option(RL_TOOLS_ENABLE_TENSORBOARD "Enable the Tensorboard logging interface" OFF)
option(RL_TOOLS_ENABLE_LIBATTOPNG "Enable the libattopng library to log images to TensorBoard" OFF)
option(RL_TOOLS_ENABLE_GTK "Enable the GTK UI library" OFF)
option(RL_TOOLS_BACKEND_ENABLE_MKL "Enable the MKL gemm library, only available on Intel CPUs" OFF)
option(RL_TOOLS_BACKEND_ENABLE_ACCELERATE "Enable the Accelerate gemm library, only available on Apple CPUs" OFF)
option(RL_TOOLS_BACKEND_ENABLE_CUDA "Enable Nvidia CUDA" OFF)
option(RL_TOOLS_RL_ENVIRONMENTS_ENABLE_MUJOCO "Enable MuJoCo" OFF)
option(RL_TOOLS_ENABLE_JSON "Enable JSON (used in the multirotor UI)" OFF)
option(RL_TOOLS_ENABLE_BOOST_BEAST "Enable BOOST beast (used in the multirotor UI)" OFF)
option(RL_TOOLS_HDF5_HIGHFIVE_FROM_CONAN "Use HighFive from conan (don't use the submodule from /external)" OFF)
option(RL_TOOLS_BUILD_TYPE "Use \"Release\" to e.g. build fat binaries with code for all CUDA architectures (not just for the GPU found in the machine used to build it)" OFF)
option(RL_TOOLS_INSTALL_INCLUDE_REDISTRIBUTABLES "Including redistributables from the redistributable submodule" OFF)
option(RL_TOOLS_ENABLE_LTO "Enable link-time optimization" OFF)
option(RL_TOOLS_ENABLE_FAST_MATH "Enable (possibly unsafe) fast math operations" OFF)
option(RL_TOOLS_COMPILER_FLAGS "Additional compiler flags" OFF)
option(RL_TOOLS_DISABLE_CPU_SPECIFIC_OPTIMIZATIONS "Disable CPU specific optimizations (e.g. for building portable Docker containers)" OFF)

if(CMAKE_BUILD_TYPE STREQUAL "Release" AND RL_TOOLS_ENABLE_LTO)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
endif()

set(BUILD_SHARED_LIBS OFF)


# Targets
# RLtools::Core: Plain C++ library with no dependencies, definitions or anything (just requiring cxx_std_17 and adding the header-search path)
# RLtools::CoreDefinitions: Core + Definitions (e.g. RL_TOOLS_BACKEND_ENABLE_MKL)
# RLtools::Backend: CoreDefinitions + Backend (e.g. MKL, Accelerate)
# RLtools::Minimal: Backend + Minimal dependencies (e.g. CLI11, Tensorboard, HDF5)
# RLtools::Optimizations: Minimal + Optimizations (e.g. -ffast-math, -march=native)
# RLtools::OptimizationsCUDA: Optimizations + CUDA optimizations (e.g. --use_fast_math)
# RLtools::RLtools: OptimizationsCUDA + everything else (e.g. UI, Mujoco)

add_library(rl_tools_core INTERFACE) # This is the most "clean" target it just contains the include directories
set_property(TARGET rl_tools_core PROPERTY EXPORT_NAME Core)
target_compile_features(rl_tools_core INTERFACE cxx_std_17)
target_include_directories(
    rl_tools_core INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
        "${PROJECT_BINARY_DIR}/RLtoolsConfigVersion.cmake"
        VERSION 0.1
        COMPATIBILITY AnyNewerVersion
        ARCH_INDEPENDENT
)
configure_package_config_file(
        "${PROJECT_SOURCE_DIR}/cmake/RLtoolsConfig.cmake.in"
        "${PROJECT_BINARY_DIR}/RLtoolsConfig.cmake"
        INSTALL_DESTINATION lib/cmake/RLtools
)
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/ DESTINATION include)
install(FILES "${PROJECT_BINARY_DIR}/RLtoolsConfigVersion.cmake"
        "${PROJECT_BINARY_DIR}/RLtoolsConfig.cmake"
        DESTINATION lib/cmake/RLtools)
add_library(rl_tools_core_definitions INTERFACE) # This target additionally contains definitions
set_property(TARGET rl_tools_core_definitions PROPERTY EXPORT_NAME CoreDefinitions)
target_link_libraries(rl_tools_core_definitions INTERFACE rl_tools_core)
add_library(rl_tools_backend INTERFACE)
set_property(TARGET rl_tools_backend PROPERTY EXPORT_NAME Backend)
add_library(rl_tools_minimal INTERFACE) # This target additionally contains main and relatively harmless dependencies (BLAS, CLI11, TENSORBOARD, HDF5), the minimal config works with the CUDA comnpiler, when adding heavier dependencies in INTERFACE mode some cuda targets fail
set_property(TARGET rl_tools_minimal PROPERTY EXPORT_NAME Minimal)
add_library(rl_tools_optimizations INTERFACE) # This target contains optimization options
set_property(TARGET rl_tools_optimizations PROPERTY EXPORT_NAME Optimizations)
add_library(rl_tools_optimizations_cuda INTERFACE) # This target contains optimization options for CUDA
set_property(TARGET rl_tools_optimizations_cuda PROPERTY EXPORT_NAME OptimizationsCUDA)
add_library(rl_tools INTERFACE) # This is the full target (containing, all configured dependencies including UI, Mujoco etc.)
set_property(TARGET rl_tools PROPERTY EXPORT_NAME RLtools)

install(TARGETS
        rl_tools_core
        rl_tools_core_definitions
        rl_tools_backend
        rl_tools_minimal
        rl_tools_optimizations
        rl_tools_optimizations_cuda
        rl_tools
        EXPORT RLtoolsTargets
        LIBRARY DESTINATION lib COMPONENT Runtime
        ARCHIVE DESTINATION lib COMPONENT Development
        RUNTIME DESTINATION bin COMPONENT Runtime
        PUBLIC_HEADER DESTINATION include COMPONENT Development
        BUNDLE DESTINATION bin COMPONENT Runtime
)
install(EXPORT RLtoolsTargets NAMESPACE RLtools:: DESTINATION lib/cmake/RLtools)


set(RL_TOOLS_INSTALL_PREFIX ".")


if(RL_TOOLS_RL_ENVIRONMENTS_ENABLE_MUJOCO)
    if(RL_TOOLS_RL_ENVIRONMENTS_MUJOCO_ENABLE_UI)
#        find_package(glfw3 REQUIRED)
        add_subdirectory(external/glfw)
        target_link_libraries(rl_tools INTERFACE glfw)
        target_compile_definitions(rl_tools_core_definitions INTERFACE RL_TOOLS_RL_ENVIRONMENTS_MUJOCO_ENABLE_UI)
        install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/redistributable/licenses/glfw.txt" DESTINATION licenses)
    endif()
    set(MUJOCO_BUILD_EXAMPLES OFF)
    set(MUJOCO_BUILD_SIMULATE OFF)
    set(MUJOCO_BUILD_TESTS OFF)
    set(MUJOCO_TEST_PYTHON_UTIL OFF)
    add_subdirectory(external/mujoco)
    #    set_property(TARGET mujoco PROPERTY EXCLUDE_FROM_ALL TRUE)
    target_link_libraries(rl_tools INTERFACE mujoco)
#    if(WIN32)
#        add_custom_command(TARGET rl_tools POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:mujoco> $<TARGET_FILE_DIR:${target}>)
#    endif()
#
#    # MuJoCo has its own install instructions which are hard to overwrite so we copy it manually:
#    if (UNIX)
#        set(MUJOCO_INSTALL_DIR ${CMAKE_INSTALL_PREFIX}/lib/)
#    else()
#        set(MUJOCO_INSTALL_DIR ${CMAKE_INSTALL_PREFIX}/bin/)
#    endif()
#    install(CODE "
#        message(STATUS \"Creating directory if it does not exist: ${MUJOCO_INSTALL_DIR}\")
#        file(MAKE_DIRECTORY ${MUJOCO_INSTALL_DIR})
#        message(STATUS \"Copying Mujoco library to: ${MUJOCO_INSTALL_DIR}\")
#        execute_process(COMMAND \${CMAKE_COMMAND} -E copy_if_different
#            \"\$<TARGET_FILE:mujoco>\"
#            \"${MUJOCO_INSTALL_DIR}\"
#        )
#    ")

    # MuJoCo is build as a shared library and bleeds the configuration to other targets
    set(BUILD_SHARED_LIBS OFF)
endif()

# dependencies
if(RL_TOOLS_BACKEND_ENABLE_MKL)
    find_package(MKL REQUIRED)
    target_compile_definitions(rl_tools_core_definitions INTERFACE RL_TOOLS_BACKEND_ENABLE_MKL)
    target_link_libraries(rl_tools_backend INTERFACE MKL::MKL)
    if(RL_TOOLS_INSTALL_INCLUDE_REDISTRIBUTABLES)
        if(WIN32)
            file(GLOB MKL_REDISTRIBUTABLES "${CMAKE_CURRENT_SOURCE_DIR}/redistributable/windows/mkl/*.dll")
            install(FILES ${MKL_REDISTRIBUTABLES} DESTINATION bin)
        else()
            file(GLOB MKL_REDISTRIBUTABLES
                "${CMAKE_CURRENT_SOURCE_DIR}/redistributable/linux/mkl/*.so"
                "${CMAKE_CURRENT_SOURCE_DIR}/redistributable/linux/mkl/*.so.[0-9]*"
            )
            install(FILES ${MKL_REDISTRIBUTABLES} DESTINATION lib)
        endif()
        install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/redistributable/licenses/mkl.pdf" DESTINATION licenses)
    endif()
    set(RL_TOOLS_BACKEND_ENABLE_BLAS ON)
endif()

if(RL_TOOLS_BACKEND_ENABLE_ACCELERATE)
    find_package(BLAS REQUIRED)
    target_compile_definitions(rl_tools_core_definitions INTERFACE RL_TOOLS_BACKEND_ENABLE_ACCELERATE)
    target_link_libraries(rl_tools_backend INTERFACE BLAS)
    set(RL_TOOLS_BACKEND_ENABLE_BLAS ON)
endif()

if(RL_TOOLS_BACKEND_ENABLE_OPENBLAS)
#    find_package(OpenBLAS REQUIRED)
    find_package(BLAS REQUIRED)
    target_compile_definitions(rl_tools_core_definitions INTERFACE RL_TOOLS_BACKEND_ENABLE_OPENBLAS)
    target_link_libraries(rl_tools_backend INTERFACE ${BLAS_LIBRARIES})

#    message(STATUS "OpenBLAS_INCLUDE_DIRS: ${OpenBLAS_INCLUDE_DIRS}")
#    message(STATUS "OpenBLAS_LIBRARIES: ${OpenBLAS_LIBRARIES}")

    set(RL_TOOLS_BACKEND_ENABLE_BLAS ON)
endif()

if(RL_TOOLS_BACKEND_ENABLE_CUDA)
    cmake_minimum_required(VERSION 3.17)
    if(RL_TOOLS_BUILD_TYPE STREQUAL "Release")
        set(CMAKE_CUDA_ARCHITECTURES "all")
    else()
        set(CMAKE_CUDA_ARCHITECTURES "native")
    endif()
    find_package(CUDAToolkit REQUIRED)
    message(STATUS "CUDAToolkit version: ${CUDAToolkit_VERSION}")
    message(STATUS "CUDAToolkit include path: ${CUDAToolkit_INCLUDE_DIRS}")
    message(STATUS "CUDAToolkit libraries: ${CUDAToolkit_LIBRARIES}")
    message(STATUS "CUDAToolkit cublas library: ${CUDAToolkit_CUBLAS_LIBRARIES}")
    target_compile_definitions(rl_tools_core_definitions INTERFACE RL_TOOLS_BACKEND_ENABLE_CUDA)
    target_link_libraries(rl_tools_backend INTERFACE CUDA::cublas)
    if(RL_TOOLS_INSTALL_INCLUDE_REDISTRIBUTABLES)
        if(WIN32)
            file(GLOB CUDA_REDISTRIBUTABLES "${CMAKE_CURRENT_SOURCE_DIR}/redistributable/windows/cuda/*.dll")
            install(FILES ${CUDA_REDISTRIBUTABLES} DESTINATION bin)
        else()
            file(GLOB CUDA_REDISTRIBUTABLES
                "${CMAKE_CURRENT_SOURCE_DIR}/redistributable/linux/cuda/*.so"
                "${CMAKE_CURRENT_SOURCE_DIR}/redistributable/linux/cuda/*.so.[0-9]*"
            )
            message(STATUS "CUDA redistributables: ${CUDA_REDISTRIBUTABLES}")
            install(FILES ${CUDA_REDISTRIBUTABLES} DESTINATION lib)
        endif()
    endif()
endif()

if(RL_TOOLS_ENABLE_CLI11)
    add_subdirectory(external/cli11)
    target_compile_definitions(rl_tools_core_definitions INTERFACE RL_TOOLS_ENABLE_CLI11)
    install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/redistributable/licenses/cli11.txt" DESTINATION licenses)
endif()

if(RL_TOOLS_ENABLE_HDF5)
    message(STATUS "HDF5 enabled")
    if(RL_TOOLS_HDF5_HIGHFIVE_FROM_CONAN)
        find_package(HighFive REQUIRED)
    else()
        add_subdirectory(external/highfive)
    endif()

    target_link_libraries(rl_tools_minimal INTERFACE HighFive)
    target_compile_definitions(rl_tools_minimal INTERFACE RL_TOOLS_ENABLE_HDF5)

    install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/redistributable/licenses/highfive.txt" DESTINATION licenses)
endif()

if(RL_TOOLS_ENABLE_TENSORBOARD)
	find_package(Protobuf REQUIRED)
    add_subdirectory(external/tensorboard)
    get_target_property(library_type tensorboard_logger TYPE)
    message(STATUS "Tensorboard library type: ${library_type}")
    target_link_libraries(rl_tools_minimal INTERFACE tensorboard_logger)
    target_compile_definitions(rl_tools_minimal INTERFACE RL_TOOLS_ENABLE_TENSORBOARD)
    install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/redistributable/licenses/protobuf.txt" DESTINATION licenses)
    install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/redistributable/licenses/tensorboard_logger.txt" DESTINATION licenses)
endif()
if(RL_TOOLS_ENABLE_LIBATTOPNG)
    add_subdirectory(external/libattopng)
    target_compile_definitions(rl_tools_core_definitions INTERFACE RL_TOOLS_ENABLE_LIBATTOPNG)
    target_link_libraries(rl_tools_minimal INTERFACE libattopng)
    install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/external/tibattopng/libattopng.h" DESTINATION include/libattopng)
endif()
if(RL_TOOLS_ENABLE_JSON)
    message(STATUS "JSON enabled")
    set(JSON_Install ON)
    add_subdirectory(external/json)
    target_link_libraries(rl_tools_minimal INTERFACE nlohmann_json::nlohmann_json)
    target_compile_definitions(rl_tools_minimal INTERFACE RL_TOOLS_ENABLE_JSON)
endif()
if(RL_TOOLS_ENABLE_BOOST_BEAST)
find_package(Boost REQUIRED COMPONENTS system filesystem)
message(STATUS "Boost version: ${Boost_VERSION}")
target_include_directories(rl_tools_minimal INTERFACE ${Boost_INCLUDE_DIRS})
target_compile_definitions(rl_tools_minimal INTERFACE RL_TOOLS_ENABLE_BOOST_BEAST)
endif()

if(RL_TOOLS_ENABLE_GTK)
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        find_package(PkgConfig REQUIRED)
        pkg_check_modules(GTK3 REQUIRED gtk+-3.0)
    endif()
    target_link_libraries(rl_tools INTERFACE ${GTK3_LIBRARIES})
    target_include_directories(rl_tools INTERFACE ${GTK3_INCLUDE_DIRS})
    target_compile_options(rl_tools INTERFACE ${GTK3_CFLAGS_OTHER})
    target_link_directories(rl_tools INTERFACE ${GTK3_LIBRARY_DIRS})
    target_compile_definitions(rl_tools INTERFACE RL_TOOLS_ENABLE_GTK)
endif()

if(RL_TOOLS_ENABLE_SDL2)
    find_package(SDL2 REQUIRED)
endif()

if(RL_TOOLS_COMMIT_HASH)
    target_compile_definitions(rl_tools_core_definitions INTERFACE RL_TOOLS_COMMIT_HASH=${RL_TOOLS_COMMIT_HASH})
endif()

if(RL_TOOLS_TEST_MACHINE_LENOVO_P1)
    target_compile_definitions(rl_tools_core_definitions INTERFACE RL_TOOLS_TEST_MACHINE_LENOVO_P1)
endif()
if(RL_TOOLS_TEST_MACHINE_MACBOOK_M1)
    target_compile_definitions(rl_tools_core_definitions INTERFACE RL_TOOLS_TEST_MACHINE_MACBOOK_M1)
endif()


if(RL_TOOLS_CONTAINER_INIT_NAN)
    target_compile_definitions(rl_tools_core_definitions INTERFACE -DRL_TOOLS_DEBUG_CONTAINER_MALLOC_INIT_NAN)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(rl_tools_core_definitions INTERFACE -DRL_TOOLS_DEBUG)
    target_compile_definitions(rl_tools_core_definitions INTERFACE -DRL_TOOLS_DEBUG_DEVICE_CUDA_CHECK_INIT)
    target_compile_definitions(rl_tools_core_definitions INTERFACE -DRL_TOOLS_DEBUG_CONTAINER_CHECK_BOUNDS)
    target_compile_definitions(rl_tools_core_definitions INTERFACE -DRL_TOOLS_DEBUG_CONTAINER_CHECK_MALLOC)
    target_compile_definitions(rl_tools_core_definitions INTERFACE -DRL_TOOLS_DEBUG_CONTAINER_MALLOC_INIT_NAN)
    target_compile_definitions(rl_tools_core_definitions INTERFACE -DRL_TOOLS_DEBUG_RL_COMPONENTS_OFF_POLICY_RUNNER_CHECK_INIT)
    target_compile_definitions(rl_tools_core_definitions INTERFACE -DRL_TOOLS_DEBUG_RL_COMPONENTS_OFF_POLICY_RUNNER_GATHER_BATCH_CHECK_REPLAY_BUFFER_POSITION)
    target_compile_definitions(rl_tools_core_definitions INTERFACE -DRL_TOOLS_DEBUG_RL_COMPONENTS_ON_POLICY_RUNNER_CHECK_INIT)
    target_compile_definitions(rl_tools_core_definitions INTERFACE -DRL_TOOLS_DEBUG_RL_ALGORITHMS_PPO_GAE_CHECK_TERMINATED_TRUNCATED)
    target_compile_definitions(rl_tools_core_definitions INTERFACE -DRL_TOOLS_DEBUG_RL_ALGORITHMS_PPO_CHECK_INIT)
    target_compile_definitions(rl_tools_core_definitions INTERFACE -DRL_TOOLS_DEBUG_RL_ENVIRONMENTS_MUJOCO_CHECK_INIT)
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
if(RL_TOOLS_BUILD_TYPE STREQUAL "Release")
#    if (UNIX)
#        if (APPLE)
#            set_target_properties(rl_tools PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE INSTALL_RPATH "@loader_path/../lib")
#        else()
#            set_target_properties(rl_tools PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE INSTALL_RPATH "$ORIGIN/../lib")
#        endif()
#    elseif (WIN32)
#    endif()
    target_compile_definitions(rl_tools_core_definitions INTERFACE -DRL_TOOLS_RELEASE)
    if(WIN32)
        target_compile_definitions(rl_tools_core_definitions INTERFACE -DRL_TOOLS_RELEASE_WINDOWS)
    elseif(APPLE)
        target_compile_definitions(rl_tools_core_definitions INTERFACE -DRL_TOOLS_RELEASE_MACOS)
    elseif(UNIX)
        target_compile_definitions(rl_tools_core_definitions INTERFACE -DRL_TOOLS_RELEASE_LINUX)
    endif()
endif()
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(NOT APPLE AND NOT WIN32)
        if(RL_TOOLS_ENABLE_FAST_MATH)
            target_compile_options(rl_tools_optimizations INTERFACE -ffast-math) # -fno-unsafe-math-optimizations) #-fno-finite-math-only
            target_compile_options(rl_tools_optimizations_cuda INTERFACE -use_fast_math --optimize 3)# --dlink-time-opt --gen-opt-lto)
            target_compile_definitions(rl_tools_optimizations INTERFACE RL_TOOLS_ENABLE_FAST_MATH)
        endif()
        if(NOT RL_TOOLS_DISABLE_CPU_SPECIFIC_OPTIMIZATIONS)
            target_compile_options(rl_tools_optimizations INTERFACE -march=native)
        endif()
    elseif(APPLE)
        if(RL_TOOLS_ENABLE_FAST_MATH)
            target_compile_options(rl_tools_optimizations INTERFACE -Ofast)
            target_compile_definitions(rl_tools_optimizations INTERFACE RL_TOOLS_ENABLE_FAST_MATH)
        endif()
    endif()
endif()

function(RL_TOOLS_INSTALL TARGET)
        if (UNIX)
            if (APPLE)
#                set_target_properties(${TARGET} PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE INSTALL_RPATH "@loader_path/../lib")
                set_target_properties(${TARGET} PROPERTIES INSTALL_RPATH "@loader_path/../lib")
            else()
                set_target_properties(${TARGET} PROPERTIES INSTALL_RPATH "$ORIGIN/../lib")
            endif()
        elseif (WIN32)
        endif()
    install(TARGETS ${TARGET} RUNTIME DESTINATION bin)
endfunction()

function(RL_TOOLS_TAG_IS_CUDA TARGET)
    set_target_properties(${TARGET} PROPERTIES IS_CUDA YES)
endfunction()


if (RL_TOOLS_ENABLE_TESTS)
    message(STATUS "Tests enabled")
    include(CTest)
    enable_testing()
    add_subdirectory(tests)
endif()

install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/release/readme.txt" DESTINATION share/rl_tools)
if(WIN32)
    install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/release/windows/readme_windows.txt" DESTINATION share/rl_tools)
elseif(APPLE)
    install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/release/mac/readme_mac.txt" DESTINATION share/rl_tools)
elseif(UNIX)
    install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/release/linux/readme_linux.txt" DESTINATION share/rl_tools)
endif()

if(RL_TOOLS_INSTALL_INCLUDE_REDISTRIBUTABLES)
    if(WIN32)
        file(GLOB MSVC_REDISTRIBUTABLES "${CMAKE_CURRENT_SOURCE_DIR}/redistributable/windows/msvc/*.dll")
        install(FILES ${MSVC_REDISTRIBUTABLES} DESTINATION bin)
    endif()
endif()

target_link_libraries(rl_tools_backend INTERFACE rl_tools_core_definitions)
target_link_libraries(rl_tools_minimal INTERFACE rl_tools_backend)
target_link_libraries(rl_tools_optimizations INTERFACE rl_tools_minimal)
target_link_libraries(rl_tools INTERFACE rl_tools_optimizations)

add_library(RLtools::Core ALIAS rl_tools_core)
add_library(RLtools::CoreDefinitions ALIAS rl_tools_core_definitions)
add_library(RLtools::Backend ALIAS rl_tools_backend)
add_library(RLtools::Minimal ALIAS rl_tools_minimal)
add_library(RLtools::Optimizations ALIAS rl_tools_optimizations)
add_library(RLtools::OptimizationsCUDA ALIAS rl_tools_optimizations_cuda)
add_library(RLtools::RLtools ALIAS rl_tools)

if(RL_TOOLS_ENABLE_TARGETS)
add_subdirectory(src)
endif()