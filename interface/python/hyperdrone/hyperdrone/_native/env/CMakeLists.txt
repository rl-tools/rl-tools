cmake_minimum_required(VERSION 3.24)
project(hyperdrone_env LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(RLTOOLS_ROOT "" CACHE PATH "Path to the rl-tools source root")
set(HYPERDRONE_VARIANT "OPTIX" CACHE STRING "Raytracing backend (OPTIX|METAL|VULKAN|WEBGPU|GENERIC)")

if(NOT RLTOOLS_ROOT)
    message(FATAL_ERROR "RLTOOLS_ROOT must point to the rl-tools source root")
endif()

set(RL_TOOLS_DISABLE_TARGETS ON CACHE BOOL "" FORCE)
set(RL_TOOLS_DISABLE_TESTS ON CACHE BOOL "" FORCE)
set(RL_TOOLS_DISABLE_GIT_DIFF ON CACHE BOOL "" FORCE)
set(RL_TOOLS_RL_ENVIRONMENTS_DISABLE_MUJOCO ON CACHE BOOL "" FORCE)
set(RL_TOOLS_WARNINGS_AS_ERRORS OFF CACHE BOOL "" FORCE)
# an explicit backend selection hard-requires raytracing inside the rl_tools subtree
set(RL_TOOLS_RENDERING_RAYTRACING_BACKEND ${HYPERDRONE_VARIANT} CACHE STRING "" FORCE)
if(HYPERDRONE_VARIANT STREQUAL "OPTIX")
    # a normal variable before enable_language shadows the compiler default it caches, also in
    # trees configured before; CUDAARCHS overrides
    if(NOT DEFINED ENV{CUDAARCHS})
        set(CMAKE_CUDA_ARCHITECTURES native)
    endif()
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()
if(HYPERDRONE_VARIANT STREQUAL "VULKAN")
    find_package(Vulkan REQUIRED COMPONENTS glslangValidator)
endif()

unset(FETCHCONTENT_SOURCE_DIR_ASSIMP CACHE)
add_subdirectory(${RLTOOLS_ROOT} rl_tools EXCLUDE_FROM_ALL)
set(RL_TOOLS_RENDERING_RAYTRACING_BACKEND_RESOLVED ${HYPERDRONE_VARIANT})
add_subdirectory(${RLTOOLS_ROOT}/src/rendering/raytracing/backends rl_tools_rt_backends EXCLUDE_FROM_ALL)

include(FetchContent)
FetchContent_Declare(
    stb
    GIT_REPOSITORY https://github.com/nothings/stb.git
    GIT_TAG        f1c79c02822848a9bed4315b12c8c8f3761e1296
    EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(stb)
add_library(hyperdrone_env_stb INTERFACE)
target_include_directories(hyperdrone_env_stb SYSTEM INTERFACE ${stb_SOURCE_DIR})

include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/hyperdrone_jit.cmake)
hyperdrone_jit_find_nanobind()

# the environment's drone rig instantiates the render core's RigModel (shared struct header,
# type registered by the render core, which the Python wrapper loads first)
nanobind_add_module(hyperdrone_env_core rotorcraft_bindings.cpp)
target_include_directories(hyperdrone_env_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../common ${CMAKE_CURRENT_SOURCE_DIR}/../render)
target_link_libraries(hyperdrone_env_core PRIVATE RLtools::RLtools hyperdrone_env_stb)
set_target_properties(hyperdrone_env_core PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
if(HYPERDRONE_VARIANT STREQUAL "OPTIX")
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(hyperdrone_env_core PRIVATE CUDA::cudart)
endif()

function(hyperdrone_component_libraries DEFINES OUT_LIBRARIES)
    hyperdrone_jit_parse_defines("${DEFINES}" CONFIG)
    if(DEFINED CONFIG_HYPERDRONE_ENV_SPEC_HEADER)
        set(${OUT_LIBRARIES} rendering_raytracing_backend rendering_raytracing_backend_depth rendering_raytracing_backend_segmentation rendering_raytracing_backend_normals PARENT_SCOPE)
    elseif(CONFIG_HYPERDRONE_ENV_OUTPUT_NORMALS)
        set(${OUT_LIBRARIES} rendering_raytracing_backend_normals PARENT_SCOPE)
    elseif(CONFIG_HYPERDRONE_ENV_OUTPUT_INSTANCE_SEGMENTATION OR CONFIG_HYPERDRONE_ENV_OUTPUT_SEMANTIC_SEGMENTATION)
        set(${OUT_LIBRARIES} rendering_raytracing_backend_segmentation PARENT_SCOPE)
    elseif(CONFIG_HYPERDRONE_ENV_OUTPUT_DEPTH)
        set(${OUT_LIBRARIES} rendering_raytracing_backend_depth PARENT_SCOPE)
    else()
        set(${OUT_LIBRARIES} rendering_raytracing_backend PARENT_SCOPE)
    endif()
endfunction()

# under OptiX impl.cpp, which holds the environment behind the binding, is compiled by nvcc so
# the World can run on the GPU (EnvConfig.device); the nanobind translation unit stays C++
if(HYPERDRONE_VARIANT STREQUAL "OPTIX")
    add_compile_definitions(HYPERDRONE_ENV_CUDA=1)
    set_source_files_properties(impl.cpp PROPERTIES LANGUAGE CUDA)
endif()

set(hyperdrone_env_libraries RLtools::RLtools hyperdrone_env_stb)
if(TARGET RLtools::HyperDroneSpy)
    list(APPEND hyperdrone_env_libraries RLtools::HyperDroneSpy rendering_raytracing_backend)
endif()

hyperdrone_jit_add_configs(
    COMPONENT env
    PYTHON_MODULE
    SOURCES environment.cpp impl.cpp
    LINK ${hyperdrone_env_libraries}
)
