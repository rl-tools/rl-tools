cmake_minimum_required(VERSION 3.24)
project(hyperdrone_render LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(RLTOOLS_ROOT "" CACHE PATH "Path to the rl-tools source root")
set(HYPERDRONE_VARIANT "OPTIX" CACHE STRING "Raytracing backend (OPTIX|METAL|VULKAN|WEBGPU|GENERIC)")

if(NOT RLTOOLS_ROOT)
    message(FATAL_ERROR "RLTOOLS_ROOT must point to the rl-tools source root")
endif()

# rl_tools_full carries the backend compile definitions and dependencies (OWL/assimp/
# nlohmann_json); targets are disabled so the dead-pinned procthor2glb subtree is never
# configured. The backend device-program libraries are pulled in separately below.
set(RL_TOOLS_ENABLE_TARGETS OFF CACHE BOOL "" FORCE)
set(RL_TOOLS_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
# the ExTrack git snapshot re-runs on every build (~10s of pure overhead per Renderer
# construction) and is useless for the JIT cache
set(RL_TOOLS_ENABLE_GIT_DIFF OFF CACHE BOOL "" FORCE)
set(RL_TOOLS_WARNINGS_AS_ERRORS OFF CACHE BOOL "" FORCE)
set(RL_TOOLS_RENDERING_ENABLE_RAYTRACING ON CACHE BOOL "" FORCE)
set(RL_TOOLS_RENDERING_RAYTRACING_BACKEND ${HYPERDRONE_VARIANT} CACHE STRING "" FORCE)
# language enablement is directory-scoped too: the backends subtree calls embed_ptx and
# needs CUDA enabled from here, not just inside the rl_tools subtree
if(HYPERDRONE_VARIANT STREQUAL "OPTIX")
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()
if(HYPERDRONE_VARIANT STREQUAL "VULKAN")
    find_package(Vulkan REQUIRED COMPONENTS glslangValidator)
endif()

# imported targets are directory-scoped; the backends subtree below needs assimp visible
# from this scope, not just inside the rl_tools subtree
find_package(assimp REQUIRED)
add_subdirectory(${RLTOOLS_ROOT} rl_tools EXCLUDE_FROM_ALL)
# the resolved-backend variable is directory-scoped inside the rl_tools subtree, so the
# backends dispatch needs it set again here
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
add_library(hyperdrone_render_stb INTERFACE)
target_include_directories(hyperdrone_render_stb SYSTEM INTERFACE ${stb_SOURCE_DIR})

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(
    COMMAND ${Python_EXECUTABLE} -m nanobind --cmake_dir
    OUTPUT_VARIABLE nanobind_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(hyperdrone_render_core host.cpp scene_bindings.cpp)
target_include_directories(hyperdrone_render_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../common)
target_link_libraries(hyperdrone_render_core PRIVATE RLtools::RLtools hyperdrone_render_stb ${CMAKE_DL_LIBS})
set_target_properties(hyperdrone_render_core PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
if(HYPERDRONE_VARIANT STREQUAL "OPTIX")
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(hyperdrone_render_core PRIVATE CUDA::cudart)
endif()

# per-config JIT artifacts: the output mode selects the backend device-program library
# (0=rgb 1=rgbd 2=depth 3=segmentation 4=rgbd_segmentation)
function(hyperdrone_component_libraries DEFINES OUT_LIBRARIES)
    hyperdrone_jit_parse_defines("${DEFINES}" CONFIG)
    if(NOT DEFINED CONFIG_HYPERDRONE_RENDER_OUTPUT_MODE)
        set(CONFIG_HYPERDRONE_RENDER_OUTPUT_MODE 0)
    endif()
    if(CONFIG_HYPERDRONE_RENDER_OUTPUT_MODE EQUAL 0)
        set(${OUT_LIBRARIES} rendering_raytracing_backend PARENT_SCOPE)
    elseif(CONFIG_HYPERDRONE_RENDER_OUTPUT_MODE EQUAL 1 OR CONFIG_HYPERDRONE_RENDER_OUTPUT_MODE EQUAL 2)
        set(${OUT_LIBRARIES} rendering_raytracing_backend_depth PARENT_SCOPE)
    else()
        set(${OUT_LIBRARIES} rendering_raytracing_backend_segmentation PARENT_SCOPE)
    endif()
endfunction()

include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/hyperdrone_jit.cmake)
hyperdrone_jit_add_configs(
    COMPONENT render
    SOURCES impl.cpp
    LINK RLtools::RLtools hyperdrone_render_stb
)
