cmake_minimum_required(VERSION 3.24)
project(hypert LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(RLTOOLS_ROOT "" CACHE PATH "Path to the rl-tools repository root")
set(HYPERT_BACKEND "OPTIX" CACHE STRING "Raytracing backend (OPTIX|METAL|VULKAN|GENERIC)")

if(NOT RLTOOLS_ROOT)
    message(FATAL_ERROR "RLTOOLS_ROOT must point to the rl-tools repository root")
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
set(RL_TOOLS_RENDERING_RAYTRACING_BACKEND ${HYPERT_BACKEND} CACHE STRING "" FORCE)
# language enablement is directory-scoped too: the backends subtree calls embed_ptx and
# needs CUDA enabled from here, not just inside the rl_tools subtree
if(HYPERT_BACKEND STREQUAL "OPTIX")
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()
if(HYPERT_BACKEND STREQUAL "VULKAN")
    find_package(Vulkan REQUIRED COMPONENTS glslangValidator)
endif()

# imported targets are directory-scoped; the backends subtree below needs assimp visible
# from this scope, not just inside the rl_tools subtree
find_package(assimp REQUIRED)
add_subdirectory(${RLTOOLS_ROOT} rl_tools EXCLUDE_FROM_ALL)
# the resolved-backend variable is directory-scoped inside the rl_tools subtree, so the
# backends dispatch needs it set again here
set(RL_TOOLS_RENDERING_RAYTRACING_BACKEND_RESOLVED ${HYPERT_BACKEND})
add_subdirectory(${RLTOOLS_ROOT}/src/rendering/raytracing/backends rl_tools_rt_backends EXCLUDE_FROM_ALL)

include(FetchContent)
FetchContent_Declare(
    stb
    GIT_REPOSITORY https://github.com/nothings/stb.git
    GIT_TAG        f1c79c02822848a9bed4315b12c8c8f3761e1296
    EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(stb)
add_library(hypert_stb INTERFACE)
target_include_directories(hypert_stb SYSTEM INTERFACE ${stb_SOURCE_DIR})

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(
    COMMAND ${Python_EXECUTABLE} -m nanobind --cmake_dir
    OUTPUT_VARIABLE nanobind_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(hypert_core core.cpp)
target_link_libraries(hypert_core PRIVATE RLtools::RLtools hypert_stb ${CMAKE_DL_LIBS})
set_target_properties(hypert_core PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# JIT renderer configurations: one line per requested constant set, appended by the Python
# frontend. Editing the file triggers reconfiguration on the next cmake --build.
set(HYPERT_JIT_CONFIG_FILE ${CMAKE_BINARY_DIR}/jit_configs.txt)
if(NOT EXISTS ${HYPERT_JIT_CONFIG_FILE})
    file(TOUCH ${HYPERT_JIT_CONFIG_FILE})
endif()
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${HYPERT_JIT_CONFIG_FILE})

file(STRINGS ${HYPERT_JIT_CONFIG_FILE} HYPERT_JIT_CONFIGS)
foreach(HYPERT_JIT_CONFIG IN LISTS HYPERT_JIT_CONFIGS)
    separate_arguments(HYPERT_JIT_FIELDS UNIX_COMMAND ${HYPERT_JIT_CONFIG})
    list(LENGTH HYPERT_JIT_FIELDS HYPERT_JIT_NUM_FIELDS)
    if(NOT HYPERT_JIT_NUM_FIELDS EQUAL 13)
        message(FATAL_ERROR "hypert: malformed jit_configs.txt line: ${HYPERT_JIT_CONFIG}")
    endif()
    list(GET HYPERT_JIT_FIELDS 0 HYPERT_JIT_KEY)
    list(GET HYPERT_JIT_FIELDS 1 HYPERT_JIT_WIDTH)
    list(GET HYPERT_JIT_FIELDS 2 HYPERT_JIT_HEIGHT)
    list(GET HYPERT_JIT_FIELDS 3 HYPERT_JIT_NUM_CAMERAS)
    list(GET HYPERT_JIT_FIELDS 4 HYPERT_JIT_NUM_PROBES)
    list(GET HYPERT_JIT_FIELDS 5 HYPERT_JIT_SHADING)
    list(GET HYPERT_JIT_FIELDS 6 HYPERT_JIT_OUTPUT_MODE)
    list(GET HYPERT_JIT_FIELDS 7 HYPERT_JIT_MB_SAMPLES)
    list(GET HYPERT_JIT_FIELDS 8 HYPERT_JIT_AA_GRID)
    list(GET HYPERT_JIT_FIELDS 9 HYPERT_JIT_NUM_OVERLAYS)
    list(GET HYPERT_JIT_FIELDS 10 HYPERT_JIT_MAX_OVERLAY_INSTANCES)
    list(GET HYPERT_JIT_FIELDS 11 HYPERT_JIT_MAX_OVERLAYS_PER_CAMERA)
    list(GET HYPERT_JIT_FIELDS 12 HYPERT_JIT_SEMANTIC_SEGMENTATION)

    if(HYPERT_JIT_OUTPUT_MODE EQUAL 0)
        set(HYPERT_JIT_BACKEND_TARGET rendering_raytracing_backend)
    elseif(HYPERT_JIT_OUTPUT_MODE EQUAL 1 OR HYPERT_JIT_OUTPUT_MODE EQUAL 2)
        set(HYPERT_JIT_BACKEND_TARGET rendering_raytracing_backend_depth)
    else()
        set(HYPERT_JIT_BACKEND_TARGET rendering_raytracing_backend_segmentation)
    endif()

    set(HYPERT_JIT_TARGET hypert_renderer_${HYPERT_JIT_KEY})
    add_library(${HYPERT_JIT_TARGET} MODULE renderer_impl.cpp)
    target_include_directories(${HYPERT_JIT_TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(${HYPERT_JIT_TARGET} PRIVATE RLtools::RLtools ${HYPERT_JIT_BACKEND_TARGET} hypert_stb)
    target_compile_definitions(${HYPERT_JIT_TARGET} PRIVATE
        HYPERT_WIDTH=${HYPERT_JIT_WIDTH}
        HYPERT_HEIGHT=${HYPERT_JIT_HEIGHT}
        HYPERT_NUM_CAMERAS=${HYPERT_JIT_NUM_CAMERAS}
        HYPERT_NUM_PROBES=${HYPERT_JIT_NUM_PROBES}
        HYPERT_SHADING=${HYPERT_JIT_SHADING}
        HYPERT_OUTPUT_MODE=${HYPERT_JIT_OUTPUT_MODE}
        HYPERT_MB_SAMPLES=${HYPERT_JIT_MB_SAMPLES}
        HYPERT_AA_GRID=${HYPERT_JIT_AA_GRID}
        HYPERT_NUM_OVERLAYS=${HYPERT_JIT_NUM_OVERLAYS}
        HYPERT_MAX_OVERLAY_INSTANCES=${HYPERT_JIT_MAX_OVERLAY_INSTANCES}
        HYPERT_MAX_OVERLAYS_PER_CAMERA=${HYPERT_JIT_MAX_OVERLAYS_PER_CAMERA}
        HYPERT_SEMANTIC_SEGMENTATION=${HYPERT_JIT_SEMANTIC_SEGMENTATION}
    )
    set_target_properties(${HYPERT_JIT_TARGET} PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/jit
        PREFIX ""
    )
endforeach()
