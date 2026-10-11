add_executable(rendering_raytracing_benchmark benchmark.cpp)
target_link_libraries(rendering_raytracing_benchmark
    PRIVATE
        RLtools::RLtools
        rendering_raytracing_backend
        rendering_raytracing_stb
)

add_executable(rendering_raytracing_benchmark_64 EXCLUDE_FROM_ALL benchmark.cpp)
target_link_libraries(rendering_raytracing_benchmark_64
    PRIVATE
        RLtools::RLtools
        rendering_raytracing_backend
        rendering_raytracing_stb
)
target_compile_definitions(rendering_raytracing_benchmark_64 PRIVATE
    RL_TOOLS_RENDERING_RAYTRACING_BENCHMARK_WIDTH=64
    RL_TOOLS_RENDERING_RAYTRACING_BENCHMARK_HEIGHT=64
)

function(add_rendering_raytracing_benchmark_dynamic SUFFIX)
    cmake_parse_arguments(ARG "" "SHADING;OUTPUT_MODE;AA;MB;CONSUME;TILE;NUM_CAMERAS" "" ${ARGN})
    set(TARGET_NAME rendering_raytracing_benchmark_dynamic_${SUFFIX})
    add_executable(${TARGET_NAME} EXCLUDE_FROM_ALL benchmark_dynamic.cpp)
    if(ARG_OUTPUT_MODE AND ARG_OUTPUT_MODE EQUAL 1)
        set(BACKEND_TARGET rendering_raytracing_backend_depth)
    else()
        set(BACKEND_TARGET rendering_raytracing_backend)
    endif()
    target_link_libraries(${TARGET_NAME} PRIVATE RLtools::RLtools ${BACKEND_TARGET} rendering_raytracing_stb)
    foreach(OPTION SHADING OUTPUT_MODE AA MB CONSUME TILE NUM_CAMERAS)
        if(DEFINED ARG_${OPTION})
            if(OPTION STREQUAL "AA")
                set(DEFINE_NAME AA_GRID)
            elseif(OPTION STREQUAL "MB")
                set(DEFINE_NAME MB_SAMPLES)
            else()
                set(DEFINE_NAME ${OPTION})
            endif()
            target_compile_definitions(${TARGET_NAME} PRIVATE RL_TOOLS_RENDERING_RAYTRACING_BENCHMARK_DYNAMIC_${DEFINE_NAME}=${ARG_${OPTION}})
        endif()
    endforeach()
endfunction()

add_rendering_raytracing_benchmark_dynamic(low SHADING 0)
add_rendering_raytracing_benchmark_dynamic(medium SHADING 1)
add_rendering_raytracing_benchmark_dynamic(high SHADING 2)
add_rendering_raytracing_benchmark_dynamic(veryhigh SHADING 3)
add_rendering_raytracing_benchmark_dynamic(high_aa2 SHADING 2 AA 2)
add_rendering_raytracing_benchmark_dynamic(high_mb4 SHADING 2 MB 4)
add_rendering_raytracing_benchmark_dynamic(high_rgbd SHADING 2 OUTPUT_MODE 1)
add_rendering_raytracing_benchmark_dynamic(high_consume SHADING 2 CONSUME 1)
add_rendering_raytracing_benchmark_dynamic(high_tile2 SHADING 2 TILE 2)
add_rendering_raytracing_benchmark_dynamic(high_n256 SHADING 2 NUM_CAMERAS 256)

add_executable(rendering_raytracing_benchmark_depth EXCLUDE_FROM_ALL benchmark.cpp)
target_link_libraries(rendering_raytracing_benchmark_depth
    PRIVATE
        RLtools::RLtools
        rendering_raytracing_backend_depth
        rendering_raytracing_stb
)
target_compile_definitions(rendering_raytracing_benchmark_depth PRIVATE RL_TOOLS_RENDERING_RAYTRACING_BENCHMARK_OUTPUT_MODE=2)

add_executable(rendering_raytracing_benchmark_rgbd EXCLUDE_FROM_ALL benchmark.cpp)
target_link_libraries(rendering_raytracing_benchmark_rgbd
    PRIVATE
        RLtools::RLtools
        rendering_raytracing_backend_depth
        rendering_raytracing_stb
)
target_compile_definitions(rendering_raytracing_benchmark_rgbd PRIVATE RL_TOOLS_RENDERING_RAYTRACING_BENCHMARK_OUTPUT_MODE=1)

if(RL_TOOLS_RENDERING_RAYTRACING_BACKEND_RESOLVED MATCHES "^(OPTIX|METAL|VULKAN)$")
    set(RL_TOOLS_RENDERING_RAYTRACING_SIM_BENCHMARK_NUM_ENVS "4096" CACHE STRING "Compile-time camera count for the render benchmark")
    if(NOT RL_TOOLS_RENDERING_RAYTRACING_SIM_BENCHMARK_NUM_ENVS MATCHES "^[1-9][0-9]*$")
        message(FATAL_ERROR "RL_TOOLS_RENDERING_RAYTRACING_SIM_BENCHMARK_NUM_ENVS must be a positive integer")
    endif()

    function(add_rendering_raytracing_sim_target target_name shading mode)
        set(SIM_SHADING ${shading})
        set(SIM_MODE ${mode})
        set(SIM_PAPER false)
        set(SIM_PHYSICS_BACKEND none)
        set(SIM_SOURCE_HASH "")
        if(mode STREQUAL "paper")
            set(SIM_PAPER true)
            set(SIM_MODE benchmark)
            set(SIM_PHYSICS_BACKEND cpu)
            if(RL_TOOLS_BACKEND_ENABLE_CUDA AND RL_TOOLS_RENDERING_RAYTRACING_BACKEND_RESOLVED STREQUAL "OPTIX")
                set(SIM_PHYSICS_BACKEND cuda)
            endif()
            set(SIM_SOURCE_IDENTITY "")
            foreach(SOURCE CMakeLists.txt simulator_matrix_config.h.in simulator_matrix.cpp
                    simulator_matrix_physics.h simulator_matrix_physics_contract.h simulator_matrix_physics.cu
                    simulator_matrix_physics_cpu.cpp)
                set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE}")
                file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE}" SOURCE_HASH)
                string(APPEND SIM_SOURCE_IDENTITY "${SOURCE}:${SOURCE_HASH}\n")
            endforeach()
            string(SHA256 SIM_SOURCE_HASH "${SIM_SOURCE_IDENTITY}")
        endif()
        if(mode STREQUAL "frame")
            set(SIM_NUM_CAMERAS 1)
            set(SIM_RESOLUTION 1000)
        else()
            set(SIM_NUM_CAMERAS ${RL_TOOLS_RENDERING_RAYTRACING_SIM_BENCHMARK_NUM_ENVS})
            set(SIM_RESOLUTION 64)
        endif()
        set(CONFIG_DIR "${CMAKE_CURRENT_BINARY_DIR}/${target_name}_config")
        configure_file(simulator_matrix_config.h.in "${CONFIG_DIR}/simulator_matrix_config.h" @ONLY)
        add_executable(${target_name} EXCLUDE_FROM_ALL simulator_matrix.cpp)
        target_include_directories(${target_name} PRIVATE "${CONFIG_DIR}")
        target_link_libraries(${target_name} PRIVATE RLtools::RLtools
            rendering_raytracing_backend rendering_raytracing_backend_depth rendering_raytracing_stb)
        if(SIM_PHYSICS_BACKEND STREQUAL "cuda")
            target_sources(${target_name} PRIVATE simulator_matrix_physics.cu)
            set_target_properties(${target_name} PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
        elseif(SIM_PHYSICS_BACKEND STREQUAL "cpu")
            target_sources(${target_name} PRIVATE simulator_matrix_physics_cpu.cpp)
        endif()
    endfunction()

    foreach(PROFILE Medium High Low VeryHigh)
        string(TOLOWER "${PROFILE}" SUFFIX)
        if(PROFILE STREQUAL "VeryHigh")
            set(SUFFIX very_high)
        endif()
        add_rendering_raytracing_sim_target(rendering_raytracing_sim_benchmark_${SUFFIX} ${PROFILE} benchmark)
        add_rendering_raytracing_sim_target(rendering_raytracing_procthor_frame_${SUFFIX} ${PROFILE} frame)
    endforeach()
    add_rendering_raytracing_sim_target(rendering_raytracing_sim_benchmark_sweep Medium sweep)
    add_rendering_raytracing_sim_target(rendering_raytracing_paper_benchmark Medium paper)
endif()

add_executable(rendering_raytracing_rig_benchmark EXCLUDE_FROM_ALL rig.cpp)
target_link_libraries(rendering_raytracing_rig_benchmark PRIVATE RLtools::RLtools rendering_raytracing_backend rendering_raytracing_stb)
