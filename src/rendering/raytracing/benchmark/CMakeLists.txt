add_executable(rendering_raytracing_benchmark benchmark.cpp)
target_link_libraries(rendering_raytracing_benchmark
    PRIVATE
        RLtools::RLtools
        owl::owl
        rendering_raytracing_device_ptx
        assimp::assimp
        rendering_raytracing_stb
)

add_executable(rendering_raytracing_benchmark_depth EXCLUDE_FROM_ALL benchmark.cpp)
target_link_libraries(rendering_raytracing_benchmark_depth
    PRIVATE
        RLtools::RLtools
        owl::owl
        rendering_raytracing_device_depth_ptx
        assimp::assimp
        rendering_raytracing_stb
)
target_compile_definitions(rendering_raytracing_benchmark_depth PRIVATE RL_TOOLS_RENDERING_RAYTRACING_BENCHMARK_OUTPUT_MODE=2)

add_executable(rendering_raytracing_benchmark_rgbd EXCLUDE_FROM_ALL benchmark.cpp)
target_link_libraries(rendering_raytracing_benchmark_rgbd
    PRIVATE
        RLtools::RLtools
        owl::owl
        rendering_raytracing_device_depth_ptx
        assimp::assimp
        rendering_raytracing_stb
)
target_compile_definitions(rendering_raytracing_benchmark_rgbd PRIVATE RL_TOOLS_RENDERING_RAYTRACING_BENCHMARK_OUTPUT_MODE=1)

set(RL_TOOLS_RENDERING_RAYTRACING_SIM_BENCHMARK_NUM_ENVS "4096" CACHE STRING "Compile-time number of parallel camera environments for the simulator matrix raytracing benchmark")

add_executable(rendering_raytracing_sim_benchmark simulator_matrix.cpp)
target_link_libraries(rendering_raytracing_sim_benchmark
    PRIVATE
        RLtools::RLtools
        owl::owl
        rendering_raytracing_device_ptx
        rendering_raytracing_device_depth_ptx
        assimp::assimp
        rendering_raytracing_stb
)
target_compile_definitions(rendering_raytracing_sim_benchmark
    PRIVATE
        RL_TOOLS_RENDERING_RAYTRACING_SIM_BENCHMARK_OUTPUT_MODE=-1
        RL_TOOLS_RENDERING_RAYTRACING_SIM_BENCHMARK_NUM_ENVS=${RL_TOOLS_RENDERING_RAYTRACING_SIM_BENCHMARK_NUM_ENVS}
)

add_executable(rendering_raytracing_sim_benchmark_sweep EXCLUDE_FROM_ALL simulator_matrix.cpp)
target_link_libraries(rendering_raytracing_sim_benchmark_sweep
    PRIVATE
        RLtools::RLtools
        owl::owl
        rendering_raytracing_device_ptx
        rendering_raytracing_device_depth_ptx
        assimp::assimp
        rendering_raytracing_stb
)
target_compile_definitions(rendering_raytracing_sim_benchmark_sweep
    PRIVATE
        RL_TOOLS_RENDERING_RAYTRACING_SIM_BENCHMARK_OUTPUT_MODE=-1
        RL_TOOLS_RENDERING_RAYTRACING_SIM_BENCHMARK_NUM_ENVS=${RL_TOOLS_RENDERING_RAYTRACING_SIM_BENCHMARK_NUM_ENVS}
        RL_TOOLS_RENDERING_RAYTRACING_SIM_BENCHMARK_SWEEP=1
        RL_TOOLS_RENDERING_RAYTRACING_SIM_BENCHMARK_EXTENDED_CSV=1
)

function(add_rendering_raytracing_sim_benchmark_profile_target target_name shading_profile)
    add_executable(${target_name} EXCLUDE_FROM_ALL simulator_matrix.cpp)
    target_link_libraries(${target_name}
        PRIVATE
            RLtools::RLtools
            owl::owl
            rendering_raytracing_device_ptx
            rendering_raytracing_device_depth_ptx
            assimp::assimp
            rendering_raytracing_stb
    )
    target_compile_definitions(${target_name}
        PRIVATE
            RL_TOOLS_RENDERING_RAYTRACING_SIM_BENCHMARK_OUTPUT_MODE=-1
            RL_TOOLS_RENDERING_RAYTRACING_SIM_BENCHMARK_NUM_ENVS=${RL_TOOLS_RENDERING_RAYTRACING_SIM_BENCHMARK_NUM_ENVS}
            RL_TOOLS_RENDERING_RAYTRACING_SIM_BENCHMARK_SHADING_PROFILE=${shading_profile}
    )
endfunction()

add_rendering_raytracing_sim_benchmark_profile_target(rendering_raytracing_sim_benchmark_medium 0)
add_rendering_raytracing_sim_benchmark_profile_target(rendering_raytracing_sim_benchmark_high 1)
add_rendering_raytracing_sim_benchmark_profile_target(rendering_raytracing_sim_benchmark_low 2)
add_rendering_raytracing_sim_benchmark_profile_target(rendering_raytracing_sim_benchmark_high_fidelity 1)
add_rendering_raytracing_sim_benchmark_profile_target(rendering_raytracing_sim_benchmark_fast_flat 2)

add_executable(rendering_raytracing_sim_benchmark_rgb EXCLUDE_FROM_ALL simulator_matrix.cpp)
target_link_libraries(rendering_raytracing_sim_benchmark_rgb
    PRIVATE
        RLtools::RLtools
        owl::owl
        rendering_raytracing_device_ptx
        assimp::assimp
        rendering_raytracing_stb
)
target_compile_definitions(rendering_raytracing_sim_benchmark_rgb
    PRIVATE
        RL_TOOLS_RENDERING_RAYTRACING_SIM_BENCHMARK_OUTPUT_MODE=0
        RL_TOOLS_RENDERING_RAYTRACING_SIM_BENCHMARK_NUM_ENVS=${RL_TOOLS_RENDERING_RAYTRACING_SIM_BENCHMARK_NUM_ENVS}
)

add_executable(rendering_raytracing_sim_benchmark_depth EXCLUDE_FROM_ALL simulator_matrix.cpp)
target_link_libraries(rendering_raytracing_sim_benchmark_depth
    PRIVATE
        RLtools::RLtools
        owl::owl
        rendering_raytracing_device_depth_ptx
        assimp::assimp
        rendering_raytracing_stb
)
target_compile_definitions(rendering_raytracing_sim_benchmark_depth
    PRIVATE
        RL_TOOLS_RENDERING_RAYTRACING_SIM_BENCHMARK_OUTPUT_MODE=2
        RL_TOOLS_RENDERING_RAYTRACING_SIM_BENCHMARK_NUM_ENVS=${RL_TOOLS_RENDERING_RAYTRACING_SIM_BENCHMARK_NUM_ENVS}
    )

function(add_rendering_raytracing_procthor_frame_target target_name output_mode ptx_target shading_profile)
    add_executable(${target_name} EXCLUDE_FROM_ALL simulator_matrix.cpp)
    target_link_libraries(${target_name}
        PRIVATE
            RLtools::RLtools
            owl::owl
            ${ptx_target}
            assimp::assimp
            rendering_raytracing_stb
    )
    target_compile_definitions(${target_name}
        PRIVATE
            RL_TOOLS_RENDERING_RAYTRACING_SIM_BENCHMARK_OUTPUT_MODE=${output_mode}
            RL_TOOLS_RENDERING_RAYTRACING_SIM_BENCHMARK_NUM_ENVS=1
            RL_TOOLS_RENDERING_RAYTRACING_SIM_BENCHMARK_WIDTH=1000
            RL_TOOLS_RENDERING_RAYTRACING_SIM_BENCHMARK_HEIGHT=1000
            RL_TOOLS_RENDERING_RAYTRACING_SIM_FRAME_TOOL=1
            RL_TOOLS_RENDERING_RAYTRACING_SIM_BENCHMARK_SHADING_PROFILE=${shading_profile}
    )
endfunction()

add_rendering_raytracing_procthor_frame_target(rendering_raytracing_procthor_frame 0 rendering_raytracing_device_ptx 0)
add_rendering_raytracing_procthor_frame_target(rendering_raytracing_procthor_frame_medium 0 rendering_raytracing_device_ptx 0)
add_rendering_raytracing_procthor_frame_target(rendering_raytracing_procthor_frame_high 0 rendering_raytracing_device_ptx 1)
add_rendering_raytracing_procthor_frame_target(rendering_raytracing_procthor_frame_low 0 rendering_raytracing_device_ptx 2)
add_rendering_raytracing_procthor_frame_target(rendering_raytracing_procthor_frame_basic 0 rendering_raytracing_device_ptx 0)
add_rendering_raytracing_procthor_frame_target(rendering_raytracing_procthor_frame_high_fidelity 0 rendering_raytracing_device_ptx 1)
add_rendering_raytracing_procthor_frame_target(rendering_raytracing_procthor_frame_fast_flat 0 rendering_raytracing_device_ptx 2)

add_rendering_raytracing_procthor_frame_target(rendering_raytracing_procthor_frame_depth 2 rendering_raytracing_device_depth_ptx 0)
add_rendering_raytracing_procthor_frame_target(rendering_raytracing_procthor_frame_depth_medium 2 rendering_raytracing_device_depth_ptx 0)
add_rendering_raytracing_procthor_frame_target(rendering_raytracing_procthor_frame_depth_high 2 rendering_raytracing_device_depth_ptx 1)
add_rendering_raytracing_procthor_frame_target(rendering_raytracing_procthor_frame_depth_low 2 rendering_raytracing_device_depth_ptx 2)
add_rendering_raytracing_procthor_frame_target(rendering_raytracing_procthor_frame_depth_basic 2 rendering_raytracing_device_depth_ptx 0)
add_rendering_raytracing_procthor_frame_target(rendering_raytracing_procthor_frame_depth_high_fidelity 2 rendering_raytracing_device_depth_ptx 1)
add_rendering_raytracing_procthor_frame_target(rendering_raytracing_procthor_frame_depth_fast_flat 2 rendering_raytracing_device_depth_ptx 2)
