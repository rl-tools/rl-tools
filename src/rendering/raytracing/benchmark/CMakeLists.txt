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

add_executable(rendering_raytracing_sim_benchmark_high_fidelity EXCLUDE_FROM_ALL simulator_matrix.cpp)
target_link_libraries(rendering_raytracing_sim_benchmark_high_fidelity
    PRIVATE
        RLtools::RLtools
        owl::owl
        rendering_raytracing_device_ptx
        rendering_raytracing_device_depth_ptx
        assimp::assimp
        rendering_raytracing_stb
)
target_compile_definitions(rendering_raytracing_sim_benchmark_high_fidelity
    PRIVATE
        RL_TOOLS_RENDERING_RAYTRACING_SIM_BENCHMARK_OUTPUT_MODE=-1
        RL_TOOLS_RENDERING_RAYTRACING_SIM_BENCHMARK_NUM_ENVS=${RL_TOOLS_RENDERING_RAYTRACING_SIM_BENCHMARK_NUM_ENVS}
        RL_TOOLS_RENDERING_RAYTRACING_SIM_BENCHMARK_SHADING_PROFILE=1
)

add_executable(rendering_raytracing_sim_benchmark_fast_flat EXCLUDE_FROM_ALL simulator_matrix.cpp)
target_link_libraries(rendering_raytracing_sim_benchmark_fast_flat
    PRIVATE
        RLtools::RLtools
        owl::owl
        rendering_raytracing_device_ptx
        rendering_raytracing_device_depth_ptx
        assimp::assimp
        rendering_raytracing_stb
)
target_compile_definitions(rendering_raytracing_sim_benchmark_fast_flat
    PRIVATE
        RL_TOOLS_RENDERING_RAYTRACING_SIM_BENCHMARK_OUTPUT_MODE=-1
        RL_TOOLS_RENDERING_RAYTRACING_SIM_BENCHMARK_NUM_ENVS=${RL_TOOLS_RENDERING_RAYTRACING_SIM_BENCHMARK_NUM_ENVS}
        RL_TOOLS_RENDERING_RAYTRACING_SIM_BENCHMARK_SHADING_PROFILE=2
)

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
