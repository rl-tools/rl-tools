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
