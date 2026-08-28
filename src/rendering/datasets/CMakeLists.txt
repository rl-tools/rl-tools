add_executable(rendering_datasets_ai2thorhab_smoke EXCLUDE_FROM_ALL ai2thorhab_smoke.cpp)
target_link_libraries(rendering_datasets_ai2thorhab_smoke
    PRIVATE
        RLtools::RLtools
        rl_tools_procthor_conversion
        rendering_raytracing_stb
        assimp::assimp
)
