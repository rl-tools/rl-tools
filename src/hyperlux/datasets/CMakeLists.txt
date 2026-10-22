add_executable(hyperlux_datasets_annotate EXCLUDE_FROM_ALL annotate.cpp)
target_link_libraries(hyperlux_datasets_annotate
    PRIVATE
        RLtools::RLtools
        rl_tools_procthor_conversion
        hyperlux_backend
        hyperlux_stb
)

add_executable(hyperlux_datasets_ai2thorhab_smoke EXCLUDE_FROM_ALL ai2thorhab_smoke.cpp)
target_link_libraries(hyperlux_datasets_ai2thorhab_smoke
    PRIVATE
        RLtools::RLtools
        rl_tools_procthor_conversion
        hyperlux_stb
        assimp::assimp
)
