if(TARGET owl::owl AND TARGET rendering_raytracing_device_ptx AND TARGET assimp::assimp)
    add_executable(rl_environments_l2f_visual_training training.cpp)
    target_link_libraries(rl_environments_l2f_visual_training PRIVATE
        RLtools::RLtools
        owl::owl
        rendering_raytracing_device_ptx
        assimp::assimp
        rendering_raytracing_stb
    )
endif()
