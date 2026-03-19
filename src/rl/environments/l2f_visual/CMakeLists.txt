if(TARGET owl::owl AND TARGET rendering_raytracing_device_ptx AND TARGET assimp::assimp)
    add_executable(rl_environments_l2f_visual_training training.cpp)
    target_link_libraries(rl_environments_l2f_visual_training PRIVATE
        RLtools::RLtools
        owl::owl
        rendering_raytracing_device_ptx
        assimp::assimp
        rendering_raytracing_stb
    )
    if(RL_TOOLS_BACKEND_ENABLE_CUDA)
        add_executable(rl_environments_l2f_visual_training_cuda training_cuda.cu)
        target_link_libraries(rl_environments_l2f_visual_training_cuda PRIVATE
            RLtools::RLtools
            owl::owl
            rendering_raytracing_device_ptx
            assimp::assimp
            rendering_raytracing_stb
        )
        set_target_properties(rl_environments_l2f_visual_training_cuda PROPERTIES
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON
        )
    endif()
endif()
