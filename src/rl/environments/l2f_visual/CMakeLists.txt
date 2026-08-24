if(TARGET owl::owl AND TARGET rendering_raytracing_device_ptx AND TARGET assimp::assimp)
#    add_executable(rl_environments_l2f_visual_training training.cpp)
#    target_link_libraries(rl_environments_l2f_visual_training PRIVATE
#        RLtools::RLtools
#        owl::owl
#        rendering_raytracing_device_ptx
#        assimp::assimp
#        rendering_raytracing_stb
#    )
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
        add_executable(rl_environments_l2f_visual_training_cuda_hyperdrone training_cuda_hyperdrone.cu)
        target_link_libraries(rl_environments_l2f_visual_training_cuda_hyperdrone PRIVATE
            RLtools::RLtools
            owl::owl
            rendering_raytracing_device_ptx
            assimp::assimp
            rendering_raytracing_stb
        )
        set_target_properties(rl_environments_l2f_visual_training_cuda_hyperdrone PROPERTIES
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON
        )
        add_executable(rl_environments_l2f_visual_imitation_cuda imitation_cuda.cu)
        target_link_libraries(rl_environments_l2f_visual_imitation_cuda PRIVATE
            RLtools::RLtools
            owl::owl
            rendering_raytracing_device_ptx
            assimp::assimp
            rendering_raytracing_stb
        )
        set_target_properties(rl_environments_l2f_visual_imitation_cuda PROPERTIES
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON
            CUDA_SEPARABLE_COMPILATION ON
        )
        add_executable(rl_environments_l2f_visual_imitation_state_estimation_cuda imitation_cuda.cu)
        target_link_libraries(rl_environments_l2f_visual_imitation_state_estimation_cuda PRIVATE
            RLtools::RLtools
            owl::owl
            rendering_raytracing_device_ptx
            assimp::assimp
            rendering_raytracing_stb
        )
        target_compile_definitions(rl_environments_l2f_visual_imitation_state_estimation_cuda PRIVATE
            RL_TOOLS_L2F_VISUAL_IMITATION_STATE_ESTIMATION
        )
        set_target_properties(rl_environments_l2f_visual_imitation_state_estimation_cuda PROPERTIES
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON
            CUDA_SEPARABLE_COMPILATION ON
        )
        add_executable(rl_environments_l2f_visual_yaw_cuda yaw.cu)
        target_link_libraries(rl_environments_l2f_visual_yaw_cuda PRIVATE
            RLtools::RLtools
            owl::owl
            rendering_raytracing_device_ptx
            assimp::assimp
            rendering_raytracing_stb
        )
        set_target_properties(rl_environments_l2f_visual_yaw_cuda PROPERTIES
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON
            CUDA_SEPARABLE_COMPILATION ON
        )
        add_executable(rl_environments_l2f_visual_inference_dyn_cuda inference_dyn_cuda.cu)
        target_link_libraries(rl_environments_l2f_visual_inference_dyn_cuda PRIVATE
            RLtools::RLtools
            owl::owl
            rendering_raytracing_device_ptx
            assimp::assimp
            rendering_raytracing_stb
        )
        set_target_properties(rl_environments_l2f_visual_inference_dyn_cuda PROPERTIES
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON
            CUDA_SEPARABLE_COMPILATION ON
        )
        add_executable(rl_environments_l2f_visual_stitch_cuda stitch.cu)
        target_link_libraries(rl_environments_l2f_visual_stitch_cuda PRIVATE
            RLtools::RLtools
            owl::owl
            rendering_raytracing_device_ptx
            assimp::assimp
            rendering_raytracing_stb
        )
        set_target_properties(rl_environments_l2f_visual_stitch_cuda PROPERTIES
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON
            CUDA_SEPARABLE_COMPILATION ON
        )
    endif()
endif()
