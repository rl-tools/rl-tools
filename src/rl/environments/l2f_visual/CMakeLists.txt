set(RL_TOOLS_L2F_VISUAL_IMITATION_N_RANKS 1 CACHE STRING "Number of GPUs (data-parallel ranks) the l2f_visual imitation targets are compiled for")
if(NOT RL_TOOLS_L2F_VISUAL_IMITATION_N_RANKS MATCHES "^[1-9][0-9]*$")
    message(FATAL_ERROR "RL_TOOLS_L2F_VISUAL_IMITATION_N_RANKS must be a positive integer")
endif()
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
        target_compile_definitions(rl_environments_l2f_visual_imitation_cuda PRIVATE RL_TOOLS_L2F_VISUAL_IMITATION_N_RANKS=${RL_TOOLS_L2F_VISUAL_IMITATION_N_RANKS})
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
        add_executable(rl_environments_l2f_visual_imitation_cuda_small imitation_cuda.cu)
        target_compile_definitions(rl_environments_l2f_visual_imitation_cuda_small PRIVATE RL_TOOLS_L2F_VISUAL_IMITATION_N_RANKS=${RL_TOOLS_L2F_VISUAL_IMITATION_N_RANKS})
        target_link_libraries(rl_environments_l2f_visual_imitation_cuda_small PRIVATE
            RLtools::RLtools
            owl::owl
            rendering_raytracing_device_ptx
            assimp::assimp
            rendering_raytracing_stb
        )
        target_compile_definitions(rl_environments_l2f_visual_imitation_cuda_small PRIVATE
            RL_TOOLS_L2F_VISUAL_IMITATION_SMALL
        )
        set_target_properties(rl_environments_l2f_visual_imitation_cuda_small PROPERTIES
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON
            CUDA_SEPARABLE_COMPILATION ON
        )
        add_executable(rl_environments_l2f_visual_imitation_state_estimation_cuda imitation_cuda.cu)
        target_compile_definitions(rl_environments_l2f_visual_imitation_state_estimation_cuda PRIVATE RL_TOOLS_L2F_VISUAL_IMITATION_N_RANKS=${RL_TOOLS_L2F_VISUAL_IMITATION_N_RANKS})
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

# hyperdrone port of the imitation target: one device-generic source, compiled as CUDA (compute
# on the GPU, bf16) when the OptiX backend is active and CUDA is available, as plain C++ (CPU
# compute, fp32) against every other raytracing backend. On CUDA machines the additional
# `_cpu` variant builds the CPU-compute path against the OptiX backend for A/B comparisons
if(RL_TOOLS_RENDERING_ENABLE_RAYTRACING)
    set(RL_TOOLS_L2F_VISUAL_IMITATION_HYPERDRONE_CUDA_COMPUTE OFF)
    if(RL_TOOLS_BACKEND_ENABLE_CUDA AND RL_TOOLS_RENDERING_RAYTRACING_BACKEND_RESOLVED STREQUAL "OPTIX")
        set(RL_TOOLS_L2F_VISUAL_IMITATION_HYPERDRONE_CUDA_COMPUTE ON)
    endif()
    function(rl_tools_add_l2f_visual_imitation_hyperdrone_target TARGET_NAME CUDA_COMPUTE)
        if(CUDA_COMPUTE)
            set(SOURCE_FILE ${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}.cu)
            file(CONFIGURE OUTPUT ${SOURCE_FILE} CONTENT "#include \"${CMAKE_CURRENT_SOURCE_DIR}/imitation_hyperdrone.cpp\"\n")
        else()
            set(SOURCE_FILE imitation_hyperdrone.cpp)
        endif()
        add_executable(${TARGET_NAME} ${SOURCE_FILE})
        target_link_libraries(${TARGET_NAME} PRIVATE
            RLtools::RLtools
            rendering_raytracing_backend
            rendering_raytracing_stb
        )
        if(CUDA_COMPUTE)
            target_compile_definitions(${TARGET_NAME} PRIVATE RL_TOOLS_L2F_VISUAL_IMITATION_N_RANKS=${RL_TOOLS_L2F_VISUAL_IMITATION_N_RANKS})
            set_target_properties(${TARGET_NAME} PROPERTIES
                CUDA_STANDARD 17
                CUDA_STANDARD_REQUIRED ON
                CUDA_SEPARABLE_COMPILATION ON
            )
        else()
            target_compile_definitions(${TARGET_NAME} PRIVATE RL_TOOLS_L2F_VISUAL_IMITATION_N_RANKS=1)
        endif()
        foreach(EXTRA_DEFINITION ${ARGN})
            target_compile_definitions(${TARGET_NAME} PRIVATE ${EXTRA_DEFINITION})
        endforeach()
    endfunction()
    rl_tools_add_l2f_visual_imitation_hyperdrone_target(rl_environments_l2f_visual_imitation_hyperdrone ${RL_TOOLS_L2F_VISUAL_IMITATION_HYPERDRONE_CUDA_COMPUTE})
    rl_tools_add_l2f_visual_imitation_hyperdrone_target(rl_environments_l2f_visual_imitation_state_estimation_hyperdrone ${RL_TOOLS_L2F_VISUAL_IMITATION_HYPERDRONE_CUDA_COMPUTE} RL_TOOLS_L2F_VISUAL_IMITATION_STATE_ESTIMATION)
    rl_tools_add_l2f_visual_imitation_hyperdrone_target(rl_environments_l2f_visual_imitation_hyperdrone_small ${RL_TOOLS_L2F_VISUAL_IMITATION_HYPERDRONE_CUDA_COMPUTE} RL_TOOLS_L2F_VISUAL_IMITATION_HYPERDRONE_SMALL)
    rl_tools_add_l2f_visual_imitation_hyperdrone_target(rl_environments_l2f_visual_imitation_state_estimation_hyperdrone_small ${RL_TOOLS_L2F_VISUAL_IMITATION_HYPERDRONE_CUDA_COMPUTE} RL_TOOLS_L2F_VISUAL_IMITATION_STATE_ESTIMATION RL_TOOLS_L2F_VISUAL_IMITATION_HYPERDRONE_SMALL)
    if(RL_TOOLS_L2F_VISUAL_IMITATION_HYPERDRONE_CUDA_COMPUTE)
        rl_tools_add_l2f_visual_imitation_hyperdrone_target(rl_environments_l2f_visual_imitation_hyperdrone_cpu OFF)
    endif()
endif()
