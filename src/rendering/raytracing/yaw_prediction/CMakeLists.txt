if(RL_TOOLS_BACKEND_ENABLE_CUDA AND RL_TOOLS_BACKEND_ENABLE_CUDNN)
    add_library(rendering_raytracing_yaw_prediction_scene STATIC scene.cpp)
    set_target_properties(rendering_raytracing_yaw_prediction_scene PROPERTIES CXX_STANDARD 20)
    target_link_libraries(rendering_raytracing_yaw_prediction_scene
        PRIVATE
            rl_tools_full
            owl::owl
            rendering_raytracing_device_ptx
            assimp::assimp
            rendering_raytracing_stb
    )

    add_executable(rendering_raytracing_yaw_prediction yaw_prediction_cuda.cu)
    set_target_properties(rendering_raytracing_yaw_prediction PROPERTIES CUDA_STANDARD 20)
    set_property(TARGET rendering_raytracing_yaw_prediction PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    target_compile_options(rendering_raytracing_yaw_prediction PRIVATE -O3)
    # Allow multiple definitions: rl_tools non-inline functions appear in both scene.cpp and .cu TUs
    if(NOT APPLE)
        target_link_options(rendering_raytracing_yaw_prediction PRIVATE -Wl,--allow-multiple-definition)
    endif()
    target_link_libraries(rendering_raytracing_yaw_prediction
        PRIVATE
            rl_tools_full
            owl::owl
            rendering_raytracing_device_ptx
            assimp::assimp
            rendering_raytracing_stb
            rendering_raytracing_yaw_prediction_scene
    )

    if(RL_TOOLS_ENABLE_HDF5)
        add_executable(rendering_raytracing_yaw_prediction_eval yaw_prediction_eval.cu)
        set_target_properties(rendering_raytracing_yaw_prediction_eval PROPERTIES CUDA_STANDARD 20)
        set_property(TARGET rendering_raytracing_yaw_prediction_eval PROPERTY CUDA_SEPARABLE_COMPILATION ON)
        target_compile_options(rendering_raytracing_yaw_prediction_eval PRIVATE -O3)
        if(NOT APPLE)
            target_link_options(rendering_raytracing_yaw_prediction_eval PRIVATE -Wl,--allow-multiple-definition)
        endif()
        target_link_libraries(rendering_raytracing_yaw_prediction_eval
            PRIVATE
                rl_tools_full
                owl::owl
                rendering_raytracing_device_ptx
                assimp::assimp
                rendering_raytracing_stb
                rendering_raytracing_yaw_prediction_scene
        )

        add_executable(rendering_raytracing_yaw_prediction_distill yaw_prediction_distill_cuda.cu)
        set_target_properties(rendering_raytracing_yaw_prediction_distill PROPERTIES CUDA_STANDARD 20)
        set_property(TARGET rendering_raytracing_yaw_prediction_distill PROPERTY CUDA_SEPARABLE_COMPILATION ON)
        target_compile_options(rendering_raytracing_yaw_prediction_distill PRIVATE -O3)
        if(NOT APPLE)
            target_link_options(rendering_raytracing_yaw_prediction_distill PRIVATE -Wl,--allow-multiple-definition)
        endif()
        target_link_libraries(rendering_raytracing_yaw_prediction_distill
            PRIVATE
                rl_tools_full
                owl::owl
                rendering_raytracing_device_ptx
                assimp::assimp
                rendering_raytracing_stb
                rendering_raytracing_yaw_prediction_scene
        )

        add_executable(rendering_raytracing_yaw_prediction_viewer yaw_prediction_viewer.cu)
        set_target_properties(rendering_raytracing_yaw_prediction_viewer PROPERTIES CUDA_STANDARD 20)
        set_property(TARGET rendering_raytracing_yaw_prediction_viewer PROPERTY CUDA_SEPARABLE_COMPILATION ON)
        target_compile_options(rendering_raytracing_yaw_prediction_viewer PRIVATE -O3)
        if(NOT APPLE)
            target_link_options(rendering_raytracing_yaw_prediction_viewer PRIVATE -Wl,--allow-multiple-definition)
        endif()
        target_link_libraries(rendering_raytracing_yaw_prediction_viewer
            PRIVATE
                rl_tools_full
                owl::owl
                rendering_raytracing_device_ptx
                assimp::assimp
                rendering_raytracing_yaw_prediction_scene
        )
    endif()
endif()
