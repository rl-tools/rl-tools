if(RL_TOOLS_RENDERING_ENABLE_RAYTRACING AND RL_TOOLS_TEST_DATA_PATH)
    add_executable(test_rl_environments_hyperdrone_rig rig.cpp)
    target_link_libraries(test_rl_environments_hyperdrone_rig
        PRIVATE
            RLtools::RLtools
            RLtools::Test
            rendering_raytracing_backend
            rendering_raytracing_stb
    )
    gtest_discover_tests(test_rl_environments_hyperdrone_rig)
    add_executable(test_rl_environments_hyperdrone_world world.cpp)
    target_link_libraries(test_rl_environments_hyperdrone_world
        PRIVATE
            RLtools::RLtools
            RLtools::Test
            rendering_raytracing_backend
            rendering_raytracing_stb
    )
    gtest_discover_tests(test_rl_environments_hyperdrone_world)
    add_executable(test_rl_environments_hyperdrone_annotations annotations.cpp)
    target_link_libraries(test_rl_environments_hyperdrone_annotations
        PRIVATE
            RLtools::RLtools
            RLtools::Test
            rendering_raytracing_backend
            rendering_raytracing_stb
    )
    gtest_discover_tests(test_rl_environments_hyperdrone_annotations)
    add_executable(test_rl_environments_hyperdrone_target_frame target_frame.cpp)
    target_link_libraries(test_rl_environments_hyperdrone_target_frame
        PRIVATE
            RLtools::RLtools
            RLtools::Test
            rendering_raytracing_backend
            rendering_raytracing_stb
    )
    gtest_discover_tests(test_rl_environments_hyperdrone_target_frame)
    add_executable(test_rl_environments_hyperdrone_moving_gate moving_gate.cpp)
    target_link_libraries(test_rl_environments_hyperdrone_moving_gate
        PRIVATE
            RLtools::RLtools
            RLtools::Test
            rendering_raytracing_backend
            rendering_raytracing_stb
    )
    gtest_discover_tests(test_rl_environments_hyperdrone_moving_gate)
    add_executable(test_rl_environments_hyperdrone_visual_inertial_localization visual_inertial_localization.cpp)
    target_link_libraries(test_rl_environments_hyperdrone_visual_inertial_localization
        PRIVATE
            RLtools::RLtools
            RLtools::Test
            rendering_raytracing_backend
            rendering_raytracing_stb
    )
    gtest_discover_tests(test_rl_environments_hyperdrone_visual_inertial_localization)
    add_executable(test_rl_environments_hyperdrone_segmentation_determinism segmentation_determinism.cpp)
    target_link_libraries(test_rl_environments_hyperdrone_segmentation_determinism
        PRIVATE
            RLtools::RLtools
            RLtools::Test
            rendering_raytracing_backend
            rendering_raytracing_backend_segmentation
            rendering_raytracing_stb
    )
    gtest_discover_tests(test_rl_environments_hyperdrone_segmentation_determinism)
    add_executable(test_rl_environments_hyperdrone_self_visible self_visible.cpp)
    target_link_libraries(test_rl_environments_hyperdrone_self_visible
        PRIVATE
            RLtools::RLtools
            RLtools::Test
            rendering_raytracing_backend
            rendering_raytracing_backend_segmentation
            rendering_raytracing_stb
    )
    gtest_discover_tests(test_rl_environments_hyperdrone_self_visible)
    add_executable(test_rl_environments_hyperdrone_multi_agent multi_agent.cpp)
    target_link_libraries(test_rl_environments_hyperdrone_multi_agent
        PRIVATE
            RLtools::RLtools
            RLtools::Test
            rendering_raytracing_backend
            rendering_raytracing_backend_segmentation
            rendering_raytracing_stb
    )
    gtest_discover_tests(test_rl_environments_hyperdrone_multi_agent)
    add_executable(test_rl_environments_hyperdrone_presets presets.cpp)
    target_link_libraries(test_rl_environments_hyperdrone_presets
        PRIVATE
            RLtools::RLtools
            RLtools::Test
            rendering_raytracing_backend
            rendering_raytracing_stb
    )
    gtest_discover_tests(test_rl_environments_hyperdrone_presets)
    add_executable(test_rl_environments_hyperdrone_episodes episodes.cpp)
    target_link_libraries(test_rl_environments_hyperdrone_episodes
        PRIVATE
            RLtools::RLtools
            RLtools::Test
            rendering_raytracing_backend
            rendering_raytracing_stb
    )
    gtest_discover_tests(test_rl_environments_hyperdrone_episodes)
    add_executable(test_rl_environments_hyperdrone_on_policy_runner_batched on_policy_runner_batched.cpp)
    target_link_libraries(test_rl_environments_hyperdrone_on_policy_runner_batched
        PRIVATE
            RLtools::RLtools
            RLtools::Test
            rendering_raytracing_backend
            rendering_raytracing_stb
    )
    gtest_discover_tests(test_rl_environments_hyperdrone_on_policy_runner_batched)
    # golden-rollout generator for the Python binding (run deliberately from the repo root)
    add_executable(test_rl_environments_hyperdrone_python_golden python_golden.cpp)
    target_link_libraries(test_rl_environments_hyperdrone_python_golden
        PRIVATE
            RLtools::RLtools
            rendering_raytracing_backend
            rendering_raytracing_stb
    )
    if(RL_TOOLS_BACKEND_ENABLE_CUDA AND TARGET owl::owl)
        add_executable(test_rl_environments_hyperdrone_rig_cuda rig_cuda.cu)
        target_link_libraries(test_rl_environments_hyperdrone_rig_cuda
            PRIVATE
                RLtools::RLtools
                RLtools::Test
                owl::owl
                rendering_raytracing_device_ptx
                assimp::assimp
                rendering_raytracing_stb
        )
        set_target_properties(test_rl_environments_hyperdrone_rig_cuda PROPERTIES
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON
        )
        gtest_discover_tests(test_rl_environments_hyperdrone_rig_cuda)
        add_executable(test_rl_environments_hyperdrone_world_cuda world_cuda.cu)
        target_link_libraries(test_rl_environments_hyperdrone_world_cuda
            PRIVATE
                RLtools::RLtools
                RLtools::Test
                owl::owl
                rendering_raytracing_device_ptx
                assimp::assimp
                rendering_raytracing_stb
        )
        set_target_properties(test_rl_environments_hyperdrone_world_cuda PROPERTIES
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON
        )
        gtest_discover_tests(test_rl_environments_hyperdrone_world_cuda)
        add_executable(test_rl_environments_hyperdrone_target_frame_cuda target_frame_cuda.cu)
        target_link_libraries(test_rl_environments_hyperdrone_target_frame_cuda
            PRIVATE
                RLtools::RLtools
                RLtools::Test
                owl::owl
                rendering_raytracing_device_ptx
                assimp::assimp
                rendering_raytracing_stb
        )
        set_target_properties(test_rl_environments_hyperdrone_target_frame_cuda PROPERTIES
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON
        )
        gtest_discover_tests(test_rl_environments_hyperdrone_target_frame_cuda)
        add_executable(test_rl_environments_hyperdrone_multi_agent_cuda multi_agent_cuda.cu)
        target_link_libraries(test_rl_environments_hyperdrone_multi_agent_cuda
            PRIVATE
                RLtools::RLtools
                RLtools::Test
                owl::owl
                rendering_raytracing_device_ptx
                rendering_raytracing_device_segmentation_ptx
                rendering_raytracing_device_depth_segmentation_ptx
                assimp::assimp
                rendering_raytracing_stb
        )
        set_target_properties(test_rl_environments_hyperdrone_multi_agent_cuda PROPERTIES
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON
        )
        gtest_discover_tests(test_rl_environments_hyperdrone_multi_agent_cuda)
        add_executable(test_rl_environments_hyperdrone_self_visible_cuda self_visible_cuda.cu)
        target_link_libraries(test_rl_environments_hyperdrone_self_visible_cuda
            PRIVATE
                RLtools::RLtools
                RLtools::Test
                owl::owl
                rendering_raytracing_device_ptx
                rendering_raytracing_device_segmentation_ptx
                rendering_raytracing_device_depth_segmentation_ptx
                assimp::assimp
                rendering_raytracing_stb
        )
        set_target_properties(test_rl_environments_hyperdrone_self_visible_cuda PROPERTIES
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON
        )
        gtest_discover_tests(test_rl_environments_hyperdrone_self_visible_cuda)
        add_executable(test_rl_environments_hyperdrone_moving_gate_cuda moving_gate_cuda.cu)
        target_link_libraries(test_rl_environments_hyperdrone_moving_gate_cuda
            PRIVATE
                RLtools::RLtools
                RLtools::Test
                owl::owl
                rendering_raytracing_device_ptx
                assimp::assimp
                rendering_raytracing_stb
        )
        set_target_properties(test_rl_environments_hyperdrone_moving_gate_cuda PROPERTIES
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON
        )
        gtest_discover_tests(test_rl_environments_hyperdrone_moving_gate_cuda)
        add_executable(test_rl_environments_hyperdrone_episodes_cuda episodes_cuda.cu)
        target_link_libraries(test_rl_environments_hyperdrone_episodes_cuda
            PRIVATE
                RLtools::RLtools
                RLtools::Test
                owl::owl
                rendering_raytracing_device_ptx
                assimp::assimp
                rendering_raytracing_stb
        )
        set_target_properties(test_rl_environments_hyperdrone_episodes_cuda PROPERTIES
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON
        )
        gtest_discover_tests(test_rl_environments_hyperdrone_episodes_cuda)
        add_executable(test_rl_environments_hyperdrone_benchmark_cuda benchmark_cuda.cu)
        target_link_libraries(test_rl_environments_hyperdrone_benchmark_cuda
            PRIVATE
                RLtools::RLtools
                RLtools::Test
                owl::owl
                rendering_raytracing_device_ptx
                assimp::assimp
                rendering_raytracing_stb
        )
        set_target_properties(test_rl_environments_hyperdrone_benchmark_cuda PROPERTIES
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON
        )
        gtest_discover_tests(test_rl_environments_hyperdrone_benchmark_cuda PROPERTIES TIMEOUT 300)
    endif()
endif()
