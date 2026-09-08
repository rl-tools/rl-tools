if(TARGET owl::owl AND TARGET rendering_raytracing_device_ptx AND TARGET assimp::assimp)
    add_executable(test_rl_environments_l2f_visual test.cpp)
    target_link_libraries(test_rl_environments_l2f_visual
        PRIVATE
            RLtools::RLtools
            RLtools::Test
            owl::owl
            rendering_raytracing_device_ptx
            assimp::assimp
            rendering_raytracing_stb
    )
    gtest_discover_tests(test_rl_environments_l2f_visual)
    if(RL_TOOLS_BACKEND_ENABLE_CUDA)
        add_executable(test_rl_environments_l2f_visual_cuda test_cuda.cu)
        target_link_libraries(test_rl_environments_l2f_visual_cuda
            PRIVATE
                RLtools::RLtools
                RLtools::Test
                owl::owl
                rendering_raytracing_device_ptx
                assimp::assimp
                rendering_raytracing_stb
        )
        set_target_properties(test_rl_environments_l2f_visual_cuda PROPERTIES
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON
        )
        gtest_discover_tests(test_rl_environments_l2f_visual_cuda)
    endif()
endif()

add_executable(test_rl_environments_l2f_visual_hyperdrone_gradient on_policy_gradient.cpp)
target_include_directories(test_rl_environments_l2f_visual_hyperdrone_gradient PRIVATE ${CMAKE_SOURCE_DIR}/src/rl/environments/l2f_visual)
target_link_libraries(test_rl_environments_l2f_visual_hyperdrone_gradient PRIVATE RLtools::RLtools GTest::gtest_main)
gtest_discover_tests(test_rl_environments_l2f_visual_hyperdrone_gradient)
