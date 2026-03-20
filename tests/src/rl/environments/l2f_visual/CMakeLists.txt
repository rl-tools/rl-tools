if(TARGET owl::owl AND TARGET rendering_raytracing_device_ptx AND TARGET assimp::assimp)
    add_executable(test_rl_environments_l2f_visual test.cpp)
    target_link_libraries(test_rl_environments_l2f_visual
        PRIVATE
            RLtools::RLtools
            GTest::gtest_main
            owl::owl
            rendering_raytracing_device_ptx
            assimp::assimp
            rendering_raytracing_stb
    )
    gtest_discover_tests(test_rl_environments_l2f_visual)
endif()
