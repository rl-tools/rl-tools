if(TARGET owl::owl AND TARGET rendering_raytracing_device_ptx AND TARGET assimp::assimp AND RL_TOOLS_TEST_DATA_PATH)
    add_executable(test_rendering_raytracing_generate_golden EXCLUDE_FROM_ALL generate_golden.cpp)
    target_link_libraries(test_rendering_raytracing_generate_golden
        PRIVATE
            RLtools::RLtools
            owl::owl
            rendering_raytracing_device_ptx
            rendering_raytracing_device_depth_ptx
            assimp::assimp
            rendering_raytracing_stb
    )
endif()

if(RL_TOOLS_RENDERING_ENABLE_RAYTRACING AND RL_TOOLS_TEST_DATA_PATH)
    add_executable(test_rendering_raytracing_golden_comparison golden_comparison.cpp)
    target_link_libraries(test_rendering_raytracing_golden_comparison
        PRIVATE
            RLtools::RLtools
            assimp::assimp
            rendering_raytracing_stb
            GTest::gtest_main
    )
    gtest_discover_tests(test_rendering_raytracing_golden_comparison PROPERTIES TIMEOUT 300)
endif()
