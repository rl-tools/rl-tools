if(RL_TOOLS_RENDERING_ENABLE_RAYTRACING AND RL_TOOLS_TEST_DATA_PATH)
    add_executable(test_rendering_raytracing_generate_golden_generic EXCLUDE_FROM_ALL generate_golden.cpp)
    target_link_libraries(test_rendering_raytracing_generate_golden_generic
        PRIVATE
            RLtools::RLtools
            assimp::assimp
            rendering_raytracing_stb
    )
    target_compile_definitions(test_rendering_raytracing_generate_golden_generic PRIVATE
        RL_TOOLS_RENDERING_RAYTRACING_GOLDEN_BACKEND_NAME="generic"
    )

    if(NOT RL_TOOLS_RENDERING_RAYTRACING_BACKEND_RESOLVED STREQUAL "GENERIC")
        string(TOLOWER ${RL_TOOLS_RENDERING_RAYTRACING_BACKEND_RESOLVED} RL_TOOLS_GOLDEN_GENERATOR_BACKEND_SUFFIX)
        set(RL_TOOLS_GOLDEN_GENERATOR_BACKEND_TARGET test_rendering_raytracing_generate_golden_${RL_TOOLS_GOLDEN_GENERATOR_BACKEND_SUFFIX})
        add_executable(${RL_TOOLS_GOLDEN_GENERATOR_BACKEND_TARGET} EXCLUDE_FROM_ALL generate_golden.cpp)
        target_link_libraries(${RL_TOOLS_GOLDEN_GENERATOR_BACKEND_TARGET}
            PRIVATE
                RLtools::RLtools
                rendering_raytracing_backend
                rendering_raytracing_backend_depth
                rendering_raytracing_stb
        )
        target_compile_definitions(${RL_TOOLS_GOLDEN_GENERATOR_BACKEND_TARGET} PRIVATE
            RL_TOOLS_RENDERING_RAYTRACING_GOLDEN_GENERATOR_ACTIVE_BACKEND
            RL_TOOLS_RENDERING_RAYTRACING_GOLDEN_BACKEND_NAME="${RL_TOOLS_GOLDEN_GENERATOR_BACKEND_SUFFIX}"
        )
    endif()
endif()

if(RL_TOOLS_RENDERING_ENABLE_RAYTRACING)
    # always present: the CPU (generic software) raytracer vs the OptiX goldens
    add_executable(test_rendering_raytracing_golden_comparison_cpu golden_comparison.cpp)
    target_link_libraries(test_rendering_raytracing_golden_comparison_cpu
        PRIVATE
            RLtools::RLtools
            assimp::assimp
            rendering_raytracing_stb
            GTest::gtest_main
    )
    gtest_discover_tests(test_rendering_raytracing_golden_comparison_cpu PROPERTIES TIMEOUT 300)

    # the backend this build is configured for vs the OptiX goldens; in GENERIC
    # builds this would duplicate the cpu target, so it is skipped
    if(NOT RL_TOOLS_RENDERING_RAYTRACING_BACKEND_RESOLVED STREQUAL "GENERIC")
        string(TOLOWER ${RL_TOOLS_RENDERING_RAYTRACING_BACKEND_RESOLVED} RL_TOOLS_GOLDEN_BACKEND_TARGET_SUFFIX)
        set(RL_TOOLS_GOLDEN_BACKEND_TARGET test_rendering_raytracing_golden_comparison_${RL_TOOLS_GOLDEN_BACKEND_TARGET_SUFFIX})
        add_executable(${RL_TOOLS_GOLDEN_BACKEND_TARGET} golden_comparison.cpp)
        target_link_libraries(${RL_TOOLS_GOLDEN_BACKEND_TARGET}
            PRIVATE
                RLtools::RLtools
                rendering_raytracing_backend
                rendering_raytracing_backend_depth
                rendering_raytracing_stb
                GTest::gtest_main
        )
        target_compile_definitions(${RL_TOOLS_GOLDEN_BACKEND_TARGET} PRIVATE RL_TOOLS_RENDERING_RAYTRACING_GOLDEN_ACTIVE_BACKEND)
        gtest_discover_tests(${RL_TOOLS_GOLDEN_BACKEND_TARGET} PROPERTIES TIMEOUT 300)
    endif()
endif()
