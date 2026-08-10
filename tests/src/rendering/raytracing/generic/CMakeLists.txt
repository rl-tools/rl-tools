add_executable(test_rendering_raytracing_generic backend.cpp)
target_link_libraries(test_rendering_raytracing_generic PRIVATE RLtools::Core GTest::gtest_main)
gtest_discover_tests(test_rendering_raytracing_generic)

add_executable(test_rendering_raytracing_generic_scene scene.cpp)
target_link_libraries(test_rendering_raytracing_generic_scene PRIVATE RLtools::RLtools assimp::assimp rendering_raytracing_stb GTest::gtest_main)
gtest_discover_tests(test_rendering_raytracing_generic_scene)

add_executable(test_rendering_raytracing_overlay_scenarios_generic overlay_scenarios.cpp)
target_link_libraries(test_rendering_raytracing_overlay_scenarios_generic PRIVATE RLtools::RLtools assimp::assimp rendering_raytracing_stb GTest::gtest_main)
gtest_discover_tests(test_rendering_raytracing_overlay_scenarios_generic
    PROPERTIES TIMEOUT 20 LABELS overlay
)

# the same backend-agnostic assertions against the build's active backend (cross-backend parity)
if(NOT RL_TOOLS_RENDERING_RAYTRACING_BACKEND_RESOLVED STREQUAL "GENERIC")
    string(TOLOWER ${RL_TOOLS_RENDERING_RAYTRACING_BACKEND_RESOLVED} RL_TOOLS_SCENE_BACKEND_LOWER)
    string(TOUPPER ${RL_TOOLS_RENDERING_RAYTRACING_BACKEND_RESOLVED} RL_TOOLS_SCENE_BACKEND_UPPER)
    set(RL_TOOLS_SCENE_BACKEND_TARGET test_rendering_raytracing_scene_${RL_TOOLS_SCENE_BACKEND_LOWER})
    add_executable(${RL_TOOLS_SCENE_BACKEND_TARGET} scene.cpp)
    target_link_libraries(${RL_TOOLS_SCENE_BACKEND_TARGET}
        PRIVATE
            RLtools::RLtools
            rendering_raytracing_backend
            rendering_raytracing_backend_depth
            rendering_raytracing_backend_segmentation
            rendering_raytracing_backend_normals
            rendering_raytracing_stb
            GTest::gtest_main
    )
    target_compile_definitions(${RL_TOOLS_SCENE_BACKEND_TARGET} PRIVATE
        RL_TOOLS_RENDERING_RAYTRACING_SCENE_TEST_ACTIVE_BACKEND
        RL_TOOLS_SCENE_SUITE=RENDERING_RAYTRACING_SCENE_${RL_TOOLS_SCENE_BACKEND_UPPER}
    )
    gtest_discover_tests(${RL_TOOLS_SCENE_BACKEND_TARGET} PROPERTIES TIMEOUT 300)

    set(RL_TOOLS_OVERLAY_BACKEND_TARGET test_rendering_raytracing_overlay_scenarios_${RL_TOOLS_SCENE_BACKEND_LOWER})
    add_executable(${RL_TOOLS_OVERLAY_BACKEND_TARGET} overlay_scenarios.cpp)
    target_link_libraries(${RL_TOOLS_OVERLAY_BACKEND_TARGET}
        PRIVATE
            RLtools::RLtools
            rendering_raytracing_backend
            rendering_raytracing_backend_depth
            rendering_raytracing_backend_segmentation
            rendering_raytracing_backend_normals
            rendering_raytracing_stb
            GTest::gtest_main
    )
    target_compile_definitions(${RL_TOOLS_OVERLAY_BACKEND_TARGET} PRIVATE
        RL_TOOLS_RENDERING_RAYTRACING_OVERLAY_TEST_ACTIVE_BACKEND
        RL_TOOLS_OVERLAY_SCENARIOS_SUITE=RENDERING_RAYTRACING_OVERLAY_SCENARIOS_${RL_TOOLS_SCENE_BACKEND_UPPER}
    )
    set(RL_TOOLS_OVERLAY_BACKEND_TIMEOUT 20)
    if(RL_TOOLS_RENDERING_RAYTRACING_BACKEND_RESOLVED STREQUAL "OPTIX")
        set(RL_TOOLS_OVERLAY_BACKEND_TIMEOUT 60)
    endif()
    gtest_discover_tests(${RL_TOOLS_OVERLAY_BACKEND_TARGET}
        PROPERTIES TIMEOUT ${RL_TOOLS_OVERLAY_BACKEND_TIMEOUT} LABELS overlay
    )
endif()
