add_executable(test_hyperlux_generic backend.cpp)
target_link_libraries(test_hyperlux_generic PRIVATE RLtools::Core GTest::gtest_main)
gtest_discover_tests(test_hyperlux_generic)

add_executable(test_hyperlux_backend_announcement backend_announcement.cpp)
target_link_libraries(test_hyperlux_backend_announcement PRIVATE RLtools::RLtools assimp::assimp hyperlux_stb GTest::gtest_main)
gtest_discover_tests(test_hyperlux_backend_announcement)

add_executable(test_hyperlux_generic_scene scene.cpp)
target_link_libraries(test_hyperlux_generic_scene PRIVATE RLtools::RLtools assimp::assimp hyperlux_stb GTest::gtest_main)
gtest_discover_tests(test_hyperlux_generic_scene)

add_executable(test_hyperlux_overlay_scenarios_generic overlay_scenarios.cpp)
target_link_libraries(test_hyperlux_overlay_scenarios_generic PRIVATE RLtools::RLtools assimp::assimp hyperlux_stb GTest::gtest_main)
gtest_discover_tests(test_hyperlux_overlay_scenarios_generic
    PROPERTIES TIMEOUT 20 LABELS overlay
)

add_executable(test_hyperlux_probes_overlay_generic probes_overlay.cpp)
target_link_libraries(test_hyperlux_probes_overlay_generic PRIVATE RLtools::RLtools assimp::assimp hyperlux_stb GTest::gtest_main)
gtest_discover_tests(test_hyperlux_probes_overlay_generic
    PROPERTIES TIMEOUT 20 LABELS overlay
)

# the same backend-agnostic assertions against the build's active backend (cross-backend parity)
if(NOT RL_TOOLS_HYPERLUX_BACKEND_RESOLVED STREQUAL "GENERIC")
    string(TOLOWER ${RL_TOOLS_HYPERLUX_BACKEND_RESOLVED} RL_TOOLS_SCENE_BACKEND_LOWER)
    string(TOUPPER ${RL_TOOLS_HYPERLUX_BACKEND_RESOLVED} RL_TOOLS_SCENE_BACKEND_UPPER)
    set(RL_TOOLS_SCENE_BACKEND_TARGET test_hyperlux_scene_${RL_TOOLS_SCENE_BACKEND_LOWER})
    add_executable(${RL_TOOLS_SCENE_BACKEND_TARGET} scene.cpp)
    target_link_libraries(${RL_TOOLS_SCENE_BACKEND_TARGET}
        PRIVATE
            RLtools::RLtools
            hyperlux_backend
            hyperlux_backend_depth
            hyperlux_backend_segmentation
            hyperlux_backend_normals
            hyperlux_stb
            GTest::gtest_main
    )
    target_compile_definitions(${RL_TOOLS_SCENE_BACKEND_TARGET} PRIVATE
        RL_TOOLS_HYPERLUX_SCENE_TEST_ACTIVE_BACKEND
        RL_TOOLS_SCENE_SUITE=HYPERLUX_SCENE_${RL_TOOLS_SCENE_BACKEND_UPPER}
    )
    gtest_discover_tests(${RL_TOOLS_SCENE_BACKEND_TARGET} PROPERTIES TIMEOUT 300)

    set(RL_TOOLS_OVERLAY_BACKEND_TARGET test_hyperlux_overlay_scenarios_${RL_TOOLS_SCENE_BACKEND_LOWER})
    add_executable(${RL_TOOLS_OVERLAY_BACKEND_TARGET} overlay_scenarios.cpp)
    target_link_libraries(${RL_TOOLS_OVERLAY_BACKEND_TARGET}
        PRIVATE
            RLtools::RLtools
            hyperlux_backend
            hyperlux_backend_depth
            hyperlux_backend_segmentation
            hyperlux_backend_normals
            hyperlux_stb
            GTest::gtest_main
    )
    target_compile_definitions(${RL_TOOLS_OVERLAY_BACKEND_TARGET} PRIVATE
        RL_TOOLS_HYPERLUX_OVERLAY_TEST_ACTIVE_BACKEND
        RL_TOOLS_OVERLAY_SCENARIOS_SUITE=HYPERLUX_OVERLAY_SCENARIOS_${RL_TOOLS_SCENE_BACKEND_UPPER}
    )
    set(RL_TOOLS_OVERLAY_BACKEND_TIMEOUT 20)
    if(RL_TOOLS_HYPERLUX_BACKEND_RESOLVED STREQUAL "OPTIX")
        set(RL_TOOLS_OVERLAY_BACKEND_TIMEOUT 60)
    endif()
    gtest_discover_tests(${RL_TOOLS_OVERLAY_BACKEND_TARGET}
        PROPERTIES TIMEOUT ${RL_TOOLS_OVERLAY_BACKEND_TIMEOUT} LABELS overlay
    )
endif()

if(NOT RL_TOOLS_HYPERLUX_BACKEND_RESOLVED STREQUAL "GENERIC")
    set(RL_TOOLS_PROBES_OVERLAY_BACKEND_TARGET test_hyperlux_probes_overlay_${RL_TOOLS_SCENE_BACKEND_LOWER})
    add_executable(${RL_TOOLS_PROBES_OVERLAY_BACKEND_TARGET} probes_overlay.cpp)
    target_link_libraries(${RL_TOOLS_PROBES_OVERLAY_BACKEND_TARGET}
        PRIVATE
            RLtools::RLtools
            hyperlux_backend
            hyperlux_stb
            GTest::gtest_main
    )
    target_compile_definitions(${RL_TOOLS_PROBES_OVERLAY_BACKEND_TARGET} PRIVATE
        RL_TOOLS_HYPERLUX_PROBES_OVERLAY_TEST_ACTIVE_BACKEND
        RL_TOOLS_PROBES_OVERLAY_SUITE=HYPERLUX_PROBES_OVERLAY_${RL_TOOLS_SCENE_BACKEND_UPPER}
    )
    gtest_discover_tests(${RL_TOOLS_PROBES_OVERLAY_BACKEND_TARGET}
        PROPERTIES TIMEOUT 60 LABELS overlay
    )
endif()
