option(RL_TOOLS_REQUIRE_RAYTRACING_GOLDENS "Fail raytracing golden tests when reference data is unavailable" OFF)
set(
    RL_TOOLS_RENDERING_RAYTRACING_GOLDEN_ARTIFACT_ROOT
    "${CMAKE_BINARY_DIR}/raytracing_golden_artifacts"
    CACHE PATH
    "Directory for raytracing golden target, current, and diff artifacts"
)

function(rl_tools_configure_raytracing_golden_target TARGET_NAME)
    target_compile_definitions(${TARGET_NAME}
        PRIVATE
            RL_TOOLS_RENDERING_RAYTRACING_GOLDEN_ARTIFACT_ROOT="${RL_TOOLS_RENDERING_RAYTRACING_GOLDEN_ARTIFACT_ROOT}"
    )
    if(RL_TOOLS_REQUIRE_RAYTRACING_GOLDENS)
        target_compile_definitions(${TARGET_NAME} PRIVATE RL_TOOLS_REQUIRE_RAYTRACING_GOLDENS=1)
    endif()
endfunction()

function(rl_tools_configure_overlay_golden_comparison TARGET_NAME BACKEND_NAME)
    rl_tools_configure_raytracing_golden_target(${TARGET_NAME})
    string(TOUPPER ${BACKEND_NAME} RL_TOOLS_OVERLAY_GOLDEN_BACKEND_UPPER)
    target_compile_definitions(${TARGET_NAME}
        PRIVATE
            RL_TOOLS_OVERLAY_GOLDEN_BACKEND_NAME="${BACKEND_NAME}"
            RL_TOOLS_OVERLAY_GOLDEN_SUITE=RENDERING_RAYTRACING_OVERLAY_GOLDEN_${RL_TOOLS_OVERLAY_GOLDEN_BACKEND_UPPER}
    )
    gtest_discover_tests(${TARGET_NAME}
        PROPERTIES TIMEOUT 300 LABELS "overlay-golden;overlay-golden-${BACKEND_NAME}"
    )
endfunction()

add_executable(test_rendering_raytracing_golden_io golden_io_test.cpp)
target_link_libraries(test_rendering_raytracing_golden_io
    PRIVATE
        RLtools::Core
        rendering_raytracing_stb
        GTest::gtest_main
)
gtest_discover_tests(test_rendering_raytracing_golden_io
    PROPERTIES TIMEOUT 20 LABELS "golden-io"
)

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
                rendering_raytracing_backend_normals
                rendering_raytracing_stb
        )
        target_compile_definitions(${RL_TOOLS_GOLDEN_GENERATOR_BACKEND_TARGET} PRIVATE
            RL_TOOLS_RENDERING_RAYTRACING_GOLDEN_GENERATOR_ACTIVE_BACKEND
            RL_TOOLS_RENDERING_RAYTRACING_GOLDEN_BACKEND_NAME="${RL_TOOLS_GOLDEN_GENERATOR_BACKEND_SUFFIX}"
        )
    endif()

    # the overlay generator stays OptiX-only; the device ptx variants are defined
    # together in the OptiX backend CMakeLists, so their availability is equivalent
    if(TARGET owl::owl AND TARGET rendering_raytracing_device_ptx AND TARGET assimp::assimp)
        add_executable(test_rendering_raytracing_generate_overlay_golden EXCLUDE_FROM_ALL generate_overlay_golden.cpp)
        target_link_libraries(test_rendering_raytracing_generate_overlay_golden
            PRIVATE
                RLtools::RLtools
                owl::owl
                rendering_raytracing_device_ptx
                rendering_raytracing_device_depth_ptx
                rendering_raytracing_device_segmentation_ptx
                rendering_raytracing_device_depth_segmentation_ptx
                rendering_raytracing_device_depth_segmentation_normals_ptx
                assimp::assimp
                nlohmann_json::nlohmann_json
                rendering_raytracing_stb
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
    rl_tools_configure_raytracing_golden_target(test_rendering_raytracing_golden_comparison_cpu)
    gtest_discover_tests(test_rendering_raytracing_golden_comparison_cpu PROPERTIES TIMEOUT 300)

    # backend-independent validation of the published overlay corpus itself
    add_executable(test_rendering_raytracing_overlay_corpus_validation overlay_corpus_validation.cpp)
    target_link_libraries(test_rendering_raytracing_overlay_corpus_validation
        PRIVATE
            RLtools::RLtools
            assimp::assimp
            nlohmann_json::nlohmann_json
            rendering_raytracing_stb
            GTest::gtest_main
    )
    rl_tools_configure_raytracing_golden_target(test_rendering_raytracing_overlay_corpus_validation)
    gtest_discover_tests(test_rendering_raytracing_overlay_corpus_validation
        PROPERTIES TIMEOUT 60 LABELS "overlay-golden;overlay-golden-corpus"
    )

    add_executable(test_rendering_raytracing_overlay_golden_comparison_generic overlay_golden_comparison.cpp)
    target_link_libraries(test_rendering_raytracing_overlay_golden_comparison_generic
        PRIVATE
            RLtools::RLtools
            assimp::assimp
            rendering_raytracing_stb
            GTest::gtest_main
    )
    rl_tools_configure_overlay_golden_comparison(
        test_rendering_raytracing_overlay_golden_comparison_generic
        generic
    )

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
                rendering_raytracing_backend_normals
                rendering_raytracing_stb
                GTest::gtest_main
        )
        target_compile_definitions(${RL_TOOLS_GOLDEN_BACKEND_TARGET} PRIVATE RL_TOOLS_RENDERING_RAYTRACING_GOLDEN_ACTIVE_BACKEND)
        rl_tools_configure_raytracing_golden_target(${RL_TOOLS_GOLDEN_BACKEND_TARGET})
        gtest_discover_tests(${RL_TOOLS_GOLDEN_BACKEND_TARGET} PROPERTIES TIMEOUT 300)

        set(RL_TOOLS_OVERLAY_GOLDEN_BACKEND_TARGET test_rendering_raytracing_overlay_golden_comparison_${RL_TOOLS_GOLDEN_BACKEND_TARGET_SUFFIX})
        add_executable(${RL_TOOLS_OVERLAY_GOLDEN_BACKEND_TARGET} overlay_golden_comparison.cpp)
        target_link_libraries(${RL_TOOLS_OVERLAY_GOLDEN_BACKEND_TARGET}
            PRIVATE
                RLtools::RLtools
                rendering_raytracing_backend
                rendering_raytracing_backend_depth
                rendering_raytracing_backend_segmentation
                rendering_raytracing_backend_normals
                rendering_raytracing_stb
                GTest::gtest_main
        )
        target_compile_definitions(${RL_TOOLS_OVERLAY_GOLDEN_BACKEND_TARGET}
            PRIVATE RL_TOOLS_RENDERING_RAYTRACING_OVERLAY_GOLDEN_ACTIVE_BACKEND
        )
        rl_tools_configure_overlay_golden_comparison(
            ${RL_TOOLS_OVERLAY_GOLDEN_BACKEND_TARGET}
            ${RL_TOOLS_GOLDEN_BACKEND_TARGET_SUFFIX}
        )
    endif()
endif()

# overlay pose ownership contract: producer-style tensor writes must render identically to the
# set_transform* verbs on every backend; self-contained (no golden data)
if(RL_TOOLS_RENDERING_ENABLE_RAYTRACING)
    function(rl_tools_configure_overlay_producer_contract TARGET_NAME BACKEND_NAME)
        string(TOUPPER ${BACKEND_NAME} RL_TOOLS_OVERLAY_PRODUCER_BACKEND_UPPER)
        target_compile_definitions(${TARGET_NAME}
            PRIVATE
                RL_TOOLS_OVERLAY_PRODUCER_SUITE=RENDERING_RAYTRACING_OVERLAY_PRODUCER_${RL_TOOLS_OVERLAY_PRODUCER_BACKEND_UPPER}
        )
        gtest_discover_tests(${TARGET_NAME}
            PROPERTIES TIMEOUT 60 LABELS "overlay-producer;overlay-producer-${BACKEND_NAME}"
        )
    endfunction()

    add_executable(test_rendering_raytracing_overlay_producer_contract_generic overlay_producer_contract.cpp)
    target_link_libraries(test_rendering_raytracing_overlay_producer_contract_generic
        PRIVATE
            RLtools::RLtools
            assimp::assimp
            rendering_raytracing_stb
            GTest::gtest_main
    )
    rl_tools_configure_overlay_producer_contract(
        test_rendering_raytracing_overlay_producer_contract_generic
        generic
    )

    if(NOT RL_TOOLS_RENDERING_RAYTRACING_BACKEND_RESOLVED STREQUAL "GENERIC")
        string(TOLOWER ${RL_TOOLS_RENDERING_RAYTRACING_BACKEND_RESOLVED} RL_TOOLS_OVERLAY_PRODUCER_BACKEND_SUFFIX)
        set(RL_TOOLS_OVERLAY_PRODUCER_BACKEND_TARGET test_rendering_raytracing_overlay_producer_contract_${RL_TOOLS_OVERLAY_PRODUCER_BACKEND_SUFFIX})
        add_executable(${RL_TOOLS_OVERLAY_PRODUCER_BACKEND_TARGET} overlay_producer_contract.cpp)
        target_link_libraries(${RL_TOOLS_OVERLAY_PRODUCER_BACKEND_TARGET}
            PRIVATE
                RLtools::RLtools
                rendering_raytracing_backend
                rendering_raytracing_backend_depth
                rendering_raytracing_backend_segmentation
                rendering_raytracing_backend_normals
                rendering_raytracing_stb
                GTest::gtest_main
        )
        target_compile_definitions(${RL_TOOLS_OVERLAY_PRODUCER_BACKEND_TARGET}
            PRIVATE RL_TOOLS_RENDERING_RAYTRACING_OVERLAY_PRODUCER_ACTIVE_BACKEND
        )
        rl_tools_configure_overlay_producer_contract(
            ${RL_TOOLS_OVERLAY_PRODUCER_BACKEND_TARGET}
            ${RL_TOOLS_OVERLAY_PRODUCER_BACKEND_SUFFIX}
        )
    endif()
endif()
