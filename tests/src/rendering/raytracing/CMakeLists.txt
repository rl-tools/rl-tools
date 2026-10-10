option(RL_TOOLS_REQUIRE_RAYTRACING_GOLDENS "Fail raytracing golden tests when reference data is unavailable" OFF)
function(rl_tools_configure_raytracing_golden_target TARGET_NAME)
    if(RL_TOOLS_REQUIRE_RAYTRACING_GOLDENS)
        target_compile_definitions(${TARGET_NAME} PRIVATE RL_TOOLS_REQUIRE_RAYTRACING_GOLDENS=1)
    endif()
endfunction()

function(rl_tools_configure_golden_generator TARGET_NAME BACKEND_NAME)
    target_compile_definitions(${TARGET_NAME} PRIVATE
        RL_TOOLS_RENDERING_RAYTRACING_GOLDEN_BACKEND_NAME="${BACKEND_NAME}"
    )
    cmake_host_system_information(RESULT RL_TOOLS_GOLDEN_HOST_CPU QUERY PROCESSOR_DESCRIPTION)
    configure_file(golden_build.h.in ${TARGET_NAME}-build.h.in @ONLY)
    file(GENERATE
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/generated/${TARGET_NAME}/$<CONFIG>/golden_build.h
        INPUT ${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}-build.h.in
        CONDITION $<COMPILE_LANGUAGE:CXX>
        TARGET ${TARGET_NAME}
    )
    target_include_directories(${TARGET_NAME} PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/generated/${TARGET_NAME}/$<CONFIG>)
endfunction()

add_executable(test_rendering_raytracing_golden_io golden_io_test.cpp)
target_link_libraries(test_rendering_raytracing_golden_io
    PRIVATE
        RLtools::Core
        nlohmann_json::nlohmann_json
        rendering_raytracing_stb
        GTest::gtest_main
)
gtest_discover_tests(test_rendering_raytracing_golden_io
    PROPERTIES TIMEOUT 20 LABELS "golden-io"
)

if(RL_TOOLS_RENDERING_ENABLE_RAYTRACING AND RL_TOOLS_TEST_DATA_PATH)
    add_executable(test_rendering_raytracing_generate_golden_cpu EXCLUDE_FROM_ALL generate_golden.cpp)
    target_link_libraries(test_rendering_raytracing_generate_golden_cpu PRIVATE
        RLtools::RLtools assimp::assimp rendering_raytracing_stb
    )
    rl_tools_configure_golden_generator(test_rendering_raytracing_generate_golden_cpu generic)

    if(NOT RL_TOOLS_RENDERING_RAYTRACING_BACKEND_RESOLVED STREQUAL "GENERIC")
        string(TOLOWER ${RL_TOOLS_RENDERING_RAYTRACING_BACKEND_RESOLVED} RL_TOOLS_GOLDEN_GENERATOR_BACKEND_SUFFIX)
        set(RL_TOOLS_GOLDEN_GENERATOR_BACKEND_TARGET test_rendering_raytracing_generate_golden_${RL_TOOLS_GOLDEN_GENERATOR_BACKEND_SUFFIX})
        add_executable(${RL_TOOLS_GOLDEN_GENERATOR_BACKEND_TARGET} EXCLUDE_FROM_ALL generate_golden.cpp)
        target_link_libraries(${RL_TOOLS_GOLDEN_GENERATOR_BACKEND_TARGET}
            PRIVATE
                RLtools::RLtools
                rendering_raytracing_backend
                rendering_raytracing_backend_depth
                rendering_raytracing_backend_segmentation
                rendering_raytracing_backend_normals
                rendering_raytracing_stb
        )
        target_compile_definitions(${RL_TOOLS_GOLDEN_GENERATOR_BACKEND_TARGET} PRIVATE RL_TOOLS_RENDERING_RAYTRACING_GOLDEN_ACTIVE_BACKEND)
        rl_tools_configure_golden_generator(${RL_TOOLS_GOLDEN_GENERATOR_BACKEND_TARGET} ${RL_TOOLS_GOLDEN_GENERATOR_BACKEND_SUFFIX})
    endif()
endif()

if(RL_TOOLS_RENDERING_ENABLE_RAYTRACING)
    # always present: the CPU reference backend
    add_executable(test_rendering_raytracing_golden_comparison_cpu golden_comparison.cpp)
    target_link_libraries(test_rendering_raytracing_golden_comparison_cpu
        PRIVATE
            RLtools::RLtools
            assimp::assimp
            rendering_raytracing_stb
            GTest::gtest_main
    )
    rl_tools_configure_raytracing_golden_target(test_rendering_raytracing_golden_comparison_cpu)
    gtest_discover_tests(test_rendering_raytracing_golden_comparison_cpu
        PROPERTIES TIMEOUT 300 LABELS "procthor-golden;procthor-golden-generic"
    )

    # the backend this build is configured for vs the CPU goldens; in GENERIC
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
                rendering_raytracing_backend_segmentation
                rendering_raytracing_backend_normals
                rendering_raytracing_stb
                GTest::gtest_main
        )
        target_compile_definitions(${RL_TOOLS_GOLDEN_BACKEND_TARGET} PRIVATE RL_TOOLS_RENDERING_RAYTRACING_GOLDEN_ACTIVE_BACKEND)
        rl_tools_configure_raytracing_golden_target(${RL_TOOLS_GOLDEN_BACKEND_TARGET})
        gtest_discover_tests(${RL_TOOLS_GOLDEN_BACKEND_TARGET}
            PROPERTIES TIMEOUT 300 LABELS "procthor-golden;procthor-golden-${RL_TOOLS_GOLDEN_BACKEND_TARGET_SUFFIX}"
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
