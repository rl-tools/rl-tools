option(RL_TOOLS_REQUIRE_RAYTRACING_GOLDENS "Fail raytracing golden tests when reference data is unavailable" OFF)
if(NOT RL_TOOLS_TEST_DATA_PATH)
    set(RL_TOOLS_TEST_DATA_PATH "${PROJECT_SOURCE_DIR}/tests/data")
endif()
function(rl_tools_configure_raytracing_golden_target TARGET_NAME)
    target_compile_definitions(${TARGET_NAME} PRIVATE RL_TOOLS_TEST_DATA_PATH=${RL_TOOLS_TEST_DATA_PATH})
    if(RL_TOOLS_REQUIRE_RAYTRACING_GOLDENS)
        target_compile_definitions(${TARGET_NAME} PRIVATE RL_TOOLS_REQUIRE_RAYTRACING_GOLDENS=1)
    endif()
endfunction()

include(golden_build.cmake)

add_executable(test_rendering_raytracing_golden_io golden_io_test.cpp)
target_link_libraries(test_rendering_raytracing_golden_io
    PRIVATE
        RLtools::RLtools
        nlohmann_json::nlohmann_json
        rendering_raytracing_stb
        GTest::gtest_main
)
gtest_discover_tests(test_rendering_raytracing_golden_io
    PROPERTIES TIMEOUT 20 LABELS "golden-io"
)

if(RL_TOOLS_RENDERING_ENABLE_RAYTRACING)
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
    # always present: the generic CPU backend
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

    # the backend this build is configured for; in GENERIC builds this would duplicate the cpu target, so it is skipped
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

    # the wasm reference: generate_golden built as a WASI module (wasm/CMakeLists.txt) renders the
    # corpus into a dump directory once per ctest run (fixture); the comparator loads the dumps natively
    if(RL_TOOLS_RENDERING_RAYTRACING_WASM_GOLDENS_AVAILABLE)
        include(ExternalProject)
        set(RL_TOOLS_GOLDEN_WASM_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/wasm)
        set(RL_TOOLS_GOLDEN_WASM_MODULE ${RL_TOOLS_GOLDEN_WASM_BINARY_DIR}/generate_golden.wasm)
        set(RL_TOOLS_GOLDEN_WASM_DUMP_DIR ${RL_TOOLS_GOLDEN_WASM_BINARY_DIR}/dump)
        ExternalProject_Add(rendering_raytracing_golden_wasm
            SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/wasm
            BINARY_DIR ${RL_TOOLS_GOLDEN_WASM_BINARY_DIR}
            CMAKE_ARGS
                -DFETCHCONTENT_BASE_DIR=${RL_TOOLS_GOLDEN_WASM_DEPENDENCIES_DIR}
                -DRL_TOOLS_DEPENDENCY_SOURCES=${FETCHCONTENT_BASE_DIR}
                -DRL_TOOLS_OFFLINE_BUILD=${RL_TOOLS_OFFLINE_BUILD}
                -DCMAKE_BUILD_TYPE=Release
                -DCONTA=$<TARGET_FILE:conta>
                -DOUTPUT_DIR=${RL_TOOLS_TEST_DATA_PATH}/rendering/raytracing/goldens/procthor-train-1
            BUILD_ALWAYS ON
            INSTALL_COMMAND ""
            BUILD_BYPRODUCTS ${RL_TOOLS_GOLDEN_WASM_MODULE}
            DEPENDS conta
        )
        add_test(NAME RENDERING_RAYTRACING_GOLDEN_WASM.render
            COMMAND ${RL_TOOLS_GOLDEN_WASM_BINARY_DIR}/generate_goldens --dump-dir ${RL_TOOLS_GOLDEN_WASM_DUMP_DIR}
        )
        set_tests_properties(RENDERING_RAYTRACING_GOLDEN_WASM.render PROPERTIES FIXTURES_SETUP golden-wasm LABELS "procthor-golden;procthor-golden-wasm" TIMEOUT 3600)
        add_executable(test_rendering_raytracing_golden_comparison_wasm golden_comparison.cpp)
        target_link_libraries(test_rendering_raytracing_golden_comparison_wasm PRIVATE RLtools::RLtools assimp::assimp rendering_raytracing_stb GTest::gtest_main)
        target_compile_definitions(test_rendering_raytracing_golden_comparison_wasm PRIVATE RL_TOOLS_RENDERING_RAYTRACING_GOLDEN_DUMP_DIRECTORY="${RL_TOOLS_GOLDEN_WASM_DUMP_DIR}")
        rl_tools_configure_raytracing_golden_target(test_rendering_raytracing_golden_comparison_wasm)
        gtest_discover_tests(test_rendering_raytracing_golden_comparison_wasm
            PROPERTIES TIMEOUT 300 LABELS "procthor-golden;procthor-golden-wasm" FIXTURES_REQUIRED golden-wasm
        )
    endif()
endif()

if(RL_TOOLS_RENDERING_RAYTRACING_BACKEND_RESOLVED STREQUAL "VULKAN")
    add_executable(test_rendering_raytracing_runtime_vulkan runtime_vulkan.cpp)
    target_link_libraries(test_rendering_raytracing_runtime_vulkan PRIVATE
        RLtools::RLtools
        rendering_raytracing_backend
        rendering_raytracing_backend_depth
        GTest::gtest_main
    )
    gtest_discover_tests(test_rendering_raytracing_runtime_vulkan
        PROPERTIES TIMEOUT 60 LABELS "raytracing-runtime"
    )
endif()

# overlay pose ownership contract: pose-sink producer writes must render identically to the
# set_pose verb on every backend; self-contained (no golden data)
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

add_executable(test_rendering_raytracing_paper_physics paper_physics.cpp)
target_link_libraries(test_rendering_raytracing_paper_physics PRIVATE RLtools::RLtools GTest::gtest_main)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set_source_files_properties(paper_physics.cpp PROPERTIES COMPILE_OPTIONS "-fno-fast-math;-ffp-contract=off")
endif()
gtest_discover_tests(test_rendering_raytracing_paper_physics PROPERTIES TIMEOUT 20 LABELS "paper-benchmark")

add_executable(test_rendering_raytracing_paper_physics_cpu
    paper_physics_cpu.cpp
    ../../../../src/rendering/raytracing/benchmark/simulator_matrix_physics_cpu.cpp
)
target_link_libraries(test_rendering_raytracing_paper_physics_cpu PRIVATE RLtools::RLtools GTest::gtest_main)
gtest_discover_tests(test_rendering_raytracing_paper_physics_cpu PROPERTIES TIMEOUT 20 LABELS "paper-benchmark")

if(RL_TOOLS_RENDERING_ENABLE_RAYTRACING)
    add_executable(test_rendering_raytracing_typed_outputs_generic typed_outputs.cpp)
    target_link_libraries(test_rendering_raytracing_typed_outputs_generic PRIVATE
        RLtools::RLtools assimp::assimp rendering_raytracing_stb GTest::gtest_main
    )
    gtest_discover_tests(test_rendering_raytracing_typed_outputs_generic
        PROPERTIES TIMEOUT 60 LABELS "raytracing-outputs;raytracing-outputs-generic"
        TEST_PREFIX "generic."
    )
    if(NOT RL_TOOLS_RENDERING_RAYTRACING_BACKEND_RESOLVED STREQUAL "GENERIC")
        string(TOLOWER ${RL_TOOLS_RENDERING_RAYTRACING_BACKEND_RESOLVED} RL_TOOLS_OUTPUTS_BACKEND_SUFFIX)
        set(RL_TOOLS_OUTPUTS_BACKEND_TARGET test_rendering_raytracing_typed_outputs_${RL_TOOLS_OUTPUTS_BACKEND_SUFFIX})
        add_executable(${RL_TOOLS_OUTPUTS_BACKEND_TARGET} typed_outputs.cpp)
        target_link_libraries(${RL_TOOLS_OUTPUTS_BACKEND_TARGET} PRIVATE
            RLtools::RLtools rendering_raytracing_backend rendering_raytracing_backend_depth
            rendering_raytracing_backend_segmentation rendering_raytracing_backend_normals
            rendering_raytracing_stb GTest::gtest_main
        )
        target_compile_definitions(${RL_TOOLS_OUTPUTS_BACKEND_TARGET} PRIVATE RL_TOOLS_RENDERING_RAYTRACING_TYPED_OUTPUTS_ACTIVE_BACKEND)
        gtest_discover_tests(${RL_TOOLS_OUTPUTS_BACKEND_TARGET}
            PROPERTIES TIMEOUT 60 LABELS "raytracing-outputs;raytracing-outputs-${RL_TOOLS_OUTPUTS_BACKEND_SUFFIX}"
            TEST_PREFIX "${RL_TOOLS_OUTPUTS_BACKEND_SUFFIX}."
        )
    endif()
    add_executable(test_rendering_raytracing_pose pose.cpp)
    target_link_libraries(test_rendering_raytracing_pose PRIVATE RLtools::RLtools assimp::assimp rendering_raytracing_stb GTest::gtest_main)
    gtest_discover_tests(test_rendering_raytracing_pose PROPERTIES TIMEOUT 20 LABELS "pose")
    add_executable(test_rendering_raytracing_empty_object_contract_generic empty_object_contract.cpp)
    target_link_libraries(test_rendering_raytracing_empty_object_contract_generic PRIVATE
        RLtools::RLtools assimp::assimp rendering_raytracing_stb GTest::gtest_main
    )
    gtest_discover_tests(test_rendering_raytracing_empty_object_contract_generic
        PROPERTIES TIMEOUT 60 LABELS "empty-object;empty-object-generic"
        TEST_PREFIX "generic."
    )
    if(NOT RL_TOOLS_RENDERING_RAYTRACING_BACKEND_RESOLVED STREQUAL "GENERIC")
        set(RL_TOOLS_EMPTY_OBJECT_BACKEND_TARGET test_rendering_raytracing_empty_object_contract_${RL_TOOLS_OUTPUTS_BACKEND_SUFFIX})
        add_executable(${RL_TOOLS_EMPTY_OBJECT_BACKEND_TARGET} empty_object_contract.cpp)
        target_link_libraries(${RL_TOOLS_EMPTY_OBJECT_BACKEND_TARGET} PRIVATE
            RLtools::RLtools rendering_raytracing_backend rendering_raytracing_backend_depth
            rendering_raytracing_backend_segmentation rendering_raytracing_backend_normals
            rendering_raytracing_stb GTest::gtest_main
        )
        target_compile_definitions(${RL_TOOLS_EMPTY_OBJECT_BACKEND_TARGET} PRIVATE RL_TOOLS_RENDERING_RAYTRACING_EMPTY_OBJECT_ACTIVE_BACKEND)
        gtest_discover_tests(${RL_TOOLS_EMPTY_OBJECT_BACKEND_TARGET}
            PROPERTIES TIMEOUT 60 LABELS "empty-object;empty-object-${RL_TOOLS_OUTPUTS_BACKEND_SUFFIX}"
            TEST_PREFIX "${RL_TOOLS_OUTPUTS_BACKEND_SUFFIX}."
        )
    endif()
    if(RL_TOOLS_RENDERING_RAYTRACING_BACKEND_RESOLVED STREQUAL "OPTIX")
        add_executable(test_rendering_raytracing_optix_lifecycle optix_lifecycle.cpp)
        target_link_libraries(test_rendering_raytracing_optix_lifecycle PRIVATE RLtools::RLtools rendering_raytracing_backend rendering_raytracing_stb GTest::gtest_main)
        gtest_discover_tests(test_rendering_raytracing_optix_lifecycle PROPERTIES TIMEOUT 60 RUN_SERIAL TRUE LABELS "raytracing-lifecycle")
    endif()
endif()
