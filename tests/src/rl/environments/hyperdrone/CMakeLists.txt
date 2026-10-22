if(NOT RL_TOOLS_DISABLE_TAR)
    add_executable(test_rl_environments_hyperdrone_controllers controllers.cpp)
    target_link_libraries(test_rl_environments_hyperdrone_controllers PRIVATE RLtools::RLtools GTest::gtest_main)
    if(TARGET hyperlux_backend)
        target_link_libraries(test_rl_environments_hyperdrone_controllers PRIVATE hyperlux_backend hyperlux_stb)
    endif()
    gtest_discover_tests(test_rl_environments_hyperdrone_controllers)
    add_executable(test_rl_environments_hyperdrone_raptor raptor.cpp)
    target_link_libraries(test_rl_environments_hyperdrone_raptor PRIVATE RLtools::RLtools GTest::gtest_main)
    gtest_discover_tests(test_rl_environments_hyperdrone_raptor)
    if(RL_TOOLS_BACKEND_ENABLE_CUDA)
        add_executable(test_rl_environments_hyperdrone_raptor_cuda raptor_cuda.cu)
        target_link_libraries(test_rl_environments_hyperdrone_raptor_cuda PRIVATE RLtools::RLtools GTest::gtest_main)
        set_target_properties(test_rl_environments_hyperdrone_raptor_cuda PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
        gtest_discover_tests(test_rl_environments_hyperdrone_raptor_cuda)
    endif()
endif()

add_executable(test_rl_environments_hyperdrone_on_policy_gradient on_policy_gradient.cpp)
target_link_libraries(test_rl_environments_hyperdrone_on_policy_gradient PRIVATE RLtools::RLtools GTest::gtest_main)
gtest_discover_tests(test_rl_environments_hyperdrone_on_policy_gradient)

add_executable(test_rl_environments_hyperdrone_yaw yaw.cpp)
target_link_libraries(test_rl_environments_hyperdrone_yaw PRIVATE RLtools::RLtools GTest::gtest_main)
gtest_discover_tests(test_rl_environments_hyperdrone_yaw)
if(RL_TOOLS_BACKEND_ENABLE_CUDA)
    add_executable(test_rl_environments_hyperdrone_yaw_cuda yaw_cuda.cu)
    target_link_libraries(test_rl_environments_hyperdrone_yaw_cuda PRIVATE RLtools::RLtools GTest::gtest_main)
    set_target_properties(test_rl_environments_hyperdrone_yaw_cuda PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
    gtest_discover_tests(test_rl_environments_hyperdrone_yaw_cuda)
endif()

if(RL_TOOLS_ENABLE_HYPERLUX)
    add_executable(test_rl_environments_hyperdrone_spy_capture spy_capture.cpp)
    target_link_libraries(test_rl_environments_hyperdrone_spy_capture PRIVATE RLtools::RLtools RLtools::Test hyperlux_backend)
    gtest_discover_tests(test_rl_environments_hyperdrone_spy_capture)
    add_executable(test_rl_environments_hyperdrone_spy_lifecycle spy_lifecycle.cpp)
    target_link_libraries(test_rl_environments_hyperdrone_spy_lifecycle PRIVATE RLtools::RLtools RLtools::Test hyperlux_backend)
    gtest_discover_tests(test_rl_environments_hyperdrone_spy_lifecycle PROPERTIES TIMEOUT 90)
    if(RL_TOOLS_BACKEND_ENABLE_CUDA AND TARGET owl::owl)
        add_executable(test_rl_environments_hyperdrone_spy_capture_cuda spy_capture_cuda.cu)
        target_link_libraries(test_rl_environments_hyperdrone_spy_capture_cuda PRIVATE RLtools::RLtools RLtools::Test owl::owl hyperlux_device_ptx assimp::assimp hyperlux_stb)
        set_target_properties(test_rl_environments_hyperdrone_spy_capture_cuda PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
        gtest_discover_tests(test_rl_environments_hyperdrone_spy_capture_cuda)
    endif()
endif()

if(RL_TOOLS_ENABLE_HYPERLUX AND RL_TOOLS_TEST_DATA_PATH)
    add_executable(test_rl_environments_hyperdrone_rotorcraft rotorcraft.cpp)
    target_link_libraries(test_rl_environments_hyperdrone_rotorcraft
        PRIVATE
            RLtools::RLtools
            RLtools::Test
            hyperlux_backend
            hyperlux_stb
    )
    gtest_discover_tests(test_rl_environments_hyperdrone_rotorcraft)
    add_executable(test_rl_environments_hyperdrone_world world.cpp)
    target_link_libraries(test_rl_environments_hyperdrone_world
        PRIVATE
            RLtools::RLtools
            RLtools::Test
            hyperlux_backend
            hyperlux_stb
    )
    gtest_discover_tests(test_rl_environments_hyperdrone_world)
    add_executable(test_rl_environments_hyperdrone_annotations annotations.cpp)
    target_link_libraries(test_rl_environments_hyperdrone_annotations
        PRIVATE
            RLtools::RLtools
            RLtools::Test
            hyperlux_backend
            hyperlux_stb
    )
    gtest_discover_tests(test_rl_environments_hyperdrone_annotations)
    add_executable(test_rl_environments_hyperdrone_target_frame target_frame.cpp)
    target_link_libraries(test_rl_environments_hyperdrone_target_frame
        PRIVATE
            RLtools::RLtools
            RLtools::Test
            hyperlux_backend
            hyperlux_stb
    )
    gtest_discover_tests(test_rl_environments_hyperdrone_target_frame)
    add_executable(test_rl_environments_hyperdrone_moving_gate moving_gate.cpp)
    target_link_libraries(test_rl_environments_hyperdrone_moving_gate
        PRIVATE
            RLtools::RLtools
            RLtools::Test
            hyperlux_backend
            hyperlux_backend_depth
            hyperlux_backend_segmentation
            hyperlux_backend_normals
            hyperlux_stb
    )
    gtest_discover_tests(test_rl_environments_hyperdrone_moving_gate)
    add_executable(test_rl_environments_hyperdrone_visual_inertial_localization visual_inertial_localization.cpp)
    target_link_libraries(test_rl_environments_hyperdrone_visual_inertial_localization
        PRIVATE
            RLtools::RLtools
            RLtools::Test
            hyperlux_backend
            hyperlux_stb
    )
    gtest_discover_tests(test_rl_environments_hyperdrone_visual_inertial_localization)
    add_executable(test_rl_environments_hyperdrone_segmentation_determinism segmentation_determinism.cpp)
    target_link_libraries(test_rl_environments_hyperdrone_segmentation_determinism
        PRIVATE
            RLtools::RLtools
            RLtools::Test
            hyperlux_backend
            hyperlux_backend_segmentation
            hyperlux_stb
    )
    gtest_discover_tests(test_rl_environments_hyperdrone_segmentation_determinism)
    add_executable(test_rl_environments_hyperdrone_self_visible self_visible.cpp)
    target_link_libraries(test_rl_environments_hyperdrone_self_visible
        PRIVATE
            RLtools::RLtools
            RLtools::Test
            hyperlux_backend
            hyperlux_backend_segmentation
            hyperlux_stb
    )
    gtest_discover_tests(test_rl_environments_hyperdrone_self_visible)
    add_executable(test_rl_environments_hyperdrone_multi_agent multi_agent.cpp)
    target_link_libraries(test_rl_environments_hyperdrone_multi_agent
        PRIVATE
            RLtools::RLtools
            RLtools::Test
            hyperlux_backend
            hyperlux_backend_segmentation
            hyperlux_stb
    )
    gtest_discover_tests(test_rl_environments_hyperdrone_multi_agent)
    add_executable(test_rl_environments_hyperdrone_presets presets.cpp)
    target_link_libraries(test_rl_environments_hyperdrone_presets
        PRIVATE
            RLtools::RLtools
            RLtools::Test
            hyperlux_backend
            hyperlux_stb
    )
    gtest_discover_tests(test_rl_environments_hyperdrone_presets)
    add_executable(test_rl_environments_hyperdrone_episodes episodes.cpp)
    target_link_libraries(test_rl_environments_hyperdrone_episodes
        PRIVATE
            RLtools::RLtools
            RLtools::Test
            hyperlux_backend
            hyperlux_stb
    )
    gtest_discover_tests(test_rl_environments_hyperdrone_episodes)
    add_executable(test_rl_environments_hyperdrone_on_policy_runner_batched on_policy_runner_batched.cpp)
    target_link_libraries(test_rl_environments_hyperdrone_on_policy_runner_batched
        PRIVATE
            RLtools::RLtools
            RLtools::Test
            hyperlux_backend
            hyperlux_stb
    )
    gtest_discover_tests(test_rl_environments_hyperdrone_on_policy_runner_batched)
    # golden-rollout generator for the Python binding (run deliberately from the repo root)
    set(HYPERDRONE_PYTHON_ENVIRONMENT_NATIVE ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../interface/python/hyperdrone/hyperdrone/_native/environment)
    add_executable(test_rl_environments_hyperdrone_python_golden python_golden.cpp ${HYPERDRONE_PYTHON_ENVIRONMENT_NATIVE}/impl.cpp)
    target_compile_definitions(test_rl_environments_hyperdrone_python_golden PRIVATE
        HYPERDRONE_ENVIRONMENT_NUM_WORLDS=1
        HYPERDRONE_ENVIRONMENT_INSTANCES=2
        HYPERDRONE_ENVIRONMENT_CAM_WIDTH=16
        HYPERDRONE_ENVIRONMENT_CAM_HEIGHT=16
        HYPERDRONE_ENVIRONMENT_SHADING=0
        HYPERDRONE_ENVIRONMENT_MB_SAMPLES=1
        HYPERDRONE_ENVIRONMENT_PRESET=0
        HYPERDRONE_ENVIRONMENT_CONTROL=0
        HYPERDRONE_ENVIRONMENT_ACTUATION=0
        HYPERDRONE_ENVIRONMENT_TASK=0
        HYPERDRONE_ENVIRONMENT_N_AGENTS=1
        HYPERDRONE_ENVIRONMENT_N_CAMERA_AGENTS=1
        HYPERDRONE_ENVIRONMENT_DEVICE_CUDA=0
        HYPERDRONE_CAMERA_CONFIG_HEADER=\"${CMAKE_CURRENT_SOURCE_DIR}/python_camera_config.h\"
        HYPERDRONE_ENVIRONMENT_DATASET=0
        HYPERDRONE_MODULE_NAME=hyperdrone_environment_golden
    )
    target_link_libraries(test_rl_environments_hyperdrone_python_golden
        PRIVATE
            RLtools::RLtools
            hyperlux_backend
            hyperlux_stb
    )
    add_executable(test_rl_environments_hyperdrone_user_task_header user_task_header.cpp ${HYPERDRONE_PYTHON_ENVIRONMENT_NATIVE}/impl.cpp)
    get_target_property(HYPERDRONE_USER_TASK_DEFINES test_rl_environments_hyperdrone_python_golden COMPILE_DEFINITIONS)
    list(FILTER HYPERDRONE_USER_TASK_DEFINES EXCLUDE REGEX "^HYPERDRONE_MODULE_NAME=")
    target_compile_definitions(test_rl_environments_hyperdrone_user_task_header PRIVATE ${HYPERDRONE_USER_TASK_DEFINES} HYPERDRONE_MODULE_NAME=hyperdrone_user_task HYPERDRONE_ENVIRONMENT_SPEC_HEADER=${PROJECT_SOURCE_DIR}/interface/python/hyperdrone/tests/environment/user_task_header.h)
    target_link_libraries(test_rl_environments_hyperdrone_user_task_header PRIVATE RLtools::RLtools RLtools::Test hyperlux_backend hyperlux_stb)
    gtest_discover_tests(test_rl_environments_hyperdrone_user_task_header)
    if(RL_TOOLS_BACKEND_ENABLE_CUDA AND TARGET owl::owl)
        add_executable(test_rl_environments_hyperdrone_rotorcraft_cuda rotorcraft_cuda.cu)
        target_link_libraries(test_rl_environments_hyperdrone_rotorcraft_cuda
            PRIVATE
                RLtools::RLtools
                RLtools::Test
                owl::owl
                hyperlux_device_ptx
                assimp::assimp
                hyperlux_stb
        )
        set_target_properties(test_rl_environments_hyperdrone_rotorcraft_cuda PROPERTIES
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON
        )
        gtest_discover_tests(test_rl_environments_hyperdrone_rotorcraft_cuda)
        add_executable(test_rl_environments_hyperdrone_world_cuda world_cuda.cu)
        target_link_libraries(test_rl_environments_hyperdrone_world_cuda
            PRIVATE
                RLtools::RLtools
                RLtools::Test
                owl::owl
                hyperlux_device_ptx
                assimp::assimp
                hyperlux_stb
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
                hyperlux_device_ptx
                assimp::assimp
                hyperlux_stb
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
                hyperlux_device_ptx
                assimp::assimp
                hyperlux_stb
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
                hyperlux_device_ptx
                assimp::assimp
                hyperlux_stb
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
                hyperlux_device_ptx
                hyperlux_device_normals_ptx
                assimp::assimp
                hyperlux_stb
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
                hyperlux_device_ptx
                assimp::assimp
                hyperlux_stb
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
                hyperlux_device_ptx
                assimp::assimp
                hyperlux_stb
        )
        set_target_properties(test_rl_environments_hyperdrone_benchmark_cuda PROPERTIES
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON
        )
        gtest_discover_tests(test_rl_environments_hyperdrone_benchmark_cuda PROPERTIES TIMEOUT 300)
    endif()
endif()

if(RL_TOOLS_ENABLE_HYPERLUX)
    add_executable(test_rl_environments_hyperdrone_drones drones.cpp)
    target_link_libraries(test_rl_environments_hyperdrone_drones
        PRIVATE
            RLtools::RLtools
            RLtools::Test
            hyperlux_backend
            hyperlux_backend_segmentation
            hyperlux_stb
    )
    gtest_discover_tests(test_rl_environments_hyperdrone_drones)
    if(RL_TOOLS_BACKEND_ENABLE_CUDA AND TARGET owl::owl)
        add_executable(test_rl_environments_hyperdrone_drones_cuda drones_cuda.cu)
        target_link_libraries(test_rl_environments_hyperdrone_drones_cuda
            PRIVATE
                RLtools::RLtools
                RLtools::Test
                owl::owl
                hyperlux_device_ptx
                assimp::assimp
                hyperlux_stb
        )
        set_target_properties(test_rl_environments_hyperdrone_drones_cuda PROPERTIES
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON
        )
        gtest_discover_tests(test_rl_environments_hyperdrone_drones_cuda)
    endif()
endif()
