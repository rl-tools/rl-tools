add_executable(
        rl_environments_pendulum_sac_arm_test
        test.cpp
)
target_link_libraries(
        rl_environments_pendulum_sac_arm_test
        PRIVATE
        RLtools::Core
)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_compile_options(rl_environments_pendulum_sac_arm_test PRIVATE -Ofast -march=native -mavx2)
endif()

add_executable(
        rl_environments_pendulum_sac_arm_test_benchmark
        test.cpp
)
target_link_libraries(
        rl_environments_pendulum_sac_arm_test_benchmark
        PRIVATE
        RLtools::Core
)
target_compile_definitions(rl_environments_pendulum_sac_arm_test_benchmark PRIVATE BENCHMARK)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_compile_options(rl_environments_pendulum_sac_arm_test PRIVATE -Ofast -march=native -mavx2)
endif()
