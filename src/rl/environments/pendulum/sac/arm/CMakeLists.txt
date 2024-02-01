add_executable(
        rl_environments_pendulum_sac_arm_test
        test.cpp
)
target_link_libraries(
        rl_environments_pendulum_sac_arm_test
        PRIVATE
        RLtools::Core
)
target_compile_options(rl_environments_pendulum_sac_arm_test PRIVATE -Ofast -march=native -mavx2)

