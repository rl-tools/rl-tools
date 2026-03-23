project(test_rl_algorithms_ppo_cuda CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

add_executable(test_rl_algorithms_ppo_e2e test_ppo_e2e.cu)
target_link_libraries(test_rl_algorithms_ppo_e2e PRIVATE RLtools::RLtools GTest::gtest_main)
target_compile_options(test_rl_algorithms_ppo_e2e PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe --diag_suppress=186>)
gtest_discover_tests(test_rl_algorithms_ppo_e2e)
