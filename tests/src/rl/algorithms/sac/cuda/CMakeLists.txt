project(test_rl_tools_rl_algorithms_sac_cuda CUDA)


if(RL_TOOLS_BACKEND_ENABLE_CUDA AND RL_TOOLS_BACKEND_ENABLE_MKL)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD  17)

add_executable(test_rl_algorithms_sac_cuda sac.cu)
target_link_libraries(test_rl_algorithms_sac_cuda PRIVATE RLtools::Minimal GTest::gtest_main)
#target_compile_options(test_rl_algorithms_sac_cuda PRIVATE -gG)
gtest_discover_tests(test_rl_algorithms_sac_cuda)

add_executable(test_rl_algorithms_sac_cuda_loop sac_loop.cu)
target_link_libraries(test_rl_algorithms_sac_cuda_loop PRIVATE RLtools::Minimal GTest::gtest_main)
#target_compile_options(test_rl_algorithms_sac_cuda PRIVATE -gG)
#gtest_discover_tests(test_rl_algorithms_sac_cuda_loop)

endif()