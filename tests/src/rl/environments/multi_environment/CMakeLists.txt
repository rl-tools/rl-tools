add_executable(test_rl_environments_multi_environment multi_environment.cpp)
target_link_libraries(test_rl_environments_multi_environment RLtools::RLtools GTest::gtest_main)
# the bit-exactness gates compare the same arithmetic through different container layouts;
# fast-math contraction differences between those instantiations are compiler noise, not
# semantics, so this TU compiles with strict FP (source properties append after interface flags)
set_source_files_properties(multi_environment.cpp PROPERTIES COMPILE_OPTIONS "-fno-fast-math;-ffp-contract=off")
gtest_discover_tests(test_rl_environments_multi_environment)

if(RL_TOOLS_BACKEND_ENABLE_CUDA)
    add_executable(test_rl_environments_multi_environment_cuda multi_environment_cuda.cu)
    target_link_libraries(test_rl_environments_multi_environment_cuda RLtools::RLtools RLtools::Test)
    set_target_properties(test_rl_environments_multi_environment_cuda PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
    )
    gtest_discover_tests(test_rl_environments_multi_environment_cuda)
endif()
