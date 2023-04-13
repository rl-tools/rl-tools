#include "full_training.cpp"
#include <gtest/gtest.h>
#ifdef LAYER_IN_C_TEST_RL_ALGORITHMS_TD3_FULL_TRAINING_DEBUG
TEST(LAYER_IN_C_RL_ALGORITHMS_TD3_FULL_TRAINING, TEST_FULL_TRAINING_DEBUG) {
#else
TEST(LAYER_IN_C_RL_ALGORITHMS_TD3_FULL_TRAINING, TEST_FULL_TRAINING) {
#endif
    auto start_time = std::chrono::high_resolution_clock::now();
    run();
    auto current_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = current_time - start_time;
    std::cout << "total time: " << elapsed_seconds.count() << "s" << std::endl;
    if(std::getenv("LAYER_IN_C_TEST_ENABLE_TIMING") != nullptr){
#ifndef LAYER_IN_C_TEST_RL_ALGORITHMS_TD3_FULL_TRAINING_DEBUG
#ifdef LAYER_IN_C_TEST_MACHINE_LENOVO_P1
        ASSERT_LT(elapsed_seconds.count(), 6); // should be 5.5s when run in isolation
#endif
#ifdef LAYER_IN_C_TEST_MACHINE_MACBOOK_M1
        ASSERT_LT(elapsed_seconds.count(), 3); // should be 2.5s when run in isolation
#endif
#endif
    }
}
