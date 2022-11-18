#include <gtest/gtest.h>

#include <rl/environments/pendulum.h>
#define DTYPE float
const DTYPE STATE_TOLERANCE = 0.00001;

template <typename T>
struct TestStruct{
    T initial_state[2];
    T action[1];
    int steps;
    T final_state[2];
    T reward;
};
typedef Pendulum<DTYPE, DefaultPendulumParams<DTYPE>> ENVIRONMENT;
template <typename T>
T run(TestStruct<T>& test_struct){
    T state[ENVIRONMENT::STATE_DIM];
//    std::mt19937 rng;
//    sample_initial_state(pendulum, state, rng);
    memcpy(state, test_struct.initial_state, sizeof(T) * ENVIRONMENT::STATE_DIM);
    T next_state[ENVIRONMENT::STATE_DIM];
    T r = 0;
    for(int i = 0; i < test_struct.steps; i++){
        r += ENVIRONMENT::step(state, test_struct.action, next_state);
        memcpy(state, next_state, sizeof(T) * ENVIRONMENT::STATE_DIM);
    }
    EXPECT_NEAR(test_struct.final_state[0], state[0], STATE_TOLERANCE);
    EXPECT_NEAR(test_struct.final_state[1], state[1], STATE_TOLERANCE);
    EXPECT_NEAR(test_struct.reward, r, STATE_TOLERANCE);
}

TestStruct<DTYPE> test_struct_0 = { .initial_state = {0, 0}, .action = {0.5}, .steps = 10, .final_state = {0.5695028285241353, 2.5905997568134187}, .reward = -1.648656098378955};
TestStruct<DTYPE> test_struct_1 = { .initial_state = {0.7853981633974483, 0.0}, .action = {0.5}, .steps = 10, .final_state = {2.892522479441979, 7.746681676609891}, .reward = -39.98352834688362};
TestStruct<DTYPE> test_struct_2 = { .initial_state = {0.7853981633974483, 2.0}, .action = {0.5}, .steps = 10, .final_state = {3.783972224962703, 7.9688564850711785}, .reward = -72.79789746517554};
TestStruct<DTYPE> test_struct_3 = { .initial_state = {0.7853981633974483, 2.0}, .action = {-0.5}, .steps = 10, .final_state = {3.113826968111937, 6.524693886545373}, .reward = -50.60128334382046};

TEST(LAYER_IN_C_RL_ENVIRONMENTS_PENDULUM_TEST, TEST_0) {
    run(test_struct_0);
}
TEST(LAYER_IN_C_RL_ENVIRONMENTS_PENDULUM_TEST, TEST_1) {
    run(test_struct_1);
}
TEST(LAYER_IN_C_RL_ENVIRONMENTS_PENDULUM_TEST, TEST_2) {
    run(test_struct_2);
}
TEST(LAYER_IN_C_RL_ENVIRONMENTS_PENDULUM_TEST, TEST_3) {
    run(test_struct_3);
}

