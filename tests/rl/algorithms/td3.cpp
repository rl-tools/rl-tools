#include <gtest/gtest.h>

#include <nn_models/three_layer_fc.h>

#include <rl/environments/pendulum.h>
#include <rl/algorithms/td3/off_policy_runner.h>
#define DTYPE float
const DTYPE STATE_TOLERANCE = 0.00001;

template<typename T>
class AdamParameters{
public:
    static constexpr T ALPHA   = 0.001;
    static constexpr T BETA_1  = 0.9;
    static constexpr T BETA_2  = 0.999;
    static constexpr T EPSILON = 1e-7;

};

typedef Pendulum<DTYPE, DefaultPendulumParams<DTYPE>> ENVIRONMENT;

TEST(LAYER_IN_C_RL_ALGORITHMS_OFF_POLICY_RUNNER_TEST, TEST_0) {
    layer_in_c::nn_models::ThreeLayerNeuralNetworkTrainingAdam<DTYPE, ENVIRONMENT::STATE_DIM,
            50, layer_in_c::nn::activation_functions::RELU,
            50, layer_in_c::nn::activation_functions::RELU,
            ENVIRONMENT::ACTION_DIM, layer_in_c::nn::activation_functions::TANH, AdamParameters<DTYPE>> policy;
    std::mt19937 rng(0);
    layer_in_c::nn_models::init_weights(policy, rng);
    OffPolicyRunner<DTYPE, ENVIRONMENT, typeof(policy), 1000, 100> off_policy_runner;
    for(int step_i = 0; step_i < 10000; step_i++){
        step(off_policy_runner, policy, rng);
    }
    std::cout << "hello" << std::endl;
}

