#include <gtest/gtest.h>

#include <layer_in_c/nn_models/models.h>
#include <layer_in_c/nn_models/operations_generic.h>
#include <layer_in_c/utils/rng_std.h>
#include <layer_in_c/rl/environments/pendulum.h>
#include <layer_in_c/rl/algorithms/td3/off_policy_runner.h>

#define DTYPE float
const DTYPE STATE_TOLERANCE = 0.00001;

namespace lic = layer_in_c;

typedef Pendulum<DTYPE, DefaultPendulumParams<DTYPE>> ENVIRONMENT;

TEST(LAYER_IN_C_RL_ALGORITHMS_OFF_POLICY_RUNNER_TEST, TEST_0) {
    typedef lic::nn_models::three_layer_fc::StructureSpecification<DTYPE, ENVIRONMENT::OBSERVATION_DIM,
        50, lic::nn::activation_functions::RELU,
        50, lic::nn::activation_functions::RELU,
        ENVIRONMENT::ACTION_DIM, layer_in_c::nn::activation_functions::TANH> STRUCTURE_SPEC;
    typedef lic::nn_models::three_layer_fc::AdamSpecification<lic::devices::Generic, STRUCTURE_SPEC, lic::nn::optimizers::adam::DefaultParameters<DTYPE>> SPEC;
    lic::nn_models::three_layer_fc::NeuralNetworkAdam<lic::devices::Generic, SPEC> policy;
    std::mt19937 rng(0);
    lic::init_weights<SPEC, layer_in_c::utils::random::stdlib::uniform<DTYPE, typeof(rng)>, typeof(rng)>(policy, rng);
    lic::rl::algorithms::td3::OffPolicyRunner<DTYPE, ENVIRONMENT, lic::rl::algorithms::td3::DefaultOffPolicyRunnerParameters<DTYPE, 5000, 100>> off_policy_runner;
    for(int step_i = 0; step_i < 10000; step_i++){
        step(off_policy_runner, policy, rng);
    }
    std::cout << "hello" << std::endl;
}

