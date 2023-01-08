#include <layer_in_c/context/cpu.h>


#include <layer_in_c/rl/environments/environments.h>
#include <layer_in_c/rl/components/off_policy_runner/off_policy_runner.h>

#include <layer_in_c/rl/environments/operations_cpu.h>
#include <layer_in_c/rl/algorithms/td3/operations_cpu.h>

#include <gtest/gtest.h>


#define DTYPE float
const DTYPE STATE_TOLERANCE = 0.00001;

namespace lic = layer_in_c;

using ENVIRONMENT_SPEC = lic::rl::environments::pendulum::Specification<DTYPE, lic::rl::environments::pendulum::DefaultParameters<DTYPE>>;
using ENVIRONMENT = lic::rl::environments::Pendulum<lic::devices::CPU, ENVIRONMENT_SPEC>;
typedef lic::rl::components::off_policy_runner::Spec<DTYPE, ENVIRONMENT, 5000, 100, lic::rl::components::off_policy_runner::DefaultParameters<DTYPE>> OffPolicyRunnerSpec;
typedef lic::rl::components::OffPolicyRunner<lic::devices::CPU, OffPolicyRunnerSpec> OffPolicyRunner;

struct PendulumStructureSpecification{
    typedef DTYPE T;
    static constexpr lic::index_t INPUT_DIM = ENVIRONMENT::OBSERVATION_DIM;
    static constexpr lic::index_t OUTPUT_DIM = ENVIRONMENT::ACTION_DIM;
    static constexpr int NUM_LAYERS = 3; // The input and output layers count towards the total number of layers
    static constexpr int HIDDEN_DIM = 30;
    static constexpr lic::nn::activation_functions::ActivationFunction HIDDEN_ACTIVATION_FUNCTION = lic::nn::activation_functions::GELU;
    static constexpr lic::nn::activation_functions::ActivationFunction OUTPUT_ACTIVATION_FUNCTION = lic::nn::activation_functions::IDENTITY;
};

TEST(LAYER_IN_C_RL_ALGORITHMS_OFF_POLICY_RUNNER_TEST, TEST_0) {
    typedef lic::nn_models::mlp::AdamSpecification<lic::devices::CPU, PendulumStructureSpecification, lic::nn::optimizers::adam::DefaultParametersTorch<DTYPE>> SPEC;
    lic::nn_models::mlp::NeuralNetworkAdam<lic::devices::CPU, SPEC> policy;
    std::mt19937 rng(0);
    lic::init_weights(policy, rng);
    OffPolicyRunner off_policy_runner;
    for(int step_i = 0; step_i < 10000; step_i++){
        lic::step(off_policy_runner, policy, rng);
    }
    std::cout << "hello" << std::endl;
}

