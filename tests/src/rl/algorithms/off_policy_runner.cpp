#include <backprop_tools/operations/cpu.h>


#include <backprop_tools/rl/environments/environments.h>
#include <backprop_tools/rl/components/off_policy_runner/off_policy_runner.h>

#include <backprop_tools/rl/environments/operations_cpu.h>
#include <backprop_tools/rl/algorithms/td3/operations_cpu.h>

#include <gtest/gtest.h>


#define DTYPE float
const DTYPE STATE_TOLERANCE = 0.00001;

namespace lic = backprop_tools;

using DEVICE = lic::devices::DefaultCPU;
using ENVIRONMENT_SPEC = lic::rl::environments::pendulum::Specification<DTYPE, DEVICE::index_t, lic::rl::environments::pendulum::DefaultParameters<DTYPE>>;
using ENVIRONMENT = lic::rl::environments::Pendulum<ENVIRONMENT_SPEC>;
typedef lic::rl::components::off_policy_runner::Specification<DTYPE, DEVICE::index_t, ENVIRONMENT, 1, 5000, 100, lic::rl::components::off_policy_runner::DefaultParameters<DTYPE>> OffPolicyRunnerSpec;
typedef lic::rl::components::OffPolicyRunner<OffPolicyRunnerSpec> OffPolicyRunner;

using PendulumStructureSpecification = lic::nn_models::mlp::StructureSpecification<DTYPE, DEVICE::index_t, ENVIRONMENT::OBSERVATION_DIM, ENVIRONMENT::ACTION_DIM, 3, 30, lic::nn::activation_functions::GELU, lic::nn::activation_functions::IDENTITY>;

TEST(BACKPROP_TOOLS_RL_ALGORITHMS_OFF_POLICY_RUNNER_TEST, TEST_0) {
    using OPTIMIZER_PARAMETERS = lic::nn::optimizers::adam::DefaultParametersTorch<DTYPE>;
    using OPTIMIZER = lic::nn::optimizers::Adam<OPTIMIZER_PARAMETERS>;
    typedef lic::nn_models::mlp::AdamSpecification<PendulumStructureSpecification> SPEC;
    DEVICE::SPEC::LOGGING logger;
    DEVICE device;
    device.logger = &logger;
    OPTIMIZER optimizer;
    lic::nn_models::mlp::NeuralNetworkAdam<SPEC> policy;
    lic::malloc(device, policy);
    auto rng = lic::random::default_engine(DEVICE::SPEC::RANDOM(), 0);
    lic::init_weights(device, policy, rng);
    OffPolicyRunner off_policy_runner;
    lic::malloc(device, off_policy_runner);
    ENVIRONMENT envs[OffPolicyRunnerSpec::N_ENVIRONMENTS];
    lic::init(device, off_policy_runner, envs);
    decltype(policy)::Buffers<OffPolicyRunnerSpec::N_ENVIRONMENTS> policy_buffers;
    lic::malloc(device, policy_buffers);
    for(int step_i = 0; step_i < 10000; step_i++){
        lic::step(device, off_policy_runner, policy, policy_buffers, rng);
    }
    lic::free(device, off_policy_runner);
}

