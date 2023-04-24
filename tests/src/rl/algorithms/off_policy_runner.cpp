#include <backprop_tools/operations/cpu.h>


#include <backprop_tools/rl/environments/environments.h>
#include <backprop_tools/rl/components/off_policy_runner/off_policy_runner.h>

#include <backprop_tools/rl/environments/operations_cpu.h>
#include <backprop_tools/rl/algorithms/td3/operations_cpu.h>

#include <gtest/gtest.h>


#define DTYPE float
const DTYPE STATE_TOLERANCE = 0.00001;

namespace bpt = backprop_tools;

using DEVICE = bpt::devices::DefaultCPU;
using ENVIRONMENT_SPEC = bpt::rl::environments::pendulum::Specification<DTYPE, DEVICE::index_t, bpt::rl::environments::pendulum::DefaultParameters<DTYPE>>;
using ENVIRONMENT = bpt::rl::environments::Pendulum<ENVIRONMENT_SPEC>;
typedef bpt::rl::components::off_policy_runner::Specification<DTYPE, DEVICE::index_t, ENVIRONMENT, 1, 5000, 100, bpt::rl::components::off_policy_runner::DefaultParameters<DTYPE>> OffPolicyRunnerSpec;
typedef bpt::rl::components::OffPolicyRunner<OffPolicyRunnerSpec> OffPolicyRunner;

using PendulumStructureSpecification = bpt::nn_models::mlp::StructureSpecification<DTYPE, DEVICE::index_t, ENVIRONMENT::OBSERVATION_DIM, ENVIRONMENT::ACTION_DIM, 3, 30, bpt::nn::activation_functions::GELU, bpt::nn::activation_functions::IDENTITY>;

TEST(BACKPROP_TOOLS_RL_ALGORITHMS_OFF_POLICY_RUNNER_TEST, TEST_0) {
    using OPTIMIZER_PARAMETERS = bpt::nn::optimizers::adam::DefaultParametersTorch<DTYPE>;
    using OPTIMIZER = bpt::nn::optimizers::Adam<OPTIMIZER_PARAMETERS>;
    typedef bpt::nn_models::mlp::AdamSpecification<PendulumStructureSpecification> SPEC;
    DEVICE::SPEC::LOGGING logger;
    DEVICE device;
    device.logger = &logger;
    OPTIMIZER optimizer;
    bpt::nn_models::mlp::NeuralNetworkAdam<SPEC> policy;
    bpt::malloc(device, policy);
    auto rng = bpt::random::default_engine(DEVICE::SPEC::RANDOM(), 0);
    bpt::init_weights(device, policy, rng);
    OffPolicyRunner off_policy_runner;
    bpt::malloc(device, off_policy_runner);
    ENVIRONMENT envs[OffPolicyRunnerSpec::N_ENVIRONMENTS];
    bpt::init(device, off_policy_runner, envs);
    decltype(policy)::Buffers<OffPolicyRunnerSpec::N_ENVIRONMENTS> policy_buffers;
    bpt::malloc(device, policy_buffers);
    for(int step_i = 0; step_i < 10000; step_i++){
        bpt::step(device, off_policy_runner, policy, policy_buffers, rng);
    }
    bpt::free(device, off_policy_runner);
}

