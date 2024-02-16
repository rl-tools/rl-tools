#include <rl_tools/operations/cpu.h>


#include <rl_tools/rl/environments/environments.h>
#include <rl_tools/rl/components/off_policy_runner/off_policy_runner.h>

#include <rl_tools/rl/environments/operations_cpu.h>
#include <rl_tools/rl/algorithms/td3/operations_cpu.h>

#include <gtest/gtest.h>


#define DTYPE float
const DTYPE STATE_TOLERANCE = 0.00001;

namespace rlt = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools;

using DEVICE = rlt::devices::DefaultCPU;
using TI = typename DEVICE::index_t;
using ENVIRONMENT_SPEC = rlt::rl::environments::pendulum::Specification<DTYPE, DEVICE::index_t, rlt::rl::environments::pendulum::DefaultParameters<DTYPE>>;
using ENVIRONMENT = rlt::rl::environments::Pendulum<ENVIRONMENT_SPEC>;
typedef rlt::rl::components::off_policy_runner::Specification<DTYPE, DEVICE::index_t, ENVIRONMENT, 1, false, 5000, 100> OffPolicyRunnerSpec;
typedef rlt::rl::components::OffPolicyRunner<OffPolicyRunnerSpec> OffPolicyRunner;

using PendulumStructureSpecification = rlt::nn_models::mlp::StructureSpecification<DTYPE, DEVICE::index_t, ENVIRONMENT::OBSERVATION_DIM, ENVIRONMENT::ACTION_DIM, 3, 30, rlt::nn::activation_functions::GELU, rlt::nn::activation_functions::IDENTITY>;

TEST(RL_TOOLS_RL_ALGORITHMS_OFF_POLICY_RUNNER_TEST, TEST_0) {
    using OPTIMIZER_SPEC = rlt::nn::optimizers::adam::Specification<DTYPE, TI>;
    using OPTIMIZER = rlt::nn::optimizers::Adam<OPTIMIZER_SPEC>;
    typedef rlt::nn_models::mlp::AdamSpecification<PendulumStructureSpecification> SPEC;
    DEVICE device;
    OPTIMIZER optimizer;
    rlt::nn_models::mlp::NeuralNetworkAdam<SPEC> policy;
    rlt::malloc(device, policy);
    auto rng = rlt::random::default_engine(DEVICE::SPEC::RANDOM(), 0);
    rlt::init_weights(device, policy, rng);
    OffPolicyRunner off_policy_runner;
    rlt::malloc(device, off_policy_runner);
    ENVIRONMENT envs[OffPolicyRunnerSpec::N_ENVIRONMENTS];
    rlt::init(device, off_policy_runner, envs);
    decltype(policy)::Buffer<OffPolicyRunnerSpec::N_ENVIRONMENTS> policy_buffers;
    rlt::malloc(device, policy_buffers);
    for(int step_i = 0; step_i < 10000; step_i++){
        rlt::step(device, off_policy_runner, policy, policy_buffers, rng);
    }
    rlt::free(device, off_policy_runner);
}

