#include <rl_tools/operations/cpu.h>
#include <rl_tools/rl/environments/pendulum/operations_generic.h>
#include <rl_tools/nn_models/random_uniform/operations_generic.h>
#include <rl_tools/rl/components/off_policy_runner/operations_generic.h>
namespace rlt = rl_tools;

#include <gtest/gtest.h>

using DEVICE = rlt::devices::DefaultCPU;
using T = float;
using TI = DEVICE::index_t;

using ENVIRONMENT_SPEC = rlt::rl::environments::pendulum::Specification<T, TI>;
using ENVIRONMENT = rlt::rl::environments::Pendulum<ENVIRONMENT_SPEC>;
using EXPLORATION_POLICY_SPEC = rlt::nn_models::random_uniform::Specification<T, TI, ENVIRONMENT::Observation::DIM, ENVIRONMENT::ACTION_DIM, rlt::nn_models::random_uniform::Range::MINUS_ONE_TO_ONE>;
using EXPLORATION_POLICY = rlt::nn_models::RandomUniform<EXPLORATION_POLICY_SPEC>;
using POLICIES = rl_tools::utils::Tuple<TI, EXPLORATION_POLICY>;
struct OFF_POLICY_RUNNER_PARAMETERS{
    static constexpr TI N_ENVIRONMENTS = 2;
    static constexpr bool ASYMMETRIC_OBSERVATIONS = !rl_tools::utils::typing::is_same_v<typename ENVIRONMENT::Observation, typename ENVIRONMENT::ObservationPrivileged>;
    static constexpr TI REPLAY_BUFFER_CAPACITY = 1000;
    static constexpr TI EPISODE_STEP_LIMIT = ENVIRONMENT::EPISODE_STEP_LIMIT;
    static constexpr bool COLLECT_EPISODE_STATS = false;
    static constexpr TI EPISODE_STATS_BUFFER_SIZE = 1000;
    static constexpr bool SAMPLE_PARAMETERS = true;
};
using OFF_POLICY_RUNNER_SPEC = rlt::rl::components::off_policy_runner::Specification<T, TI, ENVIRONMENT, POLICIES, OFF_POLICY_RUNNER_PARAMETERS>;
using OFF_POLICY_RUNNER = rlt::rl::components::OffPolicyRunner<OFF_POLICY_RUNNER_SPEC>;

TEST(RL_TOOLS_RL_COMPONENTS_OFF_POLICY_RUNNER_SQUENTIAL_BATCH, TEST){
    DEVICE device;
    DEVICE::SPEC::RANDOM::ENGINE<> rng;
    rlt::init(device);
    rlt::malloc(device, rng);
    rlt::init(device, rng, 0);
    OFF_POLICY_RUNNER off_policy_runner;
    EXPLORATION_POLICY policy;
    EXPLORATION_POLICY::Buffer<> policy_buffer;
    rlt::malloc(device, off_policy_runner);
    rlt::malloc(device, policy_buffer);
    rlt::init(device, off_policy_runner);
    rlt::step<0>(device, off_policy_runner, policy, policy_buffer, rng);
}
