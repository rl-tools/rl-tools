// this is a test to check if everything compiles without any dependencies (by replacing the dependency based math functions with dummy implementations)
#ifdef RL_TOOLS_OPERATIONS_CPU
#include <rl_tools/operations/cpu.h>
#else
#include <rl_tools/operations/dummy.h>
#endif

#include <rl_tools/rl/environments/pendulum/operations_generic.h>
#include <rl_tools/nn_models/models.h>
#include <rl_tools/nn_models/operations_generic.h>
#include <rl_tools/rl/rl.h>
#include <rl_tools/rl/components/off_policy_runner/operations_generic.h>
#include <rl_tools/rl/algorithms/td3/operations_generic.h>

#include <rl_tools/rl/utils/evaluation.h>

namespace rlt = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools;

#ifdef RL_TOOLS_OPERATIONS_CPU
using DEVICE = rlt::devices::DefaultCPU;
using NN_DEVICE = rlt::devices::DefaultCPU;
using AC_DEVICE = rlt::devices::DefaultCPU;
#else
using DEVICE = rlt::devices::DefaultDummy;
using NN_DEVICE = rlt::devices::DefaultDummy;
using AC_DEVICE = rlt::devices::DefaultDummy;
#endif
using DTYPE = float;
using TI = typename DEVICE::index_t;
typedef rlt::rl::environments::pendulum::Specification<DTYPE, DEVICE::index_t, rlt::rl::environments::pendulum::DefaultParameters<DTYPE>> PENDULUM_SPEC;
typedef rlt::rl::environments::Pendulum<PENDULUM_SPEC> ENVIRONMENT;
ENVIRONMENT env;

template <typename T>
struct TD3PendulumParameters: rlt::rl::algorithms::td3::DefaultParameters<T, AC_DEVICE::index_t>{
    constexpr static typename DEVICE::index_t CRITIC_BATCH_SIZE = 100;
    constexpr static typename DEVICE::index_t ACTOR_BATCH_SIZE = 100;
};

using TD3_PARAMETERS = TD3PendulumParameters<DTYPE>;

using ActorStructureSpec = rlt::nn_models::mlp::StructureSpecification<DTYPE, DEVICE::index_t, ENVIRONMENT::OBSERVATION_DIM, ENVIRONMENT::ACTION_DIM, 3, 64, rlt::nn::activation_functions::RELU, rlt::nn::activation_functions::TANH, TD3_PARAMETERS::ACTOR_BATCH_SIZE>;
using CriticStructureSpec = rlt::nn_models::mlp::StructureSpecification<DTYPE, DEVICE::index_t, ENVIRONMENT::OBSERVATION_DIM + ENVIRONMENT::ACTION_DIM, 1, 3, 64, rlt::nn::activation_functions::RELU, rlt::nn::activation_functions::IDENTITY, TD3_PARAMETERS::CRITIC_BATCH_SIZE>;


using OPTIMIZER_SPEC = typename rlt::nn::optimizers::adam::Specification<DTYPE, TI>;
using OPTIMIZER = rlt::nn::optimizers::Adam<OPTIMIZER_SPEC>;
using ACTOR_NETWORK_SPEC = rlt::nn_models::mlp::AdamSpecification<ActorStructureSpec>;
using ACTOR_NETWORK_TYPE = rlt::nn_models::mlp::NeuralNetworkAdam<ACTOR_NETWORK_SPEC>;

using ACTOR_TARGET_NETWORK_SPEC = rlt::nn_models::mlp::InferenceSpecification<ActorStructureSpec>;
using ACTOR_TARGET_NETWORK_TYPE = rlt::nn_models::mlp::NeuralNetwork<ACTOR_TARGET_NETWORK_SPEC>;

using CRITIC_NETWORK_SPEC = rlt::nn_models::mlp::AdamSpecification<CriticStructureSpec>;
using CRITIC_NETWORK_TYPE = rlt::nn_models::mlp::NeuralNetworkAdam<CRITIC_NETWORK_SPEC>;

using CRITIC_TARGET_NETWORK_SPEC = rlt::nn_models::mlp::InferenceSpecification<CriticStructureSpec>;
using CRITIC_TARGET_NETWORK_TYPE = rlt::nn_models::mlp::NeuralNetwork<CRITIC_TARGET_NETWORK_SPEC>;

using TD3_SPEC = rlt::rl::algorithms::td3::Specification<DTYPE, AC_DEVICE::index_t, ENVIRONMENT, ACTOR_NETWORK_TYPE, ACTOR_TARGET_NETWORK_TYPE, CRITIC_NETWORK_TYPE, CRITIC_TARGET_NETWORK_TYPE, OPTIMIZER, TD3_PARAMETERS>;
using ActorCriticType = rlt::rl::algorithms::td3::ActorCritic<TD3_SPEC>;


constexpr typename DEVICE::index_t REPLAY_BUFFER_CAP = 500000;
constexpr typename DEVICE::index_t EPISODE_STEP_LIMIT = 200;
using OFF_POLICY_RUNNER_SPEC = rlt::rl::components::off_policy_runner::Specification<
        DTYPE,
        AC_DEVICE::index_t,
        ENVIRONMENT,
        1,
        false,
        REPLAY_BUFFER_CAP,
        EPISODE_STEP_LIMIT
>;
rlt::rl::components::OffPolicyRunner<OFF_POLICY_RUNNER_SPEC> off_policy_runner;
ActorCriticType actor_critic;
const DTYPE STATE_TOLERANCE = 0.00001;
constexpr int N_WARMUP_STEPS = ActorCriticType::SPEC::PARAMETERS::ACTOR_BATCH_SIZE;
static_assert(ActorCriticType::SPEC::PARAMETERS::ACTOR_BATCH_SIZE == ActorCriticType::SPEC::PARAMETERS::CRITIC_BATCH_SIZE);

int main() {
    AC_DEVICE::SPEC::LOGGING logger;
    AC_DEVICE device;
    NN_DEVICE nn_device;
    rlt::malloc(device, off_policy_runner);
    rlt::malloc(device, actor_critic);
    auto rng = rlt::random::default_engine(decltype(device)::SPEC::RANDOM());
    rlt::init(device, actor_critic, rng);

    rlt::rl::environments::DummyUI ui;

    using CRITIC_BATCH_SPEC = rlt::rl::components::off_policy_runner::BatchSpecification<decltype(off_policy_runner)::SPEC, ActorCriticType::SPEC::PARAMETERS::CRITIC_BATCH_SIZE>;
    rlt::rl::components::off_policy_runner::Batch<CRITIC_BATCH_SPEC> critic_batch;
    rlt::rl::algorithms::td3::CriticTrainingBuffers<ActorCriticType::SPEC> critic_training_buffers;
    CRITIC_NETWORK_TYPE::Buffer<> critic_buffers[2];
    rlt::malloc(device, critic_batch);
    rlt::malloc(device, critic_training_buffers);
    rlt::malloc(device, critic_buffers[0]);
    rlt::malloc(device, critic_buffers[1]);

    using ACTOR_BATCH_SPEC = rlt::rl::components::off_policy_runner::BatchSpecification<decltype(off_policy_runner)::SPEC, ActorCriticType::SPEC::PARAMETERS::ACTOR_BATCH_SIZE>;
    rlt::rl::components::off_policy_runner::Batch<ACTOR_BATCH_SPEC> actor_batch;
    rlt::rl::algorithms::td3::ActorTrainingBuffers<ActorCriticType::SPEC> actor_training_buffers;
    ACTOR_NETWORK_TYPE::Buffer<> actor_buffers[2];
    ACTOR_NETWORK_TYPE::Buffer<OFF_POLICY_RUNNER_SPEC::N_ENVIRONMENTS> actor_buffers_eval;
    rlt::malloc(device, actor_batch);
    rlt::malloc(device, actor_training_buffers);
    rlt::malloc(device, actor_buffers_eval);
    rlt::malloc(device, actor_buffers[0]);
    rlt::malloc(device, actor_buffers[1]);

    for(int step_i = 0; step_i < 15000; step_i++){
        rlt::step(device, off_policy_runner, actor_critic.actor, actor_buffers_eval, rng);

        if(step_i > N_WARMUP_STEPS){
            if(step_i % 1000 == 0){
                rlt::log(device, device.logger, "step_i: ", step_i);
            }
            for(int critic_i = 0; critic_i < 2; critic_i++){
                rlt::target_action_noise(device, actor_critic, critic_training_buffers.target_next_action_noise, rng);
                rlt::gather_batch(device, off_policy_runner, critic_batch, rng);
                rlt::train_critic(device, actor_critic, critic_i == 0 ? actor_critic.critic_1 : actor_critic.critic_2, critic_batch, actor_critic.critic_optimizers[critic_i], actor_buffers[critic_i], critic_buffers[critic_i], critic_training_buffers);
            }
            if(step_i % 2 == 0){
                rlt::gather_batch(device, off_policy_runner, actor_batch, rng);
                rlt::train_actor(device, actor_critic, actor_batch, actor_critic.actor_optimizer, actor_buffers[0], critic_buffers[0], actor_training_buffers);
                rlt::update_critic_targets(device, actor_critic);
                rlt::update_actor_target(device, actor_critic);
            }
        }
        if(step_i % 1000 == 0){
            auto result = rlt::evaluate(device, env, ui, actor_critic.actor, rlt::rl::utils::evaluation::Specification<10, EPISODE_STEP_LIMIT>(), actor_buffers_eval, rng, true);
            rlt::log(device, device.logger, "Mean return: ", result.returns_mean);
            if(result.returns_mean > -200000){
                return 0;
            }
        }
    }
    return -1;

}
