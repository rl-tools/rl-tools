// this is a test to check if everything compiles without any dependencies (by replacing the dependency based math functions with dummy implementations)
#ifdef BACKPROP_TOOLS_OPERATIONS_CPU
#include <backprop_tools/operations/cpu.h>
#else
#include <backprop_tools/operations/dummy.h>
#endif

#include <backprop_tools/rl/environments/environments.h>
#include <backprop_tools/rl/environments/operations_generic.h>
#include <backprop_tools/nn_models/models.h>
#include <backprop_tools/nn_models/operations_generic.h>
#include <backprop_tools/rl/rl.h>
#include <backprop_tools/rl/operations_generic.h>

#include <backprop_tools/rl/utils/evaluation.h>

namespace bpt = backprop_tools;

#ifdef BACKPROP_TOOLS_OPERATIONS_CPU
using DEVICE = bpt::devices::DefaultCPU;
using NN_DEVICE = bpt::devices::DefaultCPU;
using AC_DEVICE = bpt::devices::DefaultCPU;
#else
using DEVICE = bpt::devices::DefaultDummy;
using NN_DEVICE = bpt::devices::DefaultDummy;
using AC_DEVICE = bpt::devices::DefaultDummy;
#endif
using DTYPE = float;
using TI = typename DEVICE::index_t;
typedef bpt::rl::environments::pendulum::Specification<DTYPE, DEVICE::index_t, bpt::rl::environments::pendulum::DefaultParameters<DTYPE>> PENDULUM_SPEC;
typedef bpt::rl::environments::Pendulum<PENDULUM_SPEC> ENVIRONMENT;
ENVIRONMENT env;

template <typename T>
struct TD3PendulumParameters: bpt::rl::algorithms::td3::DefaultParameters<T, AC_DEVICE::index_t>{
    constexpr static typename DEVICE::index_t CRITIC_BATCH_SIZE = 100;
    constexpr static typename DEVICE::index_t ACTOR_BATCH_SIZE = 100;
};

using TD3_PARAMETERS = TD3PendulumParameters<DTYPE>;

using ActorStructureSpec = bpt::nn_models::mlp::StructureSpecification<DTYPE, DEVICE::index_t, ENVIRONMENT::OBSERVATION_DIM, ENVIRONMENT::ACTION_DIM, 3, 64, bpt::nn::activation_functions::RELU, bpt::nn::activation_functions::TANH, TD3_PARAMETERS::ACTOR_BATCH_SIZE>;
using CriticStructureSpec = bpt::nn_models::mlp::StructureSpecification<DTYPE, DEVICE::index_t, ENVIRONMENT::OBSERVATION_DIM + ENVIRONMENT::ACTION_DIM, 1, 3, 64, bpt::nn::activation_functions::RELU, bpt::nn::activation_functions::IDENTITY, TD3_PARAMETERS::CRITIC_BATCH_SIZE>;


using OPTIMIZER_PARAMETERS = typename bpt::nn::optimizers::adam::DefaultParametersTorch<DTYPE, TI>;
using OPTIMIZER = bpt::nn::optimizers::Adam<OPTIMIZER_PARAMETERS>;
using ACTOR_NETWORK_SPEC = bpt::nn_models::mlp::AdamSpecification<ActorStructureSpec>;
using ACTOR_NETWORK_TYPE = bpt::nn_models::mlp::NeuralNetworkAdam<ACTOR_NETWORK_SPEC>;

using ACTOR_TARGET_NETWORK_SPEC = bpt::nn_models::mlp::InferenceSpecification<ActorStructureSpec>;
using ACTOR_TARGET_NETWORK_TYPE = backprop_tools::nn_models::mlp::NeuralNetwork<ACTOR_TARGET_NETWORK_SPEC>;

using CRITIC_NETWORK_SPEC = bpt::nn_models::mlp::AdamSpecification<CriticStructureSpec>;
using CRITIC_NETWORK_TYPE = backprop_tools::nn_models::mlp::NeuralNetworkAdam<CRITIC_NETWORK_SPEC>;

using CRITIC_TARGET_NETWORK_SPEC = backprop_tools::nn_models::mlp::InferenceSpecification<CriticStructureSpec>;
using CRITIC_TARGET_NETWORK_TYPE = backprop_tools::nn_models::mlp::NeuralNetwork<CRITIC_TARGET_NETWORK_SPEC>;

using TD3_SPEC = bpt::rl::algorithms::td3::Specification<DTYPE, AC_DEVICE::index_t, ENVIRONMENT, ACTOR_NETWORK_TYPE, ACTOR_TARGET_NETWORK_TYPE, CRITIC_NETWORK_TYPE, CRITIC_TARGET_NETWORK_TYPE, OPTIMIZER, TD3_PARAMETERS>;
using ActorCriticType = bpt::rl::algorithms::td3::ActorCritic<TD3_SPEC>;


constexpr typename DEVICE::index_t REPLAY_BUFFER_CAP = 500000;
constexpr typename DEVICE::index_t ENVIRONMENT_STEP_LIMIT = 200;
using OFF_POLICY_RUNNER_SPEC = bpt::rl::components::off_policy_runner::Specification<
        DTYPE,
        AC_DEVICE::index_t,
        ENVIRONMENT,
        1,
        REPLAY_BUFFER_CAP,
        ENVIRONMENT_STEP_LIMIT,
        bpt::rl::components::off_policy_runner::DefaultParameters<DTYPE>
>;
bpt::rl::components::OffPolicyRunner<OFF_POLICY_RUNNER_SPEC> off_policy_runner;
ActorCriticType actor_critic;
const DTYPE STATE_TOLERANCE = 0.00001;
constexpr int N_WARMUP_STEPS = ActorCriticType::SPEC::PARAMETERS::ACTOR_BATCH_SIZE;
static_assert(ActorCriticType::SPEC::PARAMETERS::ACTOR_BATCH_SIZE == ActorCriticType::SPEC::PARAMETERS::CRITIC_BATCH_SIZE);

int main() {
    AC_DEVICE::SPEC::LOGGING logger;
    AC_DEVICE device;
    NN_DEVICE nn_device;
    device.logger = &logger;
    nn_device.logger = &logger;
    bpt::malloc(device, off_policy_runner);
    bpt::malloc(device, actor_critic);
    auto rng = bpt::random::default_engine(decltype(device)::SPEC::RANDOM());
    bpt::init(device, actor_critic, rng);

    bool ui = false;

    using CRITIC_BATCH_SPEC = bpt::rl::components::off_policy_runner::BatchSpecification<decltype(off_policy_runner)::SPEC, ActorCriticType::SPEC::PARAMETERS::CRITIC_BATCH_SIZE>;
    bpt::rl::components::off_policy_runner::Batch<CRITIC_BATCH_SPEC> critic_batch;
    bpt::rl::algorithms::td3::CriticTrainingBuffers<ActorCriticType::SPEC> critic_training_buffers;
    CRITIC_NETWORK_TYPE::BuffersForwardBackward<> critic_buffers[2];
    bpt::malloc(device, critic_batch);
    bpt::malloc(device, critic_training_buffers);
    bpt::malloc(device, critic_buffers[0]);
    bpt::malloc(device, critic_buffers[1]);

    using ACTOR_BATCH_SPEC = bpt::rl::components::off_policy_runner::BatchSpecification<decltype(off_policy_runner)::SPEC, ActorCriticType::SPEC::PARAMETERS::ACTOR_BATCH_SIZE>;
    bpt::rl::components::off_policy_runner::Batch<ACTOR_BATCH_SPEC> actor_batch;
    bpt::rl::algorithms::td3::ActorTrainingBuffers<ActorCriticType::SPEC> actor_training_buffers;
    ACTOR_NETWORK_TYPE::Buffers<> actor_buffers[2];
    ACTOR_NETWORK_TYPE::Buffers<OFF_POLICY_RUNNER_SPEC::N_ENVIRONMENTS> actor_buffers_eval;
    bpt::malloc(device, actor_batch);
    bpt::malloc(device, actor_training_buffers);
    bpt::malloc(device, actor_buffers_eval);
    bpt::malloc(device, actor_buffers[0]);
    bpt::malloc(device, actor_buffers[1]);

    for(int step_i = 0; step_i < 15000; step_i++){
        bpt::step(device, off_policy_runner, actor_critic.actor, actor_buffers_eval, rng);

        if(step_i > N_WARMUP_STEPS){
            if(step_i % 1000 == 0){
                bpt::logging::text(device, device.logger, "step_i: ", step_i);
            }
            for(int critic_i = 0; critic_i < 2; critic_i++){
                bpt::target_action_noise(device, actor_critic, critic_training_buffers.target_next_action_noise, rng);
                bpt::gather_batch(device, off_policy_runner, critic_batch, rng);
                bpt::train_critic(device, actor_critic, critic_i == 0 ? actor_critic.critic_1 : actor_critic.critic_2, critic_batch, actor_critic.critic_optimizers[critic_i], actor_buffers[critic_i], critic_buffers[critic_i], critic_training_buffers);
            }
            if(step_i % 2 == 0){
                bpt::gather_batch(device, off_policy_runner, actor_batch, rng);
                bpt::train_actor(device, actor_critic, actor_batch, actor_critic.actor_optimizer, actor_buffers[0], critic_buffers[0], actor_training_buffers);
                bpt::update_critic_targets(device, actor_critic);
                bpt::update_actor_target(device, actor_critic);
            }
        }
        if(step_i % 1000 == 0){
            auto result = bpt::evaluate(device, env, ui, actor_critic.actor, bpt::rl::utils::evaluation::Specification<10, ENVIRONMENT_STEP_LIMIT>(), rng, true);
            bpt::logging::text(device, device.logger, "Mean return: ", result.mean);
            if(result.mean > -200000){
                return 0;
            }
        }
    }
    return -1;

}
