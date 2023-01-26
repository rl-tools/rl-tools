// this is a test to check if everything compiles without any dependencies (by replacing the dependency based math functions with dummy implementations)
#ifdef LAYER_IN_C_OPERATIONS_CPU
#include <layer_in_c/operations/cpu.h>
#else
#include <layer_in_c/operations/dummy.h>
#endif

#include <layer_in_c/rl/environments/environments.h>
#include <layer_in_c/rl/environments/operations_generic.h>
#include <layer_in_c/nn_models/models.h>
#include <layer_in_c/nn_models/operations_generic.h>
#include <layer_in_c/rl/rl.h>
#include <layer_in_c/rl/operations_generic.h>

#include <layer_in_c/rl/utils/evaluation.h>

namespace lic = layer_in_c;
using DTYPE = float;

#ifdef LAYER_IN_C_OPERATIONS_CPU
using DEVICE = lic::devices::DefaultCPU;
using NN_DEVICE = lic::devices::DefaultCPU;
using AC_DEVICE = lic::devices::DefaultCPU;
#else
using DEVICE = lic::devices::DefaultDummy;
using NN_DEVICE = lic::devices::DefaultDummy;
using AC_DEVICE = lic::devices::DefaultDummy;
#endif
typedef lic::rl::environments::pendulum::Specification<DTYPE, DEVICE::index_t, lic::rl::environments::pendulum::DefaultParameters<DTYPE>> PENDULUM_SPEC;
typedef lic::rl::environments::Pendulum<PENDULUM_SPEC> ENVIRONMENT;
ENVIRONMENT env;

template <typename T>
struct TD3PendulumParameters: lic::rl::algorithms::td3::DefaultParameters<T, AC_DEVICE::index_t>{
    constexpr static typename DEVICE::index_t CRITIC_BATCH_SIZE = 100;
    constexpr static typename DEVICE::index_t ACTOR_BATCH_SIZE = 100;
};

using TD3_PARAMETERS = TD3PendulumParameters<DTYPE>;

using ActorStructureSpec = lic::nn_models::mlp::StructureSpecification<DTYPE, DEVICE::index_t, ENVIRONMENT::OBSERVATION_DIM, ENVIRONMENT::ACTION_DIM, 3, 64, lic::nn::activation_functions::RELU, lic::nn::activation_functions::TANH, TD3_PARAMETERS::ACTOR_BATCH_SIZE>;
using CriticStructureSpec = lic::nn_models::mlp::StructureSpecification<DTYPE, DEVICE::index_t, ENVIRONMENT::OBSERVATION_DIM + ENVIRONMENT::ACTION_DIM, 1, 3, 64, lic::nn::activation_functions::RELU, lic::nn::activation_functions::IDENTITY, TD3_PARAMETERS::CRITIC_BATCH_SIZE>;


using ACTOR_NETWORK_SPEC = lic::nn_models::mlp::AdamSpecification<ActorStructureSpec, typename lic::nn::optimizers::adam::DefaultParametersTorch<DTYPE>>;
using ACTOR_NETWORK_TYPE = lic::nn_models::mlp::NeuralNetworkAdam<ACTOR_NETWORK_SPEC>;

using ACTOR_TARGET_NETWORK_SPEC = lic::nn_models::mlp::InferenceSpecification<ActorStructureSpec>;
using ACTOR_TARGET_NETWORK_TYPE = layer_in_c::nn_models::mlp::NeuralNetwork<ACTOR_TARGET_NETWORK_SPEC>;

using CRITIC_NETWORK_SPEC = lic::nn_models::mlp::AdamSpecification<CriticStructureSpec, typename lic::nn::optimizers::adam::DefaultParametersTorch<DTYPE>>;
using CRITIC_NETWORK_TYPE = layer_in_c::nn_models::mlp::NeuralNetworkAdam<CRITIC_NETWORK_SPEC>;

using CRITIC_TARGET_NETWORK_SPEC = layer_in_c::nn_models::mlp::InferenceSpecification<CriticStructureSpec>;
using CRITIC_TARGET_NETWORK_TYPE = layer_in_c::nn_models::mlp::NeuralNetwork<CRITIC_TARGET_NETWORK_SPEC>;

using TD3_SPEC = lic::rl::algorithms::td3::Specification<DTYPE, AC_DEVICE::index_t, ENVIRONMENT, ACTOR_NETWORK_TYPE, ACTOR_TARGET_NETWORK_TYPE, CRITIC_NETWORK_TYPE, CRITIC_TARGET_NETWORK_TYPE, TD3_PARAMETERS>;
using ActorCriticType = lic::rl::algorithms::td3::ActorCritic<TD3_SPEC>;


constexpr typename DEVICE::index_t REPLAY_BUFFER_CAP = 500000;
constexpr typename DEVICE::index_t ENVIRONMENT_STEP_LIMIT = 200;
AC_DEVICE::SPEC::LOGGING logger;
AC_DEVICE device(logger);
NN_DEVICE nn_device(logger);
lic::rl::components::OffPolicyRunner<
        lic::rl::components::off_policy_runner::Specification<
                DTYPE,
                AC_DEVICE::index_t,
                ENVIRONMENT,
                REPLAY_BUFFER_CAP,
                ENVIRONMENT_STEP_LIMIT,
                lic::rl::components::off_policy_runner::DefaultParameters<DTYPE>
        >
> off_policy_runner;
ActorCriticType actor_critic;
const DTYPE STATE_TOLERANCE = 0.00001;
constexpr int N_WARMUP_STEPS = ActorCriticType::SPEC::PARAMETERS::ACTOR_BATCH_SIZE;
static_assert(ActorCriticType::SPEC::PARAMETERS::ACTOR_BATCH_SIZE == ActorCriticType::SPEC::PARAMETERS::CRITIC_BATCH_SIZE);

int main() {
    lic::malloc(device, actor_critic);
    auto rng = lic::random::default_engine(decltype(device)::SPEC::RANDOM());
    lic::init(device, actor_critic, rng);

    lic::rl::components::replay_buffer::Batch<decltype(off_policy_runner.replay_buffer)::SPEC, ActorCriticType::SPEC::PARAMETERS::CRITIC_BATCH_SIZE> critic_batch;
    lic::rl::algorithms::td3::CriticTrainingBuffers<ActorCriticType::SPEC> critic_training_buffers;
    lic::malloc(device, critic_batch);
    lic::malloc(device, critic_training_buffers);

    lic::rl::components::replay_buffer::Batch<decltype(off_policy_runner.replay_buffer)::SPEC, ActorCriticType::SPEC::PARAMETERS::ACTOR_BATCH_SIZE> actor_batch;
    lic::rl::algorithms::td3::ActorTrainingBuffers<ActorCriticType::SPEC> actor_training_buffers;
    lic::malloc(device, actor_batch);
    lic::malloc(device, actor_training_buffers);

    for(int step_i = 0; step_i < 15000; step_i++){
        lic::step(device, off_policy_runner, actor_critic.actor, rng);

        if(off_policy_runner.replay_buffer.full || off_policy_runner.replay_buffer.position > N_WARMUP_STEPS){
            if(step_i % 1000 == 0){
                lic::logging::text(device.logger, "step_i: ", step_i);
            }
            for(int critic_i = 0; critic_i < 2; critic_i++){
                lic::target_action_noise(device, actor_critic, critic_training_buffers.target_next_action_noise, rng);
                lic::gather_batch(device, off_policy_runner.replay_buffer, critic_batch, rng);
                lic::train_critic(device, actor_critic, critic_i == 0 ? actor_critic.critic_1 : actor_critic.critic_2, critic_batch, critic_training_buffers);
            }
            if(step_i % 2 == 0){
                lic::gather_batch(device, off_policy_runner.replay_buffer, actor_batch, rng);
                lic::train_actor(device, actor_critic, actor_batch, actor_training_buffers);
                lic::update_targets(device, actor_critic);
            }
        }
        if(step_i % 1000 == 0){
            DTYPE mean_return = lic::evaluate<AC_DEVICE, ENVIRONMENT, decltype(actor_critic.actor), typeof(rng), ENVIRONMENT_STEP_LIMIT, true>(device, env, actor_critic.actor, 1, rng);
            lic::logging::text(logger, "Mean return: ", mean_return);
            if(mean_return > -200000){
                return 0;
            }
        }
    }
    return -1;

}
