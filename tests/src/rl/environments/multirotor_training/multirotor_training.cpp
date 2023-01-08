#include <layer_in_c/operations/cpu.h>

#include <layer_in_c/rl/environments/environments.h>
#include <layer_in_c/nn_models/models.h>
#include <layer_in_c/rl/components/off_policy_runner/off_policy_runner.h>

#include <layer_in_c/nn_models/operations_cpu.h>
#include <layer_in_c/rl/environments/multirotor/operations_cpu.h>
#include <layer_in_c/rl/components/off_policy_runner/operations_generic.h>
#include <layer_in_c/rl/algorithms/td3/operations_cpu.h>


#include <layer_in_c/rl/utils/evaluation.h>

#include <gtest/gtest.h>


namespace lic = layer_in_c;
using DTYPE = float;

using DEVICE = lic::devices::DefaultCPU;
typedef lic::rl::environments::multirotor::Specification<DTYPE, lic::rl::environments::multirotor::StaticParameters> ENVIRONMENT_SPEC;
typedef lic::rl::environments::Multirotor<DEVICE, ENVIRONMENT_SPEC> ENVIRONMENT;
lic::rl::environments::multirotor::Parameters<DTYPE, 4> parameters = lic::rl::environments::multirotor::default_parameters<DTYPE>;

struct ActorStructureSpec{
    using T = DTYPE;
    static constexpr lic::index_t INPUT_DIM = ENVIRONMENT::OBSERVATION_DIM;
    static constexpr lic::index_t OUTPUT_DIM = ENVIRONMENT::ACTION_DIM;
    static constexpr int NUM_LAYERS = 3;
    static constexpr int HIDDEN_DIM = 64;
    static constexpr lic::nn::activation_functions::ActivationFunction HIDDEN_ACTIVATION_FUNCTION = lic::nn::activation_functions::RELU;
    static constexpr lic::nn::activation_functions::ActivationFunction OUTPUT_ACTIVATION_FUNCTION = lic::nn::activation_functions::TANH;
};

struct CriticStructureSpec{
    using T = DTYPE;
    static constexpr lic::index_t INPUT_DIM = ENVIRONMENT::OBSERVATION_DIM + ENVIRONMENT::ACTION_DIM;
    static constexpr lic::index_t OUTPUT_DIM = 1;
    static constexpr int NUM_LAYERS = 3;
    static constexpr int HIDDEN_DIM = 64;
    static constexpr lic::nn::activation_functions::ActivationFunction HIDDEN_ACTIVATION_FUNCTION = lic::nn::activation_functions::RELU;
    static constexpr lic::nn::activation_functions::ActivationFunction OUTPUT_ACTIVATION_FUNCTION = lic::nn::activation_functions::IDENTITY;
};

template <typename T>
struct TD3PendulumParameters: lic::rl::algorithms::td3::DefaultParameters<T>{
    constexpr static lic::index_t CRITIC_BATCH_SIZE = 100;
    constexpr static lic::index_t ACTOR_BATCH_SIZE = 100;
};

using NN_DEVICE = lic::devices::DefaultCPU;
using ACTOR_NETWORK_SPEC = lic::nn_models::mlp::AdamSpecification<NN_DEVICE, ActorStructureSpec, typename lic::nn::optimizers::adam::DefaultParametersTorch<DTYPE>>;
using ACTOR_NETWORK_TYPE = lic::nn_models::mlp::NeuralNetworkAdam<NN_DEVICE, ACTOR_NETWORK_SPEC>;

using ACTOR_TARGET_NETWORK_SPEC = lic::nn_models::mlp::InferenceSpecification<NN_DEVICE, ActorStructureSpec>;
using ACTOR_TARGET_NETWORK_TYPE = layer_in_c::nn_models::mlp::NeuralNetwork<NN_DEVICE , ACTOR_TARGET_NETWORK_SPEC>;

using CRITIC_NETWORK_SPEC = lic::nn_models::mlp::AdamSpecification<NN_DEVICE, CriticStructureSpec, typename lic::nn::optimizers::adam::DefaultParametersTorch<DTYPE>>;
using CRITIC_NETWORK_TYPE = layer_in_c::nn_models::mlp::NeuralNetworkAdam<NN_DEVICE, CRITIC_NETWORK_SPEC>;

using CRITIC_TARGET_NETWORK_SPEC = layer_in_c::nn_models::mlp::InferenceSpecification<NN_DEVICE, CriticStructureSpec>;
using CRITIC_TARGET_NETWORK_TYPE = layer_in_c::nn_models::mlp::NeuralNetwork<NN_DEVICE, CRITIC_TARGET_NETWORK_SPEC>;

using AC_DEVICE = lic::devices::DefaultCPU;
using TD3_SPEC = lic::rl::algorithms::td3::Specification<DTYPE, ENVIRONMENT, NN_DEVICE, ACTOR_NETWORK_TYPE, ACTOR_TARGET_NETWORK_TYPE, CRITIC_NETWORK_TYPE, CRITIC_TARGET_NETWORK_TYPE, TD3PendulumParameters<DTYPE>>;
using ActorCriticType = lic::rl::algorithms::td3::ActorCritic<AC_DEVICE, TD3_SPEC>;


constexpr lic::index_t REPLAY_BUFFER_CAP = 500000;
constexpr lic::index_t ENVIRONMENT_STEP_LIMIT = 200;
lic::rl::components::OffPolicyRunner<
        DEVICE,
        lic::rl::components::off_policy_runner::Spec<
                DTYPE,
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

TEST(LAYER_IN_C_RL_ENVIRONMENTS_MULTIROTOR, TEST_FULL_TRAINING) {
    std::mt19937 rng(2);
    lic::init(actor_critic, rng);
    parameters.init = lic::rl::environments::multirotor::simple_init_parameters<DTYPE, 4>;
    ENVIRONMENT env({parameters});
    off_policy_runner.env = env;

    for(int step_i = 0; step_i < 1000000; step_i++){
        if(step_i > REPLAY_BUFFER_CAP){
            std::cout << "warning: replay buffer is rolling over" << std::endl;
        }
        lic::step(off_policy_runner, actor_critic.actor, rng);
        if(off_policy_runner.replay_buffer.full || off_policy_runner.replay_buffer.position > N_WARMUP_STEPS){
            if(step_i % 1000 == 0){
                std::cout << "step_i: " << step_i << std::endl;
            }
            DTYPE critic_1_loss = lic::train_critic(actor_critic, actor_critic.critic_1, off_policy_runner.replay_buffer, rng);
            lic::train_critic(actor_critic, actor_critic.critic_2, off_policy_runner.replay_buffer, rng);
//            std::cout << "Critic 1 loss: " << critic_1_loss << std::endl;
            if(step_i % 2 == 0){
                lic::train_actor(actor_critic, off_policy_runner.replay_buffer, rng);
                lic::update_targets(actor_critic);
            }
        }
        if(step_i % 1000 == 0){
            DTYPE mean_return = lic::evaluate<DEVICE, ENVIRONMENT, decltype(actor_critic.actor), typeof(rng), ENVIRONMENT_STEP_LIMIT, true>(DEVICE(), env, actor_critic.actor, 1, rng);
            std::cout << "Mean return: " << mean_return << std::endl;
        }
    }
}
