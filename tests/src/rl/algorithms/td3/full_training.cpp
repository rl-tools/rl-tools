#include <layer_in_c/logging/operations_cpu_wandb.h>
#include <layer_in_c/operations/cpu.h>

#include <layer_in_c/nn/operations_generic.h>
#include <layer_in_c/rl/environments/operations_generic.h>
#include <layer_in_c/nn_models/operations_generic.h>
#include <layer_in_c/rl/operations_generic.h>


#include <layer_in_c/rl/utils/evaluation.h>

#include <gtest/gtest.h>


#ifdef LAYER_IN_C_TEST_RL_ALGORITHMS_TD3_FULL_TRAINING_EVALUATE_VISUALLY
#include <layer_in_c/rl/environments/pendulum/ui.h>
#include <layer_in_c/rl/utils/evaluation_visual.h>
#endif


#ifdef LAYER_IN_C_TEST_RL_ALGORITHMS_TD3_FULL_TRAINING_OUTPUT_PLOTS
#include "plot_policy_and_value_function.h"
#endif


namespace lic = layer_in_c;
using DTYPE = float;

using DEVICE = lic::devices::DefaultCPU;
typedef lic::rl::environments::pendulum::Specification<DTYPE, lic::rl::environments::pendulum::DefaultParameters<DTYPE>> PENDULUM_SPEC;
typedef lic::rl::environments::Pendulum<DEVICE, PENDULUM_SPEC> ENVIRONMENT;
#ifdef LAYER_IN_C_TEST_RL_ALGORITHMS_TD3_FULL_TRAINING_EVALUATE_VISUALLY
typedef lic::rl::environments::pendulum::UI<DTYPE> UI;
#endif
ENVIRONMENT env;

struct ActorStructureSpec{
    using T = DTYPE;
    static constexpr typename DEVICE::index_t INPUT_DIM = ENVIRONMENT::OBSERVATION_DIM;
    static constexpr typename DEVICE::index_t OUTPUT_DIM = ENVIRONMENT::ACTION_DIM;
    static constexpr int NUM_LAYERS = 3;
    static constexpr int HIDDEN_DIM = 64;
    static constexpr lic::nn::activation_functions::ActivationFunction HIDDEN_ACTIVATION_FUNCTION = lic::nn::activation_functions::RELU;
    static constexpr lic::nn::activation_functions::ActivationFunction OUTPUT_ACTIVATION_FUNCTION = lic::nn::activation_functions::TANH;
};

struct CriticStructureSpec{
    using T = DTYPE;
    static constexpr typename DEVICE::index_t INPUT_DIM = ENVIRONMENT::OBSERVATION_DIM + ENVIRONMENT::ACTION_DIM;
    static constexpr typename DEVICE::index_t OUTPUT_DIM = 1;
    static constexpr int NUM_LAYERS = 3;
    static constexpr int HIDDEN_DIM = 64;
    static constexpr lic::nn::activation_functions::ActivationFunction HIDDEN_ACTIVATION_FUNCTION = lic::nn::activation_functions::RELU;
    static constexpr lic::nn::activation_functions::ActivationFunction OUTPUT_ACTIVATION_FUNCTION = lic::nn::activation_functions::IDENTITY;
};

struct AC_DEVICE_SPEC: lic::devices::DefaultCPUSpecification {
    using LOGGING = lic::devices::logging::CPU_WANDB;
};
using AC_DEVICE = lic::devices::CPU<AC_DEVICE_SPEC>;
template <typename T>
struct TD3PendulumParameters: lic::rl::algorithms::td3::DefaultParameters<AC_DEVICE, T>{
    constexpr static typename DEVICE::index_t CRITIC_BATCH_SIZE = 100;
    constexpr static typename DEVICE::index_t ACTOR_BATCH_SIZE = 100;
};

//using NN_DEVICE = lic::devices::DefaultCPU;
using NN_DEVICE = AC_DEVICE;
using ACTOR_NETWORK_SPEC = lic::nn_models::mlp::AdamSpecification<NN_DEVICE, ActorStructureSpec, typename lic::nn::optimizers::adam::DefaultParametersTorch<DTYPE>>;
using ACTOR_NETWORK_TYPE = lic::nn_models::mlp::NeuralNetworkAdam<NN_DEVICE, ACTOR_NETWORK_SPEC>;

using ACTOR_TARGET_NETWORK_SPEC = lic::nn_models::mlp::InferenceSpecification<NN_DEVICE, ActorStructureSpec>;
using ACTOR_TARGET_NETWORK_TYPE = layer_in_c::nn_models::mlp::NeuralNetwork<NN_DEVICE , ACTOR_TARGET_NETWORK_SPEC>;

using CRITIC_NETWORK_SPEC = lic::nn_models::mlp::AdamSpecification<NN_DEVICE, CriticStructureSpec, typename lic::nn::optimizers::adam::DefaultParametersTorch<DTYPE>>;
using CRITIC_NETWORK_TYPE = layer_in_c::nn_models::mlp::NeuralNetworkAdam<NN_DEVICE, CRITIC_NETWORK_SPEC>;

using CRITIC_TARGET_NETWORK_SPEC = layer_in_c::nn_models::mlp::InferenceSpecification<NN_DEVICE, CriticStructureSpec>;
using CRITIC_TARGET_NETWORK_TYPE = layer_in_c::nn_models::mlp::NeuralNetwork<NN_DEVICE, CRITIC_TARGET_NETWORK_SPEC>;

using TD3_SPEC = lic::rl::algorithms::td3::Specification<DTYPE, ENVIRONMENT, NN_DEVICE, ACTOR_NETWORK_TYPE, ACTOR_TARGET_NETWORK_TYPE, CRITIC_NETWORK_TYPE, CRITIC_TARGET_NETWORK_TYPE, TD3PendulumParameters<DTYPE>>;
using ActorCriticType = lic::rl::algorithms::td3::ActorCritic<AC_DEVICE, TD3_SPEC>;


constexpr typename DEVICE::index_t REPLAY_BUFFER_CAP = 500000;
constexpr typename DEVICE::index_t ENVIRONMENT_STEP_LIMIT = 200;
AC_DEVICE::SPEC::LOGGING logger;
AC_DEVICE ac_dev(logger);
//NN_DEVICE::SPEC::LOGGING nn_logger;
NN_DEVICE nn_dev(logger);
lic::rl::components::OffPolicyRunner<
    AC_DEVICE,
    lic::rl::components::off_policy_runner::Specification<
        AC_DEVICE,
        DTYPE,
        ENVIRONMENT,
        REPLAY_BUFFER_CAP,
        ENVIRONMENT_STEP_LIMIT,
        lic::rl::components::off_policy_runner::DefaultParameters<DTYPE>
    >
> off_policy_runner(ac_dev);
ActorCriticType actor_critic(ac_dev, nn_dev);
const DTYPE STATE_TOLERANCE = 0.00001;
constexpr int N_WARMUP_STEPS = ActorCriticType::SPEC::PARAMETERS::ACTOR_BATCH_SIZE;
static_assert(ActorCriticType::SPEC::PARAMETERS::ACTOR_BATCH_SIZE == ActorCriticType::SPEC::PARAMETERS::CRITIC_BATCH_SIZE);

TEST(LAYER_IN_C_RL_ALGORITHMS_TD3_FULL_TRAINING, TEST_FULL_TRAINING) {
#ifdef LAYER_IN_C_TEST_RL_ALGORITHMS_TD3_FULL_TRAINING_EVALUATE_VISUALLY
    UI ui;
#endif
    std::mt19937 rng(4);
    lic::init(actor_critic, rng);

    for(int step_i = 0; step_i < 15000; step_i++){
#ifdef LAYER_IN_C_TEST_RL_ALGORITHMS_TD3_FULL_TRAINING_OUTPUT_PLOTS
        if(step_i % 20 == 0){
            plot_policy_and_value_function<DTYPE, ENVIRONMENT, decltype(actor_critic.actor), decltype(actor_critic.critic_1)>(actor_critic.actor, actor_critic.critic_1, std::string("full_training"), step_i);
        }
#endif
        if(step_i > REPLAY_BUFFER_CAP){
            std::cout << "warning: replay buffer is rolling over" << std::endl;
        }
        lic::step(off_policy_runner, actor_critic.actor, rng);
#ifdef LAYER_IN_C_TEST_RL_ALGORITHMS_TD3_FULL_TRAINING_EVALUATE_VISUALLY
        lic::set_state(ui, off_policy_runner.state);
#endif

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
            DTYPE mean_return = lic::evaluate<AC_DEVICE, ENVIRONMENT, decltype(actor_critic.actor), typeof(rng), ENVIRONMENT_STEP_LIMIT, true>(ac_dev, env, actor_critic.actor, 1, rng);
            std::cout << "Mean return: " << mean_return << std::endl;
//            if(step_i >= 6000){
//                ASSERT_GT(mean_return, -1000);
//            }
//            if(step_i >= 14000){
//                ASSERT_GT(mean_return, -400);
//            }

//#ifdef LAYER_IN_C_TEST_RL_ALGORITHMS_TD3_FULL_TRAINING_OUTPUT_PLOTS
//            plot_policy_and_value_function<DTYPE, ENVIRONMENT, ActorCriticType::ACTOR_NETWORK_TYPE, ActorCriticType::CRITIC_NETWORK_TYPE>(actor_critic.actor, actor_critic.critic_1, std::string("full_training"), step_i);
//#endif
#ifdef LAYER_IN_C_TEST_RL_ALGORITHMS_TD3_FULL_TRAINING_EVALUATE_VISUALLY
//            for(int evaluation_i = 0; evaluation_i < 10; evaluation_i++){
//                ENVIRONMENT::State initial_state;
//                lic::sample_initial_state(env, initial_state, rng);
//                lic::evaluate_visual<ENVIRONMENT, UI, ActorCriticType::ACTOR_NETWORK_TYPE, ENVIRONMENT_STEP_LIMIT, 5>(env, ui, actor_critic.actor, initial_state);
//            }
#endif
        }
    }
}
