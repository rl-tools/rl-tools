#include <layer_in_c/nn_models/models.h>

#include <layer_in_c/rl/environments/environments.h>
#include <layer_in_c/rl/components/off_policy_runner/off_policy_runner.h>
#include <layer_in_c/rl/algorithms/td3/operations_generic.h>

#include <layer_in_c/rl/algorithms/td3/operations_cpu.h>
#include <layer_in_c/rl/environments/pendulum/operations_cpu.h>
#include <layer_in_c/rl/components/off_policy_runner/operations_cpu.h>
#include <layer_in_c/utils/rng_std.h>

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
#define DTYPE double

typedef lic::rl::environments::pendulum::Spec<DTYPE, lic::rl::environments::pendulum::DefaultParameters<DTYPE>> PENDULUM_SPEC;
typedef lic::rl::environments::Pendulum<lic::devices::CPU, PENDULUM_SPEC> ENVIRONMENT;
#ifdef LAYER_IN_C_TEST_RL_ALGORITHMS_TD3_FULL_TRAINING_EVALUATE_VISUALLY
typedef lic::rl::environments::pendulum::UI<DTYPE> UI;
#endif
ENVIRONMENT env;

template <typename T>
using TestActorNetworkDefinition = lic::rl::algorithms::td3::ActorNetworkSpecification<T, 64, 64, lic::nn::activation_functions::ActivationFunction::RELU, lic::nn::optimizers::adam::DefaultParametersTorch<DTYPE>>;

template <typename T>
using TestCriticNetworkDefinition = lic::rl::algorithms::td3::CriticNetworkSpecification<T, 64, 64, lic::nn::activation_functions::ActivationFunction::RELU, lic::nn::optimizers::adam::DefaultParametersTorch<DTYPE>>;

template <typename T>
struct TD3PendulumParameters: lic::rl::algorithms::td3::DefaultTD3Parameters<T>{
    constexpr static size_t CRITIC_BATCH_SIZE = 100;
    constexpr static size_t ACTOR_BATCH_SIZE = 100;
};

constexpr size_t REPLAY_BUFFER_CAP = 500000;
constexpr size_t ENVIRONMENT_STEP_LIMIT = 200;
typedef lic::rl::algorithms::td3::ActorCritic<lic::devices::CPU, lic::rl::algorithms::td3::ActorCriticSpecification<lic::devices::Generic, DTYPE, ENVIRONMENT, TestActorNetworkDefinition<DTYPE>, TestCriticNetworkDefinition<DTYPE>, TD3PendulumParameters<DTYPE>>> ActorCriticType;
lic::rl::components::OffPolicyRunner<
    lic::devices::CPU,
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

TEST(LAYER_IN_C_RL_ALGORITHMS_TD3_FULL_TRAINING, TEST_FULL_TRAINING) {
#ifdef LAYER_IN_C_TEST_RL_ALGORITHMS_TD3_FULL_TRAINING_EVALUATE_VISUALLY
    UI ui;
#endif
    std::mt19937 rng(2);
    lic::init<decltype(actor_critic)::DEVICE, ActorCriticType::SPEC, layer_in_c::utils::random::stdlib::uniform<DTYPE, typeof(rng)>, typeof(rng)>(actor_critic, rng);

    for(int step_i = 0; step_i < 15000; step_i++){
#ifdef LAYER_IN_C_TEST_RL_ALGORITHMS_TD3_FULL_TRAINING_OUTPUT_PLOTS
        if(step_i % 20 == 0){
            plot_policy_and_value_function<DTYPE, ENVIRONMENT, ActorCriticType::ACTOR_NETWORK_TYPE, ActorCriticType::CRITIC_NETWORK_TYPE>(actor_critic.actor, actor_critic.critic_1, std::string("full_training"), step_i);
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
            DTYPE mean_return = lic::evaluate<ENVIRONMENT, ActorCriticType::ACTOR_NETWORK_TYPE, typeof(rng), ENVIRONMENT_STEP_LIMIT, true>(ENVIRONMENT(), actor_critic.actor, 1, rng);
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
//                lic::evaluate_visual<ENVIRONMENT, UI, ActorCriticType::ACTOR_NETWORK_TYPE, ENVIRONMENT_STEP_LIMIT, 5>(ENVIRONMENT(), ui, actor_critic.actor, initial_state);
//            }
#endif
        }
    }
}
