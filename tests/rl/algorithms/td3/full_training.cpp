#include <gtest/gtest.h>

#include <layer_in_c/nn_models/models.h>
#include <layer_in_c/rl/algorithms/td3/td3.h>
#include <layer_in_c/rl/algorithms/td3/off_policy_runner.h>
#include <layer_in_c/rl/environments/pendulum.h>
#include <layer_in_c/utils/rng_std.h>

namespace lic = layer_in_c;
#define DTYPE float

typedef lic::rl::environments::pendulum::Spec<DTYPE, lic::rl::environments::pendulum::DefaultParameters<DTYPE>> PENDULUM_SPEC;
typedef lic::rl::environments::pendulum::Pendulum<lic::devices::Generic, PENDULUM_SPEC> ENVIRONMENT;
ENVIRONMENT env;

template <typename T>
using TestActorNetworkDefinition = lic::rl::algorithms::td3::ActorNetworkSpecification<T, 64, 64, lic::nn::activation_functions::ActivationFunction::RELU, lic::nn::optimizers::adam::DefaultParametersTorch<DTYPE>>;

template <typename T>
using TestCriticNetworkDefinition = lic::rl::algorithms::td3::CriticNetworkSpecification<T, 64, 64, lic::nn::activation_functions::ActivationFunction::RELU, lic::nn::optimizers::adam::DefaultParametersTorch<DTYPE>>;

const DTYPE STATE_TOLERANCE = 0.00001;
#define N_WARMUP_STEPS 100
TEST(LAYER_IN_C_RL_ALGORITHMS_TD3_TEST, TEST_FULL_TRAINING) {
    typedef lic::rl::algorithms::td3::ActorCritic<lic::devices::Generic, lic::rl::algorithms::td3::ActorCriticSpecification<DTYPE, ENVIRONMENT, TestActorNetworkDefinition<DTYPE>, TestCriticNetworkDefinition<DTYPE>, lic::rl::algorithms::td3::DefaultTD3Parameters<DTYPE>>> ActorCriticType;
    lic::rl::algorithms::td3::OffPolicyRunner<DTYPE, ENVIRONMENT, lic::rl::algorithms::td3::DefaultOffPolicyRunnerParameters<DTYPE, 50000, 200>> off_policy_runner;
    ActorCriticType actor_critic;
    std::mt19937 rng(0);
    lic::init<lic::devices::Generic, ActorCriticType::SPEC, layer_in_c::utils::random::stdlib::uniform<DTYPE, typeof(rng)>, typeof(rng)>(actor_critic, rng);

    for(int step_i = 0; step_i < 10000000; step_i++){
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
    }
}
