#include <gtest/gtest.h>

#include <nn_models/models.h>

#include <rl/environments/pendulum.h>
#include <rl/algorithms/td3/off_policy_runner.h>
#include <rl/algorithms/td3/td3.h>
#define DTYPE float
const DTYPE STATE_TOLERANCE = 0.00001;

typedef Pendulum<DTYPE, DefaultPendulumParams<DTYPE>> ENVIRONMENT;

template <typename T>
struct TD3Params{
    static constexpr T GAMMA = 0.99;
    static constexpr uint32_t CRITIC_BATCH_SIZE = 32;
};

using namespace layer_in_c;

TEST(LAYER_IN_C_RL_ALGORITHMS_TD3_TEST, TEST_0) {
    layer_in_c::nn_models::ThreeLayerNeuralNetworkTrainingAdam<DTYPE, ENVIRONMENT::STATE_DIM,
            50, layer_in_c::nn::activation_functions::RELU,
            50, layer_in_c::nn::activation_functions::RELU,
            ENVIRONMENT::ACTION_DIM, layer_in_c::nn::activation_functions::TANH, nn_models::DefaultAdamParameters<DTYPE>> policy;
    std::mt19937 rng(0);
    layer_in_c::nn_models::init_weights(policy, rng);
    OffPolicyRunner<DTYPE, ENVIRONMENT, typeof(policy), 1000, 100> off_policy_runner;
    for(int step_i = 0; step_i < 10000; step_i++){
        step(off_policy_runner, policy, rng);
    }
    std::cout << "hello" << std::endl;
    ActorCritic<DTYPE, ENVIRONMENT, TD3Params<DTYPE>> actor_critic;
    init(actor_critic, rng);
    train_critic(actor_critic, actor_critic.critic_1, off_policy_runner.replay_buffer, rng);
    train_critic(actor_critic, actor_critic.critic_2, off_policy_runner.replay_buffer, rng);
}

