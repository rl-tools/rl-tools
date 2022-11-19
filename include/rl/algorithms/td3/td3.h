#ifndef LAYER_IN_C_RL_ALGORITHMS_TD3
#define LAYER_IN_C_RL_ALGORITHMS_TD3

#include "replay_buffer.h"
#include <random>
#include <nn/nn.h>
using namespace layer_in_c;


template <typename T, typename ENVIRONMENT, typename PARAMS>
struct ActorCritic{
    typedef layer_in_c::nn_models::ThreeLayerNeuralNetworkTrainingAdam<T, ENVIRONMENT::STATE_DIM,
            50, layer_in_c::nn::activation_functions::RELU,
            50, layer_in_c::nn::activation_functions::RELU,
            ENVIRONMENT::ACTION_DIM, layer_in_c::nn::activation_functions::TANH, nn_models::DefaultAdamParameters<T>> ACTOR_NETWORK_TYPE;
    typedef layer_in_c::nn_models::ThreeLayerNeuralNetworkTrainingAdam<T, ENVIRONMENT::STATE_DIM,
            50, layer_in_c::nn::activation_functions::RELU,
            50, layer_in_c::nn::activation_functions::RELU,
            ENVIRONMENT::ACTION_DIM, layer_in_c::nn::activation_functions::LINEAR, nn_models::DefaultAdamParameters<T>> CRITIC_NETWORK_TYPE;
    ACTOR_NETWORK_TYPE actor;
    ACTOR_NETWORK_TYPE actor_target;

    CRITIC_NETWORK_TYPE critic_1;
    CRITIC_NETWORK_TYPE critic_2;
    CRITIC_NETWORK_TYPE critic_target_1;
    CRITIC_NETWORK_TYPE critic_target_2;
};
template <typename T, typename ENVIRONMENT, typename PARAMS>
void init(ActorCritic<T, ENVIRONMENT, PARAMS>& actor_critic, std::mt19937& rng){
    layer_in_c::nn_models::init_weights(actor_critic.actor, rng);
    layer_in_c::nn_models::init_weights(actor_critic.actor_target, rng);
    layer_in_c::nn_models::init_weights(actor_critic.critic_1, rng);
    layer_in_c::nn_models::init_weights(actor_critic.critic_2, rng);
    layer_in_c::nn_models::init_weights(actor_critic.critic_target_1, rng);
    layer_in_c::nn_models::init_weights(actor_critic.critic_target_2, rng);
}


template <typename T, typename ENVIRONMENT, typename PARAMS, typename CRITIC_TYPE, int CAPACITY, typename RNG>
void train_critic(ActorCritic<T, ENVIRONMENT, PARAMS>& actor_critic, CRITIC_TYPE& critic, ReplayBuffer<T, ENVIRONMENT::OBSERVATION_DIM, ENVIRONMENT::ACTION_DIM, CAPACITY>& replay_buffer, RNG& rng) {
    T loss = 0;
    zero_gradient(critic);
    std::uniform_int_distribution<uint32_t> sample_distribution(0, (replay_buffer.full ? CAPACITY : replay_buffer.position) - 1);
    for (int sample_i=0; sample_i < PARAMS::CRITIC_BATCH_SIZE; sample_i++){
        uint32_t sample_index = sample_distribution(rng);
        T next_state_action_value_input[ENVIRONMENT::OBSERVATION_DIM + ENVIRONMENT::ACTION_DIM];
        memcpy(next_state_action_value_input, replay_buffer.next_observations[sample_index], sizeof(T) * ENVIRONMENT::OBSERVATION_DIM); // setting the first part with next observations
        evaluate(actor_critic.actor_target, next_state_action_value_input, &next_state_action_value_input[ENVIRONMENT::OBSERVATION_DIM]); // setting the second part with next actions
        T min_next_state_action_value = std::min(
            evaluate(actor_critic.critic_target_1, next_state_action_value_input),
            evaluate(actor_critic.critic_target_2, next_state_action_value_input)
        );
        T state_action_value_input[ENVIRONMENT::OBSERVATION_DIM + ENVIRONMENT::ACTION_DIM];
        memcpy(state_action_value_input, replay_buffer.observations[sample_index], sizeof(T) * ENVIRONMENT::OBSERVATION_DIM); // setting the first part with the current observation
        memcpy(&state_action_value_input[ENVIRONMENT::OBSERVATION_DIM], replay_buffer.actions[sample_index], sizeof(T) * ENVIRONMENT::ACTION_DIM); // setting the first part with the current action
//        standardise<T,  OBSERVATION_DIM>(X_train[batch_i * batch_size + sample_i].data(), X_mean.data(), X_std.data(), input);
//        standardise<T, ACTION_DIM>(Y_train[batch_i * batch_size + sample_i].data(), Y_mean.data(), Y_std.data(), output);
        T target_action_value[1] = {replay_buffer.rewards[sample_index] + PARAMS::GAMMA * min_next_state_action_value * (1 - replay_buffer.terminated[sample_index])};

        forward_backward_mse(critic, state_action_value_input, target_action_value);
        loss += nn::loss_functions::mse<T, 1>(critic.output_layer.output, target_action_value);
    }
    loss /= PARAMS::CRITIC_BATCH_SIZE;
    std::cout << "Critic loss: " << loss << std::endl;
    update(critic);
}

template <typename T, typename ENVIRONMENT, typename PARAMS, int CAPACITY, typename RNG>
void train_actor(ActorCritic<T, ENVIRONMENT, PARAMS>& actor_critic, ReplayBuffer<T, ENVIRONMENT::OBSERVATION_DIM, ENVIRONMENT::ACTION_DIM, CAPACITY>& replay_buffer, RNG& rng) {
    T loss = 0;
    zero_gradient(actor_critic.actor);
    std::uniform_int_distribution<uint32_t> sample_distribution(0, (replay_buffer.full ? CAPACITY : replay_buffer.position) - 1);
    for (int sample_i=0; sample_i < PARAMS::ACTOR_BATCH_SIZE; sample_i++){
        uint32_t sample_index = sample_distribution(rng);
        T state_action_value_input[ENVIRONMENT::OBSERVATION_DIM + ENVIRONMENT::ACTION_DIM];
        memcpy(state_action_value_input, replay_buffer.observations[sample_index], sizeof(T) * ENVIRONMENT::OBSERVATION_DIM); // setting the first part with next observations

        evaluate(actor_critic.actor_target, state_action_value_input, &state_action_value_input[ENVIRONMENT::OBSERVATION_DIM]); // setting the second part with next actions
        forward(actor_critic.critic_target_1, state_action_value_input);
        T d_output[1] = {-1};
        T d_critic_input[ENVIRONMENT::OBSERVATION_DIM + ENVIRONMENT::ACTION_DIM];
        backward(actor_critic.critic_target_1, state_action_value_input, d_output, d_critic_input);
        T d_actor_input[ENVIRONMENT::OBSERVATION_DIM];
        backward(actor_critic.actor, state_action_value_input, &d_critic_input[ENVIRONMENT::OBSERVATION_DIM], d_actor_input);
    }
    loss /= PARAMS::ACTOR_BATCH_SIZE;
    std::cout << "Critic loss: " << loss << std::endl;
    update(critic);

}
#endif