#include "replay_buffer.h"
#include <random>
#include <nn/nn.h>
using namespace layer_in_c;


template <typename T, typename ENVIRONMENT, typename PARAMS>
struct ActorCritic{
    typedef layer_in_c::nn_models::ThreeLayerNeuralNetworkTrainingAdam<T, ENVIRONMENT::STATE_DIM,
            50, layer_in_c::nn::activation_functions::RELU,
            50, layer_in_c::nn::activation_functions::RELU,
            ENVIRONMENT::ACTION_DIM, layer_in_c::nn::activation_functions::TANH, DefaultAdamParameters<T>> ACTOR_NETWORK_TYPE;
    typedef layer_in_c::nn_models::ThreeLayerNeuralNetworkTrainingAdam<T, ENVIRONMENT::STATE_DIM,
            50, layer_in_c::nn::activation_functions::RELU,
            50, layer_in_c::nn::activation_functions::RELU,
            ENVIRONMENT::ACTION_DIM, layer_in_c::nn::activation_functions::LINEAR, DefaultAdamParameters<T>> CRITIC_NETWORK_TYPE;
    ACTOR_NETWORK_TYPE actor;
    ACTOR_NETWORK_TYPE actor_target;

    CRITIC_NETWORK_TYPE critic_1;
    CRITIC_NETWORK_TYPE critic_2;
    CRITIC_NETWORK_TYPE critic_target_1;
    CRITIC_NETWORK_TYPE critic_target_2;
};

template <typename T, typename ENVIRONMENT, typename PARAMS, typename CRITIC_TYPE, int CAPACITY, int BATCH_SIZE, typename RNG>
void train_critic(ActorCritic<T, ENVIRONMENT, PARAMS> actor_critic, CRITIC_TYPE critic, ReplayBuffer<T, ENVIRONMENT::OBSERVATION_DIM, ENVIRONMENT::ACTION_DIM, CAPACITY> replay_buffer, RNG& rng) {
    T loss = 0;
    zero_gradient(critic);
    int sample_distribution = std::uniform_int_distribution<uint32_t>(0, (replay_buffer.full ? CAPACITY : replay_buffer.position) - 1)(rng);
    for (int sample_i=0; sample_i < BATCH_SIZE; sample_i++){
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
        T target_action_value[1] = {replay_buffer.rewards[sample_index] + PARAMS::gamma * min_next_state_action_value * (1 - replay_buffer.terminated[sample_index])};

        forward_backward_mse(critic, state_action_value_input, target_action_value);
        loss += nn::loss_functions::mse<T, 1>(critic.output_layer.output, target_action_value);
    }
    loss /= BATCH_SIZE;
    std::cout << "Critic loss: " << loss << std::endl;
    update(critic);
}