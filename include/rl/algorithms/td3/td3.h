#ifndef LAYER_IN_C_RL_ALGORITHMS_TD3
#define LAYER_IN_C_RL_ALGORITHMS_TD3

#include "replay_buffer.h"
#include <random>
#include <nn/nn.h>
#include <utils/polyak.h>
using namespace layer_in_c;
using namespace layer_in_c::nn::layers;


template <typename T>
struct DefaultActorNetworkDefinition{
    static constexpr int LAYER_1_DIM = 50;
    static constexpr int LAYER_2_DIM = 50;
    static constexpr ActivationFunction LAYER_1_FN = ActivationFunction::RELU;
    static constexpr ActivationFunction LAYER_2_FN = ActivationFunction::RELU;
    typedef nn_models::DefaultAdamParameters<T> ADAM_PARAMETERS;
};

template <typename T>
struct DefaultCriticNetworkDefinition{
    static constexpr int LAYER_1_DIM = 50;
    static constexpr int LAYER_2_DIM = 50;
    static constexpr ActivationFunction LAYER_1_FN = ActivationFunction::RELU;
    static constexpr ActivationFunction LAYER_2_FN = ActivationFunction::RELU;
    typedef nn_models::DefaultAdamParameters<T> ADAM_PARAMETERS;
};

template <
    typename T,
    typename ENVIRONMENT,
    typename ACTOR_NETWORK_DEFINITION,
    typename CRITIC_NETWORK_DEFINITION,
    typename PARAMETERS
>
struct ActorCritic{
    typedef layer_in_c::nn_models::ThreeLayerNeuralNetworkTrainingAdam<T, ENVIRONMENT::STATE_DIM,
            ACTOR_NETWORK_DEFINITION::LAYER_1_DIM, ACTOR_NETWORK_DEFINITION::LAYER_1_FN,
            ACTOR_NETWORK_DEFINITION::LAYER_2_DIM, ACTOR_NETWORK_DEFINITION::LAYER_2_FN,
            ENVIRONMENT::ACTION_DIM, layer_in_c::nn::activation_functions::TANH, typename ACTOR_NETWORK_DEFINITION::ADAM_PARAMETERS> ACTOR_NETWORK_TYPE;
    typedef layer_in_c::nn_models::ThreeLayerNeuralNetworkInference<T, ENVIRONMENT::STATE_DIM,
            ACTOR_NETWORK_DEFINITION::LAYER_1_DIM, ACTOR_NETWORK_DEFINITION::LAYER_1_FN,
            ACTOR_NETWORK_DEFINITION::LAYER_2_DIM, ACTOR_NETWORK_DEFINITION::LAYER_2_FN,
            ENVIRONMENT::ACTION_DIM, layer_in_c::nn::activation_functions::TANH, typename ACTOR_NETWORK_DEFINITION::ADAM_PARAMETERS> ACTOR_TARGET_NETWORK_TYPE;

    static constexpr int CRITIC_INPUT_DIM = ENVIRONMENT::STATE_DIM + ENVIRONMENT::ACTION_DIM;
    typedef layer_in_c::nn_models::ThreeLayerNeuralNetworkTrainingAdam<T, CRITIC_INPUT_DIM,
            CRITIC_NETWORK_DEFINITION::LAYER_1_DIM, CRITIC_NETWORK_DEFINITION::LAYER_1_FN,
            CRITIC_NETWORK_DEFINITION::LAYER_2_DIM, CRITIC_NETWORK_DEFINITION::LAYER_2_FN,
            1, layer_in_c::nn::activation_functions::LINEAR, typename CRITIC_NETWORK_DEFINITION::ADAM_PARAMETERS> CRITIC_NETWORK_TYPE;
    typedef layer_in_c::nn_models::ThreeLayerNeuralNetworkInferenceBackward<T, CRITIC_INPUT_DIM,
            CRITIC_NETWORK_DEFINITION::LAYER_1_DIM, CRITIC_NETWORK_DEFINITION::LAYER_1_FN,
            CRITIC_NETWORK_DEFINITION::LAYER_2_DIM, CRITIC_NETWORK_DEFINITION::LAYER_2_FN,
            1, layer_in_c::nn::activation_functions::LINEAR, typename CRITIC_NETWORK_DEFINITION::ADAM_PARAMETERS> CRITIC_TARGET_1_NETWORK_TYPE;
    typedef layer_in_c::nn_models::ThreeLayerNeuralNetworkInference<T, CRITIC_INPUT_DIM,
            CRITIC_NETWORK_DEFINITION::LAYER_1_DIM, CRITIC_NETWORK_DEFINITION::LAYER_1_FN,
            CRITIC_NETWORK_DEFINITION::LAYER_2_DIM, CRITIC_NETWORK_DEFINITION::LAYER_2_FN,
            1, layer_in_c::nn::activation_functions::LINEAR, typename CRITIC_NETWORK_DEFINITION::ADAM_PARAMETERS> CRITIC_TARGET_2_NETWORK_TYPE;
    ACTOR_NETWORK_TYPE actor;
    ACTOR_TARGET_NETWORK_TYPE actor_target;

    CRITIC_NETWORK_TYPE critic_1;
    CRITIC_NETWORK_TYPE critic_2;
    CRITIC_TARGET_1_NETWORK_TYPE critic_target_1;
    CRITIC_TARGET_2_NETWORK_TYPE critic_target_2;
};

template<typename T, int INPUT_DIM, int OUTPUT_DIM, nn::activation_functions::ActivationFunction FN>
void update_target_layer(Layer<T, INPUT_DIM, OUTPUT_DIM, FN>& target, const Layer<T, INPUT_DIM, OUTPUT_DIM, FN>& source, T polyak) {
    utils::polyak_update_matrix<T, OUTPUT_DIM, INPUT_DIM>(target.weights, source.weights, polyak);
    utils::polyak_update<T, OUTPUT_DIM>(target.biases, source.biases, polyak);
}
template<typename T, typename TARGET_NETWORK_TYPE, typename SOURCE_NETWORK_TYPE>
void update_target_network(TARGET_NETWORK_TYPE& target, const SOURCE_NETWORK_TYPE& source, T polyak) {
    update_target_layer(target.layer_1, source.layer_1, polyak);
    update_target_layer(target.layer_2, source.layer_2, polyak);
    update_target_layer(target.output_layer, source.output_layer, polyak);
}

template <typename T, typename ENVIRONMENT, typename ACTOR_NETWORK_DEFINITION, typename CRITIC_NETWORK_DEFINITION, typename PARAMETERS>
void update_targets(ActorCritic<T, ENVIRONMENT, ACTOR_NETWORK_DEFINITION, CRITIC_NETWORK_DEFINITION, PARAMETERS>& actor_critic) {
    update_target_network(actor_critic.actor_target, actor_critic.actor, PARAMETERS::ACTOR_POLYAK);
    update_target_network(actor_critic.critic_target_1, actor_critic.critic_1, PARAMETERS::CRITIC_POLYAK);
    update_target_network(actor_critic.critic_target_2, actor_critic.critic_2, PARAMETERS::CRITIC_POLYAK);

}


template <typename T, typename ENVIRONMENT, typename ACTOR_NETWORK_DEFINITION, typename CRITIC_NETWORK_DEFINITION, typename PARAMETERS>
void init(ActorCritic<T, ENVIRONMENT, ACTOR_NETWORK_DEFINITION, CRITIC_NETWORK_DEFINITION, PARAMETERS>& actor_critic, std::mt19937& rng){
    layer_in_c::nn_models::init_weights(actor_critic.actor, rng);
    layer_in_c::nn_models::init_weights(actor_critic.critic_1, rng);
    layer_in_c::nn_models::init_weights(actor_critic.critic_2, rng);
    // Target networks still need to be initialised because they could be none which could destroy the use of the polyak update for assignment
    layer_in_c::nn_models::init_weights(actor_critic.actor_target, rng);
    layer_in_c::nn_models::init_weights(actor_critic.critic_target_1, rng);
    layer_in_c::nn_models::init_weights(actor_critic.critic_target_2, rng);
    update_target_network(actor_critic.actor_target, actor_critic.actor, (T)0);
    update_target_network(actor_critic.critic_target_1, actor_critic.critic_1, (T)0);
    update_target_network(actor_critic.critic_target_2, actor_critic.critic_2, (T)0);
}


template <typename T, typename ENVIRONMENT, typename ACTOR_NETWORK_DEFINITION, typename CRITIC_NETWORK_DEFINITION, typename PARAMETERS, typename CRITIC_TYPE, int CAPACITY, typename RNG>
void train_critic(ActorCritic<T, ENVIRONMENT, ACTOR_NETWORK_DEFINITION, CRITIC_NETWORK_DEFINITION, PARAMETERS>& actor_critic, CRITIC_TYPE& critic, ReplayBuffer<T, ENVIRONMENT::OBSERVATION_DIM, ENVIRONMENT::ACTION_DIM, CAPACITY>& replay_buffer, RNG& rng) {
    T loss = 0;
    zero_gradient(critic);
    std::uniform_int_distribution<uint32_t> sample_distribution(0, (replay_buffer.full ? CAPACITY : replay_buffer.position) - 1);
    for (int sample_i=0; sample_i < PARAMETERS::CRITIC_BATCH_SIZE; sample_i++){
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
        T target_action_value[1] = {replay_buffer.rewards[sample_index] + PARAMETERS::GAMMA * min_next_state_action_value * (1 - replay_buffer.terminated[sample_index])};

        forward_backward_mse(critic, state_action_value_input, target_action_value);
        loss += nn::loss_functions::mse<T, 1>(critic.output_layer.output, target_action_value);
    }
    loss /= PARAMETERS::CRITIC_BATCH_SIZE;
    std::cout << "Critic loss: " << loss << std::endl;
    update(critic);
}

template <typename T, typename ENVIRONMENT, typename ACTOR_NETWORK_DEFINITION, typename CRITIC_NETWORK_DEFINITION, typename PARAMETERS, int CAPACITY, typename RNG>
void train_actor(ActorCritic<T, ENVIRONMENT, ACTOR_NETWORK_DEFINITION, CRITIC_NETWORK_DEFINITION, PARAMETERS>& actor_critic, ReplayBuffer<T, ENVIRONMENT::OBSERVATION_DIM, ENVIRONMENT::ACTION_DIM, CAPACITY>& replay_buffer, RNG& rng) {
    T loss = 0;
    zero_gradient(actor_critic.actor);
    std::uniform_int_distribution<uint32_t> sample_distribution(0, (replay_buffer.full ? CAPACITY : replay_buffer.position) - 1);
    for (int sample_i=0; sample_i < PARAMETERS::ACTOR_BATCH_SIZE; sample_i++){
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
    loss /= PARAMETERS::ACTOR_BATCH_SIZE;
    std::cout << "Critic loss: " << loss << std::endl;
    update(actor_critic.actor);
}


#endif