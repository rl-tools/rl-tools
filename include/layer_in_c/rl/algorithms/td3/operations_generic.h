#ifndef LAYER_IN_C_RL_ALGORITHMS_TD3_OPERATIONS_GENERIC_H
#define LAYER_IN_C_RL_ALGORITHMS_TD3_OPERATIONS_GENERIC_H

#include "td3.h"

#include <layer_in_c/rl/components/replay_buffer/replay_buffer.h>
#include <layer_in_c/nn/nn.h>
#include <layer_in_c/nn_models/operations_generic.h>
#include <layer_in_c/utils/generic/polyak.h>
#include <layer_in_c/math/operations_generic.h>
#include <layer_in_c/utils/generic/memcpy.h>

namespace layer_in_c{
    template <typename DEVICE, typename SPEC>
    FUNCTION_PLACEMENT void malloc(DEVICE& device, rl::algorithms::td3::ActorCritic<SPEC>& actor_critic){
        malloc(device, actor_critic.actor);
        malloc(device, actor_critic.actor_target);
        malloc(device, actor_critic.critic_1);
        malloc(device, actor_critic.critic_2);
        malloc(device, actor_critic.critic_target_1);
        malloc(device, actor_critic.critic_target_2);
    }
    template <typename DEVICE, typename SPEC>
    FUNCTION_PLACEMENT void free(DEVICE& device, rl::algorithms::td3::ActorCritic<SPEC>& actor_critic){
        free(device, actor_critic.actor);
        free(device, actor_critic.actor_target);
        free(device, actor_critic.critic_1);
        free(device, actor_critic.critic_2);
        free(device, actor_critic.critic_target_1);
        free(device, actor_critic.critic_target_2);
    }
    template <typename DEVICE, typename SPEC>
    FUNCTION_PLACEMENT void malloc(DEVICE& device, rl::algorithms::td3::ActorTrainingBuffers<SPEC>& actor_training_buffers){
        malloc(device, actor_training_buffers.actions);
        malloc(device, actor_training_buffers.state_action_value_input);
        malloc(device, actor_training_buffers.state_action_value);
        malloc(device, actor_training_buffers.d_output);
        malloc(device, actor_training_buffers.d_critic_input);
        malloc(device, actor_training_buffers.d_actor_output);
        malloc(device, actor_training_buffers.d_actor_input);
    }
    template <typename DEVICE, typename SPEC>
    FUNCTION_PLACEMENT void free(DEVICE& device, rl::algorithms::td3::ActorTrainingBuffers<SPEC>& actor_training_buffers){
        free(device, actor_training_buffers.actions);
        free(device, actor_training_buffers.state_action_value_input);
        free(device, actor_training_buffers.state_action_value);
        free(device, actor_training_buffers.d_output);
        free(device, actor_training_buffers.d_critic_input);
        free(device, actor_training_buffers.d_actor_output);
        free(device, actor_training_buffers.d_actor_input);
    }

    template <typename DEVICE, typename SPEC>
    FUNCTION_PLACEMENT void malloc(DEVICE& device, rl::algorithms::td3::CriticTrainingBuffers<SPEC>& critic_training_buffers){
        malloc(device, critic_training_buffers.target_next_action_noise);
        malloc(device, critic_training_buffers.next_actions);
        malloc(device, critic_training_buffers.next_state_action_value_input);
        malloc(device, critic_training_buffers.target_action_value);
        malloc(device, critic_training_buffers.state_action_value_input);
        malloc(device, critic_training_buffers.next_state_action_value_critic_1);
        malloc(device, critic_training_buffers.next_state_action_value_critic_2);
    }

    template <typename DEVICE, typename SPEC>
    FUNCTION_PLACEMENT void free(DEVICE& device, rl::algorithms::td3::CriticTrainingBuffers<SPEC>& critic_training_buffers){
        free(device, critic_training_buffers.target_next_action_noise);
        free(device, critic_training_buffers.next_actions);
        free(device, critic_training_buffers.next_state_action_value_input);
        free(device, critic_training_buffers.target_action_value);
        free(device, critic_training_buffers.state_action_value_input);
        free(device, critic_training_buffers.next_state_action_value_critic_1);
        free(device, critic_training_buffers.next_state_action_value_critic_2);
    }

    template <typename DEVICE, typename SPEC, typename RNG>
    FUNCTION_PLACEMENT void init(DEVICE& device, rl::algorithms::td3::ActorCritic<SPEC>& actor_critic, RNG& rng){
        init_weights(device, actor_critic.actor   , rng);
        init_weights(device, actor_critic.critic_1, rng);
        init_weights(device, actor_critic.critic_2, rng);
        reset_optimizer_state(device, actor_critic.actor);
        reset_optimizer_state(device, actor_critic.critic_1);
        reset_optimizer_state(device, actor_critic.critic_2);

        copy(device, actor_critic.actor_target, actor_critic.actor);
        copy(device, actor_critic.critic_target_1, actor_critic.critic_1);
        copy(device, actor_critic.critic_target_2, actor_critic.critic_2);
    }
    template <typename DEVICE, typename SPEC, typename CRITIC_TYPE, typename REPLAY_BUFFER_SPEC, typename REPLAY_BUFFER_SPEC::TI BATCH_SIZE>
    FUNCTION_PLACEMENT typename SPEC::T train_critic(DEVICE& device, const rl::algorithms::td3::ActorCritic<SPEC>& actor_critic, CRITIC_TYPE& critic, rl::components::replay_buffer::Batch<REPLAY_BUFFER_SPEC, BATCH_SIZE>& batch, rl::algorithms::td3::CriticTrainingBuffers<SPEC>& training_buffers) {
        // requires training_buffers.target_next_action_noise to be populated
        using T = typename SPEC::T;
        using TI = typename DEVICE::index_t;
        static_assert(BATCH_SIZE == SPEC::PARAMETERS::CRITIC_BATCH_SIZE);
        constexpr auto OBSERVATION_DIM = SPEC::ENVIRONMENT::OBSERVATION_DIM;
        constexpr auto ACTION_DIM = SPEC::ENVIRONMENT::ACTION_DIM;
        zero_gradient(device, critic);

        evaluate(device, actor_critic.actor_target, batch.next_observations, training_buffers.next_actions);
        for(TI batch_step_i = 0; batch_step_i < BATCH_SIZE; batch_step_i++){
            for(TI action_i=0; action_i < SPEC::ENVIRONMENT::ACTION_DIM; action_i++){
                T noisy_next_action = training_buffers.next_actions.data[batch_step_i * ACTION_DIM + action_i] + training_buffers.target_next_action_noise.data[batch_step_i * ACTION_DIM + action_i];
                noisy_next_action = math::clamp<T>(noisy_next_action, -1, 1);
                training_buffers.next_actions.data[batch_step_i * ACTION_DIM + action_i] = noisy_next_action;
            }
        }
        hcat(device, batch.next_observations, training_buffers.next_actions, training_buffers.next_state_action_value_input);
        evaluate(device, actor_critic.critic_target_1, training_buffers.next_state_action_value_input, training_buffers.next_state_action_value_critic_1);
        evaluate(device, actor_critic.critic_target_2, training_buffers.next_state_action_value_input, training_buffers.next_state_action_value_critic_2);

        T mean_target_action_value = 0;
        for(TI batch_step_i = 0; batch_step_i < BATCH_SIZE; batch_step_i++){
            utils::memcpy(training_buffers.state_action_value_input.data + batch_step_i * (OBSERVATION_DIM + ACTION_DIM), batch.observations.data + batch_step_i * OBSERVATION_DIM, OBSERVATION_DIM);
            utils::memcpy(training_buffers.state_action_value_input.data + batch_step_i * (OBSERVATION_DIM + ACTION_DIM) + OBSERVATION_DIM, batch.actions.data + batch_step_i * ACTION_DIM, ACTION_DIM);
            T min_next_state_action_value = math::min(
                    training_buffers.next_state_action_value_critic_1.data[batch_step_i],
                    training_buffers.next_state_action_value_critic_2.data[batch_step_i]
            );
            T current_target_action_value = batch.rewards.data[batch_step_i] + (SPEC::PARAMETERS::IGNORE_TERMINATION || !batch.terminated.data[batch_step_i] ? SPEC::PARAMETERS::GAMMA * min_next_state_action_value : 0);
            training_buffers.target_action_value.data[batch_step_i] = current_target_action_value;
            mean_target_action_value += current_target_action_value;
            if(batch_step_i == 0){
                add_scalar(device.logger, "mean_target_action_value_sample", mean_target_action_value, 100);
            }
        }
        mean_target_action_value /= BATCH_SIZE;
        add_scalar(device.logger, "mean_target_action_value", mean_target_action_value, 100);

        forward_backward_mse(device, critic, training_buffers.state_action_value_input, training_buffers.target_action_value);
        static_assert(CRITIC_TYPE::SPEC::OUTPUT_LAYER::SPEC::ACTIVATION_FUNCTION == nn::activation_functions::IDENTITY); // Ensuring the critic output activation is identity so that we can just use the pre_activations to get the loss value
        T loss = nn::loss_functions::mse(device, critic.output_layer.pre_activations, training_buffers.target_action_value);

        update(device, critic);
        return loss;
    }
    template <typename DEVICE, typename SPEC, typename OUTPUT_SPEC, typename RNG>
    FUNCTION_PLACEMENT void target_action_noise(DEVICE& device, const rl::algorithms::td3::ActorCritic<SPEC>& actor_critic, Matrix<OUTPUT_SPEC> target_action_noise, RNG& rng ) {
        static_assert(OUTPUT_SPEC::ROWS == SPEC::PARAMETERS::CRITIC_BATCH_SIZE);
        static_assert(OUTPUT_SPEC::COLS == SPEC::ENVIRONMENT::ACTION_DIM);
        typedef typename SPEC::T T;
        for(typename DEVICE::index_t batch_sample_i=0; batch_sample_i < SPEC::PARAMETERS::CRITIC_BATCH_SIZE; batch_sample_i++){
            for(typename DEVICE::index_t action_i=0; action_i < SPEC::ENVIRONMENT::ACTION_DIM; action_i++){
                target_action_noise.data[batch_sample_i * SPEC::ENVIRONMENT::ACTION_DIM + action_i] = math::clamp(
                        random::normal_distribution(typename DEVICE::SPEC::RANDOM(), (T)0, SPEC::PARAMETERS::TARGET_NEXT_ACTION_NOISE_STD, rng),
                        -SPEC::PARAMETERS::TARGET_NEXT_ACTION_NOISE_CLIP,
                        SPEC::PARAMETERS::TARGET_NEXT_ACTION_NOISE_CLIP
                );
            }
        }
    }
    template <typename DEVICE, typename SPEC, typename REPLAY_BUFFER_SPEC, typename REPLAY_BUFFER_SPEC::TI BATCH_SIZE>
    FUNCTION_PLACEMENT typename SPEC::T train_actor(DEVICE& device, rl::algorithms::td3::ActorCritic<SPEC>& actor_critic, rl::components::replay_buffer::Batch<REPLAY_BUFFER_SPEC, BATCH_SIZE>& batch, rl::algorithms::td3::ActorTrainingBuffers<SPEC>& training_buffers) {
        using T = typename SPEC::T;
        using TI = typename DEVICE::index_t;
        static_assert(BATCH_SIZE == SPEC::PARAMETERS::ACTOR_BATCH_SIZE);
        constexpr auto OBSERVATION_DIM = SPEC::ENVIRONMENT::OBSERVATION_DIM;
        constexpr auto ACTION_DIM = SPEC::ENVIRONMENT::ACTION_DIM;

        zero_gradient(device, actor_critic.actor);
        forward(device, actor_critic.actor, batch.observations, training_buffers.actions);
        hcat(device, batch.observations, training_buffers.actions, training_buffers.state_action_value_input);
        auto& critic = actor_critic.critic_1;
        forward(device, critic, training_buffers.state_action_value_input, training_buffers.state_action_value);
        set(device, training_buffers.d_output, (T)-1/BATCH_SIZE);
        backward(device, critic, training_buffers.state_action_value_input, training_buffers.d_output, training_buffers.d_critic_input);
        slice(device, training_buffers.d_actor_output, training_buffers.d_critic_input, 0, OBSERVATION_DIM);
        backward(device, actor_critic.actor, batch.observations, training_buffers.d_actor_output, training_buffers.d_actor_input);
        T actor_value = sum(device, training_buffers.state_action_value)/BATCH_SIZE;

        update(device, actor_critic.actor);
        return actor_value;
    }

    template<typename DEVICE, typename SPEC>
    FUNCTION_PLACEMENT void update_target_layer(DEVICE& device, nn::layers::dense::Layer<SPEC>& target, const nn::layers::dense::Layer<SPEC>& source, typename SPEC::T polyak) {
        utils::polyak::update(device, target.weights, source.weights, polyak);
        utils::polyak::update(device, target.biases , source.biases , polyak);
    }
    template<typename T, typename DEVICE, typename TARGET_SPEC, typename SOURCE_SPEC>
    FUNCTION_PLACEMENT void update_target_network(DEVICE& device, nn_models::mlp::NeuralNetwork<TARGET_SPEC>& target, const nn_models::mlp::NeuralNetwork<SOURCE_SPEC>& source, T polyak) {
        using TargetNetworkType = typename utils::typing::remove_reference<decltype(target)>::type;
        update_target_layer(device, target.input_layer, source.input_layer, polyak);
        for(typename DEVICE::index_t layer_i=0; layer_i < TargetNetworkType::NUM_HIDDEN_LAYERS; layer_i++){
            update_target_layer(device, target.hidden_layers[layer_i], source.hidden_layers[layer_i], polyak);
        }
        update_target_layer(device, target.output_layer, source.output_layer, polyak);
    }

    template <typename DEVICE, typename SPEC>
    FUNCTION_PLACEMENT void update_critic_targets(DEVICE& device, rl::algorithms::td3::ActorCritic<SPEC>& actor_critic) {
        update_target_network(device, actor_critic.critic_target_1, actor_critic.critic_1, SPEC::PARAMETERS::CRITIC_POLYAK);
        update_target_network(device, actor_critic.critic_target_2, actor_critic.critic_2, SPEC::PARAMETERS::CRITIC_POLYAK);

    }
    template <typename DEVICE, typename SPEC>
    FUNCTION_PLACEMENT void update_actor_target(DEVICE& device, rl::algorithms::td3::ActorCritic<SPEC>& actor_critic) {
        update_target_network(device, actor_critic.actor_target   , actor_critic.   actor, SPEC::PARAMETERS::ACTOR_POLYAK);
    }


}

#endif
