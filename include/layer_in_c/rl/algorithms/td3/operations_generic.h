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
    template <typename DEVICE, typename SPEC, typename RNG>
    FUNCTION_PLACEMENT void init(DEVICE& device, rl::algorithms::td3::ActorCritic<SPEC>& actor_critic, RNG& rng){
        init_weights(device, actor_critic.actor   , rng);
        init_weights(device, actor_critic.critic_1, rng);
        init_weights(device, actor_critic.critic_2, rng);
        reset_optimizer_state(device, actor_critic.actor);
        reset_optimizer_state(device, actor_critic.critic_1);
        reset_optimizer_state(device, actor_critic.critic_2);

        copy(actor_critic.actor_target, actor_critic.actor);
        copy(actor_critic.critic_target_1, actor_critic.critic_1);
        copy(actor_critic.critic_target_2, actor_critic.critic_2);
    }
    template <typename DEVICE, typename SPEC, typename CRITIC_TYPE, auto REPLAY_BUFFER_CAPACITY, typename RNG, bool DETERMINISTIC=false>
    FUNCTION_PLACEMENT typename SPEC::T train_critic(
            DEVICE& device,
            const rl::algorithms::td3::ActorCritic<SPEC>& actor_critic,
            CRITIC_TYPE& critic,
            const rl::components::ReplayBuffer<
                    rl::components::replay_buffer::Specification<
                            typename SPEC::T,
                            typename DEVICE::index_t,
                            SPEC::ENVIRONMENT::OBSERVATION_DIM,
                            SPEC::ENVIRONMENT::ACTION_DIM,
                            REPLAY_BUFFER_CAPACITY
                    >
            >& replay_buffer,
            typename SPEC::T target_next_action_noise[SPEC::PARAMETERS::CRITIC_BATCH_SIZE][SPEC::ENVIRONMENT::ACTION_DIM],
            RNG& rng
    ) {
        typedef typename SPEC::T T;
        utils::assert_exit(device, replay_buffer.full || replay_buffer.position >= SPEC::PARAMETERS::CRITIC_BATCH_SIZE, "Error: replay buffer not full enough for training critic");
        T loss = 0;
        zero_gradient(device, critic);
        for(typename DEVICE::index_t batch_step_i=0; batch_step_i < SPEC::PARAMETERS::CRITIC_BATCH_SIZE; batch_step_i++){
            typename DEVICE::index_t sample_index_max = (replay_buffer.full ? REPLAY_BUFFER_CAPACITY : replay_buffer.position) - 1;
            typename DEVICE::index_t sample_index = DETERMINISTIC ? batch_step_i : random::uniform_int_distribution(typename DEVICE::SPEC::RANDOM(), (typename DEVICE::index_t)0, sample_index_max, rng);
            T next_state_action_value_input[SPEC::ENVIRONMENT::OBSERVATION_DIM + SPEC::ENVIRONMENT::ACTION_DIM];
            utils::memcpy(next_state_action_value_input, replay_buffer.next_observations[sample_index], SPEC::ENVIRONMENT::OBSERVATION_DIM); // setting the first part with next observations
            evaluate(device, actor_critic.actor_target, next_state_action_value_input, &next_state_action_value_input[SPEC::ENVIRONMENT::OBSERVATION_DIM]); // setting the second part with next actions
            for(typename DEVICE::index_t action_i=0; action_i < SPEC::ENVIRONMENT::ACTION_DIM; action_i++){
                T noisy_next_action = next_state_action_value_input[SPEC::ENVIRONMENT::OBSERVATION_DIM + action_i] + target_next_action_noise[batch_step_i][action_i];
                noisy_next_action = math::clamp<T>(noisy_next_action, -1, 1);
                next_state_action_value_input[SPEC::ENVIRONMENT::OBSERVATION_DIM + action_i] = noisy_next_action;
            }
            T next_state_action_value_critic_1 = evaluate(device, actor_critic.critic_target_1, next_state_action_value_input);
            T next_state_action_value_critic_2 = evaluate(device, actor_critic.critic_target_2, next_state_action_value_input);

            T min_next_state_action_value = math::min(
                    next_state_action_value_critic_1,
                    next_state_action_value_critic_2
            );
            T state_action_value_input[SPEC::ENVIRONMENT::OBSERVATION_DIM + SPEC::ENVIRONMENT::ACTION_DIM];
            utils::memcpy(state_action_value_input, replay_buffer.observations[sample_index], SPEC::ENVIRONMENT::OBSERVATION_DIM); // setting the first part with the current observation
            utils::memcpy(&state_action_value_input[SPEC::ENVIRONMENT::OBSERVATION_DIM], replay_buffer.actions[sample_index], SPEC::ENVIRONMENT::ACTION_DIM); // setting the first part with the current action
            T target_action_value[1] = {replay_buffer.rewards[sample_index] + SPEC::PARAMETERS::GAMMA * min_next_state_action_value * (!replay_buffer.terminated[sample_index])};

            forward_backward_mse<DEVICE, typename CRITIC_TYPE::SPEC, SPEC::PARAMETERS::CRITIC_BATCH_SIZE>(device, critic, state_action_value_input, target_action_value);
            static_assert(CRITIC_TYPE::SPEC::OUTPUT_LAYER::SPEC::ACTIVATION_FUNCTION == nn::activation_functions::IDENTITY); // Ensuring the critic output activation is identity so that we can just use the pre_activations to get the loss value
            T loss_sample = nn::loss_functions::mse<DEVICE, T, 1, SPEC::PARAMETERS::CRITIC_BATCH_SIZE>(device, critic.output_layer.pre_activations, target_action_value);
            loss += loss_sample;
        }
        update(device, critic);
        return loss;
    }
    template <typename DEVICE, typename SPEC, typename CRITIC_TYPE, auto REPLAY_BUFFER_CAPACITY, typename RNG>
    FUNCTION_PLACEMENT typename SPEC::T train_critic(
            DEVICE& device,
            const rl::algorithms::td3::ActorCritic<SPEC>& actor_critic,
            CRITIC_TYPE& critic,
            const rl::components::ReplayBuffer<
                    rl::components::replay_buffer::Specification<
                            typename SPEC::T,
                            typename DEVICE::index_t,
                            SPEC::ENVIRONMENT::OBSERVATION_DIM,
                            SPEC::ENVIRONMENT::ACTION_DIM,
                            REPLAY_BUFFER_CAPACITY
                    >
            >& replay_buffer,
            RNG& rng
    ) {
        typedef typename SPEC::T T;
        T action_noise[SPEC::PARAMETERS::CRITIC_BATCH_SIZE][SPEC::ENVIRONMENT::ACTION_DIM];
        for(typename DEVICE::index_t batch_sample_i=0; batch_sample_i < SPEC::PARAMETERS::CRITIC_BATCH_SIZE; batch_sample_i++){
            for(typename DEVICE::index_t action_i=0; action_i < SPEC::ENVIRONMENT::ACTION_DIM; action_i++){
                action_noise[batch_sample_i][action_i] = math::clamp(
                        random::normal_distribution(typename DEVICE::SPEC::RANDOM(), (T)0, SPEC::PARAMETERS::TARGET_NEXT_ACTION_NOISE_STD, rng),
                        -SPEC::PARAMETERS::TARGET_NEXT_ACTION_NOISE_CLIP,
                        SPEC::PARAMETERS::TARGET_NEXT_ACTION_NOISE_CLIP
                );
            }
        }
        return train_critic(device, actor_critic, critic, replay_buffer, action_noise, rng);
    }
    template <typename DEVICE, typename SPEC, auto REPLAY_BUFFER_CAPACITY, typename RNG, bool DETERMINISTIC = false>
    FUNCTION_PLACEMENT typename SPEC::T train_actor(
            DEVICE& device,
            rl::algorithms::td3::ActorCritic<SPEC>& actor_critic,
            rl::components::ReplayBuffer<
                    rl::components::replay_buffer::Specification<
                            typename SPEC::T,
                            typename DEVICE::index_t,
                            SPEC::ENVIRONMENT::OBSERVATION_DIM,
                            SPEC::ENVIRONMENT::ACTION_DIM,
                            REPLAY_BUFFER_CAPACITY
                    >
            >& replay_buffer,
            RNG& rng
    ) {
        typedef typename SPEC::T T;
        typedef typename SPEC::PARAMETERS PARAMETERS;
        typedef typename SPEC::ENVIRONMENT ENVIRONMENT;
        T actor_value = 0;
        zero_gradient(device, actor_critic.actor);
        typename DEVICE::index_t sample_index_max = (replay_buffer.full ? REPLAY_BUFFER_CAPACITY : replay_buffer.position) - 1;
        for (typename DEVICE::index_t sample_i=0; sample_i < PARAMETERS::ACTOR_BATCH_SIZE; sample_i++){
            typename DEVICE::index_t sample_index = DETERMINISTIC ? sample_i : random::uniform_int_distribution(typename DEVICE::SPEC::RANDOM(), (typename DEVICE::index_t)0, sample_index_max, rng);
            T state_action_value_input[ENVIRONMENT::OBSERVATION_DIM + ENVIRONMENT::ACTION_DIM];
            utils::memcpy(state_action_value_input, replay_buffer.observations[sample_index], ENVIRONMENT::OBSERVATION_DIM); // setting the first part with next observations
            forward(device, actor_critic.actor, state_action_value_input, &state_action_value_input[ENVIRONMENT::OBSERVATION_DIM]);

            auto& critic = actor_critic.critic_1;
            T critic_output = forward_univariate(device, critic, state_action_value_input);
            actor_value += critic_output/SPEC::PARAMETERS::ACTOR_BATCH_SIZE;
            T d_output[1] = {-(T)1/SPEC::PARAMETERS::ACTOR_BATCH_SIZE}; // we want to maximise the critic output using gradient descent
            T d_critic_input[ENVIRONMENT::OBSERVATION_DIM + ENVIRONMENT::ACTION_DIM];
            backward(device, critic, state_action_value_input, d_output, d_critic_input);
            T d_actor_input[ENVIRONMENT::OBSERVATION_DIM];
            backward(device, actor_critic.actor, state_action_value_input, &d_critic_input[ENVIRONMENT::OBSERVATION_DIM], d_actor_input);
        }
        update(device, actor_critic.actor);
        return actor_value;
    }
    template<typename DEVICE, typename SPEC>
    FUNCTION_PLACEMENT void update_target_layer(DEVICE& device, nn::layers::dense::Layer<SPEC>& target, const nn::layers::dense::Layer<SPEC>& source, typename SPEC::T polyak) {
        utils::polyak::update_matrix<DEVICE, typename SPEC::T, SPEC::OUTPUT_DIM, SPEC::INPUT_DIM>(device, target.weights, source.weights, polyak);
        utils::polyak::update       <DEVICE, typename SPEC::T, SPEC::OUTPUT_DIM                 >(device, target.biases , source.biases , polyak);
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
    FUNCTION_PLACEMENT void update_targets(DEVICE& device, rl::algorithms::td3::ActorCritic<SPEC>& actor_critic) {
        update_target_network(device, actor_critic.actor_target   , actor_critic.   actor, SPEC::PARAMETERS::ACTOR_POLYAK);
        update_target_network(device, actor_critic.critic_target_1, actor_critic.critic_1, SPEC::PARAMETERS::CRITIC_POLYAK);
        update_target_network(device, actor_critic.critic_target_2, actor_critic.critic_2, SPEC::PARAMETERS::CRITIC_POLYAK);

    }


}

#endif
