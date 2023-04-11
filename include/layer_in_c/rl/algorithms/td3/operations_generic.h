#ifndef LAYER_IN_C_RL_ALGORITHMS_TD3_OPERATIONS_GENERIC_H
#define LAYER_IN_C_RL_ALGORITHMS_TD3_OPERATIONS_GENERIC_H

#include "td3.h"

#include <layer_in_c/rl/components/replay_buffer/replay_buffer.h>
#include <layer_in_c/rl/components/off_policy_runner/off_policy_runner.h>
#include <layer_in_c/nn/nn.h>
#include <layer_in_c/nn_models/operations_generic.h>
#include <layer_in_c/utils/polyak/operations_generic.h>
#include <layer_in_c/math/operations_generic.h>
#include <layer_in_c/utils/generic/memcpy.h>

namespace layer_in_c{
    template <typename DEVICE, typename SPEC>
    void malloc(DEVICE& device, rl::algorithms::td3::ActorCritic<SPEC>& actor_critic){
        malloc(device, actor_critic.actor);
        malloc(device, actor_critic.actor_target);
        malloc(device, actor_critic.critic_1);
        malloc(device, actor_critic.critic_2);
        malloc(device, actor_critic.critic_target_1);
        malloc(device, actor_critic.critic_target_2);
    }
    template <typename DEVICE, typename SPEC>
    void free(DEVICE& device, rl::algorithms::td3::ActorCritic<SPEC>& actor_critic){
        free(device, actor_critic.actor);
        free(device, actor_critic.actor_target);
        free(device, actor_critic.critic_1);
        free(device, actor_critic.critic_2);
        free(device, actor_critic.critic_target_1);
        free(device, actor_critic.critic_target_2);
    }
    template <typename DEVICE, typename SPEC>
    void malloc(DEVICE& device, rl::algorithms::td3::ActorTrainingBuffers<SPEC>& actor_training_buffers){
        using BUFFERS = rl::algorithms::td3::ActorTrainingBuffers<SPEC>;
        malloc(device, actor_training_buffers.state_action_value_input);
        actor_training_buffers.observations = lic::view<DEVICE, typename decltype(actor_training_buffers.state_action_value_input)::SPEC, BUFFERS::BATCH_SIZE, BUFFERS::OBSERVATION_DIM>(device, actor_training_buffers.state_action_value_input, 0, 0);
        actor_training_buffers.actions      = lic::view<DEVICE, typename decltype(actor_training_buffers.state_action_value_input)::SPEC, BUFFERS::BATCH_SIZE, BUFFERS::ACTION_DIM     >(device, actor_training_buffers.state_action_value_input, 0, BUFFERS::OBSERVATION_DIM);
        malloc(device, actor_training_buffers.state_action_value);
        malloc(device, actor_training_buffers.d_output);
        malloc(device, actor_training_buffers.d_critic_input);
        malloc(device, actor_training_buffers.d_actor_output);
        malloc(device, actor_training_buffers.d_actor_input);
    }
    template <typename DEVICE, typename SPEC>
    void free(DEVICE& device, rl::algorithms::td3::ActorTrainingBuffers<SPEC>& actor_training_buffers){
        free(device, actor_training_buffers.state_action_value_input);
        actor_training_buffers.observations._data = nullptr;
        actor_training_buffers.actions._data      = nullptr;
        free(device, actor_training_buffers.state_action_value);
        free(device, actor_training_buffers.d_output);
        free(device, actor_training_buffers.d_critic_input);
        free(device, actor_training_buffers.d_actor_output);
        free(device, actor_training_buffers.d_actor_input);
    }

    template <typename DEVICE, typename SPEC>
    void malloc(DEVICE& device, rl::algorithms::td3::CriticTrainingBuffers<SPEC>& critic_training_buffers){
        using BUFFERS = rl::algorithms::td3::CriticTrainingBuffers<SPEC>;
        malloc(device, critic_training_buffers.target_next_action_noise);
        malloc(device, critic_training_buffers.next_state_action_value_input);
        critic_training_buffers.next_observations = view<DEVICE, typename decltype(critic_training_buffers.next_state_action_value_input)::SPEC, BUFFERS::BATCH_SIZE, BUFFERS::OBSERVATION_DIM>(device, critic_training_buffers.next_state_action_value_input, 0, 0);
        critic_training_buffers.next_actions      = view<DEVICE, typename decltype(critic_training_buffers.next_state_action_value_input)::SPEC, BUFFERS::BATCH_SIZE, BUFFERS::ACTION_DIM     >(device, critic_training_buffers.next_state_action_value_input, 0, BUFFERS::OBSERVATION_DIM);
        malloc(device, critic_training_buffers.action_value);
        malloc(device, critic_training_buffers.target_action_value);
        malloc(device, critic_training_buffers.next_state_action_value_critic_1);
        malloc(device, critic_training_buffers.next_state_action_value_critic_2);
    }

    template <typename DEVICE, typename SPEC>
    void free(DEVICE& device, rl::algorithms::td3::CriticTrainingBuffers<SPEC>& critic_training_buffers){
        free(device, critic_training_buffers.target_next_action_noise);
        free(device, critic_training_buffers.next_state_action_value_input);
        critic_training_buffers.next_observations._data = nullptr;
        critic_training_buffers.next_actions._data = nullptr;
        free(device, critic_training_buffers.action_value);
        free(device, critic_training_buffers.target_action_value);
        free(device, critic_training_buffers.next_state_action_value_critic_1);
        free(device, critic_training_buffers.next_state_action_value_critic_2);
    }

    template <typename DEVICE, typename SPEC, typename OPTIMIZER, typename RNG>
    void init(DEVICE& device, rl::algorithms::td3::ActorCritic<SPEC>& actor_critic, OPTIMIZER& optimizer, RNG& rng){
        init_weights(device, actor_critic.actor   , rng);
        init_weights(device, actor_critic.critic_1, rng);
        init_weights(device, actor_critic.critic_2, rng);
        zero_gradient(device, actor_critic.actor);
        zero_gradient(device, actor_critic.critic_1);
        zero_gradient(device, actor_critic.critic_2);
        reset_optimizer_state(device, actor_critic.actor, optimizer);
        reset_optimizer_state(device, actor_critic.critic_1, optimizer);
        reset_optimizer_state(device, actor_critic.critic_2, optimizer);

        copy(device, device, actor_critic.actor_target, actor_critic.actor);
        copy(device, device, actor_critic.critic_target_1, actor_critic.critic_1);
        copy(device, device, actor_critic.critic_target_2, actor_critic.critic_2);
    }
    template <typename DEVICE, typename SPEC, typename OUTPUT_SPEC, typename RNG>
    void target_action_noise(DEVICE& device, const rl::algorithms::td3::ActorCritic<SPEC>& actor_critic, Matrix<OUTPUT_SPEC>& target_action_noise, RNG& rng ) {
        static_assert(OUTPUT_SPEC::ROWS == SPEC::PARAMETERS::CRITIC_BATCH_SIZE);
        static_assert(OUTPUT_SPEC::COLS == SPEC::ENVIRONMENT::ACTION_DIM);
        typedef typename SPEC::T T;
        for(typename DEVICE::index_t batch_sample_i=0; batch_sample_i < SPEC::PARAMETERS::CRITIC_BATCH_SIZE; batch_sample_i++){
            for(typename DEVICE::index_t action_i=0; action_i < SPEC::ENVIRONMENT::ACTION_DIM; action_i++){
                set(target_action_noise, batch_sample_i, action_i, math::clamp(typename DEVICE::SPEC::MATH(),
                        random::normal_distribution(typename DEVICE::SPEC::RANDOM(), (T)0, SPEC::PARAMETERS::TARGET_NEXT_ACTION_NOISE_STD, rng),
                        -SPEC::PARAMETERS::TARGET_NEXT_ACTION_NOISE_CLIP,
                        SPEC::PARAMETERS::TARGET_NEXT_ACTION_NOISE_CLIP
                ));
            }
        }
    }
    template <typename DEVICE, typename SPEC>
    void noisy_next_actions(DEVICE& device, rl::algorithms::td3::CriticTrainingBuffers<SPEC>& training_buffers) {
        using T = typename SPEC::T;
        using TI = typename DEVICE::index_t;
        using BUFFERS = rl::algorithms::td3::CriticTrainingBuffers<SPEC>;
        constexpr TI BATCH_SIZE = BUFFERS::BATCH_SIZE;
        for(TI batch_step_i = 0; batch_step_i < BATCH_SIZE; batch_step_i++){
            for(TI action_i=0; action_i < SPEC::ENVIRONMENT::ACTION_DIM; action_i++){
                T noisy_next_action = get(training_buffers.next_actions, batch_step_i, action_i) + get(training_buffers.target_next_action_noise, batch_step_i, action_i);
                noisy_next_action = math::clamp<T>(typename DEVICE::SPEC::MATH(), noisy_next_action, -1, 1);
                set(training_buffers.next_actions, batch_step_i, action_i, noisy_next_action);
            }
        }
    }
    template <typename DEVICE, typename OFF_POLICY_RUNNER_SPEC, auto BATCH_SIZE, typename SPEC>
    void target_actions(DEVICE& device, rl::components::off_policy_runner::Batch<rl::components::off_policy_runner::BatchSpecification<OFF_POLICY_RUNNER_SPEC, BATCH_SIZE>>& batch, rl::algorithms::td3::CriticTrainingBuffers<SPEC>& training_buffers) {
        using T = typename SPEC::T;
        using TI = typename DEVICE::index_t;
        using BUFFERS = rl::algorithms::td3::CriticTrainingBuffers<SPEC>;
        static_assert(BATCH_SIZE == BUFFERS::BATCH_SIZE);
        constexpr auto OBSERVATION_DIM = SPEC::ENVIRONMENT::OBSERVATION_DIM;
        constexpr auto ACTION_DIM = SPEC::ENVIRONMENT::ACTION_DIM;
        for(TI batch_step_i = 0; batch_step_i < BATCH_SIZE; batch_step_i++){
            T min_next_state_action_value = math::min(typename DEVICE::SPEC::MATH(),
                    get(training_buffers.next_state_action_value_critic_1, batch_step_i, 0),
                    get(training_buffers.next_state_action_value_critic_2, batch_step_i, 0)
            );
            T reward = get(batch.rewards, 0, batch_step_i);
            bool terminated = get(batch.terminated, 0, batch_step_i);
            T future_value = SPEC::PARAMETERS::IGNORE_TERMINATION || !terminated ? SPEC::PARAMETERS::GAMMA * min_next_state_action_value : 0;
            T current_target_action_value = reward + future_value;
            set(training_buffers.target_action_value, batch_step_i, 0, current_target_action_value); // todo: improve pitch of target action values etc. (by transformig it into row vectors instead of column vectors)
        }
    }
    template <typename DEVICE, typename SPEC, typename CRITIC_TYPE, typename OFF_POLICY_RUNNER_SPEC, auto BATCH_SIZE, typename OPTIMIZER, typename ACTOR_BUFFERS, typename CRITIC_BUFFERS>
    void train_critic(DEVICE& device, const rl::algorithms::td3::ActorCritic<SPEC>& actor_critic, CRITIC_TYPE& critic, rl::components::off_policy_runner::Batch<rl::components::off_policy_runner::BatchSpecification<OFF_POLICY_RUNNER_SPEC, BATCH_SIZE>>& batch, OPTIMIZER& optimizer, ACTOR_BUFFERS& actor_buffers, CRITIC_BUFFERS& critic_buffers, rl::algorithms::td3::CriticTrainingBuffers<SPEC>& training_buffers) {
        // requires training_buffers.target_next_action_noise to be populated
        using T = typename SPEC::T;
        using TI = typename DEVICE::index_t;
        static_assert(BATCH_SIZE == SPEC::PARAMETERS::CRITIC_BATCH_SIZE);
        static_assert(BATCH_SIZE == CRITIC_BUFFERS::BATCH_SIZE);
        static_assert(BATCH_SIZE == ACTOR_BUFFERS::BATCH_SIZE);
        zero_gradient(device, critic);

        evaluate(device, actor_critic.actor_target, batch.next_observations, training_buffers.next_actions, actor_buffers);
        noisy_next_actions(device, training_buffers);
        copy(device, device, training_buffers.next_observations, batch.next_observations);
        evaluate(device, actor_critic.critic_target_1, training_buffers.next_state_action_value_input, training_buffers.next_state_action_value_critic_1, critic_buffers);
        evaluate(device, actor_critic.critic_target_2, training_buffers.next_state_action_value_input, training_buffers.next_state_action_value_critic_2, critic_buffers);

        target_actions(device, batch, training_buffers);
        forward_backward_mse(device, critic, batch.observations_and_actions, training_buffers.target_action_value, critic_buffers);
        update(device, critic, optimizer);
    }
    template <typename DEVICE, typename SPEC, typename CRITIC_TYPE, typename OFF_POLICY_RUNNER_SPEC, auto BATCH_SIZE>
    typename SPEC::T critic_loss(DEVICE& device, const rl::algorithms::td3::ActorCritic<SPEC>& actor_critic, CRITIC_TYPE& critic, rl::components::off_policy_runner::Batch<rl::components::off_policy_runner::BatchSpecification<OFF_POLICY_RUNNER_SPEC, BATCH_SIZE>>& batch, typename SPEC::ACTOR_NETWORK_TYPE::template Buffers<BATCH_SIZE>& actor_buffers, typename CRITIC_TYPE::template BuffersForwardBackward<BATCH_SIZE>& critic_buffers, rl::algorithms::td3::CriticTrainingBuffers<SPEC>& training_buffers) {
        // requires training_buffers.target_next_action_noise to be populated
        using T = typename SPEC::T;
        using TI = typename DEVICE::index_t;
        static_assert(BATCH_SIZE == SPEC::PARAMETERS::CRITIC_BATCH_SIZE);

        evaluate(device, actor_critic.actor_target, batch.next_observations, training_buffers.next_actions, actor_buffers);
        noisy_next_actions(device, training_buffers);
        copy(device, device, training_buffers.next_observations, batch.next_observations);
        evaluate(device, actor_critic.critic_target_1, training_buffers.next_state_action_value_input, training_buffers.next_state_action_value_critic_1, critic_buffers);
        evaluate(device, actor_critic.critic_target_2, training_buffers.next_state_action_value_input, training_buffers.next_state_action_value_critic_2, critic_buffers);

        target_actions(device, batch, training_buffers);
        evaluate(device, critic, batch.observations_and_actions, training_buffers.action_value, critic_buffers);
        return nn::loss_functions::mse(device, training_buffers.action_value, training_buffers.target_action_value);
    }
    template <typename DEVICE, typename SPEC, typename OFF_POLICY_RUNNER_SPEC, auto BATCH_SIZE, typename OPTIMIZER, typename ACTOR_BUFFERS, typename CRITIC_BUFFERS>
    void train_actor(DEVICE& device, rl::algorithms::td3::ActorCritic<SPEC>& actor_critic, rl::components::off_policy_runner::Batch<rl::components::off_policy_runner::BatchSpecification<OFF_POLICY_RUNNER_SPEC, BATCH_SIZE>>& batch, OPTIMIZER& optimizer, ACTOR_BUFFERS& actor_buffers, CRITIC_BUFFERS& critic_buffers, rl::algorithms::td3::ActorTrainingBuffers<SPEC>& training_buffers) {
        using T = typename SPEC::T;
        using TI = typename DEVICE::index_t;
        static_assert(BATCH_SIZE == SPEC::PARAMETERS::ACTOR_BATCH_SIZE);
        static_assert(BATCH_SIZE == CRITIC_BUFFERS::BATCH_SIZE);
        static_assert(BATCH_SIZE == ACTOR_BUFFERS::BATCH_SIZE);
        constexpr auto OBSERVATION_DIM = SPEC::ENVIRONMENT::OBSERVATION_DIM;
        constexpr auto ACTION_DIM = SPEC::ENVIRONMENT::ACTION_DIM;

        zero_gradient(device, actor_critic.actor);
        forward(device, actor_critic.actor, batch.observations, training_buffers.actions);
        copy(device, device, training_buffers.observations, batch.observations);
        auto& critic = actor_critic.critic_1;
        forward(device, critic, training_buffers.state_action_value_input, training_buffers.state_action_value);
        set_all(device, training_buffers.d_output, (T)-1/BATCH_SIZE);
        backward(device, critic, training_buffers.state_action_value_input, training_buffers.d_output, training_buffers.d_critic_input, critic_buffers);
        auto d_actor_output = view<DEVICE, typename decltype(training_buffers.d_critic_input)::SPEC, BATCH_SIZE, ACTION_DIM>(device, training_buffers.d_critic_input, 0, OBSERVATION_DIM);
        backward(device, actor_critic.actor, batch.observations, d_actor_output, training_buffers.d_actor_input, actor_buffers);

        update(device, actor_critic.actor, optimizer);
    }

    template <typename DEVICE, typename SPEC, typename OFF_POLICY_RUNNER_SPEC, auto BATCH_SIZE>
    typename SPEC::T actor_value(DEVICE& device, rl::algorithms::td3::ActorCritic<SPEC>& actor_critic, rl::components::off_policy_runner::Batch<rl::components::off_policy_runner::BatchSpecification<OFF_POLICY_RUNNER_SPEC, BATCH_SIZE>>& batch, typename SPEC::ACTOR_NETWORK_TYPE::template Buffers<BATCH_SIZE>& actor_buffers, typename SPEC::CRITIC_NETWORK_TYPE::template Buffers<BATCH_SIZE>& critic_buffers, rl::algorithms::td3::ActorTrainingBuffers<SPEC>& training_buffers) {
        using T = typename SPEC::T;
        using TI = typename DEVICE::index_t;
        static_assert(BATCH_SIZE == SPEC::PARAMETERS::ACTOR_BATCH_SIZE);

        evaluate(device, actor_critic.actor, batch.observations, training_buffers.actions, actor_buffers);
        copy(device, device, training_buffers.observations, batch.observations);
        auto& critic = actor_critic.critic_1;
        evaluate(device, critic, training_buffers.state_action_value_input, training_buffers.state_action_value, critic_buffers);
        return mean(device, training_buffers.state_action_value);
    }

    template<typename DEVICE, typename TARGET_SPEC, typename SOURCE_SPEC>
    void update_target_layer(DEVICE& device, nn::layers::dense::Layer<TARGET_SPEC>& target, const nn::layers::dense::Layer<SOURCE_SPEC>& source, typename TARGET_SPEC::T polyak) {
        utils::polyak::update(device, target.weights.parameters, source.weights.parameters, polyak);
        utils::polyak::update(device, target.biases.parameters , source.biases.parameters , polyak);
    }
    template<typename T, typename DEVICE, typename TARGET_SPEC, typename SOURCE_SPEC>
    void update_target_network(DEVICE& device, nn_models::mlp::NeuralNetwork<TARGET_SPEC>& target, const nn_models::mlp::NeuralNetwork<SOURCE_SPEC>& source, T polyak) {
        using TargetNetworkType = typename utils::typing::remove_reference<decltype(target)>::type;
        update_target_layer(device, target.input_layer, source.input_layer, polyak);
        for(typename DEVICE::index_t layer_i=0; layer_i < TargetNetworkType::NUM_HIDDEN_LAYERS; layer_i++){
            update_target_layer(device, target.hidden_layers[layer_i], source.hidden_layers[layer_i], polyak);
        }
        update_target_layer(device, target.output_layer, source.output_layer, polyak);
    }

    template <typename DEVICE, typename SPEC>
    void update_critic_targets(DEVICE& device, rl::algorithms::td3::ActorCritic<SPEC>& actor_critic) {
        update_target_network(device, actor_critic.critic_target_1, actor_critic.critic_1, SPEC::PARAMETERS::CRITIC_POLYAK);
        update_target_network(device, actor_critic.critic_target_2, actor_critic.critic_2, SPEC::PARAMETERS::CRITIC_POLYAK);
    }
    template <typename DEVICE, typename SPEC>
    void update_actor_target(DEVICE& device, rl::algorithms::td3::ActorCritic<SPEC>& actor_critic) {
        update_target_network(device, actor_critic.actor_target   , actor_critic.   actor, SPEC::PARAMETERS::ACTOR_POLYAK);
    }

    template <typename DEVICE, typename SPEC>
    bool is_nan(DEVICE& device, rl::algorithms::td3::ActorCritic<SPEC>& ac) {
        bool found_nan = false;
        found_nan = found_nan || is_nan(device, ac.actor);
        found_nan = found_nan || is_nan(device, ac.critic_1);
        found_nan = found_nan || is_nan(device, ac.critic_2);
        found_nan = found_nan || is_nan(device, ac.actor_target);
        found_nan = found_nan || is_nan(device, ac.critic_target_1);
        found_nan = found_nan || is_nan(device, ac.critic_target_2);
        return found_nan;
    }
    template <typename TARGET_DEVICE, typename SOURCE_DEVICE, typename TARGET_SPEC, typename SOURCE_SPEC>
    void copy(TARGET_DEVICE& target_device, SOURCE_DEVICE& source_device, rl::algorithms::td3::ActorCritic<TARGET_SPEC>& target, rl::algorithms::td3::ActorCritic<SOURCE_SPEC>& source){
        copy(target_device, source_device, target.actor   , source.actor);
        copy(target_device, source_device, target.critic_1, source.critic_1);
        copy(target_device, source_device, target.critic_2, source.critic_2);

        copy(target_device, source_device, target.actor_target   , source.actor_target);
        copy(target_device, source_device, target.critic_target_1, source.critic_target_1);
        copy(target_device, source_device, target.critic_target_2, source.critic_target_2);
    }
    template <typename TARGET_DEVICE, typename SOURCE_DEVICE, typename TARGET_SPEC, typename SOURCE_SPEC>
    void copy(TARGET_DEVICE& target_device, SOURCE_DEVICE& source_device, rl::algorithms::td3::CriticTrainingBuffers<TARGET_SPEC>& target, rl::algorithms::td3::CriticTrainingBuffers<SOURCE_SPEC>& source){
        copy(target_device, source_device, target.target_next_action_noise, source.target_next_action_noise);
        copy(target_device, source_device, target.next_state_action_value_input, source.next_state_action_value_input);
        copy(target_device, source_device, target.target_action_value, source.target_action_value);
        copy(target_device, source_device, target.next_state_action_value_critic_1, source.next_state_action_value_critic_1);
        copy(target_device, source_device, target.next_state_action_value_critic_2, source.next_state_action_value_critic_2);
    }
}

#endif
