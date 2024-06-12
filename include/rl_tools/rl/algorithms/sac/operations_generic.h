#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ALGORITHMS_SAC_OPERATIONS_GENERIC_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ALGORITHMS_SAC_OPERATIONS_GENERIC_H

#include "sac.h"

#include "../../../nn/layers/sample_and_squash/operations_generic.h"
#include "../../../rl/components/replay_buffer/replay_buffer.h"
#include "../../../rl/components/off_policy_runner/off_policy_runner.h"
#include "../../../nn/nn.h"
#include "../../../utils/polyak/operations_generic.h"
#include "../../../math/operations_generic.h"
#include "../../../utils/generic/memcpy.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template <typename DEVICE, typename SPEC>
    void malloc(DEVICE& device, rl::algorithms::sac::ActorCritic<SPEC>& actor_critic){
        malloc(device, actor_critic.actor);
        malloc(device, actor_critic.critic_1);
        malloc(device, actor_critic.critic_2);
        malloc(device, actor_critic.critic_target_1);
        malloc(device, actor_critic.critic_target_2);
    }
    template <typename DEVICE, typename SPEC>
    void free(DEVICE& device, rl::algorithms::sac::ActorCritic<SPEC>& actor_critic){
        free(device, actor_critic.actor);
        free(device, actor_critic.critic_1);
        free(device, actor_critic.critic_2);
        free(device, actor_critic.critic_target_1);
        free(device, actor_critic.critic_target_2);
    }
    template <typename DEVICE, typename SPEC>
    void malloc(DEVICE& device, rl::algorithms::sac::ActorTrainingBuffers<SPEC>& actor_training_buffers){
        using BUFFERS = rl::algorithms::sac::ActorTrainingBuffers<SPEC>;
        malloc(device, actor_training_buffers.state_action_value_input);
        actor_training_buffers.observations = view(device, actor_training_buffers.state_action_value_input, matrix::ViewSpec<BUFFERS::BATCH_SIZE, BUFFERS::CRITIC_OBSERVATION_DIM>{}, 0, 0);
        actor_training_buffers.actions      = view(device, actor_training_buffers.state_action_value_input, matrix::ViewSpec<BUFFERS::BATCH_SIZE, BUFFERS::ACTION_DIM>{}, 0, BUFFERS::CRITIC_OBSERVATION_DIM);
//        malloc(device, actor_training_buffers.state_action_value);
        malloc(device, actor_training_buffers.d_output);
        malloc(device, actor_training_buffers.d_critic_1_input);
        malloc(device, actor_training_buffers.d_critic_2_input);
        malloc(device, actor_training_buffers.d_critic_action_input);
        malloc(device, actor_training_buffers.action_sample);
        malloc(device, actor_training_buffers.action_noise);
        malloc(device, actor_training_buffers.d_actor_output);
        malloc(device, actor_training_buffers.d_actor_output_squashing);
        malloc(device, actor_training_buffers.d_squashing_input);
        malloc(device, actor_training_buffers.d_actor_input);
    }
    template <typename DEVICE, typename SPEC>
    void free(DEVICE& device, rl::algorithms::sac::ActorTrainingBuffers<SPEC>& actor_training_buffers){
        free(device, actor_training_buffers.state_action_value_input);
        actor_training_buffers.observations._data = nullptr;
        actor_training_buffers.actions._data      = nullptr;
//        free(device, actor_training_buffers.state_action_value);
        free(device, actor_training_buffers.d_output);
        free(device, actor_training_buffers.d_critic_1_input);
        free(device, actor_training_buffers.d_critic_2_input);
        free(device, actor_training_buffers.d_critic_action_input);
        free(device, actor_training_buffers.action_sample);
        free(device, actor_training_buffers.action_noise);
        free(device, actor_training_buffers.d_actor_output);
        free(device, actor_training_buffers.d_actor_output_squashing);
        free(device, actor_training_buffers.d_squashing_input);
        free(device, actor_training_buffers.d_actor_input);
    }

    template <typename DEVICE, typename SPEC>
    void malloc(DEVICE& device, rl::algorithms::sac::CriticTrainingBuffers<SPEC>& critic_training_buffers){
        using BUFFERS = rl::algorithms::sac::CriticTrainingBuffers<SPEC>;
        malloc(device, critic_training_buffers.next_state_action_value_input_full);
        critic_training_buffers.next_state_action_value_input = view(device, critic_training_buffers.next_state_action_value_input_full, matrix::ViewSpec<BUFFERS::BATCH_SIZE, BUFFERS::CRITIC_OBSERVATION_DIM + BUFFERS::ACTION_DIM>{}, 0, 0);
        critic_training_buffers.next_observations             = view(device, critic_training_buffers.next_state_action_value_input_full, matrix::ViewSpec<BUFFERS::BATCH_SIZE, BUFFERS::CRITIC_OBSERVATION_DIM>{}, 0, 0);
        critic_training_buffers.next_actions_distribution     = view(device, critic_training_buffers.next_state_action_value_input_full, matrix::ViewSpec<BUFFERS::BATCH_SIZE, BUFFERS::ACTION_DIM*2>{}, 0, BUFFERS::CRITIC_OBSERVATION_DIM);
        critic_training_buffers.next_actions_mean             = view(device, critic_training_buffers.next_state_action_value_input_full, matrix::ViewSpec<BUFFERS::BATCH_SIZE, BUFFERS::ACTION_DIM>{}, 0, BUFFERS::CRITIC_OBSERVATION_DIM);
        malloc(device, critic_training_buffers.action_value);
        malloc(device, critic_training_buffers.target_action_value);
        malloc(device, critic_training_buffers.next_state_action_value_critic_1);
        malloc(device, critic_training_buffers.next_state_action_value_critic_2);
        malloc(device, critic_training_buffers.d_output);
        malloc(device, critic_training_buffers.d_input);
        malloc(device, critic_training_buffers.next_action_log_probs);
    }

    template <typename DEVICE, typename SPEC>
    void free(DEVICE& device, rl::algorithms::sac::CriticTrainingBuffers<SPEC>& critic_training_buffers){
        free(device, critic_training_buffers.next_state_action_value_input_full);
        critic_training_buffers.next_state_action_value_input._data = nullptr;
        critic_training_buffers.next_observations._data = nullptr;
        critic_training_buffers.next_actions_distribution._data = nullptr;
        critic_training_buffers.next_actions_mean._data = nullptr;
        free(device, critic_training_buffers.action_value);
        free(device, critic_training_buffers.target_action_value);
        free(device, critic_training_buffers.next_state_action_value_critic_1);
        free(device, critic_training_buffers.next_state_action_value_critic_2);
        free(device, critic_training_buffers.d_output);
        free(device, critic_training_buffers.d_input);
        free(device, critic_training_buffers.next_action_log_probs);
    }

    template <typename DEVICE, typename SPEC, typename RNG>
    void init(DEVICE& device, rl::algorithms::sac::ActorCritic<SPEC>& actor_critic, RNG& rng){
        init_weights(device, actor_critic.actor   , rng);
        init_weights(device, actor_critic.critic_1, rng);
        init_weights(device, actor_critic.critic_2, rng);
        zero_gradient(device, actor_critic.actor);
        zero_gradient(device, actor_critic.critic_1);
        zero_gradient(device, actor_critic.critic_2);
        reset_optimizer_state(device, actor_critic.actor_optimizer, actor_critic.actor);
        reset_optimizer_state(device, actor_critic.critic_optimizers[0], actor_critic.critic_1);
        reset_optimizer_state(device, actor_critic.critic_optimizers[1], actor_critic.critic_2);
//        set(actor_critic.log_alpha.parameters, 0, 0, math::log(typename DEVICE::SPEC::MATH{}, SPEC::PARAMETERS::ALPHA));


        copy(device, device, actor_critic.critic_1, actor_critic.critic_target_1);
        copy(device, device, actor_critic.critic_2, actor_critic.critic_target_2);
    }
    template <typename DEVICE, typename BATCH_SPEC, typename SPEC, typename NEXT_ACTION_LOG_PROBS_SPEC, typename ALPHA_PARAMETER, typename TI_SAMPLE>
    RL_TOOLS_FUNCTION_PLACEMENT void target_actions_per_sample(DEVICE& device, rl::components::off_policy_runner::Batch<BATCH_SPEC>& batch, rl::algorithms::sac::CriticTrainingBuffers<SPEC>& training_buffers, const Matrix<NEXT_ACTION_LOG_PROBS_SPEC>& next_action_log_probs, ALPHA_PARAMETER alpha, TI_SAMPLE batch_step_i){
        using T = typename SPEC::T;
        using TI = typename DEVICE::index_t;
        using BUFFERS = rl::algorithms::sac::CriticTrainingBuffers<SPEC>;
        using BATCH = rl::components::off_policy_runner::Batch<BATCH_SPEC>;
        constexpr TI BATCH_SIZE = BATCH::BATCH_SIZE;
        static_assert(BATCH_SIZE == BUFFERS::BATCH_SIZE);
        T min_next_state_action_value = math::min(device.math,
                                                  get(training_buffers.next_state_action_value_critic_1, batch_step_i, 0),
                                                  get(training_buffers.next_state_action_value_critic_2, batch_step_i, 0)
        );
        T reward = get(batch.rewards, 0, batch_step_i);
        bool terminated = get(batch.terminated, 0, batch_step_i);
        T entropy_bonus = -alpha * get(next_action_log_probs, batch_step_i, 0);
        T min_next_state_action_value_entropy_bonus = min_next_state_action_value + entropy_bonus;
        T future_value = SPEC::PARAMETERS::IGNORE_TERMINATION || !terminated ? SPEC::PARAMETERS::GAMMA * min_next_state_action_value_entropy_bonus : 0;
        T current_target_action_value = reward + future_value;
        set(training_buffers.target_action_value, batch_step_i, 0, current_target_action_value); // todo: improve pitch of target action values etc. (by transformig it into row vectors instead of column vectors)
    }
    template <typename DEVICE, typename BATCH_SPEC, typename SPEC, typename NEXT_ACTION_LOG_PROBS_SPEC, typename ALPHA_PARAMETER>
    void target_actions(DEVICE& device, rl::components::off_policy_runner::Batch<BATCH_SPEC>& batch, rl::algorithms::sac::CriticTrainingBuffers<SPEC>& training_buffers, const Matrix<NEXT_ACTION_LOG_PROBS_SPEC>& next_action_log_probs, ALPHA_PARAMETER& log_alpha) {
        using T = typename SPEC::T;
        using TI = typename DEVICE::index_t;
        using BUFFERS = rl::algorithms::sac::CriticTrainingBuffers<SPEC>;
        using BATCH = rl::components::off_policy_runner::Batch<BATCH_SPEC>;
        constexpr TI BATCH_SIZE = BATCH::BATCH_SIZE;
        static_assert(BATCH_SIZE == BUFFERS::BATCH_SIZE);
        T alpha = math::exp(typename DEVICE::SPEC::MATH{}, get(log_alpha.parameters, 0, 0));
        for(TI batch_step_i = 0; batch_step_i < BATCH_SIZE; batch_step_i++){
            target_actions_per_sample(device, batch, training_buffers, next_action_log_probs, alpha, batch_step_i);
        }
    }
    template <typename DEVICE, typename SPEC, typename CRITIC_TYPE, typename OFF_POLICY_RUNNER_SPEC, auto BATCH_SIZE, typename OPTIMIZER, typename ACTOR_BUFFERS, typename CRITIC_BUFFERS, typename ACTION_NOISE_SPEC, typename RNG>
    void train_critic(DEVICE& device, rl::algorithms::sac::ActorCritic<SPEC>& actor_critic, CRITIC_TYPE& critic, rl::components::off_policy_runner::Batch<rl::components::off_policy_runner::BatchSpecification<OFF_POLICY_RUNNER_SPEC, BATCH_SIZE>>& batch, OPTIMIZER& optimizer, ACTOR_BUFFERS& actor_buffers, CRITIC_BUFFERS& critic_buffers, rl::algorithms::sac::CriticTrainingBuffers<SPEC>& training_buffers, Matrix<ACTION_NOISE_SPEC>& action_noise, RNG& rng){
        using T = typename SPEC::T;
        using TI = typename DEVICE::index_t;
        constexpr TI ACTION_DIM = SPEC::ENVIRONMENT::ACTION_DIM;
        static_assert(BATCH_SIZE == SPEC::PARAMETERS::CRITIC_BATCH_SIZE);
        static_assert(BATCH_SIZE == CRITIC_BUFFERS::BATCH_SIZE);
        static_assert(BATCH_SIZE == ACTOR_BUFFERS::BATCH_SIZE);
        static_assert(BATCH_SIZE == ACTION_NOISE_SPEC::ROWS);
        static_assert(ACTION_DIM == ACTION_NOISE_SPEC::COLS);

        zero_gradient(device, critic);


        auto sample_and_squash_layer = get_last_layer(actor_critic.actor);
        auto sample_and_squash_buffer = get_last_buffer(actor_buffers);
        copy(device, device, action_noise, sample_and_squash_buffer.noise);
        using SAMPLE_AND_SQUASH_MODE = nn::Mode<nn::layers::sample_and_squash::mode::ExternalNoise<nn::mode::Default>>;
        forward(device, actor_critic.actor, batch.next_observations, training_buffers.next_actions_mean, actor_buffers, rng, SAMPLE_AND_SQUASH_MODE{});
        copy(device, device, batch.next_observations_privileged, training_buffers.next_observations);
        evaluate(device, actor_critic.critic_target_1, training_buffers.next_state_action_value_input, training_buffers.next_state_action_value_critic_1, critic_buffers, rng);
        evaluate(device, actor_critic.critic_target_2, training_buffers.next_state_action_value_input, training_buffers.next_state_action_value_critic_2, critic_buffers, rng);

        auto last_layer = get_last_layer(actor_critic.actor);
        auto next_action_log_probs = view_transpose(device, last_layer.log_probabilities);
        target_actions(device, batch, training_buffers, next_action_log_probs, sample_and_squash_layer.log_alpha);
        forward(device, critic, batch.observations_and_actions, critic_buffers, rng);
        nn::loss_functions::mse::gradient(device, output(critic), training_buffers.target_action_value, training_buffers.d_output, 0.5); // SB3/SBX uses 1/2, CleanRL doesn't
        backward(device, critic, batch.observations_and_actions, training_buffers.d_output, critic_buffers);
        step(device, optimizer, critic);
    }
    template <typename DEVICE, typename SPEC, typename CRITIC_TYPE, typename OFF_POLICY_RUNNER_SPEC, auto BATCH_SIZE, typename RNG>
    typename SPEC::T critic_loss(DEVICE& device, const rl::algorithms::sac::ActorCritic<SPEC>& actor_critic, CRITIC_TYPE& critic, rl::components::off_policy_runner::Batch<rl::components::off_policy_runner::BatchSpecification<OFF_POLICY_RUNNER_SPEC, BATCH_SIZE>>& batch, typename SPEC::ACTOR_NETWORK_TYPE::template Buffers<BATCH_SIZE>& actor_buffers, typename CRITIC_TYPE::template Buffers<BATCH_SIZE>& critic_buffers, rl::algorithms::sac::CriticTrainingBuffers<SPEC>& training_buffers, RNG& rng) {
        // todo: needs to be updated
        using T = typename SPEC::T;
        using TI = typename DEVICE::index_t;
        static_assert(BATCH_SIZE == SPEC::PARAMETERS::CRITIC_BATCH_SIZE);

        evaluate(device, actor_critic.actor, batch.next_observations, training_buffers.next_actions_distribution, actor_buffers, rng);
        copy(device, device, batch.next_observations_privileged, training_buffers.next_observations);
        evaluate(device, actor_critic.critic_target_1, training_buffers.next_state_action_value_input, training_buffers.next_state_action_value_critic_1, critic_buffers, rng);
        evaluate(device, actor_critic.critic_target_2, training_buffers.next_state_action_value_input, training_buffers.next_state_action_value_critic_2, critic_buffers, rng);

        T log_alpha = get(actor_critic.log_alpha, 0, 0);
        T alpha = math::exp(typename DEVICE::SPEC::MATH{}, log_alpha);
        target_actions(device, batch, training_buffers, alpha);
        evaluate(device, critic, batch.observations_and_actions, training_buffers.action_value, critic_buffers, rng);
        return nn::loss_functions::mse::evaluate(device, training_buffers.action_value, training_buffers.target_action_value, 0.5);
    }
    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void min_value_d_output_per_sample(DEVICE& device, rl::algorithms::sac::ActorCritic<SPEC>& actor_critic, rl::algorithms::sac::ActorTrainingBuffers<SPEC>& training_buffers, typename DEVICE::index_t batch_i) {
        auto critic_1_output = output(actor_critic.critic_1);
        auto critic_2_output = output(actor_critic.critic_2);
        using TI = typename DEVICE::index_t;
        using T = typename SPEC::T;
        constexpr TI ACTION_DIM = SPEC::ENVIRONMENT::ACTION_DIM;

        bool critic_1_value = get(critic_1_output, batch_i, 0) < get(critic_2_output, batch_i, 0);
        for(TI action_i=0; action_i < ACTION_DIM; action_i++){
            T d_input = 0;
            if(critic_1_value) {
                d_input = get(training_buffers.d_critic_1_input, batch_i, SPEC::CRITIC_NETWORK_TYPE::INPUT_DIM - ACTION_DIM + action_i);
            }
            else{
                d_input = get(training_buffers.d_critic_2_input, batch_i, SPEC::CRITIC_NETWORK_TYPE::INPUT_DIM - ACTION_DIM + action_i);
            }
            set(training_buffers.d_actor_output_squashing, batch_i, action_i, (T)d_input);
        }
    }
    template <typename DEVICE, typename SPEC>
    void min_value_d_output(DEVICE& device, rl::algorithms::sac::ActorCritic<SPEC>& actor_critic, rl::algorithms::sac::ActorTrainingBuffers<SPEC>& training_buffers) {
        using BUFFERS = rl::algorithms::sac::ActorTrainingBuffers<SPEC>;
        using TI = typename DEVICE::index_t;
        constexpr TI BATCH_SIZE = BUFFERS::BATCH_SIZE;
        for(TI batch_i=0; batch_i < BATCH_SIZE; batch_i++){
            min_value_d_output_per_sample(device, actor_critic, training_buffers, batch_i);
        }
    }
    template <typename DEVICE, typename SPEC, typename BATCH_SPEC, typename OPTIMIZER, typename ACTOR_BUFFERS, typename CRITIC_BUFFERS, typename ACTION_NOISE_SPEC, typename RNG>
    void train_actor(DEVICE& device, rl::algorithms::sac::ActorCritic<SPEC>& actor_critic, rl::components::off_policy_runner::Batch<BATCH_SPEC>& batch, OPTIMIZER& optimizer, ACTOR_BUFFERS& actor_buffers, CRITIC_BUFFERS& critic_buffers, rl::algorithms::sac::ActorTrainingBuffers<SPEC>& training_buffers, Matrix<ACTION_NOISE_SPEC>& action_noise, RNG& rng) {
        using T = typename SPEC::T;
        using TI = typename DEVICE::index_t;
        constexpr TI BATCH_SIZE = BATCH_SPEC::BATCH_SIZE;
        static_assert(BATCH_SIZE == SPEC::PARAMETERS::ACTOR_BATCH_SIZE);
        static_assert(BATCH_SIZE == CRITIC_BUFFERS::BATCH_SIZE);
        static_assert(BATCH_SIZE == ACTOR_BUFFERS::BATCH_SIZE);
        constexpr auto ACTION_DIM = SPEC::ENVIRONMENT::ACTION_DIM;
        static_assert(SPEC::ACTOR_NETWORK_TYPE::OUTPUT_DIM == ACTION_DIM);

        auto sample_and_squashing_buffer = get_last_buffer(actor_buffers);

        zero_gradient(device, actor_critic.actor);
        copy(device, device, action_noise, sample_and_squashing_buffer.noise);
        using SAMPLE_AND_SQUASH_MODE = nn::Mode<nn::layers::sample_and_squash::mode::ExternalNoise<nn::mode::Default>>;
        forward(device, actor_critic.actor, batch.observations, training_buffers.actions, actor_buffers, rng, SAMPLE_AND_SQUASH_MODE{});
        copy(device, device, batch.observations_privileged, training_buffers.observations);
        forward(device, actor_critic.critic_1, training_buffers.state_action_value_input, critic_buffers, rng);
        forward(device, actor_critic.critic_2, training_buffers.state_action_value_input, critic_buffers, rng);
        // we minimize the negative of the actor loss
        // todo: evaluate only backpropagating the active values
        // note: the alpha * entropy term is minimized according to d_action_d_action_distribution
        set_all(device, training_buffers.d_output, (T)-1/BATCH_SIZE);
        backward_input(device, actor_critic.critic_1, training_buffers.d_output, training_buffers.d_critic_1_input, critic_buffers);
        backward_input(device, actor_critic.critic_2, training_buffers.d_output, training_buffers.d_critic_2_input, critic_buffers);
        min_value_d_output(device, actor_critic, training_buffers);
        backward(device, actor_critic.actor, batch.observations, training_buffers.d_actor_output_squashing, actor_buffers);
        step(device, optimizer, actor_critic.actor);
    }

    template <typename DEVICE, typename SPEC, typename OFF_POLICY_RUNNER_SPEC, auto BATCH_SIZE, typename RNG>
    typename SPEC::T actor_value(DEVICE& device, rl::algorithms::sac::ActorCritic<SPEC>& actor_critic, rl::components::off_policy_runner::Batch<rl::components::off_policy_runner::BatchSpecification<OFF_POLICY_RUNNER_SPEC, BATCH_SIZE>>& batch, typename SPEC::ACTOR_NETWORK_TYPE::template Buffers<BATCH_SIZE>& actor_buffers, typename SPEC::CRITIC_NETWORK_TYPE::template Buffers<BATCH_SIZE>& critic_buffers, rl::algorithms::sac::ActorTrainingBuffers<SPEC>& training_buffers, RNG& rng) {
        // todo: needs to be updated
        using T = typename SPEC::T;
        using TI = typename DEVICE::index_t;
        static_assert(BATCH_SIZE == SPEC::PARAMETERS::ACTOR_BATCH_SIZE);

        evaluate(device, actor_critic.actor, batch.observations, training_buffers.actions, actor_buffers, rng);
        copy(device, device, batch.observations, training_buffers.observations);
        auto& critic = actor_critic.critic_1;
        evaluate(device, critic, training_buffers.state_action_value_input, training_buffers.state_action_value, critic_buffers, rng);
        return mean(device, training_buffers.state_action_value);
    }

    namespace rl::algorithms::sac{
        template<typename DEVICE, typename SOURCE_SPEC, typename TARGET_SPEC>
        void update_target_module(DEVICE& device, const  nn::layers::dense::LayerForward<SOURCE_SPEC>& source, nn::layers::dense::LayerForward<TARGET_SPEC>& target, typename SOURCE_SPEC::T polyak) {
            rl_tools::utils::polyak::update(device, source.weights.parameters, target.weights.parameters, polyak);
            rl_tools::utils::polyak::update(device, source.biases.parameters , target.biases.parameters , polyak);
        }
        template<typename T, typename DEVICE, typename SOURCE_SPEC, typename TARGET_SPEC>
        void update_target_module(DEVICE& device, const  nn_models::mlp::NeuralNetworkForward<SOURCE_SPEC>& source, nn_models::mlp::NeuralNetworkForward<TARGET_SPEC>& target, T polyak) {
            using TargetNetworkType = nn_models::mlp::NeuralNetworkForward<TARGET_SPEC>;
            update_target_module(device, source.input_layer, target.input_layer, polyak);
            for(typename DEVICE::index_t layer_i=0; layer_i < TargetNetworkType::NUM_HIDDEN_LAYERS; layer_i++){
                update_target_module(device, source.hidden_layers[layer_i], target.hidden_layers[layer_i], polyak);
            }
            update_target_module(device, source.output_layer, target.output_layer, polyak);
        }
        template<typename T, typename DEVICE, typename SOURCE_SPEC, typename TARGET_SPEC>
        void update_target_module(DEVICE& device, const  nn_models::sequential::ModuleForward<SOURCE_SPEC>& source, nn_models::sequential::ModuleForward<TARGET_SPEC>& target, T polyak) {
            update_target_module(device, source.content, target.content, polyak);
            if constexpr(!rl_tools::utils::typing::is_same_v<typename SOURCE_SPEC::NEXT_MODULE, nn_models::sequential::OutputModule>){
                update_target_module(device, source.next_module, target.next_module, polyak);
            }
        }
    }

    template <typename DEVICE, typename SPEC>
    void update_critic_targets(DEVICE& device, rl::algorithms::sac::ActorCritic<SPEC>& actor_critic) {
        rl::algorithms::sac::update_target_module(device, actor_critic.critic_1, actor_critic.critic_target_1, SPEC::PARAMETERS::CRITIC_POLYAK);
        rl::algorithms::sac::update_target_module(device, actor_critic.critic_2, actor_critic.critic_target_2, SPEC::PARAMETERS::CRITIC_POLYAK);
    }

    template <typename DEVICE, typename SPEC>
    bool is_nan(DEVICE& device, rl::algorithms::sac::ActorCritic<SPEC>& ac) {
        bool found_nan = false;
        found_nan = found_nan || is_nan(device, ac.actor);
        found_nan = found_nan || is_nan(device, ac.critic_1);
        found_nan = found_nan || is_nan(device, ac.critic_2);
        found_nan = found_nan || is_nan(device, ac.critic_target_1);
        found_nan = found_nan || is_nan(device, ac.critic_target_2);
        return found_nan;
    }
    template <typename SOURCE_DEVICE, typename TARGET_DEVICE, typename SOURCE_SPEC, typename TARGET_SPEC>
    void copy(SOURCE_DEVICE& source_device, TARGET_DEVICE& target_device, rl::algorithms::sac::ActorCritic<SOURCE_SPEC>& source, rl::algorithms::sac::ActorCritic<TARGET_SPEC>& target){
        copy(source_device, target_device, source.actor   , target.actor);
        copy(source_device, target_device, source.critic_1, target.critic_1);
        copy(source_device, target_device, source.critic_2, target.critic_2);

        copy(source_device, target_device, source.critic_target_1, target.critic_target_1);
        copy(source_device, target_device, source.critic_target_2, target.critic_target_2);

        copy(source_device, target_device, source.actor_optimizer, target.actor_optimizer);
        copy(source_device, target_device, source.critic_optimizers[0], target.critic_optimizers[0]);
        copy(source_device, target_device, source.critic_optimizers[1], target.critic_optimizers[1]);
        copy(source_device, target_device, source.alpha_optimizer, target.alpha_optimizer);
    }
    template <typename SOURCE_DEVICE, typename TARGET_DEVICE, typename SOURCE_SPEC, typename TARGET_SPEC>
    void copy(SOURCE_DEVICE& source_device, TARGET_DEVICE& target_device, rl::algorithms::sac::ActorTrainingBuffers<SOURCE_SPEC>& source, rl::algorithms::sac::ActorTrainingBuffers<TARGET_SPEC>& target){
        copy(source_device, target_device, source.state_action_value_input, target.state_action_value_input);
        copy(source_device, target_device, source.d_output, target.d_output);
        copy(source_device, target_device, source.d_critic_1_input, target.d_critic_1_input);
        copy(source_device, target_device, source.d_critic_2_input, target.d_critic_2_input);
        copy(source_device, target_device, source.d_critic_action_input, target.d_critic_action_input);
        copy(source_device, target_device, source.action_sample, target.action_sample);
        copy(source_device, target_device, source.action_noise, target.action_noise);
        copy(source_device, target_device, source.d_actor_output, target.d_actor_output);
        copy(source_device, target_device, source.d_actor_input, target.d_actor_input);
        copy(source_device, target_device, source.d_actor_output_squashing, target.d_actor_output_squashing);
    }
    template <typename SOURCE_DEVICE, typename TARGET_DEVICE, typename SOURCE_SPEC, typename TARGET_SPEC>
    void copy(SOURCE_DEVICE& source_device, TARGET_DEVICE& target_device, rl::algorithms::sac::CriticTrainingBuffers<SOURCE_SPEC>& source, rl::algorithms::sac::CriticTrainingBuffers<TARGET_SPEC>& target){
        copy(source_device, target_device, source.next_state_action_value_input_full, target.next_state_action_value_input_full);
        copy(source_device, target_device, source.action_value, target.action_value);
        copy(source_device, target_device, source.target_action_value, target.target_action_value);
        copy(source_device, target_device, source.next_state_action_value_critic_1, target.next_state_action_value_critic_1);
        copy(source_device, target_device, source.next_state_action_value_critic_2, target.next_state_action_value_critic_2);
        copy(source_device, target_device, source.d_input, target.d_input);
        copy(source_device, target_device, source.d_output, target.d_output);
        copy(source_device, target_device, source.next_action_log_probs, target.next_action_log_probs);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
