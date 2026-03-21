#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ALGORITHMS_PPO_OPERATIONS_GENERIC_TRAIN_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ALGORITHMS_PPO_OPERATIONS_GENERIC_TRAIN_H

#include "operations_generic.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template <typename DEVICE, typename PPO_SPEC, typename DATASET_SPEC, typename ACTOR_OPTIMIZER, typename CRITIC_OPTIMIZER, typename BUFFERS_SPEC, typename ACTOR_BUFFER, typename CRITIC_BUFFER, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void train(DEVICE& device, rl::algorithms::PPO<PPO_SPEC>& ppo, rl::components::on_policy_runner::Dataset<DATASET_SPEC>& dataset, ACTOR_OPTIMIZER& actor_optimizer, CRITIC_OPTIMIZER& critic_optimizer, rl::algorithms::ppo::Buffers<BUFFERS_SPEC>& ppo_buffers, ACTOR_BUFFER& actor_buffers, CRITIC_BUFFER& critic_buffers, RNG& rng){
#ifdef RL_TOOLS_DEBUG_RL_ALGORITHMS_PPO_CHECK_INIT
        utils::assert_exit(device, ppo.initialized, "PPO not initialized");
#endif
        using T = typename PPO_SPEC::TYPE_POLICY::DEFAULT;
        using TI = typename PPO_SPEC::TI;
        static_assert(utils::typing::is_same_v<typename PPO_SPEC::ENVIRONMENT, typename DATASET_SPEC::SPEC::ENVIRONMENT>, "environment mismatch");
        using ENVIRONMENT = typename PPO_SPEC::ENVIRONMENT;
        using DATASET = rl::components::on_policy_runner::Dataset<DATASET_SPEC>;
        static_assert(DATASET::STEPS_TOTAL > 1);
        constexpr TI N_EPOCHS = PPO_SPEC::PARAMETERS::N_EPOCHS;
        constexpr TI BATCH_SIZE = PPO_SPEC::PARAMETERS::BATCH_SIZE;
        constexpr TI N_BATCHES = DATASET::STEPS_TOTAL/BATCH_SIZE;
        static_assert(N_BATCHES > 0);
        constexpr TI ACTION_DIM = ENVIRONMENT::ACTION_DIM;
        constexpr TI N_AGENTS = PPO_SPEC::ENVIRONMENT::N_AGENTS;
        static_assert(ACTION_DIM % N_AGENTS == 0);
        constexpr TI PER_AGENT_ACTION_DIM = PPO_SPEC::ENVIRONMENT::ACTION_DIM/N_AGENTS;

        using OBS_SHAPE = typename DATASET::OBS_SHAPE;
        using OBS_PRIV_SHAPE = typename DATASET::OBS_PRIV_SHAPE;

        // batch needs observations, original log-probs, advantages
        T policy_kl_divergence = 0; // KL( current || old ) todo: make hyperparameter that swaps the order
        if(PPO_SPEC::PARAMETERS::ADAPTIVE_LEARNING_RATE) {
            auto& last_layer = get_last_layer(ppo.actor);
            auto log_std = matrix_view(device, last_layer.log_std.parameters);
            copy(device, device, log_std, ppo_buffers.rollout_log_std);
        }
        for(TI epoch_i = 0; epoch_i < N_EPOCHS; epoch_i++){
            static_assert(!PPO_SPEC::PARAMETERS::STATEFUL_ACTOR_AND_CRITIC || (N_EPOCHS == 1), "Stateful actor and critic implies single epoch");
            static_assert(!PPO_SPEC::PARAMETERS::STATEFUL_ACTOR_AND_CRITIC || (PPO_SPEC::PARAMETERS::TRUNCATE_ON_EACH_ITERATION == true), "Stateful actor and critic implies that the OnPolicyRunner should truncate in the beginning of each iteration, to prevent hidden state spillover.");
            static_assert(!PPO_SPEC::PARAMETERS::STATEFUL_ACTOR_AND_CRITIC || (BATCH_SIZE == DATASET_SPEC::STEPS_PER_ENV * DATASET_SPEC::SPEC::N_ENVIRONMENTS), "Stateful actor and critic implies single batch");
            static_assert(!PPO_SPEC::PARAMETERS::STATEFUL_ACTOR_AND_CRITIC || !PPO_SPEC::PARAMETERS::SHUFFLE_EPOCH, "Stateful actor and critic implies epoch shuffling");
            if constexpr(PPO_SPEC::PARAMETERS::SHUFFLE_EPOCH){ // shuffling
                // Create matrix views of observation tensors for shuffling
                auto observations_matrix = matrix_view(device, dataset.all_observations);
                auto observations_privileged_matrix = matrix_view(device, dataset.all_observations_privileged);
                for(TI dataset_i = 0; dataset_i < DATASET::STEPS_TOTAL; dataset_i++){
                    TI sample_index = random::uniform_int_distribution(device.random, dataset_i, DATASET::STEPS_TOTAL-1, rng);
                    {
                        auto target_row = row(device, observations_matrix, dataset_i);
                        auto source_row = row(device, observations_matrix, sample_index);
                        swap(device, target_row, source_row);
                    }
                    {
                        auto target_row = row(device, observations_privileged_matrix, dataset_i);
                        auto source_row = row(device, observations_privileged_matrix, sample_index);
                        swap(device, target_row, source_row);
                    }
                    if(PPO_SPEC::PARAMETERS::ADAPTIVE_LEARNING_RATE){
                        auto target_row = row(device, dataset.actions_mean, dataset_i);
                        auto source_row = row(device, dataset.actions_mean, sample_index);
                        swap(device, target_row, source_row);
                    }
                    {
                        auto target_row = row(device, dataset.actions, dataset_i);
                        auto source_row = row(device, dataset.actions, sample_index);
                        swap(device, target_row, source_row);
                    }
                    swap(device, dataset.advantages      , dataset.advantages      , dataset_i, 0, sample_index, 0);
                    swap(device, dataset.action_log_probs, dataset.action_log_probs, dataset_i, 0, sample_index, 0);
                    swap(device, dataset.target_values   , dataset.target_values   , dataset_i, 0, sample_index, 0);
                }
            }
            static_assert(N_BATCHES > 0);
            for(TI batch_i = 0; batch_i < N_BATCHES; batch_i++){
                T batch_policy_kl_divergence = 0; // KL( current || old ) todo: make hyperparameter that swaps the order
                zero_gradient(device, ppo.critic);
                zero_gradient(device, ppo.actor); // has to be reset before accumulating the action-log-std gradient

                auto batch_offset = batch_i * BATCH_SIZE;
                auto batch_observations            = view_range(device, dataset.all_observations              , batch_offset, tensor::ViewSpec<0, BATCH_SIZE>{});
                auto batch_observations_privileged = view_range(device, dataset.all_observations_privileged   , batch_offset, tensor::ViewSpec<0, BATCH_SIZE>{});
                auto batch_actions_mean            = view(device, dataset.actions_mean               , matrix::ViewSpec<BATCH_SIZE, ACTION_DIM                >(), batch_offset, 0);
                auto batch_actions                 = view(device, dataset.actions                    , matrix::ViewSpec<BATCH_SIZE, ACTION_DIM                >(), batch_offset, 0);
                auto batch_action_log_probs        = view(device, dataset.action_log_probs           , matrix::ViewSpec<BATCH_SIZE, 1                         >(), batch_offset, 0);
                auto batch_advantages              = view(device, dataset.advantages                 , matrix::ViewSpec<BATCH_SIZE, 1                         >(), batch_offset, 0);
                auto batch_target_values           = view(device, dataset.target_values              , matrix::ViewSpec<BATCH_SIZE, 1                         >(), batch_offset, 0);
                auto batch_reset                   = view(device, dataset.reset                      , matrix::ViewSpec<BATCH_SIZE, 1                         >(), batch_offset, 0);

                T advantage_mean = 0;
                T advantage_std = 0;
                if(PPO_SPEC::PARAMETERS::NORMALIZE_ADVANTAGE) {
                    ppo_advantage_normalization(device, batch_advantages, advantage_mean, advantage_std);
                }

                static constexpr TI STEPS = PPO_SPEC::PARAMETERS::STATEFUL_ACTOR_AND_CRITIC ? DATASET_SPEC::STEPS_PER_ENV : 1;
                static constexpr TI FORWARD_BATCH_SIZE = PPO_SPEC::PARAMETERS::STATEFUL_ACTOR_AND_CRITIC ? DATASET_SPEC::SPEC::N_ENVIRONMENTS : BATCH_SIZE;
                using ACTOR_INPUT_SHAPE = tensor::Prepend<tensor::Prepend<OBS_SHAPE, FORWARD_BATCH_SIZE>, STEPS>;
                auto batch_observations_reshaped = reshape_row_major(device, batch_observations, ACTOR_INPUT_SHAPE{});
                auto current_batch_actions_tensor = to_tensor(device, ppo_buffers.current_batch_actions);
                auto current_batch_actions_tensor_reshaped = reshape_row_major(device, current_batch_actions_tensor, tensor::Shape<TI, STEPS, FORWARD_BATCH_SIZE, ACTION_DIM>{});
                auto batch_reset_tensor_flat = to_tensor(device, batch_reset);
                auto batch_reset_tensor = reshape_row_major(device, batch_reset_tensor_flat, tensor::Shape<TI, STEPS, FORWARD_BATCH_SIZE, 1>{});
                Mode<nn::layers::gru::ResetMode<mode::Rollout<>, nn::layers::gru::ResetModeSpecification<TI, decltype(batch_reset_tensor)>>> mode;
                mode.reset_container = batch_reset_tensor;
                forward(device, ppo.actor, batch_observations_reshaped, current_batch_actions_tensor_reshaped, actor_buffers, rng, mode);

                ppo_compute_actor_loss_gradient(device, ppo, ppo_buffers, batch_actions, batch_actions_mean, batch_action_log_probs, batch_advantages, advantage_mean, advantage_std, policy_kl_divergence, batch_policy_kl_divergence, rng);
                if(PPO_SPEC::PARAMETERS::ADAPTIVE_LEARNING_RATE){
                    batch_policy_kl_divergence /= BATCH_SIZE;
                    auto& actor_optimizer_parameters = get_ref(device, actor_optimizer.parameters, 0);
                    if(batch_policy_kl_divergence > 2 * PPO_SPEC::PARAMETERS::ADAPTIVE_LEARNING_RATE_POLICY_KL_THRESHOLD){
                        actor_optimizer_parameters.alpha = math::max(device.math, actor_optimizer_parameters.alpha * PPO_SPEC::PARAMETERS::ADAPTIVE_LEARNING_RATE_DECAY, PPO_SPEC::PARAMETERS::ADAPTIVE_LEARNING_RATE_MIN);
                    }
                    if(batch_policy_kl_divergence < 0.5 * PPO_SPEC::PARAMETERS::ADAPTIVE_LEARNING_RATE_POLICY_KL_THRESHOLD){
                        actor_optimizer_parameters.alpha = math::min(device.math, actor_optimizer_parameters.alpha / PPO_SPEC::PARAMETERS::ADAPTIVE_LEARNING_RATE_DECAY, PPO_SPEC::PARAMETERS::ADAPTIVE_LEARNING_RATE_MAX);
                    }
                }
                auto d_action_d_log_prob_action_tensor = to_tensor(device, ppo_buffers.d_action_log_prob_d_action);
                auto d_action_d_log_prob_action_tensor_reshaped = reshape_row_major(device, d_action_d_log_prob_action_tensor, tensor::Shape<TI, STEPS, FORWARD_BATCH_SIZE, ACTION_DIM>{});
                backward(device, ppo.actor, batch_observations_reshaped, d_action_d_log_prob_action_tensor_reshaped, actor_buffers, mode);

                using CRITIC_INPUT_SHAPE = tensor::Prepend<tensor::Prepend<OBS_PRIV_SHAPE, FORWARD_BATCH_SIZE>, STEPS>;
                auto batch_observations_privileged_reshaped = reshape_row_major(device, batch_observations_privileged, CRITIC_INPUT_SHAPE{});
                {
                    forward(device, ppo.critic, batch_observations_privileged_reshaped, critic_buffers, rng, mode);
                    auto output_tensor = output(device, ppo.critic);
                    static_assert(sizeof(output_tensor) <= sizeof(void*));
                    auto output_matrix_view = matrix_view(device, output_tensor);
                    nn::loss_functions::mse::gradient(device, output_matrix_view, batch_target_values, ppo_buffers.d_critic_output, 0.5);
                    auto d_critic_output_tensor = to_tensor(device, ppo_buffers.d_critic_output);
                    auto d_critic_output_tensor_reshaped = reshape_row_major(device, d_critic_output_tensor, tensor::Shape<TI, STEPS, FORWARD_BATCH_SIZE, 1>{});
                    backward(device, ppo.critic, batch_observations_privileged_reshaped, d_critic_output_tensor_reshaped, critic_buffers, mode);
                }
                auto output_tensor = output(device, ppo.critic);
                auto output_matrix_view = matrix_view(device, output_tensor);
                T critic_loss = nn::loss_functions::mse::evaluate(device, output_matrix_view, batch_target_values);
                add_scalar(device, device.logger, "ppo/critic_loss", critic_loss);
                step(device, actor_optimizer, ppo.actor);
                step(device, critic_optimizer, ppo.critic);
            }
        }
        if(PPO_SPEC::PARAMETERS::ADAPTIVE_LEARNING_RATE) {
            policy_kl_divergence /= N_EPOCHS * N_BATCHES * BATCH_SIZE;
            add_scalar(device, device.logger, "ppo/policy_kl", policy_kl_divergence);
        }
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
