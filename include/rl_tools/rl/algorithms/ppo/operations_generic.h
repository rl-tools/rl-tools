#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ALGORITHMS_PPO_OPERATIONS_GENERIC_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ALGORITHMS_PPO_OPERATIONS_GENERIC_H

#include "ppo.h"
#include "../../../nn/loss_functions/mse/operations_generic.h"
#include "../../../rl/components/on_policy_runner/on_policy_runner.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, rl::algorithms::ppo::Buffers<SPEC>& buffers){
        malloc(device, buffers.current_batch_actions);
        malloc(device, buffers.d_critic_output);
        malloc(device, buffers.d_action_log_prob_d_action);
        malloc(device, buffers.d_action_log_prob_d_action_log_std);
        malloc(device, buffers.rollout_log_std);
    }
    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, rl::algorithms::ppo::Buffers<SPEC>& buffers){
        free(device, buffers.current_batch_actions);
        free(device, buffers.d_critic_output);
        free(device, buffers.d_action_log_prob_d_action);
        free(device, buffers.d_action_log_prob_d_action_log_std);
        free(device, buffers.rollout_log_std);
    }
    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, rl::algorithms::PPO<SPEC>& ppo){
        malloc(device, ppo.actor);
        malloc(device, ppo.critic);
    }
    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, rl::algorithms::PPO<SPEC>& ppo){
        free(device, ppo.actor);
        free(device, ppo.critic);
    }
    template <typename DEVICE, typename SPEC, typename ACTOR_OPTIMIZER, typename CRITIC_OPTIMIZER, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void init(DEVICE& device, rl::algorithms::PPO<SPEC>& ppo, ACTOR_OPTIMIZER& actor_optimizer, CRITIC_OPTIMIZER& critic_optimizer, RNG& rng){
#ifdef RL_TOOLS_DEBUG_RL_ALGORITHMS_PPO_CHECK_INIT
        ppo.initialized = true;
#endif
        init_weights(device, ppo.actor, rng);
        init(device, actor_optimizer);
        reset_forward_state(device, ppo.actor);
        zero_gradient(device, ppo.actor);
        reset_optimizer_state(device, actor_optimizer, ppo.actor);
        auto& last_layer = get_last_layer(ppo.actor);
        set_all(device, last_layer.log_std.parameters, math::log(device.math, SPEC::PARAMETERS::INITIAL_ACTION_STD));
        init_weights(device, ppo.critic, rng);
        init(device, critic_optimizer);
        reset_forward_state(device, ppo.actor);
        zero_gradient(device, ppo.actor);
        reset_optimizer_state(device, critic_optimizer, ppo.critic);
    }
    template <typename DEVICE, typename DATASET_SPEC, typename PPO_PARAMETERS>
    RL_TOOLS_FUNCTION_PLACEMENT void estimate_generalized_advantages(DEVICE& device, rl::components::on_policy_runner::Dataset<DATASET_SPEC>& dataset, PPO_PARAMETERS ppo_parameters_tag){
        using OPR_SPEC = typename DATASET_SPEC::SPEC;
        using BUFFER = decltype(dataset);
        using T = typename DATASET_SPEC::SPEC::TYPE_POLICY::DEFAULT;
        using TI = typename DEVICE::index_t;
        constexpr TI STEPS_PER_ENV = DATASET_SPEC::STEPS_PER_ENV;
        for(TI env_i = 0; env_i < OPR_SPEC::N_ENVIRONMENTS; env_i++){
            T previous_value = get(dataset.all_values, STEPS_PER_ENV * OPR_SPEC::N_ENVIRONMENTS + env_i, 0);
            T previous_advantage = 0;
            for(TI step_forward_i = 0; step_forward_i < STEPS_PER_ENV; step_forward_i++){
                TI step_backward_i = (STEPS_PER_ENV - 1 - step_forward_i);
                TI pos = step_backward_i * OPR_SPEC::N_ENVIRONMENTS + env_i;
                bool terminated = get(dataset.terminated, pos, 0);
                bool truncated = get(dataset.truncated, pos, 0);
#ifdef RL_TOOLS_DEBUG_RL_ALGORITHMS_PPO_GAE_CHECK_TERMINATED_TRUNCATED
                utils::assert_exit(device, !terminated || (terminated && truncated), "terminationn should imply truncation");
#endif
                T current_step_value = get(dataset.values, pos, 0);
                bool terminated_actual = terminated && !PPO_PARAMETERS::IGNORE_TERMINATION;
                T next_step_value = terminated_actual ? 0 : previous_value;

                T td_error = get(dataset.rewards, pos, 0) + PPO_PARAMETERS::GAMMA * next_step_value - current_step_value;
                if(truncated){
                    if(!terminated){ // e.g. time limited or random truncation
                        td_error = 0;
                    }
                    previous_advantage = 0;
                }
                T advantage = PPO_PARAMETERS::LAMBDA * PPO_PARAMETERS::GAMMA * previous_advantage + td_error;
                set(dataset.advantages, pos, 0, advantage);
                set(dataset.target_values, pos, 0, advantage + current_step_value);
                previous_advantage = advantage;
                previous_value = current_step_value;
            }
        }
    }
    template <typename DEVICE, typename BATCH_ADVANTAGES_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void ppo_advantage_normalization(DEVICE& device, Matrix<BATCH_ADVANTAGES_SPEC>& batch_advantages, typename BATCH_ADVANTAGES_SPEC::T& advantage_mean, typename BATCH_ADVANTAGES_SPEC::T& advantage_std){
        using T = typename BATCH_ADVANTAGES_SPEC::T;
        using TI = typename DEVICE::index_t;
        constexpr TI BATCH_SIZE = BATCH_ADVANTAGES_SPEC::ROWS;
        for(TI batch_step_i = 0; batch_step_i < BATCH_SIZE; batch_step_i++){
            T advantage = get(batch_advantages, batch_step_i, 0);
            advantage_mean += advantage;
            advantage_std += advantage * advantage;
        }
        advantage_mean /= BATCH_SIZE;
        advantage_std /= BATCH_SIZE;
        advantage_std = math::sqrt(device.math, math::max(device.math, (T)0, advantage_std - advantage_mean * advantage_mean));
    }
    template <typename DEVICE, typename PPO_SPEC, typename BUFFERS_SPEC, typename BATCH_ACTIONS_SPEC, typename BATCH_ACTIONS_MEAN_SPEC, typename BATCH_ACTION_LOG_PROBS_SPEC, typename BATCH_ADVANTAGES_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void ppo_compute_actor_loss_gradient(DEVICE& device, rl::algorithms::PPO<PPO_SPEC>& ppo, rl::algorithms::ppo::Buffers<BUFFERS_SPEC>& ppo_buffers, Matrix<BATCH_ACTIONS_SPEC>& batch_actions, Matrix<BATCH_ACTIONS_MEAN_SPEC>& batch_actions_mean, Matrix<BATCH_ACTION_LOG_PROBS_SPEC>& batch_action_log_probs, Matrix<BATCH_ADVANTAGES_SPEC>& batch_advantages, typename PPO_SPEC::TYPE_POLICY::DEFAULT advantage_mean, typename PPO_SPEC::TYPE_POLICY::DEFAULT advantage_std, typename PPO_SPEC::TYPE_POLICY::DEFAULT& policy_kl_divergence, typename PPO_SPEC::TYPE_POLICY::DEFAULT& batch_policy_kl_divergence, RNG& rng){
        using T = typename PPO_SPEC::TYPE_POLICY::DEFAULT;
        using TI = typename PPO_SPEC::TI;
        constexpr TI BATCH_SIZE = PPO_SPEC::PARAMETERS::BATCH_SIZE;
        constexpr TI ACTION_DIM = PPO_SPEC::ENVIRONMENT::ACTION_DIM;
        constexpr TI N_AGENTS = PPO_SPEC::ENVIRONMENT::N_AGENTS;
        constexpr TI PER_AGENT_ACTION_DIM = ACTION_DIM / N_AGENTS;
        for(TI batch_step_i = 0; batch_step_i < BATCH_SIZE; batch_step_i++){
            T action_log_prob = 0;
            for(TI action_i = 0; action_i < ACTION_DIM; action_i++){
                T current_action = get(ppo_buffers.current_batch_actions, batch_step_i, action_i);
                T rollout_action = get(batch_actions, batch_step_i, action_i);
                auto& last_layer = get_last_layer(ppo.actor);
                T current_action_log_std = get(device, last_layer.log_std.parameters, action_i % PER_AGENT_ACTION_DIM);
                T current_action_std = math::exp(device.math, current_action_log_std);
                if(PPO_SPEC::PARAMETERS::ADAPTIVE_LEARNING_RATE){
                    T rollout_action_log_std = get(ppo_buffers.rollout_log_std, 0, action_i);
                    T rollout_action_std = math::exp(device.math, rollout_action_log_std);
                    T rollout_action_mean = get(batch_actions_mean, batch_step_i, action_i);
                    T action_mean_diff = rollout_action_mean - current_action;
                    T kl = rollout_action_log_std - current_action_log_std;
                    kl += (current_action_std * current_action_std + action_mean_diff * action_mean_diff)/(2 * rollout_action_std * rollout_action_std + PPO_SPEC::PARAMETERS::POLICY_KL_EPSILON);
                    kl += (T)-0.5;
                    kl = math::max(device.math, kl, (T)0);
                    policy_kl_divergence += kl;
                    batch_policy_kl_divergence += kl;
                }
                action_log_prob += random::normal_distribution::log_prob(device.random, current_action, current_action_log_std, rollout_action);
                set(ppo_buffers.d_action_log_prob_d_action, batch_step_i, action_i, random::normal_distribution::d_log_prob_d_mean(device.random, current_action, current_action_log_std, rollout_action));
                T current_entropy = current_action_log_std + math::log(device.math, 2 * math::PI<T>)/(T)2 + (T)1/(T)2;
                T current_entropy_loss = -(T)1/BATCH_SIZE * PPO_SPEC::PARAMETERS::ACTION_ENTROPY_COEFFICIENT * current_entropy;
                if(PPO_SPEC::PARAMETERS::LEARN_ACTION_STD){
                    T d_entropy_loss_d_current_action_log_std = -(T)1/BATCH_SIZE * PPO_SPEC::PARAMETERS::ACTION_ENTROPY_COEFFICIENT;
                    auto& last_layer2 = get_last_layer(ppo.actor);
                    increment(device, last_layer2.log_std.gradient, d_entropy_loss_d_current_action_log_std, action_i % PER_AGENT_ACTION_DIM);
                    T d_action_log_prob_d_current_action_log_std = random::normal_distribution::d_log_prob_d_log_std(device.random, current_action, current_action_log_std, rollout_action);
                    set(ppo_buffers.d_action_log_prob_d_action_log_std, batch_step_i, action_i, d_action_log_prob_d_current_action_log_std);
                }
            }
            T rollout_action_log_prob = get(batch_action_log_probs, batch_step_i, 0);
            T advantage = get(batch_advantages, batch_step_i, 0);
            if(PPO_SPEC::PARAMETERS::NORMALIZE_ADVANTAGE){
                advantage = (advantage - advantage_mean) / (advantage_std + PPO_SPEC::PARAMETERS::ADVANTAGE_EPSILON);
            }
            T log_ratio = action_log_prob - rollout_action_log_prob;
            T ratio = math::exp(device.math, log_ratio);
            T clipped_ratio = math::clamp(device.math, ratio, 1 - PPO_SPEC::PARAMETERS::EPSILON_CLIP, 1 + PPO_SPEC::PARAMETERS::EPSILON_CLIP);
            bool clipped = ratio != clipped_ratio;
            T normal_advantage = ratio * advantage;
            T clipped_advantage = clipped_ratio * advantage;
            T slippage = 0.0;
            bool ratio_min_switch = normal_advantage - clipped_advantage <= slippage;
            T d_loss_d_pessimistic_surrogate = -(T)1/BATCH_SIZE;
            T d_pessimistic_surrogate_d_normal_advantage = ratio_min_switch ? 1 : 0;
            T d_pessimistic_surrogate_d_clipped_advantage = ratio_min_switch ? 0 : 1;
            T d_normal_advantage_d_ratio = advantage;
            T d_clipped_advantage_d_clipped_ratio = advantage;
            T d_clipped_ratio_d_ratio = clipped ? 0 : 1;
            T d_pessimistic_surrogate_d_ratio = d_pessimistic_surrogate_d_normal_advantage * d_normal_advantage_d_ratio + d_pessimistic_surrogate_d_clipped_advantage * d_clipped_advantage_d_clipped_ratio * d_clipped_ratio_d_ratio;
            T d_loss_d_ratio = d_loss_d_pessimistic_surrogate * d_pessimistic_surrogate_d_ratio;
            T d_ratio_d_action_log_prob = ratio;
            T d_loss_d_action_log_prob = d_loss_d_ratio * d_ratio_d_action_log_prob;
            for(TI action_i = 0; action_i < ACTION_DIM; action_i++){
                multiply(ppo_buffers.d_action_log_prob_d_action, batch_step_i, action_i, d_loss_d_action_log_prob);
                if(PPO_SPEC::PARAMETERS::LEARN_ACTION_STD){
                    T current_d_action_log_prob_d_action_log_std = get(ppo_buffers.d_action_log_prob_d_action_log_std, batch_step_i, action_i);
                    auto& last_layer3 = get_last_layer(ppo.actor);
                    increment(device, last_layer3.log_std.gradient, d_loss_d_action_log_prob * current_d_action_log_prob_d_action_log_std, action_i % PER_AGENT_ACTION_DIM);
                }
            }
        }
    }

    template <typename DEVICE_SOURCE, typename DEVICE_TARGET, typename PPO_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void copy(DEVICE_SOURCE& device_source, DEVICE_TARGET& device_target, const rl::algorithms::PPO<PPO_SPEC>& source, rl::algorithms::PPO<PPO_SPEC>& target){
        copy(device_source, device_target, source.actor, target.actor);
        copy(device_source, device_target, source.critic, target.critic);
#ifdef RL_TOOLS_DEBUG_RL_ALGORITHMS_PPO_CHECK_INIT
        target.initialized = source.initialized;
#endif
    }
    template <typename DEVICE, typename SPEC_1, typename SPEC_2>
    RL_TOOLS_FUNCTION_PLACEMENT typename SPEC_1::TYPE_POLICY::DEFAULT abs_diff(DEVICE& device, rl::algorithms::PPO<SPEC_1>& p1, rl::algorithms::PPO<SPEC_2>& p2){
        using T = typename SPEC_1::TYPE_POLICY::DEFAULT;
        T acc = 0;
        acc += abs_diff(device, p1.actor, p2.actor);
        acc += abs_diff(device, p1.critic, p2.critic);
        return acc;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
