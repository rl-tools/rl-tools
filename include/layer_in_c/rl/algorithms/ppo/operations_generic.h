#ifndef LAYER_IN_C_RL_ALGORITHMS_PPO_OPERATIONS_GENERIC_H
#define LAYER_IN_C_RL_ALGORITHMS_PPO_OPERATIONS_GENERIC_H

#include "ppo.h"
#include <layer_in_c/rl/components/on_policy_runner/on_policy_runner.h>

namespace layer_in_c{
    template <typename DEVICE, typename SPEC>
    void malloc(DEVICE& device, rl::algorithms::ppo::Buffers<SPEC>& buffers){
        malloc(device, buffers.current_batch_actions);
        malloc(device, buffers.d_batch_observations);
        malloc(device, buffers.d_action_log_prob_d_action);
        malloc(device, buffers.d_action_log_prob_d_action_log_std);
    }
    template <typename DEVICE, typename SPEC>
    void free(DEVICE& device, rl::algorithms::ppo::Buffers<SPEC>& buffers){
        free(device, buffers.current_batch_actions);
        free(device, buffers.d_batch_observations);
        free(device, buffers.d_action_log_prob_d_action);
        free(device, buffers.d_action_log_prob_d_action_log_std);
    }
    template <typename DEVICE, typename SPEC>
    void malloc(DEVICE& device, rl::algorithms::PPO<SPEC>& ppo){
        malloc(device, ppo.actor);
        malloc(device, ppo.critic);
    }
    template <typename DEVICE, typename SPEC>
    void free(DEVICE& device, rl::algorithms::PPO<SPEC>& ppo){
        free(device, ppo.actor);
        free(device, ppo.critic);
    }
    template <typename DEVICE, typename SPEC, typename OPTIMIZER, typename RNG>
    void init(DEVICE& device, rl::algorithms::PPO<SPEC>& ppo, OPTIMIZER& optimizer, RNG& rng){
        init_weights(device, ppo.actor, rng);
        reset_optimizer_state(device, ppo.actor, optimizer);
        set_all(device, ppo.actor.action_log_std.parameters, math::log(typename DEVICE::SPEC::MATH(), SPEC::PARAMETERS::INITIAL_ACTION_STD));
        init_weights(device, ppo.critic, rng);
        reset_optimizer_state(device, ppo.critic, optimizer);
#ifdef LAYER_IN_C_DEBUG_RL_ALGORITHMS_PPO_CHECK_INIT
        ppo.initialized = true;
#endif
    }
    template <typename DEVICE, typename PPO_SPEC, typename OPR_SPEC, auto STEPS_PER_ENV>
    void estimate_generalized_advantages(DEVICE& device, rl::algorithms::PPO<PPO_SPEC>& ppo, rl::components::on_policy_runner::Buffer<rl::components::on_policy_runner::BufferSpecification<OPR_SPEC, STEPS_PER_ENV>>& buffer){
#ifdef LAYER_IN_C_DEBUG_RL_ALGORITHMS_PPO_CHECK_INIT
        utils::assert_exit(device, ppo.initialized, "PPO not initialized");
#endif
        using BUFFER = decltype(buffer);
        using T = typename PPO_SPEC::T;
        using TI = typename PPO_SPEC::TI;
        evaluate(device, ppo.critic, buffer.all_observations, buffer.all_values);
        for(TI env_i = 0; env_i < OPR_SPEC::N_ENVIRONMENTS; env_i++){
            T previous_value = get(buffer.all_values, STEPS_PER_ENV * OPR_SPEC::N_ENVIRONMENTS + env_i, 0);
            T previous_advantage = 0;
            for(TI step_forward_i = 0; step_forward_i < STEPS_PER_ENV; step_forward_i++){
                TI step_backward_i = (STEPS_PER_ENV - 1 - step_forward_i);
                TI pos = step_backward_i * OPR_SPEC::N_ENVIRONMENTS + env_i;
                bool terminated = get(buffer.terminated, pos, 0);
                bool truncated = get(buffer.truncated, pos, 0);
#ifdef LAYER_IN_C_DEBUG_RL_ALGORITHMS_PPO_GAE_CHECK_TERMINATED_TRUNCATED
                utils::assert_exit(device, !terminated || terminated && truncated, "terminationn should imply truncation");
#endif
                T current_step_value = get(buffer.values, pos, 0);
//                T next_step_value = terminated || truncated ? 0 : previous_value;
                T next_step_value = terminated && !PPO_SPEC::PARAMETERS::IGNORE_TERMINATION ? 0 : previous_value;

                T td_error = get(buffer.rewards, pos, 0) + PPO_SPEC::PARAMETERS::GAMMA * next_step_value - current_step_value;
//                previous_advantage = terminated || truncated ? 0 : previous_advantage;
                if(truncated){
                    if(!terminated){ // e.g. time limited or random truncation
                        td_error = 0;
                    }
                    previous_advantage = 0;
                }
                T advantage = PPO_SPEC::PARAMETERS::LAMBDA * PPO_SPEC::PARAMETERS::GAMMA * previous_advantage + td_error;
                set(buffer.advantages, pos, 0, advantage);
                set(buffer.target_values, pos, 0, advantage + current_step_value);
                previous_advantage = advantage;
                previous_value = current_step_value;
            }
        }
    }
    template <typename DEVICE, typename PPO_SPEC, typename OPR_SPEC, auto STEPS_PER_ENV, typename OPTIMIZER, typename RNG>
    void train(DEVICE& device, rl::algorithms::PPO<PPO_SPEC>& ppo, rl::components::on_policy_runner::Buffer<rl::components::on_policy_runner::BufferSpecification<OPR_SPEC, STEPS_PER_ENV>>& buffer, OPTIMIZER& optimizer, rl::algorithms::ppo::Buffers<PPO_SPEC>& ppo_buffers, typename PPO_SPEC::ACTOR_NETWORK_TYPE::template BuffersForwardBackward<PPO_SPEC::BATCH_SIZE>& actor_buffers, typename PPO_SPEC::CRITIC_NETWORK_TYPE::template BuffersForwardBackward<PPO_SPEC::BATCH_SIZE>& critic_buffers, RNG& rng){
#ifdef LAYER_IN_C_DEBUG_RL_ALGORITHMS_PPO_CHECK_INIT
        utils::assert_exit(device, ppo.initialized, "PPO not initialized");
#endif
        using T = typename PPO_SPEC::T;
        using TI = typename PPO_SPEC::TI;
        static_assert(utils::typing::is_same_v<typename PPO_SPEC::ENVIRONMENT, typename OPR_SPEC::ENVIRONMENT>, "environment mismatch");
        using BUFFER = rl::components::on_policy_runner::Buffer<rl::components::on_policy_runner::BufferSpecification<OPR_SPEC, STEPS_PER_ENV>>;
        static_assert(BUFFER::STEPS_TOTAL > 0);
        constexpr TI N_EPOCHS = PPO_SPEC::PARAMETERS::N_EPOCHS;
        constexpr TI BATCH_SIZE = PPO_SPEC::BATCH_SIZE;
        constexpr TI N_BATCHES = BUFFER::STEPS_TOTAL/BATCH_SIZE;
        static_assert(N_BATCHES > 0);
        constexpr TI ACTION_DIM = OPR_SPEC::ENVIRONMENT::ACTION_DIM;
        constexpr TI OBSERVATION_DIM = OPR_SPEC::ENVIRONMENT::OBSERVATION_DIM;
        // batch needs observations, original log-probs, advantages
        for(TI epoch_i = 0; epoch_i < N_EPOCHS; epoch_i++){
            // shuffling
            for(TI buffer_i = 0; buffer_i < BUFFER::STEPS_TOTAL; buffer_i++){
                TI sample_index = random::uniform_int_distribution(typename DEVICE::SPEC::RANDOM(), buffer_i, BUFFER::STEPS_TOTAL-1, rng);
                {
                    auto target_row = row(device, buffer.observations, buffer_i);
                    auto source_row = row(device, buffer.observations, sample_index);
                    swap(device, target_row, source_row);
                }
                {
                    auto target_row = row(device, buffer.actions, buffer_i);
                    auto source_row = row(device, buffer.actions, sample_index);
                    swap(device, target_row, source_row);
                }
                swap(device, buffer.advantages      , buffer.advantages      , buffer_i, 0, sample_index, 0);
                swap(device, buffer.action_log_probs, buffer.action_log_probs, buffer_i, 0, sample_index, 0);
                swap(device, buffer.target_values   , buffer.target_values   , buffer_i, 0, sample_index, 0);
            }
            for(TI batch_i = 0; batch_i < N_BATCHES; batch_i++){
                zero_gradient(device, ppo.critic);
                zero_gradient(device, ppo.actor); // has to be reset before accumulating the action-log-std gradient

                auto batch_offset = batch_i * BATCH_SIZE;
                auto batch_observations     = view(device, buffer.observations    , matrix::ViewSpec<BATCH_SIZE, OBSERVATION_DIM>(), batch_offset, 0);
                auto batch_actions          = view(device, buffer.actions         , matrix::ViewSpec<BATCH_SIZE, ACTION_DIM     >(), batch_offset, 0);
                auto batch_action_log_probs = view(device, buffer.action_log_probs, matrix::ViewSpec<BATCH_SIZE, 1              >(), batch_offset, 0);
                auto batch_advantages       = view(device, buffer.advantages      , matrix::ViewSpec<BATCH_SIZE, 1              >(), batch_offset, 0);
                auto batch_target_values    = view(device, buffer.target_values   , matrix::ViewSpec<BATCH_SIZE, 1              >(), batch_offset, 0);

                T advantage_mean = 0;
                T advantage_std = 0;
                for(TI batch_step_i = 0; batch_step_i < BATCH_SIZE; batch_step_i++){
                    T advantage = get(batch_advantages, batch_step_i, 0);
                    advantage_mean += advantage;
                    advantage_std += advantage * advantage;
                }
                advantage_mean /= BATCH_SIZE;
                advantage_std /= BATCH_SIZE;

                advantage_std = math::sqrt(typename DEVICE::SPEC::MATH(), advantage_std - advantage_mean * advantage_mean);
//                add_scalar(device, device.logger, "ppo/advantage/mean", advantage_mean);
//                add_scalar(device, device.logger, "ppo/advantage/std", advantage_std);
                forward(device, ppo.actor, batch_observations, ppo_buffers.current_batch_actions);
//                auto abs_diff = abs_diff(device, batch_actions, buffer.actions);

                for(TI batch_step_i = 0; batch_step_i < BATCH_SIZE; batch_step_i++){
                    T action_log_prob = 0;
                    for(TI action_i = 0; action_i < ACTION_DIM; action_i++){
                        T current_action = get(ppo_buffers.current_batch_actions, batch_step_i, action_i);
                        T action = get(batch_actions, batch_step_i, action_i);
                        T action_log_std = get(ppo.actor.action_log_std.parameters, 0, action_i);
                        T action_std = math::exp(typename DEVICE::SPEC::MATH(), action_log_std);
                        T action_diff_by_action_std = (current_action - action) / action_std;
                        action_log_prob += -0.5 * action_diff_by_action_std * action_diff_by_action_std - action_log_std - 0.5 * math::log(typename DEVICE::SPEC::MATH(), 2 * math::PI<T>);
                        set(ppo_buffers.d_action_log_prob_d_action, batch_step_i, action_i, - action_diff_by_action_std / action_std);
//                      d_action_log_prob_d_action_std =  (-action_diff_by_action_std) * (-action_diff_by_action_std)      / action_std - 1 / action_std)
//                      d_action_log_prob_d_action_std = ((-action_diff_by_action_std) * (-action_diff_by_action_std) - 1) / action_std)
//                      d_action_log_prob_d_action_std = (action_diff_by_action_std * action_diff_by_action_std - 1) / action_std
//                      d_action_log_prob_d_action_log_std = (action_diff_by_action_std * action_diff_by_action_std - 1) / action_std * exp(action_log_std)
//                      d_action_log_prob_d_action_log_std = (action_diff_by_action_std * action_diff_by_action_std - 1) / action_std * action_std
//                      d_action_log_prob_d_action_log_std =  action_diff_by_action_std * action_diff_by_action_std - 1
                        T current_entropy = action_log_std + math::log(typename DEVICE::SPEC::MATH(), 2 * math::PI<T>)/(T)2 + (T)1/(T)2;
                        T current_entropy_loss = -(T)1/BATCH_SIZE * PPO_SPEC::PARAMETERS::ACTION_ENTROPY_COEFFICIENT * current_entropy;
                        // todo: think about possible implementation detail: clipping entropy bonus as well (because it changes the distribution)
                        if(PPO_SPEC::PARAMETERS::LEARN_ACTION_STD){
                            T current_d_entropy_loss_d_action_log_std = -(T)1/BATCH_SIZE * PPO_SPEC::PARAMETERS::ACTION_ENTROPY_COEFFICIENT;
                            increment(ppo.actor.action_log_std.gradient, 0, action_i, current_d_entropy_loss_d_action_log_std);
                            T current_d_action_log_prob_d_action_log_std = action_diff_by_action_std * action_diff_by_action_std - 1;
                            set(ppo_buffers.d_action_log_prob_d_action_log_std, batch_step_i, action_i, current_d_action_log_prob_d_action_log_std);
                        }
                    }
                    T old_action_log_prob = get(batch_action_log_probs, batch_step_i, 0);
                    T advantage = get(batch_advantages, batch_step_i, 0);
                    if(PPO_SPEC::PARAMETERS::NORMALIZE_ADVANTAGE){
                        advantage = (advantage - advantage_mean) / (advantage_std + PPO_SPEC::PARAMETERS::ADVANTAGE_EPSILON);
                    }
                    T log_ratio = action_log_prob - old_action_log_prob;
                    T ratio = math::exp(typename DEVICE::SPEC::MATH(), log_ratio);
                    // todo: test relative clipping (clipping in log space makes more sense thatn clipping in exp space)
                    T clipped_ratio = math::clamp(ratio, 1 - PPO_SPEC::PARAMETERS::EPSILON_CLIP, 1 + PPO_SPEC::PARAMETERS::EPSILON_CLIP);
                    T normal_advantage = ratio * advantage;
                    T clipped_advantage = clipped_ratio * advantage;
                    T slippage = 0.0;
                    bool ratio_min_switch = normal_advantage - clipped_advantage <= slippage;
                    T clipped_surrogate = ratio_min_switch ? normal_advantage : clipped_advantage;

                    T d_loss_d_clipped_surrogate = -(T)1/BATCH_SIZE;
                    T d_clipped_surrogate_d_ratio = ratio_min_switch ? advantage : 0;
                    T d_ratio_d_action_log_prob = ratio;
                    T d_loss_d_action_log_prob = d_loss_d_clipped_surrogate * d_clipped_surrogate_d_ratio * d_ratio_d_action_log_prob;
                    for(TI action_i = 0; action_i < ACTION_DIM; action_i++){
                        multiply(ppo_buffers.d_action_log_prob_d_action, batch_step_i, action_i, d_loss_d_action_log_prob);
                        if(PPO_SPEC::PARAMETERS::LEARN_ACTION_STD){
                            T current_d_action_log_prob_d_action_log_std = get(ppo_buffers.d_action_log_prob_d_action_log_std, batch_step_i, action_i);
                            increment(ppo.actor.action_log_std.gradient, 0, action_i, d_loss_d_action_log_prob * current_d_action_log_prob_d_action_log_std);
                        }
                    }
                }
                backward(device, ppo.actor, batch_observations, ppo_buffers.d_action_log_prob_d_action, ppo_buffers.d_batch_observations, actor_buffers);
                forward_backward_mse(device, ppo.critic, batch_observations, batch_target_values, critic_buffers);
                update(device, ppo.actor, optimizer);
                update(device, ppo.critic, optimizer);
            }
        }
    }

}
#endif
