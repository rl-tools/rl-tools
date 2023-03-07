#ifndef LAYER_IN_C_RL_ALGORITHMS_PPO_OPERATIONS_GENERIC_H
#define LAYER_IN_C_RL_ALGORITHMS_PPO_OPERATIONS_GENERIC_H

#include "ppo.h"
#include <layer_in_c/rl/components/on_policy_runner/on_policy_runner.h>

namespace layer_in_c{
    template <typename DEVICE, typename SPEC>
    void malloc(DEVICE& device, rl::algorithms::PPO<SPEC>& ppo){
        malloc(device, ppo.actor);
        malloc(device, ppo.actor_log_std);
        malloc(device, ppo.critic);
    }
    template <typename DEVICE, typename SPEC>
    void free(DEVICE& device, rl::algorithms::PPO<SPEC>& ppo){
        free(device, ppo.actor);
        free(device, ppo.actor_log_std);
        free(device, ppo.critic);
    }
    template <typename DEVICE, typename SPEC, typename OPTIMIZER, typename RNG>
    void init(DEVICE& device, rl::algorithms::PPO<SPEC>& ppo, OPTIMIZER& optimizer, RNG& rng){
        init_weights(device, ppo.actor, rng);
        reset_optimizer_state(device, ppo.actor, optimizer);
        set_all(device, ppo.actor_log_std, SPEC::PARAMETERS::ACTOR_LOG_STD);
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
                T next_step_value = terminated ? 0 : previous_value;
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
    void train(DEVICE& device, rl::algorithms::PPO<PPO_SPEC>& ppo, rl::components::on_policy_runner::Buffer<rl::components::on_policy_runner::BufferSpecification<OPR_SPEC, STEPS_PER_ENV>>& buffer, OPTIMIZER& optimizer, typename PPO_SPEC::ACTOR_NETWORK_TYPE::template BuffersForwardBackward<PPO_SPEC::BATCH_SIZE>& actor_buffers, typename PPO_SPEC::CRITIC_NETWORK_TYPE::template BuffersForwardBackward<PPO_SPEC::BATCH_SIZE>& critic_buffers, RNG& rng){
#ifdef LAYER_IN_C_DEBUG_RL_ALGORITHMS_PPO_CHECK_INIT
        utils::assert_exit(device, ppo.initialized, "PPO not initialized");
#endif
        using T = typename PPO_SPEC::T;
        using TI = typename PPO_SPEC::TI;
        static_assert(utils::typing::is_same_v<typename PPO_SPEC::ENVIRONMENT, typename OPR_SPEC::ENVIRONMENT>, "environment mismatch");
        using BUFFER = rl::components::on_policy_runner::Buffer<rl::components::on_policy_runner::BufferSpecification<OPR_SPEC, STEPS_PER_ENV>>;
        constexpr TI N_EPOCHS = PPO_SPEC::PARAMETERS::N_EPOCHS;
        constexpr TI BATCH_SIZE = PPO_SPEC::BATCH_SIZE;
        constexpr TI ACTION_DIM = OPR_SPEC::ENVIRONMENT::ACTION_DIM;
        constexpr TI OBSERVATION_DIM = OPR_SPEC::ENVIRONMENT::OBSERVATION_DIM;
        // batch needs observations, original log-probs, advantages
        Matrix<matrix::Specification<T, TI, BATCH_SIZE, ACTION_DIM>> batch_actions;
        Matrix<matrix::Specification<T, TI, BATCH_SIZE, ACTION_DIM>> current_batch_actions;
        Matrix<matrix::Specification<T, TI, 1, BATCH_SIZE>> batch_advantages;
        Matrix<matrix::Specification<T, TI, 1, BATCH_SIZE>> batch_action_log_probs;
        Matrix<matrix::Specification<T, TI, BATCH_SIZE, 1>> batch_target_values;
        Matrix<matrix::Specification<T, TI, BATCH_SIZE, OBSERVATION_DIM>> batch_observations;
        Matrix<matrix::Specification<T, TI, BATCH_SIZE, OBSERVATION_DIM>> d_batch_observations;
        Matrix<matrix::Specification<T, TI, BATCH_SIZE, ACTION_DIM>> d_action_log_prob_d_action;
//        Matrix<matrix::Specification<T, TI, 1, ACTION_DIM>> d_action_log_prob_d_action_log_std;
        malloc(device, batch_actions);
        malloc(device, current_batch_actions);
        malloc(device, batch_advantages);
        malloc(device, batch_action_log_probs);
        malloc(device, batch_target_values);
        malloc(device, batch_observations);
        malloc(device, d_batch_observations);
        malloc(device, d_action_log_prob_d_action);
//        malloc(device, d_action_log_prob_d_action_log_std);
        for(TI epoch_i = 0; epoch_i < N_EPOCHS; epoch_i++){
            constexpr TI N_BATCHES = BUFFER::STEPS_TOTAL/BATCH_SIZE;
            static_assert(N_BATCHES > 0);
            for(TI batch_i = 0; batch_i < N_BATCHES; batch_i++){
                for(TI batch_step_i = 0; batch_step_i < BATCH_SIZE; batch_step_i++){
                    static_assert(BUFFER::STEPS_TOTAL > 0);
                    TI sample_index = random::uniform_int_distribution(typename DEVICE::SPEC::RANDOM(), (typename DEVICE::index_t) 0, BUFFER::STEPS_TOTAL-1, rng);
//                    TI sample_index = batch_i * BATCH_SIZE + batch_step_i; // todo: make stochastic again
                    {
                        auto target_row = row(device, batch_observations, batch_step_i);
                        auto source_row = row(device, buffer.observations, sample_index);
                        copy(device, device, target_row, source_row);
                    }
                    {
                        auto target_row = row(device, batch_actions, batch_step_i);
                        auto source_row = row(device, buffer.actions, sample_index);
                        copy(device, device, target_row, source_row);
                    }
                    set(batch_advantages, 0, batch_step_i, get(buffer.advantages, sample_index, 0));
                    set(batch_action_log_probs, 0, batch_step_i, get(buffer.action_log_probs, sample_index, 0));
                    set(batch_target_values, batch_step_i, 0, get(buffer.target_values, sample_index, 0));
                }
                forward(device, ppo.actor, batch_observations, current_batch_actions);
//                auto abs_diff = abs_diff(device, batch_actions, buffer.actions);
                for(TI batch_step_i = 0; batch_step_i < BATCH_SIZE; batch_step_i++){
                    T action_log_prob = 0;
                    for(TI action_i = 0; action_i < ACTION_DIM; action_i++){
                        T current_action = get(current_batch_actions, batch_step_i, action_i);
                        T action = get(batch_actions, batch_step_i, action_i);
                        T action_std = math::exp(typename DEVICE::SPEC::MATH(), get(ppo.actor_log_std, 0, action_i));
                        T action_diff_by_action_std = (current_action - action) / action_std;
                        action_log_prob += -0.5 * action_diff_by_action_std * action_diff_by_action_std - math::log(typename DEVICE::SPEC::MATH(), action_std) - 0.5 * math::log(typename DEVICE::SPEC::MATH(), 2 * math::PI<T>);
                        set(d_action_log_prob_d_action, batch_step_i, action_i, - action_diff_by_action_std / action_std);
//                        set(d_action_log_prob_d_action_log_std, 0, action_i, -1 / action_std + action_diff_by_action_std * action_diff_by_action_std / action_std);
                    }
                    T old_action_log_prob = get(batch_action_log_probs, 0, batch_step_i);
                    T advantage = get(batch_advantages, 0, batch_step_i);
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
                        multiply(d_action_log_prob_d_action, batch_step_i, action_i, d_loss_d_action_log_prob);
                    }
                }
                zero_gradient(device, ppo.actor);
                backward(device, ppo.actor, batch_observations, d_action_log_prob_d_action, d_batch_observations, actor_buffers);
                update(device, ppo.actor, optimizer);

                zero_gradient(device, ppo.critic);
                forward_backward_mse(device, ppo.critic, batch_observations, batch_target_values, critic_buffers);
                update(device, ppo.critic, optimizer);
            }
        }
        free(device, batch_actions);
        free(device, batch_advantages);
        free(device, batch_action_log_probs);
        free(device, batch_target_values);
        free(device, batch_observations);
        free(device, d_batch_observations);
        free(device, d_action_log_prob_d_action);
//        free(device, d_action_log_prob_d_action_log_std);
    }

}
#endif
