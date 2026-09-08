#pragma once

namespace rl_tools::rl::environments::l2f_visual::training{
    template<typename T, typename TI>
    struct PolicyGradientStatistics{
        T actor_loss = 0;
        T entropy = 0;
        T approx_kl = 0;
        T ratio = 0;
        T advantage_mean = 0;
        T advantage_std = 0;
        TI samples = 0;
        TI clipped_samples = 0;
    };
}

namespace rl_tools{
    template <typename DEVICE, typename PPO_SPEC, typename BUFFERS_SPEC, typename ACTION_SPEC, typename LOG_PROB_SPEC, typename ADVANTAGE_SPEC, typename PPO_SPEC::TI T_UPDATE_SAMPLES>
    RL_TOOLS_FUNCTION_PLACEMENT auto training_hyperdrone_actor_loss_gradient(DEVICE& device, rl::algorithms::PPO<PPO_SPEC>& ppo, rl::algorithms::ppo::Buffers<BUFFERS_SPEC>& ppo_buffers, const Matrix<ACTION_SPEC>& batch_actions, const Matrix<LOG_PROB_SPEC>& batch_action_log_probs, const Matrix<ADVANTAGE_SPEC>& batch_advantages, utils::typing::integral_constant<typename PPO_SPEC::TI, T_UPDATE_SAMPLES>){
        using T = typename PPO_SPEC::TYPE_POLICY::DEFAULT;
        using TI = typename PPO_SPEC::TI;
        constexpr TI BATCH_SIZE = PPO_SPEC::PARAMETERS::BATCH_SIZE;
        constexpr TI ACTION_DIM = PPO_SPEC::ENVIRONMENT::ACTION_DIM;
        static_assert(PPO_SPEC::ENVIRONMENT::N_AGENTS == 1, "Independent action standard deviations are required");
        static_assert(T_UPDATE_SAMPLES >= BATCH_SIZE);
        static_assert(ACTION_SPEC::ROWS == BATCH_SIZE && ACTION_SPEC::COLS == ACTION_DIM);
        static_assert(LOG_PROB_SPEC::ROWS == BATCH_SIZE && LOG_PROB_SPEC::COLS == 1);
        static_assert(ADVANTAGE_SPEC::ROWS == BATCH_SIZE && ADVANTAGE_SPEC::COLS == 1);
        rl::environments::l2f_visual::training::PolicyGradientStatistics<T, TI> statistics;
        auto& last_layer = get_last_layer(ppo.actor);
        T advantage_mean = 0, advantage_std = 0;
        for(TI i = 0; i < BATCH_SIZE; i++){
            T adv = get(batch_advantages, i, 0);
            advantage_mean += adv;
            advantage_std += adv * adv;
        }
        advantage_mean /= BATCH_SIZE;
        advantage_std /= BATCH_SIZE;
        advantage_std = math::sqrt(device.math, math::max(device.math, (T)0, advantage_std - advantage_mean * advantage_mean));
        statistics.advantage_mean += advantage_mean;
        statistics.advantage_std += advantage_std;

        for(TI batch_step_i = 0; batch_step_i < BATCH_SIZE; batch_step_i++){
            T action_log_prob = 0;
            T action_entropy = 0;
            for(TI action_i = 0; action_i < ACTION_DIM; action_i++){
                T current_action = get(ppo_buffers.current_batch_actions, batch_step_i, action_i);
                T rollout_action = get(batch_actions, batch_step_i, action_i);
                T current_action_log_std = get(device, last_layer.log_std.parameters, action_i);
                action_log_prob += random::normal_distribution::log_prob(device.random, current_action, current_action_log_std, rollout_action);
                action_entropy += current_action_log_std + math::log(device.math, static_cast<T>(2) * math::PI<T>) / static_cast<T>(2) + static_cast<T>(0.5);
                set(ppo_buffers.d_action_log_prob_d_action, batch_step_i, action_i, random::normal_distribution::d_log_prob_d_mean(device.random, current_action, current_action_log_std, rollout_action));
                if(PPO_SPEC::PARAMETERS::LEARN_ACTION_STD){
                    T d_entropy_loss_d_current_action_log_std = -(T)1/T_UPDATE_SAMPLES * PPO_SPEC::PARAMETERS::ACTION_ENTROPY_COEFFICIENT;
                    increment(device, last_layer.log_std.gradient, d_entropy_loss_d_current_action_log_std, action_i);
                    set(ppo_buffers.d_action_log_prob_d_action_log_std, batch_step_i, action_i, random::normal_distribution::d_log_prob_d_log_std(device.random, current_action, current_action_log_std, rollout_action));
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
            statistics.actor_loss += -action_log_prob * advantage;
            statistics.entropy += action_entropy;
            statistics.approx_kl += (ratio - (T)1) - log_ratio;
            statistics.ratio += ratio;
            statistics.clipped_samples += clipped ? 1 : 0;
            statistics.samples++;
            T d_loss_d_action_log_prob = -advantage / static_cast<T>(T_UPDATE_SAMPLES);
            for(TI action_i = 0; action_i < ACTION_DIM; action_i++){
                multiply(ppo_buffers.d_action_log_prob_d_action, batch_step_i, action_i, d_loss_d_action_log_prob);
                if(PPO_SPEC::PARAMETERS::LEARN_ACTION_STD){
                    T current_d = get(ppo_buffers.d_action_log_prob_d_action_log_std, batch_step_i, action_i);
                    increment(device, last_layer.log_std.gradient, d_loss_d_action_log_prob * current_d, action_i);
                }
            }
        }
        return statistics;
    }
}
