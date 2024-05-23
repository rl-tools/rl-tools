#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_UTILS_EVALUATION_OPERATIONS_GENERIC_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_UTILS_EVALUATION_OPERATIONS_GENERIC_H
/*
 * This file relies on the environments methods hence it should be included after the operations of the environments that it will be used with
 */

#include "../../../math/operations_generic.h"
#include "../../environments/operations_generic.h"

#include "evaluation.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    namespace rl::utils::evaluation{
        template <typename SPEC>
        auto& get_data(Result<SPEC> result, Data<SPEC>& alternative_data){
            if constexpr(SPEC::INCLUDE_DATA){
                return result.data;
            }
            else{
                return alternative_data;
            }
        }
    }
    template<typename DEVICE, typename ENVIRONMENT, typename UI, typename POLICY, typename RNG, typename SPEC, typename POLICY_EVALUATION_BUFFERS>
    void evaluate(DEVICE& device, ENVIRONMENT&, UI& ui, const POLICY& policy, rl::utils::evaluation::Result<SPEC>& results, POLICY_EVALUATION_BUFFERS& policy_evaluation_buffers, RNG &rng, bool deterministic = false){
        using T = typename POLICY::T;
        using TI = typename DEVICE::index_t;
        static_assert(ENVIRONMENT::OBSERVATION_DIM == POLICY::INPUT_DIM, "Observation and policy input dimensions must match");
        static_assert(ENVIRONMENT::ACTION_DIM == POLICY::OUTPUT_DIM || (2*ENVIRONMENT::ACTION_DIM == POLICY::OUTPUT_DIM), "Action and policy output dimensions must match");
        static constexpr bool STOCHASTIC_POLICY = POLICY::OUTPUT_DIM == 2*ENVIRONMENT::ACTION_DIM;
        results.returns_mean = 0;
        results.returns_std = 0;
        results.episode_length_mean = 0;
        results.episode_length_std = 0;

        MatrixStatic<matrix::Specification<T, TI, SPEC::N_EPISODES, ENVIRONMENT::ACTION_DIM * (STOCHASTIC_POLICY ? 2 : 1)>> actions_buffer_full;
        MatrixStatic<matrix::Specification<T, TI, SPEC::N_EPISODES, ENVIRONMENT::OBSERVATION_DIM>> observations;
        auto actions_buffer = view(device, actions_buffer_full, matrix::ViewSpec<SPEC::N_EPISODES, ENVIRONMENT::ACTION_DIM>{});

        rl::utils::evaluation::Data<SPEC> local_data;
        auto& data = get_data(results, local_data);

        ENVIRONMENT envs[SPEC::N_EPISODES];


        for(TI env_i = 0; env_i < SPEC::N_EPISODES; env_i++){
            auto& env = envs[env_i];
            malloc(device, env);
            init(device, env);
            results.returns[env_i] = 0;
            results.episode_length[env_i] = 0;
            auto& current_state = data.states[env_i][0];
            if(deterministic) {
                rl_tools::initial_state(device, env, current_state);
            }
            else{
                sample_initial_state(device, env, current_state, rng);
            }
        }
        for(TI step_i = 0; step_i < SPEC::STEP_LIMIT; step_i++) {
            for(TI env_i = 0; env_i < SPEC::N_EPISODES; env_i++) {
                auto observation = row(device, observations, env_i);
                auto& state = data.states[env_i][step_i];
                auto& env = envs[env_i];
                observe(device, env, state, observation, rng);
            }
            constexpr TI BATCH_SIZE = POLICY_EVALUATION_BUFFERS::BATCH_SIZE;
            constexpr TI NUM_FORWARD_PASSES = SPEC::N_EPISODES / BATCH_SIZE;
            for(TI forward_pass_i = 0; forward_pass_i < NUM_FORWARD_PASSES; forward_pass_i++){
                auto observations_chunk = view(device, observations, matrix::ViewSpec<BATCH_SIZE, ENVIRONMENT::OBSERVATION_DIM>{}, forward_pass_i*BATCH_SIZE, 0);
                auto actions_buffer_chunk = view(device, actions_buffer_full, matrix::ViewSpec<BATCH_SIZE, ENVIRONMENT::ACTION_DIM * (STOCHASTIC_POLICY ? 2 : 1)>{}, forward_pass_i*BATCH_SIZE, 0);
                evaluate(device, policy, observations_chunk, actions_buffer_chunk, policy_evaluation_buffers, rng);
            }
            if constexpr(SPEC::N_EPISODES % BATCH_SIZE != 0){
                auto observations_chunk = view(device, observations, matrix::ViewSpec<SPEC::N_EPISODES % BATCH_SIZE, ENVIRONMENT::OBSERVATION_DIM>{}, NUM_FORWARD_PASSES*BATCH_SIZE, 0);
                auto actions_buffer_chunk = view(device, actions_buffer_full, matrix::ViewSpec<SPEC::N_EPISODES % BATCH_SIZE, ENVIRONMENT::ACTION_DIM * (STOCHASTIC_POLICY ? 2 : 1)>{}, NUM_FORWARD_PASSES*BATCH_SIZE, 0);
                evaluate(device, policy, observations_chunk, actions_buffer_chunk, policy_evaluation_buffers, rng);
            }



            if constexpr(STOCHASTIC_POLICY){ // todo: This is a special case for SAC, will be uneccessary once (https://github.com/rl-tools/rl-tools/blob/72a59eabf4038502c3be86a4f772bd72526bdcc8/TODO.md?plain=1#L22) is implemented
                for(TI env_i = 0; env_i < SPEC::N_EPISODES; env_i++) {
                    for (TI action_i = 0; action_i < ENVIRONMENT::ACTION_DIM; action_i++) {
                        set(actions_buffer, env_i, action_i, math::tanh<T>(device.math, get(actions_buffer, env_i, action_i)));
                    }
                }
            }
            for(TI env_i = 0; env_i < SPEC::N_EPISODES; env_i++) {
                for (TI action_i = 0; action_i < ENVIRONMENT::ACTION_DIM; action_i++) {
                    data.actions[env_i][step_i][action_i] = get(actions_buffer, env_i, action_i);
                }
            }
            for(TI env_i = 0; env_i < SPEC::N_EPISODES; env_i++) {
                if(step_i > 0){
                    if(data.terminated[env_i][step_i-1]){
                        continue;
                    }
                }
                auto& env = envs[env_i];
                typename ENVIRONMENT::State next_state;
                auto& state = data.states[env_i][step_i];
                auto action = row(device, actions_buffer, env_i);
                T dt = step(device, env, state, action, next_state, rng);
                set_state(device, env, ui, state);
                set_action(device, env, ui, action);
                render(device, env, ui);
                T r = reward(device, env, state, action, next_state, rng);
                data.rewards[env_i][step_i] = r;
                if(step_i != SPEC::STEP_LIMIT - 1){
                    data.states[env_i][step_i+1] = next_state;
                }
                bool terminated_flag = false;
                if(step_i > 0){
                    terminated_flag = data.terminated[env_i][step_i-1];
                }
                if(!terminated_flag){
                    results.returns[env_i] += r;
                    results.episode_length[env_i] += 1;
                }
                data.terminated[env_i][step_i] = terminated_flag || terminated(device, env, next_state, rng);
            }
        }
        for(TI env_i = 0; env_i < SPEC::N_EPISODES; env_i++) {
            auto &env = envs[env_i];
            free(device, env);
        }
        for(TI env_i = 0; env_i < SPEC::N_EPISODES; env_i++){
            results.returns[env_i] = results.returns[env_i];
            results.returns_mean += results.returns[env_i];
            results.returns_std += results.returns[env_i]*results.returns[env_i];
            results.episode_length[env_i] = results.episode_length[env_i];
            results.episode_length_mean += results.episode_length[env_i];
            results.episode_length_std += results.episode_length[env_i]*results.episode_length[env_i];
        }
        results.returns_mean /= SPEC::N_EPISODES;
        results.returns_std = math::sqrt(device.math, results.returns_std/SPEC::N_EPISODES - results.returns_mean*results.returns_mean);
        results.episode_length_mean /= SPEC::N_EPISODES;
        results.episode_length_std = math::sqrt(device.math, results.episode_length_std/SPEC::N_EPISODES - results.episode_length_mean*results.episode_length_mean);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
