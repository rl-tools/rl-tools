#include "../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_UTILS_EVALUATION_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_UTILS_EVALUATION_H
/*
 * This file relies on the environments methods hence it should be included after the operations of the environments that it will be used with
 */

#include "../../math/operations_generic.h"
#include "../environments/operations_generic.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::utils::evaluation{
    template <typename T, typename TI, typename ENV_STATE>
    struct State{
        T episode_return = 0;
        TI episode_step = 0;

        ENV_STATE state;
    };
    template <auto T_N_EPISODES, auto T_STEP_LIMIT>
    struct Specification{
        constexpr static auto N_EPISODES = T_N_EPISODES;
        constexpr static auto STEP_LIMIT = T_STEP_LIMIT;
    };
    template <typename T, typename TI, auto T_N_EPISODES>
    struct Result{
        constexpr static auto N_EPISODES = T_N_EPISODES;
        T returns[N_EPISODES];
        T returns_mean;
        T returns_std;
        TI episode_length[N_EPISODES];
        T episode_length_mean;
        T episode_length_std;
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{

    template<typename DEVICE, typename ENVIRONMENT, typename UI, typename POLICY, typename EVAL_STATE, typename OBSERVATION_MEAN_SPEC, typename OBSERVATION_STD_SPEC, typename POLICY_EVAL_BUFFERS, typename RNG>
    bool evaluate_step(DEVICE& device, ENVIRONMENT& env, UI& ui, const POLICY& policy, EVAL_STATE& eval_state, Matrix<OBSERVATION_MEAN_SPEC>& observation_mean, Matrix<OBSERVATION_STD_SPEC>& observation_std, POLICY_EVAL_BUFFERS& policy_eval_buffers, RNG& rng) {
        using T = typename POLICY::T;
        using TI = typename DEVICE::index_t;
        static_assert(ENVIRONMENT::ACTION_DIM == POLICY::OUTPUT_DIM || (2*ENVIRONMENT::ACTION_DIM == POLICY::OUTPUT_DIM));
        constexpr bool STOCHASTIC_POLICY = ENVIRONMENT::ACTION_DIM*2 == POLICY::OUTPUT_DIM;
        typename ENVIRONMENT::State state = eval_state.state;

#ifndef RL_TOOLS_DISABLE_DYNAMIC_MEMORY_ALLOCATIONS
        MatrixDynamic<matrix::Specification<T, TI, 1, ENVIRONMENT::ACTION_DIM * (STOCHASTIC_POLICY ? 2 : 1)>> action_full;
        MatrixDynamic<matrix::Specification<T, TI, 1, ENVIRONMENT::OBSERVATION_DIM>> observation, observation_normalized;
#else
        MatrixStatic<matrix::Specification<T, TI, 1, ENVIRONMENT::ACTION_DIM * (STOCHASTIC_POLICY ? 2 : 1)>> action_full;
        MatrixStatic<matrix::Specification<T, TI, 1, ENVIRONMENT::OBSERVATION_DIM>> observation, observation_normalized;
#endif
        malloc(device, observation);
        malloc(device, observation_normalized);
        malloc(device, action_full);
        observe(device, env, state, observation, rng);
        normalize(device, observation_mean, observation_std, observation, observation_normalized);

        auto action = view(device, action_full, matrix::ViewSpec<1, ENVIRONMENT::ACTION_DIM>{});
        evaluate(device, policy, observation_normalized, action_full, policy_eval_buffers);

        for(TI action_i=0; action_i<ENVIRONMENT::ACTION_DIM; action_i++){
            set(action, 0, action_i, math::clamp<T>(device.math, get(action, 0, action_i), -1, 1));
        }
        typename ENVIRONMENT::State next_state;
        T dt = step(device, env, state, action, next_state, rng);
        set_state(device, env, ui, state);
        set_action(device, env, ui, action);
        render(device, env, ui);
        T r = reward(device, env, state, action, next_state, rng);
        state = next_state;
        eval_state.episode_return += r;
        eval_state.episode_step += 1;
        eval_state.state = state;
        free(device, observation);
        free(device, observation_normalized);
        free(device, action_full);
        return terminated(device, env, state, rng);
    }
    template<typename DEVICE, typename ENVIRONMENT, typename UI, typename POLICY, typename SPEC, typename OBSERVATION_MEAN_SPEC, typename OBSERVATION_STD_SPEC, typename POLICY_EVALUATION_BUFFERS, typename RNG>
    auto evaluate(DEVICE& device, ENVIRONMENT& env, UI& ui, const POLICY& policy, const typename ENVIRONMENT::State initial_state, const SPEC& eval_spec_tag, Matrix<OBSERVATION_MEAN_SPEC>& observation_mean, Matrix<OBSERVATION_STD_SPEC>& observation_std, POLICY_EVALUATION_BUFFERS& policy_evaluation_buffers, RNG& rng) {
        using T = typename POLICY::T;
        using TI = typename DEVICE::index_t;
        rl::utils::evaluation::State<T, TI, typename ENVIRONMENT::State> state;
        state.state = initial_state;
        state.episode_return = 0;
        state.episode_step = 0;
        for (TI i = 0; i < SPEC::STEP_LIMIT; i++) {
            if(evaluate_step(device, env, ui, policy, state, observation_mean, observation_std, policy_evaluation_buffers, rng)){
                break;
            }
        }
        return state;
    }
    template<typename DEVICE, typename ENVIRONMENT, typename UI, typename POLICY, typename RNG, typename SPEC, typename OBSERVATION_MEAN_SPEC, typename OBSERVATION_STD_SPEC, typename POLICY_EVALUATION_BUFFERS>
    auto evaluate(DEVICE& device, ENVIRONMENT& env, UI& ui, const POLICY& policy, const SPEC& eval_spec_tag, Matrix<OBSERVATION_MEAN_SPEC>& observation_mean, Matrix<OBSERVATION_STD_SPEC>& observation_std, POLICY_EVALUATION_BUFFERS& policy_evaluation_buffers, RNG &rng, bool deterministic = false){
        using T = typename POLICY::T;
        using TI = typename DEVICE::index_t;
        static_assert(ENVIRONMENT::OBSERVATION_DIM == POLICY::INPUT_DIM, "Observation and policy input dimensions must match");
        static_assert(ENVIRONMENT::ACTION_DIM == POLICY::OUTPUT_DIM || (2*ENVIRONMENT::ACTION_DIM == POLICY::OUTPUT_DIM), "Action and policy output dimensions must match");
        static bool STOCHASTIC_POLICY = ENVIRONMENT::ACTION_DIM == 2*POLICY::OUTPUT_DIM;
        rl::utils::evaluation::Result<T, TI, SPEC::N_EPISODES> results;
        results.returns_mean = 0;
        results.returns_std = 0;
        results.episode_length_mean = 0;
        results.episode_length_std = 0;
        for(TI i = 0; i < SPEC::N_EPISODES; i++) {
            typename ENVIRONMENT::State initial_state;
            if(deterministic) {
                rl_tools::initial_state(device, env, initial_state);
            }
            else{
                sample_initial_state(device, env, initial_state, rng);
            }
            auto final_state = evaluate(device, env, ui, policy, initial_state, eval_spec_tag, observation_mean, observation_std, policy_evaluation_buffers, rng);
            results.returns[i] = final_state.episode_return;
            results.returns_mean += final_state.episode_return;
            results.returns_std += final_state.episode_return*final_state.episode_return;
            results.episode_length[i] = final_state.episode_step;
            results.episode_length_mean += final_state.episode_step;
            results.episode_length_std += final_state.episode_step*final_state.episode_step;
        }
        results.returns_mean /= SPEC::N_EPISODES;
        results.returns_std = math::sqrt(device.math, results.returns_std/SPEC::N_EPISODES - results.returns_mean*results.returns_mean);
        results.episode_length_mean /= SPEC::N_EPISODES;
        results.episode_length_std = math::sqrt(device.math, results.episode_length_std/SPEC::N_EPISODES - results.episode_length_mean*results.episode_length_mean);
        return results;
    }
    template<typename DEVICE, typename ENVIRONMENT, typename UI, typename POLICY, typename RNG, typename SPEC, typename POLICY_EVALUATION_BUFFERS>
    auto evaluate(DEVICE& device, ENVIRONMENT& env, UI& ui, const POLICY& policy, const SPEC& eval_spec_tag, POLICY_EVALUATION_BUFFERS& policy_evaluation_buffers, RNG &rng, bool deterministic = false){
        using T = typename POLICY::T;
        using TI = typename DEVICE::index_t;
        MatrixDynamic<matrix::Specification<T, TI, 1, ENVIRONMENT::OBSERVATION_DIM>> observation_mean;
        MatrixDynamic<matrix::Specification<T, TI, 1, ENVIRONMENT::OBSERVATION_DIM>> observation_std;
        malloc(device, observation_mean);
        malloc(device, observation_std);
        set_all(device, observation_mean, 0);
        set_all(device, observation_std, 1);
        auto results = evaluate(device, env, ui, policy, eval_spec_tag, observation_mean, observation_std, policy_evaluation_buffers, rng, deterministic);
        free(device, observation_mean);
        free(device, observation_std);
        return results;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
