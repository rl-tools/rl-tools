#ifndef BACKPROP_TOOLS_RL_UTILS_EVALUATION_H
#define BACKPROP_TOOLS_RL_UTILS_EVALUATION_H
/*
 * This file relies on the environments methods hence it should be included after the operations of the environments that it will be used with
 */

#include <backprop_tools/rl/environments/environments.h>
#include <backprop_tools/math/operations_generic.h>

namespace backprop_tools::rl::utils::evaluation{
    template <typename T, typename ENV_STATE>
    struct State{
        T episode_return = 0;
        ENV_STATE state;
    };
    template <auto T_N_EPISODES, auto T_STEP_LIMIT>
    struct Specification{
        constexpr static auto N_EPISODES = T_N_EPISODES;
        constexpr static auto STEP_LIMIT = T_STEP_LIMIT;
    };
    template <typename T, auto T_N_EPISODES>
    struct Result{
        constexpr static auto N_EPISODES = T_N_EPISODES;
        T returns[N_EPISODES];
        T mean;
        T std;
    };
}

namespace backprop_tools {

    template <typename DEVICE, typename UI, typename STATE>
    void set_state(DEVICE& dev, UI& ui, const STATE& state){
        // dummy implementation for the case where no ui should be used
    }

    template<typename DEVICE, typename ENVIRONMENT, typename UI, typename POLICY, typename EVAL_STATE, typename OBSERVATION_MEAN_SPEC, typename OBSERVATION_STD_SPEC, typename RNG>
    bool evaluate_step(DEVICE& device, ENVIRONMENT& env, UI& ui, const POLICY& policy, EVAL_STATE& eval_state, Matrix<OBSERVATION_MEAN_SPEC>& observation_mean, Matrix<OBSERVATION_STD_SPEC>& observation_std, RNG& rng) {
        using T = typename POLICY::T;
        using TI = typename DEVICE::index_t;
        typename ENVIRONMENT::State state = eval_state.state;

#ifndef BACKPROP_TOOLS_DISABLE_DYNAMIC_MEMORY_ALLOCATIONS
        MatrixDynamic<matrix::Specification<T, TI, 1, ENVIRONMENT::ACTION_DIM>> action;
        MatrixDynamic<matrix::Specification<T, TI, 1, ENVIRONMENT::OBSERVATION_DIM>> observation, observation_normalized;
#else
        MatrixStatic<matrix::Specification<T, TI, 1, ENVIRONMENT::ACTION_DIM>> action;
        MatrixStatic<matrix::Specification<T, TI, 1, ENVIRONMENT::OBSERVATION_DIM>> observation, observation_normalized;
#endif
        malloc(device, observation);
        malloc(device, observation_normalized);
        malloc(device, action);
        observe(device, env, state, observation);
        normalize(device, observation_mean, observation_std, observation, observation_normalized);

        evaluate(device, policy, observation_normalized, action);
        for(TI action_i=0; action_i<ENVIRONMENT::ACTION_DIM; action_i++){
            set(action, 0, action_i, math::clamp<T>(typename DEVICE::SPEC::MATH(), get(action, 0, action_i), -1, 1));
        }
        typename ENVIRONMENT::State next_state;
        T dt = step(device, env, state, action, next_state);
        set_state(device, ui, state);
        T r = reward(device, env, state, action, next_state);
        state = next_state;
        eval_state.episode_return += r;
        eval_state.state = state;
        free(device, observation);
        free(device, observation_normalized);
        free(device, action);
        return terminated(device, env, state, rng);
    }
    template<typename DEVICE, typename ENVIRONMENT, typename UI, typename POLICY, typename SPEC, typename OBSERVATION_MEAN_SPEC, typename OBSERVATION_STD_SPEC, typename RNG>
    typename POLICY::T evaluate(DEVICE& device, ENVIRONMENT& env, UI& ui, const POLICY& policy, const typename ENVIRONMENT::State initial_state, const SPEC& eval_spec_tag, Matrix<OBSERVATION_MEAN_SPEC>& observation_mean, Matrix<OBSERVATION_STD_SPEC>& observation_std, RNG& rng) {
        using T = typename POLICY::T;
        using TI = typename DEVICE::index_t;
        rl::utils::evaluation::State<T, typename ENVIRONMENT::State> state;
        state.state = initial_state;
        for (TI i = 0; i < SPEC::STEP_LIMIT; i++) {
            if(evaluate_step(device, env, ui, policy, state, observation_mean, observation_std, rng)){
                break;
            }
        }
        return state.episode_return;
    }
    template<typename DEVICE, typename ENVIRONMENT, typename UI, typename POLICY, typename RNG, typename SPEC, typename OBSERVATION_MEAN_SPEC, typename OBSERVATION_STD_SPEC>
    rl::utils::evaluation::Result<typename POLICY::T, SPEC::N_EPISODES> evaluate(DEVICE& device, ENVIRONMENT& env, UI& ui, const POLICY& policy, const SPEC& eval_spec_tag, Matrix<OBSERVATION_MEAN_SPEC>& observation_mean, Matrix<OBSERVATION_STD_SPEC>& observation_std, RNG &rng, bool deterministic = false){
        using T = typename POLICY::T;
        using TI = typename DEVICE::index_t;
        static_assert(ENVIRONMENT::OBSERVATION_DIM == POLICY::INPUT_DIM, "Observation and policy input dimensions must match");
        static_assert(ENVIRONMENT::ACTION_DIM == POLICY::OUTPUT_DIM, "Action and policy output dimensions must match");
        rl::utils::evaluation::Result<T, SPEC::N_EPISODES> results;
        results.mean = 0;
        results.std = 0;
        for(TI i = 0; i < SPEC::N_EPISODES; i++) {
            typename ENVIRONMENT::State initial_state;
            if(deterministic) {
                backprop_tools::initial_state(device, env, initial_state);
            }
            else{
                sample_initial_state(device, env, initial_state, rng);
            }
            T r = evaluate(device, env, ui, policy, initial_state, eval_spec_tag, observation_mean, observation_std, rng);
            results.returns[i] = r;
            results.mean += r;
            results.std += r*r;
        }
        results.mean /= SPEC::N_EPISODES;
        results.std = math::sqrt(typename DEVICE::SPEC::MATH(), results.std/SPEC::N_EPISODES - results.mean*results.mean);
        return results;
    }
    template<typename DEVICE, typename ENVIRONMENT, typename UI, typename POLICY, typename RNG, typename SPEC>
    auto evaluate(DEVICE& device, ENVIRONMENT& env, UI& ui, const POLICY& policy, const SPEC& eval_spec_tag, RNG &rng, bool deterministic = false){
        using T = typename POLICY::T;
        using TI = typename DEVICE::index_t;
        MatrixDynamic<matrix::Specification<T, TI, 1, ENVIRONMENT::OBSERVATION_DIM>> observation_mean;
        MatrixDynamic<matrix::Specification<T, TI, 1, ENVIRONMENT::OBSERVATION_DIM>> observation_std;
        malloc(device, observation_mean);
        malloc(device, observation_std);
        set_all(device, observation_mean, 0);
        set_all(device, observation_std, 1);
        auto results = evaluate(device, env, ui, policy, eval_spec_tag, observation_mean, observation_std, rng, deterministic);
        free(device, observation_mean);
        free(device, observation_std);
        return results;
    }
}

#endif
