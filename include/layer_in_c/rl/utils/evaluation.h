#ifndef LAYER_IN_C_RL_UTILS_EVALUATION_H
#define LAYER_IN_C_RL_UTILS_EVALUATION_H
/*
 * This file relies on the environments methods hence it should be included after the operations of the environments that it will be used with
 */

#include <layer_in_c/rl/environments/environments.h>
#include <layer_in_c/math/operations_generic.h>

namespace layer_in_c::rl::utils::evaluation{
    template <typename T, typename ENV_STATE>
    struct State{
        T episode_return = 0;
        ENV_STATE state;
    };
}

namespace layer_in_c {

    template <typename DEVICE, typename STATE>
    void set_state(DEVICE& dev, bool ui, const STATE& state){
        // dummy implementation for the case where no ui should be used
    }
    template<typename DEVICE, typename ENVIRONMENT, typename UI, typename POLICY, typename EVAL_STATE>
    bool evaluate_step(DEVICE& device, const ENVIRONMENT& env, UI& ui, const POLICY& policy, EVAL_STATE& eval_state) {
        using T = typename POLICY::T;
        using TI = typename DEVICE::index_t;
        typename ENVIRONMENT::State state = eval_state.state;

        Matrix<matrix::Specification<T, TI, 1, ENVIRONMENT::ACTION_DIM>> action;
        Matrix<matrix::Specification<T, TI, 1, ENVIRONMENT::OBSERVATION_DIM>> observation;
        malloc(device, observation);
        malloc(device, action);
        observe(device, env, state, observation);

        evaluate(device, policy, observation, action);
        for(TI action_i=0; action_i<ENVIRONMENT::ACTION_DIM; action_i++){
            set(action, 0, action_i, math::clamp<T>(get(action, 0, action_i), -1, 1));
        }
        typename ENVIRONMENT::State next_state;
        T dt = step(device, env, state, action, next_state);
        set_state(device, ui, state);
        T r = reward(device, env, state, action, next_state);
        state = next_state;
        eval_state.episode_return += r;
        eval_state.state = state;
        free(device, observation);
        malloc(device, action);
        return terminated(device, env, state);
    }
    template<typename DEVICE, typename ENVIRONMENT, typename UI, typename POLICY, typename DEVICE::index_t STEP_LIMIT>
    typename POLICY::T evaluate(DEVICE& device, const ENVIRONMENT& env, UI& ui, const POLICY& policy, const typename ENVIRONMENT::State initial_state) {
        using T = typename POLICY::T;
        using TI = typename DEVICE::index_t;
        rl::utils::evaluation::State<T, typename ENVIRONMENT::State> state;
        state.state = initial_state;
        for (TI i = 0; i < STEP_LIMIT; i++) {
            if(evaluate_step(device, env, ui, policy, state)){
                break;
            }
        }
        return state.episode_return;
    }
    template<typename DEVICE, typename ENVIRONMENT, typename UI, typename POLICY, typename RNG, auto STEP_LIMIT, bool DETERMINISTIC>
    typename POLICY::T evaluate(DEVICE& device, const ENVIRONMENT& env, UI& ui, const POLICY& policy, typename DEVICE::index_t N, RNG &rng) {
        using T = typename POLICY::T;
        using TI = typename DEVICE::index_t;
        static_assert(ENVIRONMENT::OBSERVATION_DIM == POLICY::INPUT_DIM, "Observation and policy input dimensions must match");
        static_assert(ENVIRONMENT::ACTION_DIM == POLICY::OUTPUT_DIM, "Action and policy output dimensions must match");
        T episode_returns[N];
        for(TI i = 0; i < N; i++) {
            typename ENVIRONMENT::State initial_state;
            if(DETERMINISTIC) {
                layer_in_c::initial_state(device, env, initial_state);
            }
            else{
                sample_initial_state(device, env, initial_state, rng);
            }
            episode_returns[i] = evaluate<DEVICE, ENVIRONMENT, UI, POLICY, STEP_LIMIT>(device, env, ui, policy, initial_state);
        }
        T mean = 0;
        for(TI i = 0; i < N; i++) {
            mean += episode_returns[i];
        }
        mean /= N;
        T variance = 0;
        for(TI i = 0; i < N; i++) {
            variance += (episode_returns[i] - mean) * (episode_returns[i] - mean);
        }
        variance /= N;
        T standard_deviation = math::sqrt(typename DEVICE::SPEC::MATH(), variance);
        logging::text(device.logger, "Mean: ", mean, ", Standard deviation: ", standard_deviation);
        return mean;
    }


}

#endif
