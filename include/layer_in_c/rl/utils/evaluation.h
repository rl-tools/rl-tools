#ifndef LAYER_IN_C_RL_UTILS_EVALUATION_H
#define LAYER_IN_C_RL_UTILS_EVALUATION_H
/*
 * This file relies on the environments methods hence it should be included after the operations of the environments that it will be used with
 */

#include <layer_in_c/rl/environments/environments.h>
#include <layer_in_c/math/operations_generic.h>

namespace layer_in_c {
    template<typename DEVICE, typename ENVIRONMENT, typename POLICY, auto STEP_LIMIT>
    typename POLICY::T evaluate(DEVICE& device, const ENVIRONMENT env, POLICY &policy, const typename ENVIRONMENT::State initial_state) {
        using T = typename POLICY::T;
        using TI = typename DEVICE::index_t;
        typename ENVIRONMENT::State state;
        state = initial_state;
        T episode_return = 0;
        for (TI i = 0; i < STEP_LIMIT; i++) {
            T observation_mem[ENVIRONMENT::OBSERVATION_DIM];
            T* observation;
            if constexpr(ENVIRONMENT::REQUIRES_OBSERVATION){
                observation = observation_mem;
                observe(device, env, state, observation);
            }
            else{
                static_assert(sizeof(state.state)/sizeof(state.state[0]) == ENVIRONMENT::OBSERVATION_DIM, "The environments state dimension must match the environment's observation dimension.");
                observation = state.state;
            }
            T action[ENVIRONMENT::ACTION_DIM];
            Matrix<MatrixSpecification<T, TI, 1, ENVIRONMENT::ACTION_DIM>> action_matrix = {action};
            Matrix<MatrixSpecification<T, TI, 1, ENVIRONMENT::OBSERVATION_DIM>> observation_matrix = {observation};

            evaluate(device, policy, observation_matrix, action_matrix);
            T action_clipped[ENVIRONMENT::ACTION_DIM];
            for(TI action_i=0; action_i<ENVIRONMENT::ACTION_DIM; action_i++){
                action_clipped[action_i] = math::clamp<T>(action[action_i], -1, 1);
            }
            typename ENVIRONMENT::State next_state;
            step(device, env, state, action_clipped, next_state);
            T r = reward(device, env, state, action_clipped, next_state);
            state = next_state;
            episode_return += r;
            bool terminated_flag = terminated(device, env, state);
            if (terminated_flag) {
                break;
            }
        }
        return episode_return;
    }
    template<typename DEVICE, typename ENVIRONMENT, typename POLICY, typename RNG, auto STEP_LIMIT, bool DETERMINISTIC>
    typename POLICY::T evaluate(DEVICE& device, const ENVIRONMENT env, POLICY &policy, typename DEVICE::index_t N, RNG &rng) {
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
            episode_returns[i] = evaluate<DEVICE, ENVIRONMENT, POLICY, STEP_LIMIT>(device, env, policy, initial_state);
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
