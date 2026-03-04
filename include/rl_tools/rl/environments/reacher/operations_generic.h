#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_REACHER_OPERATIONS_GENERIC_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_REACHER_OPERATIONS_GENERIC_H
#include "reacher.h"
#include "../operations_generic.h"
RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::environments::reacher{
    template <typename T>
    RL_TOOLS_FUNCTION_PLACEMENT T clip(T x, T min, T max){
        return x < min ? min : (x > max ? max : x);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT static void malloc(DEVICE& device, const rl::environments::Reacher<SPEC>& env){}
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT static void free(DEVICE& device, const rl::environments::Reacher<SPEC>& env){}
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT static void init(DEVICE& device, const rl::environments::Reacher<SPEC>& env){}
    template<typename DEVICE, typename SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT static void sample_initial_parameters(DEVICE& device, const rl::environments::Reacher<SPEC>& env, typename rl::environments::Reacher<SPEC>::Parameters& parameters, RNG& rng){}
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT static void initial_parameters(DEVICE& device, const rl::environments::Reacher<SPEC>& env, typename rl::environments::Reacher<SPEC>::Parameters& parameters){}
    template<typename DEVICE, typename SPEC, typename STATE_SPEC>
    static void initial_state(DEVICE& device, const rl::environments::Reacher<SPEC>& env, typename rl::environments::Reacher<SPEC>::Parameters& parameters, typename rl::environments::reacher::State<STATE_SPEC>& state){
        state.x = 0;
        state.y = 0;
        state.target_x = 0.5;
        state.target_y = 0.5;
    }
    template<typename DEVICE, typename SPEC, typename STATE_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT static void sample_initial_state(DEVICE& device, const rl::environments::Reacher<SPEC>& env, typename rl::environments::Reacher<SPEC>::Parameters& parameters, typename rl::environments::reacher::State<STATE_SPEC>& state, RNG& rng){
        using T = typename SPEC::T;
        using PARAMS = typename SPEC::PARAMETERS;
        state.x = random::uniform_real_distribution(device.random, -PARAMS::ARENA_SIZE, PARAMS::ARENA_SIZE, rng);
        state.y = random::uniform_real_distribution(device.random, -PARAMS::ARENA_SIZE, PARAMS::ARENA_SIZE, rng);
        state.target_x = random::uniform_real_distribution(device.random, -PARAMS::ARENA_SIZE, PARAMS::ARENA_SIZE, rng);
        state.target_y = random::uniform_real_distribution(device.random, -PARAMS::ARENA_SIZE, PARAMS::ARENA_SIZE, rng);
    }
    template<typename DEVICE, typename SPEC, typename STATE_SPEC, typename ACTION_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT typename SPEC::T step(DEVICE& device, const rl::environments::Reacher<SPEC>& env, typename rl::environments::Reacher<SPEC>::Parameters& parameters, const typename rl::environments::reacher::State<STATE_SPEC>& state, const Matrix<ACTION_SPEC>& action, typename rl::environments::reacher::State<STATE_SPEC>& next_state, RNG& rng){
        static_assert(ACTION_SPEC::ROWS == 1);
        static_assert(ACTION_SPEC::COLS == 2);
        using T = typename SPEC::T;
        using PARAMS = typename SPEC::PARAMETERS;
        T vx = rl::environments::reacher::clip(get(action, 0, 0), (T)-PARAMS::ACTION_LIMIT, (T)PARAMS::ACTION_LIMIT) * PARAMS::MAX_VELOCITY;
        T vy = rl::environments::reacher::clip(get(action, 0, 1), (T)-PARAMS::ACTION_LIMIT, (T)PARAMS::ACTION_LIMIT) * PARAMS::MAX_VELOCITY;
        next_state.x = rl::environments::reacher::clip(state.x + vx * PARAMS::DT, -PARAMS::ARENA_SIZE, PARAMS::ARENA_SIZE);
        next_state.y = rl::environments::reacher::clip(state.y + vy * PARAMS::DT, -PARAMS::ARENA_SIZE, PARAMS::ARENA_SIZE);
        next_state.target_x = state.target_x;
        next_state.target_y = state.target_y;
        return PARAMS::DT;
    }
    template<typename DEVICE, typename SPEC, typename ACTION_SPEC, typename STATE_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT static typename SPEC::T reward(DEVICE& device, const rl::environments::Reacher<SPEC>& env, typename rl::environments::Reacher<SPEC>::Parameters& parameters, const typename rl::environments::reacher::State<STATE_SPEC>& state, const Matrix<ACTION_SPEC>& action, const typename rl::environments::reacher::State<STATE_SPEC>& next_state, RNG& rng){
        using T = typename SPEC::T;
        T dx = next_state.x - next_state.target_x;
        T dy = next_state.y - next_state.target_y;
        T distance = math::sqrt(device.math, dx * dx + dy * dy);
        return -distance;
    }
    template<typename DEVICE, typename SPEC, typename STATE_SPEC, typename OBS_TYPE_SPEC, typename OBS_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT static void observe(DEVICE& device, const rl::environments::Reacher<SPEC>& env, const typename rl::environments::Reacher<SPEC>::Parameters& parameters, const typename rl::environments::reacher::State<STATE_SPEC>& state, const typename rl::environments::reacher::ObservationDense<OBS_TYPE_SPEC>&, Matrix<OBS_SPEC>& observation, RNG& rng){
        static_assert(OBS_SPEC::ROWS == 1);
        static_assert(OBS_SPEC::COLS == 4);
        set(observation, 0, 0, state.x);
        set(observation, 0, 1, state.y);
        set(observation, 0, 2, state.target_x);
        set(observation, 0, 3, state.target_y);
    }
    template<typename DEVICE, typename SPEC, typename STATE_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT static bool terminated(DEVICE& device, const rl::environments::Reacher<SPEC>& env, typename rl::environments::Reacher<SPEC>::Parameters& parameters, const typename rl::environments::reacher::State<STATE_SPEC> state, RNG& rng){
        using T = typename SPEC::T;
        using PARAMS = typename SPEC::PARAMETERS;
        T dx = state.x - state.target_x;
        T dy = state.y - state.target_y;
        T distance = math::sqrt(device.math, dx * dx + dy * dy);
        return distance < PARAMS::TARGET_RADIUS;
    }
    template<typename DEVICE, typename STATE_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT typename STATE_SPEC::T abs_diff(DEVICE& device, const rl::environments::reacher::State<STATE_SPEC>& s1, const rl::environments::reacher::State<STATE_SPEC>& s2){
        using T = typename STATE_SPEC::T;
        T acc = 0;
        acc += math::abs(device.math, s1.x - s2.x);
        acc += math::abs(device.math, s1.y - s2.y);
        acc += math::abs(device.math, s1.target_x - s2.target_x);
        acc += math::abs(device.math, s1.target_y - s2.target_y);
        return acc;
    }
    template<typename DEVICE, typename T>
    RL_TOOLS_FUNCTION_PLACEMENT T abs_diff(DEVICE& device, const rl::environments::reacher::DefaultParameters<T>& p1, const rl::environments::reacher::DefaultParameters<T>& p2){
        return 0;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
