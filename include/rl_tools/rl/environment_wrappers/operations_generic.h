#include "../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENT_WRAPPERS_OPERATIONS_GENERIC_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENT_WRAPPERS_OPERATIONS_GENERIC_H
#include "wrappers.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template<typename DEVICE, typename ENVIRONMENT>
    RL_TOOLS_FUNCTION_PLACEMENT static void malloc(DEVICE& device, rl::environment_wrappers::Wrapper<ENVIRONMENT>& env){
        malloc(device, env.env);
    }
    template<typename DEVICE, typename ENVIRONMENT>
    RL_TOOLS_FUNCTION_PLACEMENT static void free(DEVICE& device, rl::environment_wrappers::Wrapper<ENVIRONMENT>& env){
        free(device, env.env);
    }
    template<typename DEVICE, typename ENVIRONMENT>
    RL_TOOLS_FUNCTION_PLACEMENT static void init(DEVICE& device, rl::environment_wrappers::Wrapper<ENVIRONMENT>& env){
        malloc(device, env.env);
    }
    template<typename DEVICE, typename ENVIRONMENT, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT static void sample_initial_state(DEVICE& device, const rl::environment_wrappers::Wrapper<ENVIRONMENT>& env, typename rl::environment_wrappers::Wrapper<ENVIRONMENT>::State& state, RNG& rng){
        sample_initial_state(device, env.env, state, rng);
    }
    template<typename DEVICE, typename ENVIRONMENT>
    static void initial_state(DEVICE& device, const rl::environment_wrappers::Wrapper<ENVIRONMENT>& env, typename rl::environment_wrappers::Wrapper<ENVIRONMENT>::State& state){
        initial_state(device, env.env, state);
    }
    template<typename DEVICE, typename ENVIRONMENT, typename ACTION_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT typename ENVIRONMENT::T step(DEVICE& device, const rl::environment_wrappers::Wrapper<ENVIRONMENT>& env, const typename rl::environment_wrappers::Wrapper<ENVIRONMENT>::State& state, const Matrix<ACTION_SPEC>& action, typename rl::environment_wrappers::Wrapper<ENVIRONMENT>::State& next_state, RNG& rng) {
        static_assert(ACTION_SPEC::ROWS == 1);
        return step(device, env.env, state, action, next_state, rng);
    }
    template<typename DEVICE, typename ENVIRONMENT, typename ACTION_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT static typename ENVIRONMENT::T reward(DEVICE& device, const rl::environment_wrappers::Wrapper<ENVIRONMENT>& env, const typename rl::environment_wrappers::Wrapper<ENVIRONMENT>::State& state, const Matrix<ACTION_SPEC>& action, const typename rl::environment_wrappers::Wrapper<ENVIRONMENT>::State& next_state, RNG& rng){
        return reward(device, env.env, state, action, next_state, rng);
    }

    template<typename DEVICE, typename ENVIRONMENT, typename OBS_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT static void observe(DEVICE& device, const rl::environment_wrappers::Wrapper<ENVIRONMENT>& env, const typename rl::environment_wrappers::Wrapper<ENVIRONMENT>::State& state, Matrix<OBS_SPEC>& observation, RNG& rng){
        static_assert(OBS_SPEC::ROWS == 1);
        observe(device, env.env, state, observation, rng);
    }
    template<typename DEVICE, typename ENVIRONMENT, typename OBS_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT static void observe_privileged(DEVICE& device, const rl::environment_wrappers::Wrapper<ENVIRONMENT>& env, const typename rl::environment_wrappers::Wrapper<ENVIRONMENT>::State& state, Matrix<OBS_SPEC>& observation, RNG& rng){
        static_assert(OBS_SPEC::ROWS == 1);
        observe_privileged(device, env.env, state, observation, rng);
    }
    template<typename DEVICE, typename ENVIRONMENT, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT static bool terminated(DEVICE& device, const rl::environment_wrappers::Wrapper<ENVIRONMENT>& env, const typename rl::environment_wrappers::Wrapper<ENVIRONMENT>::State state, RNG& rng){
        return terminated(device, env.env, state, rng);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
