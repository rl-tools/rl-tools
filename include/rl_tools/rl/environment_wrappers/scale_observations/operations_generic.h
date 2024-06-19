#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENT_WRAPPERS_SCALE_OBSERVATIONS_OPERATIONS_GENERIC_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENT_WRAPPERS_SCALE_OBSERVATIONS_OPERATIONS_GENERIC_H

#include "../../../math/operations_generic.h"
#include "wrapper.h"
#include "../operations_generic.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template<typename DEVICE, typename SPEC, typename ENVIRONMENT, typename OBS_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT static void observe(DEVICE& device, const rl::environment_wrappers::ScaleObservations<SPEC, ENVIRONMENT>& env, const typename rl::environment_wrappers::ScaleObservations<SPEC, ENVIRONMENT>::Parameters& parameters, const typename rl::environment_wrappers::ScaleObservations<SPEC, ENVIRONMENT>::State& state, Matrix<OBS_SPEC>& observation, RNG& rng){
        using T = typename SPEC::T;
        using TI = typename DEVICE::index_t;
        static_assert(OBS_SPEC::ROWS == 1);
        observe(device, env.env, parameters, state, observation, rng);
        for(TI observation_i = 0; observation_i < OBS_SPEC::COLS; observation_i++){
            T observation_value = get(observation, 0, observation_i);
            set(observation, 0, observation_i, observation_value * SPEC::SCALE);
        }
    }
    template<typename DEVICE, typename SPEC, typename ENVIRONMENT, typename OBS_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT static void observe_privileged(DEVICE& device, const rl::environment_wrappers::ScaleObservations<SPEC, ENVIRONMENT>& env, const typename rl::environment_wrappers::ScaleObservations<SPEC, ENVIRONMENT>::Parameters& parameters, const typename rl::environment_wrappers::ScaleObservations<SPEC, ENVIRONMENT>::State& state, Matrix<OBS_SPEC>& observation, RNG& rng){
        static_assert(OBS_SPEC::ROWS == 1);
        observe_privileged(device, env.env, parameters, state, observation, rng);
    }
}

RL_TOOLS_NAMESPACE_WRAPPER_END

#endif

