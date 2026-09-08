#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_BATCH_OPERATIONS_GENERIC_COMMON_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_BATCH_OPERATIONS_GENERIC_COMMON_H
#include "environment.h"
#include "../../../random/operations_generic_array.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template <typename DEVICE, typename ENV_SPEC, typename PARAM_SPEC, typename TI, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void sample_initial_parameters_instance(DEVICE& device, Tensor<ENV_SPEC>& environments, Tensor<PARAM_SPEC>& parameters, TI i, RNG& rng){
        sample_initial_parameters(device, get_ref(device, environments, i), get_ref(device, parameters, i), rng);
    }
    template <typename DEVICE, typename ENV_SPEC, typename PARAM_SPEC, typename STATE_SPEC, typename TI, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void sample_initial_state_instance(DEVICE& device, Tensor<ENV_SPEC>& environments, Tensor<PARAM_SPEC>& parameters, Tensor<STATE_SPEC>& states, TI i, RNG& rng){
        sample_initial_state(device, get_ref(device, environments, i), get_ref(device, parameters, i), get_ref(device, states, i), rng);
    }
    template <typename DEVICE, typename ENV, typename PARAM, typename STATE, typename OBSERVATION, typename OBS_SPEC, typename TI, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void observe_instance(DEVICE& device, ENV& environment, PARAM& parameters, STATE& state, OBSERVATION observation_type, Tensor<OBS_SPEC>& observations, TI i, RNG& rng){
        auto observation = view(device, observations, i);
        auto observation_matrix = matrix_view(device, observation);
        observe(device, environment, parameters, state, observation_type, observation_matrix, rng);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
