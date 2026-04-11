#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_INFERENCE_APPLICATIONS_L2F_OPERATIONS_DYN_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_INFERENCE_APPLICATIONS_L2F_OPERATIONS_DYN_H

#include "operations_generic.h"
#include "../../../dyn/policy_adapter.h"
#include "../../../dyn/operations_generic.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template <typename DEVICE, typename TI, typename OBS_TENSOR, typename ACTION_TENSOR, typename RNG, typename MODE>
    RL_TOOLS_FUNCTION_PLACEMENT void evaluate_step(DEVICE& device, dyn::Layer<TI>& layer, OBS_TENSOR& observation, dyn::PolicyState<TI>& state, ACTION_TENSOR& action, dyn::PolicyBuffer<TI>& buffer, RNG&, MODE){
        rl_tools::evaluate_step(device, layer, observation, state.inner, action, buffer.inner);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
