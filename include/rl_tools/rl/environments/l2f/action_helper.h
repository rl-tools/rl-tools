#include <rl_tools/version.h>
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_L2F_ACTION_HELPER_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_L2F_ACTION_HELPER_H

#include "multirotor.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::environments::l2f::action_helper{
    template<typename DEVICE, typename SPEC, typename T>
    RL_TOOLS_FUNCTION_PLACEMENT T clamp_ctbr(DEVICE& device, const Multirotor<SPEC>&, T value){
        if constexpr(SPEC::STATIC_PARAMETERS::ACTION_INTERFACE == parameters::ActionInterface::CTBR){
            return math::clamp(device.math, value, (T)-1, (T)1);
        }
        else{
            return value;
        }
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
