#include "../../../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_L2F_PARAMETERS_REWARD_FUNCTIONS_DEFAULT_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_L2F_PARAMETERS_REWARD_FUNCTIONS_DEFAULT_H

#include "../../multirotor.h"
#include "squared.h"
RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::environments::multirotor::parameters::reward_functions{
    template<typename T>
    constexpr Squared<T> squared = {
            false, // non-negative
            1.0, // scale
            20, // constant
            0, // termination penalty
            20, // position
            2.5, // orientation
            0.5, // linear_velocity
            0, // angular_velocity
            0, // linear_acceleration
            0, // angular_acceleration
            0.5, // action
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif