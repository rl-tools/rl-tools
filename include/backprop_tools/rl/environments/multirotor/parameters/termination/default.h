#include "../../../../../version.h"
#if (defined(BACKPROP_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(BACKPROP_TOOLS_RL_ENVIRONMENTS_MULTIROTOR_PARAMETERS_TERMINATION_DEFAULT_H)) && (BACKPROP_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define BACKPROP_TOOLS_RL_ENVIRONMENTS_MULTIROTOR_PARAMETERS_TERMINATION_DEFAULT_H

#include "../../multirotor.h"

#include "../../../../../math/operations_generic.h"

BACKPROP_TOOLS_NAMESPACE_WRAPPER_START
namespace backprop_tools::rl::environments::multirotor::parameters::termination{
    template<typename T, typename TI, TI ACTION_DIM, typename REWARD_FUNCTION>
    constexpr typename backprop_tools::rl::environments::multirotor::ParametersBase<T, TI, 4, REWARD_FUNCTION>::MDP::Termination classic = {
            true,           // enable
            0.6,            // position
            10,         // linear velocity
            10 // angular velocity
    };
    template<typename T, typename TI, TI ACTION_DIM, typename REWARD_FUNCTION>
    constexpr typename backprop_tools::rl::environments::multirotor::ParametersBase<T, TI, 4, REWARD_FUNCTION>::MDP::Termination fast_learning = {
        true,           // enable
        0.6,            // position
        1000,         // linear velocity
        1000 // angular velocity
    };
}
BACKPROP_TOOLS_NAMESPACE_WRAPPER_END

#endif