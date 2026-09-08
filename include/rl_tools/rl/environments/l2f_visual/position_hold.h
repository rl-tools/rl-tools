#include "../../../version.h"
#include "../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_L2F_VISUAL_POSITION_HOLD_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_L2F_VISUAL_POSITION_HOLD_H


RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::environments::l2f_visual::position_hold{
    template<typename T_TYPE_POLICY, typename T_TI>
    struct Specification{
        using TYPE_POLICY = T_TYPE_POLICY;
        using T = typename TYPE_POLICY::DEFAULT;
        using TI = T_TI;
        static constexpr T GRAVITY = 9.81;
        static constexpr TI ACTION_HISTORY_LENGTH = 8;
        static constexpr TI BASE_OBSERVATION_DIM = 62;
        static constexpr TI ACTION_DIM = 4;
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
