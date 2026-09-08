#include "../../../version.h"
#include "../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_L2F_VISUAL_INERTIAL_VELOCITY_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_L2F_VISUAL_INERTIAL_VELOCITY_H


RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::environments::l2f_visual::inertial_velocity{
    template<typename T_TYPE_POLICY, typename T_TI>
    struct Specification{
        using TYPE_POLICY = T_TYPE_POLICY;
        using T = typename TYPE_POLICY::DEFAULT;
        using TI = T_TI;
        static constexpr T TIME_CONSTANT = 5;
        static constexpr T GRAVITY = 9.81;
    };
    template<typename T_SPEC>
    struct State{
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        T velocity[3];
        T previous_gyro[3];
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
