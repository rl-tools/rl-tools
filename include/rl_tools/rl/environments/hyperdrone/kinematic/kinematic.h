#include "../../../../version.h"
#include "../../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_KINEMATIC_KINEMATIC_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_KINEMATIC_KINEMATIC_H

#include "../../l2f/multirotor.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::environments::hyperdrone {
    template <typename T_T>
    struct KinematicLimits {
        static constexpr T_T LINEAR_VELOCITY = 1; // m/s, per body axis
        static constexpr T_T YAW_RATE = 1; // rad/s
    };
    // a level rigid body that tracks its velocity command instantly: the normalized action [linear_velocity (3), yaw_rate] in [-1, 1]^4, scaled by LIMITS, is the body FLU (level, hence heading-frame) linear velocity and the yaw rate, held over the integration interval and integrated exactly. It reuses the l2f layout (state, parameters, observation, reward, termination) and overloads only the transition and the initial state; rotor states only animate the props
    template <typename T_SPEC, typename T_LIMITS = KinematicLimits<typename T_SPEC::T>>
    struct KinematicMultirotor: Multirotor<T_SPEC> {
        using LIMITS = T_LIMITS;
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
