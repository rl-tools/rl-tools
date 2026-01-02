#include "../../../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_L2F_PARAMETERS_TRAJECTORIES_TRAJECTORY_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_L2F_PARAMETERS_TRAJECTORIES_TRAJECTORY_H
#include "../../multirotor.h"

#include <rl_tools/math/operations_generic.h>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::environments::l2f::parameters::trajectories{
    template<typename T>
    struct Step{
        T position[3];
        T yaw;
        T linear_velocity[3];
        T yaw_velocity;
    };
    template<typename T_T, typename T_TI, T_TI T_LENGTH, T_TI T_DT>
    struct TrajectorySpecification{
        using T = T_T;
        using TI = T_TI;
        static constexpr TI LENGTH = T_LENGTH;
        static constexpr T DT = (T)T_DT / (T)1000000; // T_DT is in microseconds
    };
    template<typename T_SPEC>
    struct Trajectory{
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        static constexpr TI LENGTH = SPEC::LENGTH;
        static constexpr T DT = SPEC::DT;
        Step<typename SPEC::T> steps[LENGTH];
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif