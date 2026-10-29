#include "../../../../../version.h"
#include "../../../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_TASKS_VISUAL_INERTIAL_LOCALIZATION_TRAJECTORY_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_TASKS_VISUAL_INERTIAL_LOCALIZATION_TRAJECTORY_H

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::environments::hyperdrone::tasks::visual_inertial_localization::trajectory {
    template <typename T_T, typename T_TI>
    struct Specification {
        using T = T_T;
        using TI = T_TI;
    };
    template <typename T_SPEC>
    struct Segment {
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        T control[6][3] = {};
        T duration = 0;
    };
    template <typename T_SPEC>
    struct Sample {
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        T position[3] = {};
        T velocity[3] = {};
        T acceleration[3] = {};
        T jerk[3] = {};
    };
    template <typename T_SPEC>
    struct Bounds {
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        T speed = 0;
        T acceleration = 0;
        T jerk = 0;
    };
    template <typename T_SPEC>
    struct Phase {
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        T rate = 0;
        T acceleration = 0;
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
