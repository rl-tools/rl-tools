#include "../../../../../version.h"
#include "../../../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_TASKS_VISUAL_INERTIAL_LOCALIZATION_COLLISION_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_TASKS_VISUAL_INERTIAL_LOCALIZATION_COLLISION_H

#include "../../../../../hyperlux/backends/generic/operations_generic.h"

#include <vector>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::environments::hyperdrone::tasks::visual_inertial_localization::collision {
    template <typename T_T, typename T_TI>
    struct Specification {
        using T = T_T;
        using TI = T_TI;
    };

    template <typename T>
    struct Triangle {
        T vertices[3][3];
    };

    template <typename T_SPEC>
    struct Scene {
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        std::vector<Triangle<T>> triangles;
        std::vector<hyperlux::backends::generic::BVHNode<T, TI>> nodes;
        std::vector<TI> primitives;
        bool valid = false;
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
