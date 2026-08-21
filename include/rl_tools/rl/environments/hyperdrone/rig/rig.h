#include "../../../../version.h"
#include "../../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_RIG_RIG_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_RIG_RIG_H

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::environments::hyperdrone::rig {
    // assembly articulation over the body/prop_* root-node convention: one part per glTF
    // scene-root node, geometry in the part-local frame. Pivots are part-local AABB centers
    // because node origins are not required to sit at the rotor hubs in real assets.
    template <typename T_T, typename T_TI, T_TI T_MAX_PROPS = 4>
    struct Rotorcraft {
        using T = T_T;
        using TI = T_TI;
        static constexpr TI MAX_PROPS = T_MAX_PROPS;
        TI body_part;
        TI num_props;
        TI prop_parts[MAX_PROPS];
        T prop_pivots[MAX_PROPS][2];
        T prop_directions[MAX_PROPS];
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
