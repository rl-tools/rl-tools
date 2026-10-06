#include "../../../../version.h"
#include "../../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_RIG_RIG_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_RIG_RIG_H

#include "../../../../rendering/rig/rig.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::environments::hyperdrone::rig {
    // the body/prop_* assembly convention as a rendering rig: link 0 is the assembly frame (fixed
    // at identity), each prop_* part a revolute link about its part-local AABB centre with the spin
    // direction of its pivot quadrant (X configuration) baked into the axis sign, every other part
    // rides link 0 at its rest frame; coordinate i is the phase of the i-th prop in part order.
    // Pivots are AABB centres because node origins need not sit at the rotor hubs in real assets.
    template <typename T_TI, T_TI T_MAX_PROPS = 4, T_TI T_MAX_PARTS = 8>
    struct Rotorcraft: rendering::rig::Model<rendering::rig::Specification<float, T_TI, 1 + T_MAX_PROPS, T_MAX_PARTS>> {
        static constexpr T_TI MAX_PROPS = T_MAX_PROPS;
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
