#include "../../../../version.h"
#include "../../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_RIG_RIG_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_RIG_RIG_H

#include "../../../../rendering/rig/rig.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::environments::hyperdrone::rig {
    inline constexpr float ROTOR_POSITION_WARNING_THRESHOLD = 0.10f;
    // the body/prop_* assembly convention as a rendering rig: link 0 is the assembly frame (fixed
    // at identity or the nominal wheelbase scale), each prop_* part a revolute link about its hub (the part's AABB centre, since
    // node origins need not sit at the rotor hubs in real assets), every other part rides link 0 at
    // its rest frame. Built from the convention alone, coordinate i is the phase of the i-th prop
    // in part order and the spin follows the Quad-X convention (front-right CCW) from the hub's
    // quadrant; built against a dynamics model, coordinate i is the phase of rotor i (hubs matched
    // to the nearest rotor) spinning against the rotor's reaction torque.
    template <typename T_TI, T_TI T_MAX_PROPS = 4, T_TI T_MAX_PARTS = 8>
    struct Rotorcraft: rendering::rig::Model<rendering::rig::Specification<float, T_TI, 1 + T_MAX_PROPS, T_MAX_PARTS>> {
        static constexpr T_TI MAX_PROPS = T_MAX_PROPS;
        float wheelbase = 0;
        float rotor_position_deviation = 0; // Maximum matched XY hub error / dynamics wheelbase after uniform fitting.
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
