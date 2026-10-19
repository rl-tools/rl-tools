#include "../../../../version.h"
#include "../../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_RIG_RIG_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_RIG_RIG_H

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::environments::hyperdrone::rig {
    inline constexpr float ROTOR_POSITION_WARNING_THRESHOLD = 0.10f;
    // a body/prop_* assembly articulated by init: every prop_* part turns about its hub (the centre
    // of its geometry, since node origins need not sit at the rotor hubs in real assets), every
    // other part is fixed. Without rotors coordinate i is the phase of the i-th prop in part order,
    // spinning per Quad-X (front-right CCW) by its hub's quadrant; fitted to a dynamics model's
    // rotors it is the phase of rotor i (hubs matched to the nearest rotor) spinning against the
    // rotor's reaction torque, and scale maps the assembly onto the rotors' wheelbase
    struct Rotorcraft {
        float wheelbase = 0; // largest XY distance between two hubs, assembly units
        float scale = 1;
        float rotor_position_deviation = 0; // largest matched XY hub error / wheelbase after fitting
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
