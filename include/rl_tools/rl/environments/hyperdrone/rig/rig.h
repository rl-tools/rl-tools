#include "../../../../version.h"
#include "../../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_RIG_RIG_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_RIG_RIG_H

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::environments::hyperdrone::rig {
    inline constexpr float ROTOR_POSITION_WARNING_THRESHOLD = 0.10f;
    // a body/prop_* assembly articulated by init: prop p (the p-th prop_* part in part order) turns
    // about +z through its hub (the centre of its geometry, since node origins need not sit at the
    // rotor hubs in real assets) as coordinate p, every other part is fixed. The asset knows only
    // that; which rotor drives which prop, and which way, is the convention held here and applied
    // by coordinates: without rotors prop p follows rotor p spinning per Quad-X (front-right CCW)
    // by its hub's quadrant; fitted to a dynamics model's rotors each hub pairs with its nearest
    // rotor and spins against the rotor's reaction torque, and scale maps the assembly onto the
    // rotors' wheelbase
    struct Rotorcraft {
        static constexpr size_t MAX_PROPS = 8;
        float wheelbase = 0; // largest XY distance between two hubs, assembly units
        float scale = 1;
        float rotor_position_deviation = 0; // largest matched XY hub error / wheelbase after fitting
        size_t num_props = 0;
        size_t prop_rotor[MAX_PROPS] = {}; // the rotor driving prop p
        float prop_spin[MAX_PROPS] = {};   // +1 or -1 about +z
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
