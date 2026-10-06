#include "../../../../version.h"
#include "../../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_RIG_OPERATIONS_GENERIC_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_RIG_OPERATIONS_GENERIC_H

#include "rig.h"
#include "../../../../rendering/rig/operations_generic.h"
#include "../../l2f/quaternion_helper.h"


RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::environments::hyperdrone::rig {
    template <typename DEVICE, typename T>
    RL_TOOLS_FUNCTION_PLACEMENT void make_body_transform(DEVICE&, const T orientation[4], const T position[3], const T scene_translation[3], T scene_yaw_cos, T scene_yaw_sin, float out[12]){
        T R[3][3];
        rl::environments::l2f::quaternion_to_rotation_matrix<DEVICE, T>(orientation, R);
        const T yc = scene_yaw_cos;
        const T ys = scene_yaw_sin;
        for(unsigned column = 0; column < 3; column++){
            out[0 * 4 + column] = (float)(yc * R[0][column] - ys * R[1][column]);
            out[1 * 4 + column] = (float)(ys * R[0][column] + yc * R[1][column]);
            out[2 * 4 + column] = (float)R[2][column];
        }
        out[0 * 4 + 3] = (float)(yc * position[0] - ys * position[1] + scene_translation[0]);
        out[1 * 4 + 3] = (float)(ys * position[0] + yc * position[1] + scene_translation[1]);
        out[2 * 4 + 3] = (float)(position[2] + scene_translation[2]);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
