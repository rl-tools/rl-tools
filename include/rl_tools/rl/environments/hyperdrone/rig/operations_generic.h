#include "../../../../version.h"
#include "../../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_RIG_OPERATIONS_GENERIC_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_RIG_OPERATIONS_GENERIC_H

#include "rig.h"
#include "../../l2f/quaternion_helper.h"

#ifndef RL_TOOLS_FUNCTION_PLACEMENT
#define RL_TOOLS_FUNCTION_PLACEMENT
#endif

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::environments::hyperdrone::rig {
    template <typename DEVICE, typename T>
    RL_TOOLS_FUNCTION_PLACEMENT void prop_spin_transform(DEVICE& device, T pivot_x, T pivot_y, T angle, float out[12]){
        const float c = (float)math::cos(device.math, angle);
        const float s = (float)math::sin(device.math, angle);
        const float px = (float)pivot_x;
        const float py = (float)pivot_y;
        out[0] = c;  out[1] = -s; out[2]  = 0; out[3]  = px - c * px + s * py;
        out[4] = s;  out[5] = c;  out[6]  = 0; out[7]  = py - s * px - c * py;
        out[8] = 0;  out[9] = 0;  out[10] = 1; out[11] = 0;
    }

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

namespace rl_tools {
    // device-capable producer twin of the host set_transform_pair(renderer, ..., rig, ...) verb:
    // writes the shutter-open/close entries straight into the renderer's transforms_pair slab
    // (open entries at [0, TOTAL_SLOTS), close entries at [TOTAL_SLOTS, 2*TOTAL_SLOTS)), the
    // drone_device.cu kernel-producer path
    template <typename DEVICE, typename T, typename TI, TI MAX_PROPS>
    RL_TOOLS_FUNCTION_PLACEMENT void set_transform_pair(DEVICE& device, float* transforms_pair, TI total_slots, TI first_slot, const rl::environments::hyperdrone::rig::Rotorcraft<T, TI, MAX_PROPS>& rig, const float body_open[12], const float body_close[12], const T phase_open[MAX_PROPS], const T phase_close[MAX_PROPS]){
        for(TI element_i = 0; element_i < 12; element_i++){
            transforms_pair[(first_slot + rig.body_part) * 12 + element_i] = body_open[element_i];
            transforms_pair[(total_slots + first_slot + rig.body_part) * 12 + element_i] = body_close[element_i];
        }
        for(TI prop_i = 0; prop_i < rig.num_props; prop_i++){
            const TI slot = first_slot + rig.prop_parts[prop_i];
            rl::environments::hyperdrone::rig::prop_spin_transform(device, rig.prop_pivots[prop_i][0], rig.prop_pivots[prop_i][1], rig.prop_directions[prop_i] * phase_open[prop_i], transforms_pair + slot * 12);
            rl::environments::hyperdrone::rig::prop_spin_transform(device, rig.prop_pivots[prop_i][0], rig.prop_pivots[prop_i][1], rig.prop_directions[prop_i] * phase_close[prop_i], transforms_pair + (total_slots + slot) * 12);
        }
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
