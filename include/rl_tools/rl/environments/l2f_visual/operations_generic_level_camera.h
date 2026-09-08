#include "../../../version.h"
#include "../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_L2F_VISUAL_OPERATIONS_GENERIC_LEVEL_CAMERA_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_L2F_VISUAL_OPERATIONS_GENERIC_LEVEL_CAMERA_H

#include "level_camera.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void init(DEVICE& device, rl::environments::l2f_visual::level_camera::Camera<SPEC>& camera, const typename SPEC::T gravity[3], const typename SPEC::T mount_forward[3], const typename SPEC::T mount_up[3], typename SPEC::T fov){
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        T level[3][3];
        T norm = math::sqrt(device.math, gravity[0]*gravity[0] + gravity[1]*gravity[1] + gravity[2]*gravity[2]);
        for(TI axis = 0; axis < 3; axis++){
            level[2][axis] = gravity[axis] / math::max(device.math, norm, static_cast<T>(1e-6));
            level[0][axis] = (axis == 0 ? static_cast<T>(1) : static_cast<T>(0)) - level[2][axis] * level[2][0];
        }
        norm = math::sqrt(device.math, level[0][0]*level[0][0] + level[0][1]*level[0][1] + level[0][2]*level[0][2]);
        for(TI axis = 0; axis < 3; axis++){
            level[0][axis] /= math::max(device.math, norm, static_cast<T>(1e-6));
        }
        T mount[3][3];
        for(TI axis = 0; axis < 3; axis++){
            const TI next = (axis + 1) % 3;
            const TI last = (axis + 2) % 3;
            level[1][axis] = level[2][next]*level[0][last] - level[2][last]*level[0][next];
            mount[0][axis] = mount_forward[axis];
            mount[2][axis] = mount_up[axis];
            mount[1][axis] = mount_up[next]*mount_forward[last] - mount_up[last]*mount_forward[next];
        }
        for(TI output = 0; output < 3; output++){
            for(TI input = 0; input < 3; input++){
                T value = 0;
                for(TI axis = 0; axis < 3; axis++){
                    for(TI component = 0; component < 3; component++){
                        value += mount[output][axis] * level[component][axis] * mount[input][component];
                    }
                }
                camera.rotation[output][input] = value;
            }
        }
        const T half_fov = fov * math::PI<T> / static_cast<T>(360);
        camera.focal = static_cast<T>(SPEC::WIDTH) * math::cos(device.math, half_fov) / (static_cast<T>(2) * math::sin(device.math, half_fov));
    }
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void project(DEVICE&, const rl::environments::l2f_visual::level_camera::Camera<SPEC>& camera, typename SPEC::T x, typename SPEC::T y, typename SPEC::T& source_x, typename SPEC::T& source_y){
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        const T center_x = static_cast<T>(SPEC::WIDTH - 1) / static_cast<T>(2);
        const T center_y = static_cast<T>(SPEC::HEIGHT - 1) / static_cast<T>(2);
        const T ray[3] = {static_cast<T>(1), -(x-center_x)/camera.focal, -(y-center_y)/camera.focal};
        T source[3] = {};
        for(TI output = 0; output < 3; output++){
            for(TI input = 0; input < 3; input++){
                source[output] += camera.rotation[output][input] * ray[input];
            }
        }
        source_x = center_x - camera.focal * source[1] / source[0];
        source_y = center_y - camera.focal * source[2] / source[0];
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
