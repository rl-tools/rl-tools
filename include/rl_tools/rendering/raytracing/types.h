#include "../../version.h"
#include "../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RENDERING_RAYTRACING_TYPES_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RENDERING_RAYTRACING_TYPES_H

#ifndef RL_TOOLS_FUNCTION_PLACEMENT
#define RL_TOOLS_FUNCTION_PLACEMENT
#endif

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {
    namespace rendering::raytracing{
        struct CollisionResult {
            float distance;
            int hit;
        };

        struct SceneLight {
            int type; // 0=directional, 1=point, 2=spot
            float position[3];
            float direction[3];
            float color[3];
            float attenuation_constant;
            float attenuation_linear;
            float attenuation_quadratic;
            float cos_inner_cone;
            float cos_outer_cone;
        };

        template <typename T_T>
        struct Camera {
            T_T pos[3];
            T_T dir_00[3];
            T_T dir_du[3];
            T_T dir_dv[3];
        };

        // user-facing FOV values are degrees everywhere; trig consumers convert through this —
        // the double-precision product keeps float FOVs bit-identical to the legacy radian
        // constants (e.g. 80 degrees reproduces the golden-pinned 1.3962634015954636 exactly)
        template <typename T>
        RL_TOOLS_FUNCTION_PLACEMENT constexpr T degrees_to_radians(T degrees){
            return (T)((double)degrees * 0.017453292519943295);
        }

    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
