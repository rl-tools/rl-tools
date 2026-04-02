#include "../../version.h"
#include "../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RENDERING_RAYTRACING_TYPES_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RENDERING_RAYTRACING_TYPES_H

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {
    namespace rendering::raytracing{
        struct CollisionResult {
            float distance;
            int hit;
        };

        template <typename T_T>
        struct CameraData {
            T_T pos[3];
            T_T dir_00[3];
            T_T dir_du[3];
            T_T dir_dv[3];
        };

    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
