#include "../../../version.h"
#include "../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RENDERING_RAYTRACING_SCENE_SCENE_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RENDERING_RAYTRACING_SCENE_SCENE_H

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rendering::raytracing::scene {
    template <typename T>
    struct IndoorPosition {
        T position[3];
        T yaw;
        T score;
    };

    template <typename T>
    struct SceneConfig {
        T scene_center[3];
        T scene_radius;
    };

    template <typename T_T, typename T_TI, T_TI T_MAX_INDOOR_POSITIONS = 256>
    struct SceneSpecification {
        using T = T_T;
        using TI = T_TI;
        static constexpr TI MAX_INDOOR_POSITIONS = T_MAX_INDOOR_POSITIONS;
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
