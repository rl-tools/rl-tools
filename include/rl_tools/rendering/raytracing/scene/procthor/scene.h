#include "../../../../version.h"
#include "../../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RENDERING_RAYTRACING_SCENE_PROCTHOR_SCENE_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RENDERING_RAYTRACING_SCENE_PROCTHOR_SCENE_H

#include "../scene.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rendering::raytracing::scene::procthor {
    template <typename T_SPEC>
    struct Scene {
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;

        IndoorPosition<T> indoor_positions[SPEC::MAX_INDOOR_POSITIONS];
        TI num_indoor_positions = 0;

        SceneConfig<T> config;
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
