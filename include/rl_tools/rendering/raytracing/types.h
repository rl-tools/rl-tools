#include "../../version.h"
#include "../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RENDERING_RAYTRACING_TYPES_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RENDERING_RAYTRACING_TYPES_H

#include "../types.h"
#include "../camera.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {
    // the raytracing namespace consumes the rendering-level content/camera vocabulary under its own name
    namespace rendering::raytracing{
        using rendering::SceneLight;
        using rendering::CollisionResult;
        using rendering::Camera;
        using rendering::ShadingOptions;
        using rendering::Low;
        using rendering::Medium;
        using rendering::High;
        using rendering::VeryHigh;
        using rendering::degrees_to_radians;
        namespace vec3 = rendering::vec3;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
