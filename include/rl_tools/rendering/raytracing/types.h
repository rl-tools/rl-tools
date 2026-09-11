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

        // per-pixel flow status, one uint32 next to each flow vector. Priority is the listed
        // order: the first condition that holds decides. MISS and BEHIND carry a zero vector;
        // OUT_OF_FRAME and OCCLUDED retain the geometric flow, so the status is the mask.
        // Cross-backend contract mirrored as literals in the CUDA, GLSL, MSL and WGSL sources.
        struct FlowStatus{
            static constexpr unsigned int VALID = 0;
            static constexpr unsigned int MISS = 1;         // primary ray misses the scene
            static constexpr unsigned int BEHIND = 2;       // carried point at or behind the other camera
            static constexpr unsigned int OUT_OF_FRAME = 3; // projects outside the other image; visibility ray skipped
            static constexpr unsigned int OCCLUDED = 4;     // visibility ray from the other camera hits before the carried point
        };
        // relative margin of the flow visibility ray: any hit with t < L * (1 - epsilon) along
        // the segment from the other camera to the carried point (length L) means occluded.
        // Shared constant, mirrored in the device sources.
        template <typename T>
        constexpr T FLOW_VISIBILITY_EPSILON = (T)1e-4;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
