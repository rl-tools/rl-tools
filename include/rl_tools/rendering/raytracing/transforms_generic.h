#include "../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RENDERING_RAYTRACING_TRANSFORMS_GENERIC_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RENDERING_RAYTRACING_TRANSFORMS_GENERIC_H

#include "../../rl_tools.h"
#include "../transforms.h"


// the renderer's internal name for the shared 3x4 transform helpers (host verbs and device kernels)
RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    namespace rendering::raytracing::detail{
        using rendering::compose_transforms;
        using rendering::transform_is_identity;
        using rendering::transform_point;
        using rendering::transform_vector;
        using rendering::invert_transform;
        using rendering::quaternion_from_transform;
        using rendering::rotate_by_quaternion;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
