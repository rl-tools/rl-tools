#include "../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_HYPERLUX_TRANSFORMS_GENERIC_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_HYPERLUX_TRANSFORMS_GENERIC_H

#include "../rl_tools.h"
#include "transforms.h"


// the renderer's internal name for the shared 3x4 transform helpers (host verbs and device kernels)
RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    namespace hyperlux::detail{
        using hyperlux::compose_transforms;
        using hyperlux::transform_is_identity;
        using hyperlux::transform_point;
        using hyperlux::transform_vector;
        using hyperlux::invert_transform;
        using hyperlux::quaternion_from_transform;
        using hyperlux::rotate_by_quaternion;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
