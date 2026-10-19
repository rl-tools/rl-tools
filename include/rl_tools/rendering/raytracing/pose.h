#include "../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RENDERING_RAYTRACING_POSE_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RENDERING_RAYTRACING_POSE_H

#include "../../rl_tools.h"
#include "../types.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {
    namespace rendering::raytracing{
        struct OverlayIndex{
            size_t index;
        };

        // returned by spawn: the contiguous slot run holding one instantiated asset
        struct OverlayPlacement{
            size_t first_slot;
            size_t num_parts;
            size_t first_part; // into the renderer's flattened asset parts
        };

        // the state of a placement: a rigid base pose, a uniform scale applied after it (motion
        // blur interpolates the rigid base) and the asset's joint coordinates (nullptr: all zero)
        struct Pose{
            float base[12] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
            float scale = 1;
            const float* coordinates = nullptr;
        };

        // a pool asset part as the pose evaluation reads it, flattened at init: its global object,
        // its frame relative to its reference (the part frame of its nearest articulated ancestor,
        // the assembly frame for none) and its joint (unit axis)
        struct AssetPart{
            static constexpr size_t NO_REFERENCE = ~(size_t)0;
            size_t object = 0;
            float origin[12] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
            size_t reference = NO_REFERENCE;
            rendering::Joint joint;
        };

        // the resident pose inputs a device producer writes through set_pose (see pose_sink): the
        // rows in the transforms layout, the motion slabs under dynamic motion blur and the asset parts
        template <typename T_SPEC>
        struct PoseSink{
            using SPEC = T_SPEC;
            float* transforms = nullptr;
            float* transforms_motion = nullptr;
            const AssetPart* parts = nullptr;
        };
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
