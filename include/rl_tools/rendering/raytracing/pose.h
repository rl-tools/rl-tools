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

        // the rigid state of a placement: a base pose and a uniform scale applied after it (motion
        // blur interpolates the rigid base); the asset's joint coordinates travel beside it as a
        // verb argument
        struct Pose{
            float base[12] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
            float scale = 1;
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

        // the resident pose inputs a device producer writes through set_pose and advance_flow (see
        // pose_sink): the rows in the transforms layout, the motion slabs under dynamic motion
        // blur, the previous instant under flow and the asset parts
        template <typename T_SPEC>
        struct PoseSink{
            using SPEC = T_SPEC;
            float* transforms = nullptr;
            float* transforms_motion = nullptr;
            float* transforms_previous = nullptr;
            const AssetPart* parts = nullptr;
        };

        // the per-slot structure the renderer consumes next to a slot's rows: active, global object
        // and segmentation class (segmentation::INHERIT: the asset's authored classes)
        struct OverlaySlotStructure{
            unsigned int active;
            unsigned int object;
            unsigned int segmentation_class;
        };

        // the state of one overlay as a plain value: its slot structures and its rows in the
        // transforms layout, moved whole with copy between renderers built from the same scene and
        // pool (the slots name global objects) at equal capacity. A member of configurations
        // without overlays too, hence the one-element floor
        template <typename T_SPEC>
        struct OverlaySnapshot{
            using SPEC = T_SPEC;
            static constexpr size_t MAX_OVERLAY_INSTANCES = SPEC::MAX_OVERLAY_INSTANCES;
            static constexpr size_t CAPACITY = MAX_OVERLAY_INSTANCES > 0 ? MAX_OVERLAY_INSTANCES : 1;
            OverlaySlotStructure slots[CAPACITY];
            float rows[CAPACITY * 12];
        };
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
