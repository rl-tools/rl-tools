#include "../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RENDERING_RAYTRACING_SCENE_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RENDERING_RAYTRACING_SCENE_H

#include "types.h"
#include "../scene.h"

#include <cstdint>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {
    namespace rendering::raytracing{
        using rendering::Texture;
        using rendering::Mesh;
        using rendering::Object;
        using rendering::ObjectAssembly;
        using rendering::AssetPool;
        using rendering::AssetHandle;
        using rendering::Instance;
        using rendering::Scene;
        using rendering::SceneMetadata;
        using rendering::Bundle;

        struct Placement{
            size_t first_instance; // parts are placed as consecutive instances
            size_t num_instances;
        };

        struct OverlayIndex{
            size_t index;
        };

        // returned by spawn: the contiguous slot run holding one instantiated asset
        struct OverlayPlacement{
            size_t first_slot;
            size_t num_parts;
            size_t first_part; // into the renderer's flattened per-part tables (for rigid moves)
        };
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
