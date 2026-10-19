#include "../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RENDERING_RAYTRACING_SCENE_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RENDERING_RAYTRACING_SCENE_H

#include "types.h"
#include "pose.h"
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
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
