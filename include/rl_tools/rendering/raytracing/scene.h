#include "../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RENDERING_RAYTRACING_SCENE_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RENDERING_RAYTRACING_SCENE_H

#include "types.h"

#include <vector>
#include <cstdint>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {
    namespace rendering::raytracing{
        struct Texture{
            std::vector<uint8_t> pixels; // RGBA8
            int width = 0;
            int height = 0;
            bool present() const {
                return !pixels.empty() && width > 0 && height > 0;
            }
        };

        struct Mesh{
            std::vector<float> vertices; // vec3f flat
            std::vector<int> indices;    // vec3i flat
            std::vector<float> tex_coords; // vec2f flat
            std::vector<float> normals;
            float color[3];
            Texture texture;
            Texture normal_map;
            Texture metallic_roughness_map;
            Texture emissive_map;
            Texture occlusion_map;
            float metallic = 0.0f;
            float roughness = 1.0f;
            float emissive[3] = {0, 0, 0};
            float opacity = 1.0f;
            int alpha_mode = 0;
            float alpha_cutoff = 0.5f;
        };

        struct Object{
            std::vector<Mesh> meshes;
            std::vector<SceneLight> lights;
        };

        // Host-side truth about what is rendered; filled by load/add, consumed by init(device,
        // renderer, scene). The scene must stay alive while the renderer uses it: the generic
        // backend renders directly from the mesh memory owned here.
        struct Scene{
            std::vector<Mesh> meshes;
            std::vector<SceneLight> lights;
        };
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
