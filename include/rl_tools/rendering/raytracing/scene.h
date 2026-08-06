#include "../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RENDERING_RAYTRACING_SCENE_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RENDERING_RAYTRACING_SCENE_H

#include "types.h"

#include <vector>
#include <string>
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
            std::vector<SceneLight> lights; // object-local frame; transformed by the instance placing the object
            std::string name; // the glTF root node's name for assembly parts; empty otherwise
        };

        // Load product of a split GLB: one part per glTF scene-root node, each an Object in the
        // root node's local frame (its origin is the articulation pivot) plus the root's placement
        // within the assembly frame. Deliberately one grouping level, no trees: deeper glTF
        // hierarchy carries no articulation semantics and the Scene stays a flat instance list.
        struct ObjectAssembly{
            struct Part{
                size_t object;       // index into ObjectAssembly::objects
                float transform[12]; // part frame relative to the assembly frame, 3x4 row-major [R|t]
            };
            std::vector<Object> objects;
            std::vector<Part> parts;
        };

        struct Placement{
            size_t first_instance; // parts are placed as consecutive instances
            size_t num_instances;
        };

        // Typed overlay addressing: declare the layout as chained constexpr ranges so the
        // Specification's NUM_OVERLAYS is the last range's end() and cannot drift from it.
        struct OverlayIndex{
            size_t index;
        };
        struct OverlayRange{
            size_t first;
            size_t count;
            constexpr OverlayIndex operator[](size_t offset) const { return {first + offset}; }
            constexpr size_t end() const { return first + count; }
        };

        // returned by spawn: the contiguous slot run holding one instantiated asset
        struct OverlayPlacement{
            size_t first_slot;
            size_t num_parts;
            size_t first_part; // into the renderer's flattened per-part tables (for rigid moves)
        };

        // Assets available to dynamic overlays: registered before init (BLASes built once, frozen
        // after), instantiated per step via spawn. Must outlive rendering, like Scene.
        struct AssetPool{
            std::vector<ObjectAssembly> assemblies;
        };
        struct AssetHandle{
            size_t index;
        };

        struct Instance{
            size_t object;        // index into Scene::objects
            float transform[12];  // object→world, 3x4 row-major [R|t]
            bool identity;
        };

        // Host-side truth about what is rendered; filled by load/add, consumed by init(device,
        // renderer, scene). The scene must stay alive while the renderer uses it: the generic
        // backend renders directly from the mesh memory owned here.
        struct Scene{
            std::vector<Object> objects;
            std::vector<Instance> instances;
            std::vector<SceneLight> lights;
        };
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
