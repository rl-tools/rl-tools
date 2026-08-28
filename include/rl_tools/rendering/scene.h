#include "../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RENDERING_SCENE_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RENDERING_SCENE_H

#include "types.h"

#include <vector>
#include <string>
#include <cstdint>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {
    namespace rendering{
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
            uint32_t segmentation_class = 0; // user-assigned (e.g. from Object::name via a user taxonomy); reported by SEMANTIC_SEGMENTATION specs
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

        // Assets available to dynamic overlays: registered before init (BLASes built once, frozen
        // after), instantiated per step via spawn. Must outlive rendering, like Scene; a boundary
        // that outlives its caller (e.g. language bindings) must own copies of both.
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

        // Host-side truth about what is rendered; produced by the datasets layer, consumed by the
        // renderer. The scene must stay alive while the renderer uses it: the generic backend
        // renders directly from the mesh memory owned here.
        struct Scene{
            // background + ambient as content: filled by dataset loaders (or authored), resolved
            // by the renderer at init. DEFAULT keeps the backend's historical behavior; GRADIENT
            // and EQUIRECT are declared for datasets to carry — backends that cannot render them
            // yet fall back to a solid horizon background.
            struct Environment{
                enum class Mode { DEFAULT, SOLID, GRADIENT, EQUIRECT };
                Mode mode = Mode::DEFAULT;
                float horizon[3] = {0, 0, 0};  // SOLID: the background; GRADIENT: horizon endpoint
                float zenith[3] = {0, 0, 0};   // GRADIENT: zenith endpoint
                float ambient[3] = {0.10f, 0.10f, 0.10f};
                Texture equirect;              // EQUIRECT panorama; not yet consumed by the backends
            };
            std::vector<Object> objects;
            std::vector<Instance> instances;
            std::vector<SceneLight> lights;
            Environment environment;
        };

        // loader-authoritative scene facts: the renderer consumes max_ray_length (probe budget,
        // depth miss sentinel), producers consume the bounds (camera rigs, free-space search),
        // and content_hash (SHA-1 hex, the conta identity) keys AssetLibrary deduplication
        template <typename T>
        struct SceneMetadata{
            T center[3] = {0, 0, 0};
            T half_extent[3] = {0, 0, 0};
            T max_ray_length = 0;
            std::string content_hash;
        };

        template <typename T>
        struct Bundle{
            Scene scene;
            SceneMetadata<T> metadata;
        };
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
