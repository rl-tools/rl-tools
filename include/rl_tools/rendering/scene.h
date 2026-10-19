#include "../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RENDERING_SCENE_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RENDERING_SCENE_H

#include "types.h"

#include <vector>
#include <string>
#include <cstdint>
#include <map>
#include <memory>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {
    namespace rendering{
        struct Texture{
            std::shared_ptr<const std::vector<uint8_t>> pixels; // RGBA8, shared by meshes and asset copies
            int width = 0;
            int height = 0;
            bool present() const {
                return pixels && !pixels->empty() && width > 0 && height > 0;
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
            bool unlit = false;
        };

        struct SegmentationTaxonomy{
            std::string name;
            std::map<uint32_t, std::string> classes;
        };

        struct Object{
            std::vector<Mesh> meshes;
            std::vector<SceneLight> lights; // object-local frame; transformed by the instance placing the object
            uint32_t segmentation_class = 0;
            std::shared_ptr<const SegmentationTaxonomy> segmentation_taxonomy;
            bool segmentation_class_valid = true; // false when welding erased class boundaries
            std::string name; // the glTF root node's name for assembly parts; empty otherwise
        };

        // Load product of a split GLB: one part per selected glTF node (the scene roots by
        // default), each an Object in the node's local frame plus the node's placement within the
        // assembly frame. The part list stays flat (the Scene is a flat instance list); a part
        // whose node sits inside another selected node records that part as its parent (listed
        // before it), which its joint moves it relative to.
        struct ObjectAssembly{
            struct Part{
                static constexpr size_t NO_PARENT = ~(size_t)0;
                size_t object;       // index into ObjectAssembly::objects
                float transform[12]; // part frame relative to the assembly frame, 3x4 row-major [R|t]
                size_t parent = NO_PARENT;
                Joint joint;
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

        // Loader-authoritative scene facts: the renderer consumes max_ray_length (probe budget), producers consume the bounds (camera rigs, free-space search), and content_hash (SHA-1 hex, the conta identity) keys AssetLibrary deduplication.
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
