#include "../../version.h"
#include "../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RENDERING_DATASETS_OPERATIONS_CPU_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RENDERING_DATASETS_OPERATIONS_CPU_H

#include "../scene.h"
#include "../camera.h"
#include "../transforms.h"

#include <conta/conta.h>

#include <string>
#include <limits>
#include <algorithm>
#include <iostream>

#define RL_TOOLS_RENDERING_DATASETS_LOG(message) do { std::cout << "\033[0;34m" << "#rl_tools::rendering::datasets: " << message << "\033[0m" << std::endl; } while(false)
#define RL_TOOLS_RENDERING_DATASETS_LOG_ERR(message) do { std::cerr << "\033[0;31m" << "#rl_tools::rendering::datasets: " << message << "\033[0m" << std::endl; } while(false)

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {
    namespace rendering::datasets{
        // every path-valued dataset field accepts either a filesystem path or a "conta:HASH"
        // reference; this is the single choke point that turns references into local files
        template <typename DEVICE>
        bool resolve_reference(DEVICE& device, const std::string& reference, std::string& path){
            constexpr const char PREFIX[] = "conta:";
            if(reference.rfind(PREFIX, 0) == 0){
                std::string error;
                if(!conta::resolve(reference.substr(sizeof(PREFIX) - 1), path, error)){
                    RL_TOOLS_RENDERING_DATASETS_LOG_ERR(error);
                    return false;
                }
                return true;
            }
            path = reference;
            return true;
        }

        // SHA-1 hex over the file bytes: the conta identity, used verbatim as the AssetLibrary
        // deduplication key and the per-episode scene provenance
        template <typename DEVICE>
        bool content_hash(DEVICE& device, const std::string& path, std::string& hex){
            return conta::detail::sha1_file(path, hex);
        }

        template <typename DEVICE, typename T>
        void compute_bounds(DEVICE& device, const rendering::Scene& scene, rendering::SceneMetadata<T>& metadata){
            float bbox_min[3] = {std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max()};
            float bbox_max[3] = {std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest()};
            for(const auto& instance : scene.instances){
                for(const auto& mesh : scene.objects[instance.object].meshes){
                    for(size_t vertex_i = 0; vertex_i + 2 < mesh.vertices.size(); vertex_i += 3){
                        float world[3] = {mesh.vertices[vertex_i], mesh.vertices[vertex_i + 1], mesh.vertices[vertex_i + 2]};
                        if(!instance.identity){
                            const float local[3] = {world[0], world[1], world[2]};
                            rendering::transform_point(instance.transform, local, world);
                        }
                        for(int d = 0; d < 3; d++){
                            bbox_min[d] = std::min(bbox_min[d], world[d]);
                            bbox_max[d] = std::max(bbox_max[d], world[d]);
                        }
                    }
                }
            }
            RL_TOOLS_RENDERING_DATASETS_LOG("Bounding box: [" << bbox_min[0] << "," << bbox_min[1] << "," << bbox_min[2] << "] - ["
                  << bbox_max[0] << "," << bbox_max[1] << "," << bbox_max[2] << "]");

            float center[3], size[3];
            for(int d = 0; d < 3; d++){
                center[d] = 0.5f * (bbox_min[d] + bbox_max[d]);
                size[d] = bbox_max[d] - bbox_min[d];
            }
            float max_dim = std::max({size[0], size[1], size[2]});
            // the default max ray length is the round trip to a diagonal whole-scene viewpoint —
            // datasets with their own far-plane semantics (e.g. outdoor) overwrite it after load
            float look_from[3] = {center[0] + max_dim * 1.5f, center[1] + max_dim * 1.5f, center[2] + max_dim * 0.8f};
            float look_offset[3];
            rendering::vec3::sub(look_from, center, look_offset);
            for(int d = 0; d < 3; d++){
                metadata.center[d] = center[d];
                metadata.half_extent[d] = size[d] * 0.5f;
            }
            metadata.max_ray_length = rendering::vec3::length(look_offset) * 2.0f;
        }

        template <typename DEVICE, typename T>
        void compute_bounds(DEVICE& device, rendering::Bundle<T>& bundle){
            compute_bounds(device, bundle.scene, bundle.metadata);
        }
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
