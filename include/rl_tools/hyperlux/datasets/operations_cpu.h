#include "../../version.h"
#include "../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_HYPERLUX_DATASETS_OPERATIONS_CPU_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_HYPERLUX_DATASETS_OPERATIONS_CPU_H

#include "../scene.h"
#include "../camera.h"
#include "../transforms.h"

#include <conta/conta.h>

#include <string>
#include <limits>
#include <algorithm>
#include <cctype>
#include <iostream>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {
    template <typename DEVICE, typename T>
    bool contains(DEVICE& device, const hyperlux::SceneRegion& region, const T* position){
        for(int axis = 0; axis < 3; axis++){
            if(position[axis] < region.min[axis] || position[axis] > region.max[axis]){
                return false;
            }
        }
        if(region.polygon.size() < 3){
            return false;
        }
        const double x = position[0], y = position[1];
        bool inside = false;
        for(size_t i = 0, j = region.polygon.size() - 1; i < region.polygon.size(); j = i++){
            const double ax = region.polygon[j][0], ay = region.polygon[j][1];
            const double bx = region.polygon[i][0], by = region.polygon[i][1];
            const double cross = (x - ax) * (by - ay) - (y - ay) * (bx - ax);
            if(cross == 0 && x >= std::min(ax, bx) && x <= std::max(ax, bx) && y >= std::min(ay, by) && y <= std::max(ay, by)){
                return true;
            }
            if((ay > y) != (by > y) && x < ax + (y - ay) * (bx - ax) / (by - ay)){
                inside = !inside;
            }
        }
        return inside;
    }

    template <typename DEVICE, typename T>
    bool contains_indoor_position(DEVICE& device, const hyperlux::SceneRegions& regions, const T* position){
        if(!regions.annotated){
            return true;
        }
        bool indoor = false;
        for(const auto& region : regions.volumes){
            if(contains(device, region, position)){
                if(!region.indoor){
                    return false;
                }
                indoor = true;
            }
        }
        return indoor;
    }

    namespace hyperlux::datasets{
        inline constexpr char LOG_PREFIX[] = "#rl_tools::hyperlux::datasets: ";

        inline bool natural_less(const std::string& a, const std::string& b){
            size_t i = 0, j = 0;
            while(i < a.size() && j < b.size()){
                if(std::isdigit((unsigned char)a[i]) && std::isdigit((unsigned char)b[j])){
                    size_t i_end = i, j_end = j;
                    while(i_end < a.size() && std::isdigit((unsigned char)a[i_end])){
                        i_end++;
                    }
                    while(j_end < b.size() && std::isdigit((unsigned char)b[j_end])){
                        j_end++;
                    }
                    size_t i_num = i, j_num = j;
                    while(i_num < i_end - 1 && a[i_num] == '0'){
                        i_num++;
                    }
                    while(j_num < j_end - 1 && b[j_num] == '0'){
                        j_num++;
                    }
                    const size_t i_len = i_end - i_num, j_len = j_end - j_num;
                    if(i_len != j_len){
                        return i_len < j_len;
                    }
                    const int order = a.compare(i_num, i_len, b, j_num, j_len);
                    if(order != 0){
                        return order < 0;
                    }
                    i = i_end;
                    j = j_end;
                }
                else{
                    if(a[i] != b[j]){
                        return (unsigned char)a[i] < (unsigned char)b[j];
                    }
                    i++;
                    j++;
                }
            }
            return a.size() - i < b.size() - j;
        }

        // every path-valued dataset field accepts either a filesystem path or a "conta:HASH"
        // reference; this is the single choke point that turns references into local files
        template <typename DEVICE>
        bool resolve_reference(DEVICE& device, const std::string& reference, std::string& path){
            constexpr const char PREFIX[] = "conta:";
            if(reference.rfind(PREFIX, 0) == 0){
                std::string error;
                if(!conta::resolve(reference.substr(sizeof(PREFIX) - 1), path, error)){
                    log(device, device.logger, devices::logging::Verbosity::error, hyperlux::datasets::LOG_PREFIX, error);
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
        void compute_bounds(DEVICE& device, const hyperlux::Scene& scene, hyperlux::SceneMetadata<T>& metadata){
            float bbox_min[3] = {std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max()};
            float bbox_max[3] = {std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest()};
            for(const auto& instance : scene.instances){
                for(const auto& mesh : scene.objects[instance.object].meshes){
                    for(size_t vertex_i = 0; vertex_i + 2 < mesh.vertices.size(); vertex_i += 3){
                        float world[3] = {mesh.vertices[vertex_i], mesh.vertices[vertex_i + 1], mesh.vertices[vertex_i + 2]};
                        if(!instance.identity){
                            const float local[3] = {world[0], world[1], world[2]};
                            hyperlux::transform_point(instance.transform, local, world);
                        }
                        for(int d = 0; d < 3; d++){
                            bbox_min[d] = std::min(bbox_min[d], world[d]);
                            bbox_max[d] = std::max(bbox_max[d], world[d]);
                        }
                    }
                }
            }
            log(device, device.logger, devices::logging::Verbosity::debug, hyperlux::datasets::LOG_PREFIX, "Bounding box: [", bbox_min[0], ",", bbox_min[1], ",", bbox_min[2], "] - [",
                  bbox_max[0], ",", bbox_max[1], ",", bbox_max[2], "]");

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
            hyperlux::vec3::sub(look_from, center, look_offset);
            for(int d = 0; d < 3; d++){
                metadata.center[d] = center[d];
                metadata.half_extent[d] = size[d] * 0.5f;
            }
            metadata.max_ray_length = hyperlux::vec3::length(look_offset) * 2.0f;
        }

        template <typename DEVICE, typename T>
        void compute_bounds(DEVICE& device, hyperlux::Bundle<T>& bundle){
            compute_bounds(device, bundle.scene, bundle.metadata);
        }
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
