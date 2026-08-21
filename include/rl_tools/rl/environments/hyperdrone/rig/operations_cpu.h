#include "../../../../version.h"
#include "../../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_RIG_OPERATIONS_CPU_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_RIG_OPERATIONS_CPU_H

#include "rig.h"
#include "operations_generic.h"
#include "../../../../rendering/raytracing/operations_cpu_mux.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {
    // "body"/"prop_*" root-node convention -> parts, part-local AABB pivots, spin directions
    // from the pivot quadrant (X configuration). Returns false on convention violations
    // (fail-fast validation, e.g. for rigged-drone datasets).
    template <typename DEVICE, typename T, typename TI, TI MAX_PROPS>
    bool init(DEVICE& device, rl::environments::hyperdrone::rig::Rotorcraft<T, TI, MAX_PROPS>& rig, const rendering::raytracing::ObjectAssembly& assembly){
        rig.body_part = 0;
        rig.num_props = 0;
        if(assembly.parts.empty() || assembly.objects[assembly.parts[0].object].name != "body"){
            return false;
        }
        for(size_t part_i = 0; part_i < assembly.parts.size(); part_i++){
            const auto& object = assembly.objects[assembly.parts[part_i].object];
            if(object.name.rfind("prop_", 0) == 0){
                if(rig.num_props >= MAX_PROPS){
                    return false;
                }
                T low[3] = {0, 0, 0}, high[3] = {0, 0, 0};
                bool first = true;
                for(const auto& mesh : object.meshes){
                    for(size_t vertex_i = 0; vertex_i + 2 < mesh.vertices.size(); vertex_i += 3){
                        for(unsigned dim = 0; dim < 3; dim++){
                            const T value = mesh.vertices[vertex_i + dim];
                            low[dim] = first ? value : (value < low[dim] ? value : low[dim]);
                            high[dim] = first ? value : (value > high[dim] ? value : high[dim]);
                        }
                        first = false;
                    }
                }
                if(first){
                    return false;
                }
                const TI prop_i = rig.num_props++;
                rig.prop_parts[prop_i] = (TI)part_i;
                rig.prop_pivots[prop_i][0] = (low[0] + high[0]) / 2;
                rig.prop_pivots[prop_i][1] = (low[1] + high[1]) / 2;
                rig.prop_directions[prop_i] = rig.prop_pivots[prop_i][0] * rig.prop_pivots[prop_i][1] > 0 ? (T)1 : (T)-1;
            }
            else if(object.name == "body" && part_i != 0){
                return false;
            }
        }
        return true;
    }

    template <typename DEVICE, typename SPEC, typename BACKEND, typename T, typename TI, TI MAX_PROPS>
    void set_transform_pair(DEVICE& device, rendering::raytracing::Renderer<SPEC, BACKEND>& renderer, rendering::raytracing::OverlayIndex overlay, const rendering::raytracing::OverlayPlacement& placement, const rl::environments::hyperdrone::rig::Rotorcraft<T, TI, MAX_PROPS>& rig, const float body_open[12], const float body_close[12], const T phase_open[MAX_PROPS], const T phase_close[MAX_PROPS]){
        set_transform_pair(device, renderer, overlay, placement, body_open, body_close);
        for(TI prop_i = 0; prop_i < rig.num_props; prop_i++){
            float spin_open[12], spin_close[12];
            rl::environments::hyperdrone::rig::prop_spin_transform(device, rig.prop_pivots[prop_i][0], rig.prop_pivots[prop_i][1], rig.prop_directions[prop_i] * phase_open[prop_i], spin_open);
            rl::environments::hyperdrone::rig::prop_spin_transform(device, rig.prop_pivots[prop_i][0], rig.prop_pivots[prop_i][1], rig.prop_directions[prop_i] * phase_close[prop_i], spin_close);
            set_transform_pair(device, renderer, overlay, placement, rig.prop_parts[prop_i], spin_open, spin_close);
        }
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
