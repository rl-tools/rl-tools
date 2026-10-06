#include "../../../../version.h"
#include "../../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_RIG_OPERATIONS_CPU_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_RIG_OPERATIONS_CPU_H

#include "rig.h"
#include "operations_generic.h"
#include "../../../../rendering/rig/operations_cpu.h"
#include "../../../../rendering/raytracing/operations_cpu_mux.h"

#include <cstring>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {
    namespace rl::environments::hyperdrone::rig::detail {
        inline bool part_bounds(const rendering::Object& object, float low[3], float high[3]){
            bool first = true;
            for(const auto& mesh : object.meshes){
                for(size_t vertex_i = 0; vertex_i + 2 < mesh.vertices.size(); vertex_i += 3){
                    for(unsigned dim = 0; dim < 3; dim++){
                        const float value = mesh.vertices[vertex_i + dim];
                        low[dim] = first || value < low[dim] ? value : low[dim];
                        high[dim] = first || value > high[dim] ? value : high[dim];
                    }
                    first = false;
                }
            }
            return !first;
        }
    }

    // "body"/"prop_*" part-name convention -> rig; returns false on convention violations
    // (fail-fast validation, e.g. for rigged-drone datasets)
    template <typename DEVICE, typename TI, TI MAX_PROPS, TI MAX_PARTS>
    bool init(DEVICE& device, rl::environments::hyperdrone::rig::Rotorcraft<TI, MAX_PROPS, MAX_PARTS>& rig, const rendering::ObjectAssembly& assembly){
        using RIG = rl::environments::hyperdrone::rig::Rotorcraft<TI, MAX_PROPS, MAX_PARTS>;
        using SPEC = typename RIG::SPEC;
        if(assembly.parts.empty() || assembly.parts.size() > MAX_PARTS || assembly.objects[assembly.parts[0].object].name != "body"){
            return false;
        }
        rig.num_links = 1;
        rig.num_parts = (TI)assembly.parts.size();
        rig.links[0] = rendering::rig::Link<SPEC>{};
        for(TI part_i = 0; part_i < rig.num_parts; part_i++){
            const auto& part = assembly.parts[part_i];
            const auto& object = assembly.objects[part.object];
            rig.parts[part_i] = rendering::rig::Part<SPEC>{};
            if(object.name.rfind("prop_", 0) == 0){
                float low[3], high[3];
                if(rig.num_links > MAX_PROPS || !rl::environments::hyperdrone::rig::detail::part_bounds(object, low, high)){
                    return false;
                }
                const float pivot[3] = {(low[0] + high[0]) / 2, (low[1] + high[1]) / 2, 0};
                const float to_pivot[12] = {1, 0, 0, pivot[0], 0, 1, 0, pivot[1], 0, 0, 1, pivot[2]};
                auto& link = rig.links[rig.num_links];
                link = rendering::rig::Link<SPEC>{};
                link.parent = 0;
                link.joint = rendering::rig::Joint::REVOLUTE;
                link.axis[0] = 0;
                link.axis[1] = 0;
                link.axis[2] = pivot[0] * pivot[1] > 0 ? 1.0f : -1.0f;
                rendering::compose_transforms(part.transform, to_pivot, link.origin);
                rig.parts[part_i].link = rig.num_links;
                for(unsigned axis = 0; axis < 3; axis++){
                    rig.parts[part_i].offset[axis * 4 + 3] = -pivot[axis];
                }
                rig.num_links++;
            }
            else{
                if(object.name == "body" && part_i != 0){
                    return false;
                }
                rig.parts[part_i].link = 0;
                std::memcpy(rig.parts[part_i].offset, part.transform, sizeof(rig.parts[part_i].offset));
            }
        }
        return init(device, static_cast<rendering::rig::Model<SPEC>&>(rig));
    }

    // phases: the prop angles in coordinate order (the i-th prop_* part); body: the assembly pose
    template <typename DEVICE, typename SPEC, typename BACKEND, typename TI, TI MAX_PROPS, TI MAX_PARTS, typename T>
    void set_transform(DEVICE& device, rendering::raytracing::Renderer<SPEC, BACKEND>& renderer, rendering::raytracing::OverlayIndex overlay, const rendering::raytracing::OverlayPlacement& placement, const rl::environments::hyperdrone::rig::Rotorcraft<TI, MAX_PROPS, MAX_PARTS>& rig, const float body[12], const T* phases){
        using RIG_SPEC = typename rl::environments::hyperdrone::rig::Rotorcraft<TI, MAX_PROPS, MAX_PARTS>::SPEC;
        float coordinates[RIG_SPEC::MAX_COORDINATES];
        for(TI prop_i = 0; prop_i < rig.num_coordinates; prop_i++){
            coordinates[prop_i] = (float)phases[prop_i];
        }
        rendering::rig::Buffer<RIG_SPEC> buffer;
        float poses[MAX_PARTS][12];
        evaluate(device, rig, body, coordinates, buffer, &poses[0][0]);
        set_transforms(device, renderer, overlay, placement, &poses[0][0]);
    }

    template <typename DEVICE, typename SPEC, typename BACKEND, typename TI, TI MAX_PROPS, TI MAX_PARTS, typename T>
    void set_transform_pair(DEVICE& device, rendering::raytracing::Renderer<SPEC, BACKEND>& renderer, rendering::raytracing::OverlayIndex overlay, const rendering::raytracing::OverlayPlacement& placement, const rl::environments::hyperdrone::rig::Rotorcraft<TI, MAX_PROPS, MAX_PARTS>& rig, const float body_open[12], const float body_close[12], const T* phase_open, const T* phase_close){
        using RIG_SPEC = typename rl::environments::hyperdrone::rig::Rotorcraft<TI, MAX_PROPS, MAX_PARTS>::SPEC;
        float coordinates_open[RIG_SPEC::MAX_COORDINATES], coordinates_close[RIG_SPEC::MAX_COORDINATES];
        for(TI prop_i = 0; prop_i < rig.num_coordinates; prop_i++){
            coordinates_open[prop_i] = (float)phase_open[prop_i];
            coordinates_close[prop_i] = (float)phase_close[prop_i];
        }
        rendering::rig::Buffer<RIG_SPEC> buffer;
        float samples[SPEC::MOTION_BLUR_SAMPLES][MAX_PARTS][12], close[MAX_PARTS][12];
        sample(device, rig, body_open, body_close, coordinates_open, coordinates_close, (TI)SPEC::MOTION_BLUR_SAMPLES, buffer, &samples[0][0][0], (TI)(rig.num_parts * 12), &close[0][0]);
        set_motion_transforms(device, renderer, overlay, placement, &samples[0][0][0], &close[0][0]);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
