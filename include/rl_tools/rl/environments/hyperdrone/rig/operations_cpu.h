#include "../../../../version.h"
#include "../../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_RIG_OPERATIONS_CPU_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_RIG_OPERATIONS_CPU_H

#include "rig.h"
#include "operations_generic.h"
#include "../../../../rendering/rig/operations_cpu.h"
#include "../../../../rendering/raytracing/operations_cpu_common.h"

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

        template <typename TI, TI MAX_PROPS>
        struct Props {
            TI count = 0;
            TI parts[MAX_PROPS];
            float pivots[MAX_PROPS][3]; // hub in the part frame
            float hubs[MAX_PROPS][3];   // hub in the assembly frame
        };

        // validates the body/prop_* convention (the first part is "body", a prop_* part has geometry)
        template <typename TI, TI MAX_PROPS>
        bool collect_props(const rendering::ObjectAssembly& assembly, TI max_parts, Props<TI, MAX_PROPS>& props){
            if(assembly.parts.empty() || assembly.parts.size() > max_parts || assembly.objects[assembly.parts[0].object].name != "body"){
                return false;
            }
            props.count = 0;
            for(TI part_i = 0; part_i < (TI)assembly.parts.size(); part_i++){
                const auto& part = assembly.parts[part_i];
                const auto& object = assembly.objects[part.object];
                if(object.name.rfind("prop_", 0) == 0){
                    float low[3] = {0, 0, 0}, high[3] = {0, 0, 0};
                    if(props.count >= MAX_PROPS || !part_bounds(object, low, high)){
                        return false;
                    }
                    float* pivot = props.pivots[props.count];
                    pivot[0] = (low[0] + high[0]) / 2;
                    pivot[1] = (low[1] + high[1]) / 2;
                    pivot[2] = 0;
                    rendering::transform_point(part.transform, pivot, props.hubs[props.count]);
                    props.parts[props.count] = part_i;
                    props.count++;
                }
                else if(object.name == "body" && part_i != 0){
                    return false;
                }
            }
            return true;
        }

        // prop p becomes link 1 + coordinate_of_prop[p], revolute about its hub along the part's z
        // with the sign spin[p]; every other part rides link 0 at its rest frame
        template <typename DEVICE, typename TI, TI MAX_PROPS, TI MAX_PARTS>
        bool assemble(DEVICE& device, Rotorcraft<TI, MAX_PROPS, MAX_PARTS>& rig, const rendering::ObjectAssembly& assembly, const Props<TI, MAX_PROPS>& props, const TI* coordinate_of_prop, const float* spin){
            using SPEC = typename Rotorcraft<TI, MAX_PROPS, MAX_PARTS>::SPEC;
            rig.num_links = 1 + props.count;
            rig.num_parts = (TI)assembly.parts.size();
            rig.links[0] = rendering::rig::Link<SPEC>{};
            for(TI part_i = 0; part_i < rig.num_parts; part_i++){
                rig.parts[part_i] = rendering::rig::Part<SPEC>{};
                rig.parts[part_i].link = 0;
                std::memcpy(rig.parts[part_i].offset, assembly.parts[part_i].transform, sizeof(rig.parts[part_i].offset));
            }
            for(TI prop_i = 0; prop_i < props.count; prop_i++){
                const TI link_i = 1 + coordinate_of_prop[prop_i];
                const float* pivot = props.pivots[prop_i];
                const float to_pivot[12] = {1, 0, 0, pivot[0], 0, 1, 0, pivot[1], 0, 0, 1, pivot[2]};
                auto& link = rig.links[link_i];
                link = rendering::rig::Link<SPEC>{};
                link.parent = 0;
                link.joint = rendering::rig::Joint::REVOLUTE;
                link.axis[0] = 0;
                link.axis[1] = 0;
                link.axis[2] = spin[prop_i];
                rendering::compose_transforms(assembly.parts[props.parts[prop_i]].transform, to_pivot, link.origin);
                auto& part = rig.parts[props.parts[prop_i]];
                part = rendering::rig::Part<SPEC>{};
                part.link = link_i;
                for(unsigned axis = 0; axis < 3; axis++){
                    part.offset[axis * 4 + 3] = -pivot[axis];
                }
            }
            return init(device, static_cast<rendering::rig::Model<SPEC>&>(rig));
        }
    }

    // "body"/"prop_*" part-name convention -> rig: coordinate i is the phase of the i-th prop in
    // part order, spinning per the Quad-X convention (front-right CCW) by its hub's quadrant in
    // the assembly frame; false on convention violations (fail-fast for rigged-drone datasets)
    template <typename DEVICE, typename TI, TI MAX_PROPS, TI MAX_PARTS>
    bool init(DEVICE& device, rl::environments::hyperdrone::rig::Rotorcraft<TI, MAX_PROPS, MAX_PARTS>& rig, const rendering::ObjectAssembly& assembly){
        rl::environments::hyperdrone::rig::detail::Props<TI, MAX_PROPS> props;
        if(!rl::environments::hyperdrone::rig::detail::collect_props(assembly, MAX_PARTS, props)){
            return false;
        }
        TI coordinate_of_prop[MAX_PROPS];
        float spin[MAX_PROPS];
        for(TI prop_i = 0; prop_i < props.count; prop_i++){
            coordinate_of_prop[prop_i] = prop_i;
            spin[prop_i] = props.hubs[prop_i][0] * props.hubs[prop_i][1] < 0 ? 1.0f : -1.0f;
        }
        return rl::environments::hyperdrone::rig::detail::assemble(device, rig, assembly, props, coordinate_of_prop, spin);
    }

    // the same convention matched to a dynamics model: each hub pairs with the nearest rotor in
    // the assembly XY plane (a bijection, else false), coordinate i is the phase of rotor i and
    // the prop spins against the rotor's reaction torque
    template <typename DEVICE, typename TI, TI MAX_PROPS, TI MAX_PARTS, typename T>
    bool init(DEVICE& device, rl::environments::hyperdrone::rig::Rotorcraft<TI, MAX_PROPS, MAX_PARTS>& rig, const rendering::ObjectAssembly& assembly, const T rotor_positions[][3], const T rotor_torque_directions[][3], TI num_rotors){
        rl::environments::hyperdrone::rig::detail::Props<TI, MAX_PROPS> props;
        if(!rl::environments::hyperdrone::rig::detail::collect_props(assembly, MAX_PARTS, props) || props.count != num_rotors){
            return false;
        }
        TI coordinate_of_prop[MAX_PROPS];
        float spin[MAX_PROPS];
        bool matched[MAX_PROPS] = {};
        for(TI prop_i = 0; prop_i < props.count; prop_i++){
            TI nearest = 0;
            T nearest_distance = 0;
            for(TI rotor_i = 0; rotor_i < num_rotors; rotor_i++){
                const T dx = (T)props.hubs[prop_i][0] - rotor_positions[rotor_i][0];
                const T dy = (T)props.hubs[prop_i][1] - rotor_positions[rotor_i][1];
                const T distance = dx * dx + dy * dy;
                if(rotor_i == 0 || distance < nearest_distance){
                    nearest = rotor_i;
                    nearest_distance = distance;
                }
            }
            const T torque = rotor_torque_directions[nearest][2];
            if(matched[nearest] || torque == 0){
                return false;
            }
            matched[nearest] = true;
            coordinate_of_prop[prop_i] = nearest;
            spin[prop_i] = torque > 0 ? -1.0f : 1.0f;
        }
        return rl::environments::hyperdrone::rig::detail::assemble(device, rig, assembly, props, coordinate_of_prop, spin);
    }

    // the rig verbs: evaluate/sample the rig at the base pose and rig.num_coordinates joint
    // values (for a Rotorcraft the prop phases in coordinate order), then publish the part poses
    template <typename DEVICE, typename SPEC, typename BACKEND, typename RIG_SPEC, typename T>
    void set_transform(DEVICE& device, rendering::raytracing::Renderer<SPEC, BACKEND>& renderer, rendering::raytracing::OverlayIndex overlay, const rendering::raytracing::OverlayPlacement& placement, const rendering::rig::Model<RIG_SPEC>& rig, const float base[12], const T* coordinates){
        using TI = typename RIG_SPEC::TI;
        float coordinates_float[RIG_SPEC::MAX_COORDINATES];
        for(TI coordinate_i = 0; coordinate_i < rig.num_coordinates; coordinate_i++){
            coordinates_float[coordinate_i] = (float)coordinates[coordinate_i];
        }
        rendering::rig::Buffer<RIG_SPEC> buffer;
        float poses[RIG_SPEC::MAX_PARTS][12];
        evaluate(device, rig, base, coordinates_float, buffer, &poses[0][0]);
        set_transforms(device, renderer, overlay, placement, &poses[0][0]);
    }

    template <typename DEVICE, typename SPEC, typename BACKEND, typename RIG_SPEC, typename T>
    void set_transform_pair(DEVICE& device, rendering::raytracing::Renderer<SPEC, BACKEND>& renderer, rendering::raytracing::OverlayIndex overlay, const rendering::raytracing::OverlayPlacement& placement, const rendering::rig::Model<RIG_SPEC>& rig, const float base_open[12], const float base_close[12], const T* coordinates_open, const T* coordinates_close){
        using TI = typename RIG_SPEC::TI;
        float coordinates_open_float[RIG_SPEC::MAX_COORDINATES], coordinates_close_float[RIG_SPEC::MAX_COORDINATES];
        for(TI coordinate_i = 0; coordinate_i < rig.num_coordinates; coordinate_i++){
            coordinates_open_float[coordinate_i] = (float)coordinates_open[coordinate_i];
            coordinates_close_float[coordinate_i] = (float)coordinates_close[coordinate_i];
        }
        rendering::rig::Buffer<RIG_SPEC> buffer;
        float samples[SPEC::MOTION_BLUR_SAMPLES][RIG_SPEC::MAX_PARTS][12], close[RIG_SPEC::MAX_PARTS][12];
        sample(device, rig, base_open, base_close, coordinates_open_float, coordinates_close_float, (TI)SPEC::MOTION_BLUR_SAMPLES, buffer, &samples[0][0][0], (TI)(rig.num_parts * 12), &close[0][0]);
        set_motion_transforms(device, renderer, overlay, placement, &samples[0][0][0], &close[0][0]);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
