#include "../../../../version.h"
#include "../../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_ROTORCRAFT_OPERATIONS_CPU_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_ROTORCRAFT_OPERATIONS_CPU_H

#include "rotorcraft.h"
#include "operations_generic.h"
#include "../../../../hyperlux/scene.h"
#include "../../../../hyperlux/transforms.h"

#include <vector>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {
    namespace rl::environments::hyperdrone::rotorcraft::detail {
        inline bool part_bounds(const hyperlux::Object& object, float low[3], float high[3]){
            bool first = true;
            for(const auto& mesh : object.meshes){
                for(size_t vertex_i = 0; vertex_i + 2 < mesh.vertices.size(); vertex_i += 3){
                    for(unsigned dim = 0; dim < 3; dim++){
                        const float value = mesh.vertices[vertex_i + dim];
                        if(!hyperlux::is_finite(value)){
                            return false;
                        }
                        low[dim] = first || value < low[dim] ? value : low[dim];
                        high[dim] = first || value > high[dim] ? value : high[dim];
                    }
                    first = false;
                }
            }
            return !first;
        }

        struct Prop {
            size_t part;
            float pivot[3]; // hub in the part frame
            float hub[3];   // hub in the assembly frame
        };

        inline bool is_prop(const hyperlux::ObjectAssembly& assembly, size_t part_i){
            return assembly.objects[assembly.parts[part_i].object].name.rfind("prop_", 0) == 0;
        }

        inline bool is_body(const hyperlux::ObjectAssembly& assembly, size_t part_i){
            return assembly.objects[assembly.parts[part_i].object].name == "body";
        }

        inline bool locate_hub(const hyperlux::ObjectAssembly::Part& part, Prop& prop){
            hyperlux::transform_point(part.transform, prop.pivot, prop.hub);
            return hyperlux::is_finite(prop.hub, 3);
        }

        // the body/prop_* convention from the geometry: the first part is "body", a prop_* part
        // has geometry and its hub is the centre of it (node origins need not sit at the rotor
        // hubs in real assets), at most Rotorcraft::MAX_PROPS props
        inline bool collect_props(const hyperlux::ObjectAssembly& assembly, std::vector<Prop>& props){
            props.clear();
            if(assembly.parts.empty() || !is_body(assembly, 0)){
                return false;
            }
            for(size_t part_i = 0; part_i < assembly.parts.size(); part_i++){
                const auto& part = assembly.parts[part_i];
                if(!hyperlux::is_finite(part.transform, 12)){
                    return false;
                }
                if(is_prop(assembly, part_i)){
                    float low[3] = {0, 0, 0}, high[3] = {0, 0, 0};
                    if(!part_bounds(assembly.objects[part.object], low, high)){
                        return false;
                    }
                    Prop prop{part_i, {(low[0] + high[0]) / 2, (low[1] + high[1]) / 2, 0}, {}};
                    if(!locate_hub(part, prop)){
                        return false;
                    }
                    props.push_back(prop);
                }
                else if(is_body(assembly, part_i) && part_i != 0){
                    return false;
                }
            }
            return props.size() <= Rotorcraft::MAX_PROPS;
        }

        // the props as articulate wrote them: every prop_* part turns about +z as its prop index,
        // every other part is fixed
        inline bool articulated_props(const hyperlux::ObjectAssembly& assembly, std::vector<Prop>& props){
            props.clear();
            if(assembly.parts.empty() || !is_body(assembly, 0)){
                return false;
            }
            for(size_t part_i = 0; part_i < assembly.parts.size(); part_i++){
                const auto& part = assembly.parts[part_i];
                const auto& joint = part.joint;
                if(!is_prop(assembly, part_i)){
                    if(joint.type != hyperlux::Joint::Type::FIXED || (is_body(assembly, part_i) && part_i != 0)){
                        return false;
                    }
                    continue;
                }
                if(joint.type != hyperlux::Joint::Type::REVOLUTE || joint.axis[0] != 0 || joint.axis[1] != 0 || joint.axis[2] != 1 || joint.coordinate != props.size() || props.size() == Rotorcraft::MAX_PROPS){
                    return false;
                }
                Prop prop{part_i, {joint.pivot[0], joint.pivot[1], joint.pivot[2]}, {}};
                if(!locate_hub(part, prop)){
                    return false;
                }
                props.push_back(prop);
            }
            return true;
        }

        template <typename DEVICE>
        float hub_wheelbase(DEVICE& device, const std::vector<Prop>& props){
            std::vector<float> hubs;
            for(const auto& prop : props){
                hubs.insert(hubs.end(), prop.hub, prop.hub + 3);
            }
            return rotor_wheelbase(device, reinterpret_cast<const float(*)[3]>(hubs.data()), props.size());
        }
    }

    // writes the joints of a body/prop_* assembly: prop p (the p-th prop_* part) turns about +z
    // through its hub as coordinate p, every other part is fixed; false on a convention violation,
    // leaving the assembly unchanged. Joints bind when the assembly enters the pool, so articulate
    // before add
    template <typename DEVICE>
    bool articulate(DEVICE& device, hyperlux::ObjectAssembly& assembly){
        std::vector<rl::environments::hyperdrone::rotorcraft::detail::Prop> props;
        if(!rl::environments::hyperdrone::rotorcraft::detail::collect_props(assembly, props)){
            return false;
        }
        for(auto& part : assembly.parts){
            part.joint = hyperlux::Joint{};
        }
        for(size_t prop_i = 0; prop_i < props.size(); prop_i++){
            auto& joint = assembly.parts[props[prop_i].part].joint;
            joint.type = hyperlux::Joint::Type::REVOLUTE;
            joint.axis[0] = 0;
            joint.axis[1] = 0;
            joint.axis[2] = 1;
            for(unsigned dim = 0; dim < 3; dim++){
                joint.pivot[dim] = props[prop_i].pivot[dim];
            }
            joint.coordinate = prop_i;
        }
        return true;
    }

    // the Quad-X convention of an articulated assembly; false unless its joints are the ones
    // articulate writes
    template <typename DEVICE>
    bool init(DEVICE& device, rl::environments::hyperdrone::Rotorcraft& rotorcraft, const hyperlux::ObjectAssembly& assembly){
        std::vector<rl::environments::hyperdrone::rotorcraft::detail::Prop> props;
        if(!rl::environments::hyperdrone::rotorcraft::detail::articulated_props(assembly, props)){
            return false;
        }
        rotorcraft = {};
        rotorcraft.wheelbase = rl::environments::hyperdrone::rotorcraft::detail::hub_wheelbase(device, props);
        rotorcraft.num_props = props.size();
        for(size_t prop_i = 0; prop_i < props.size(); prop_i++){
            rotorcraft.prop_rotor[prop_i] = prop_i;
            rotorcraft.prop_spin[prop_i] = props[prop_i].hub[0] * props[prop_i].hub[1] < 0 ? 1.0f : -1.0f;
        }
        return true;
    }

    // the convention fitted to a dynamics model's rotors: each hub pairs with the nearest rotor in
    // the XY plane after normalizing both layouts by their wheelbase (a bijection, else false) and
    // the prop spins against the rotor's reaction torque
    template <typename DEVICE, typename T, typename TI>
    bool init(DEVICE& device, rl::environments::hyperdrone::Rotorcraft& rotorcraft, const hyperlux::ObjectAssembly& assembly, const T rotor_positions[][3], const T rotor_torque_directions[][3], TI num_rotors){
        std::vector<rl::environments::hyperdrone::rotorcraft::detail::Prop> props;
        if(!rl::environments::hyperdrone::rotorcraft::detail::articulated_props(assembly, props) || props.size() != (size_t)num_rotors){
            return false;
        }
        const float asset_wheelbase = rl::environments::hyperdrone::rotorcraft::detail::hub_wheelbase(device, props);
        const T dynamics_wheelbase = rotor_wheelbase(device, rotor_positions, num_rotors);
        if(!(asset_wheelbase > 0) || !(dynamics_wheelbase > 0)){
            return false;
        }
        const float scale = (float)(dynamics_wheelbase / (T)asset_wheelbase);
        if(!hyperlux::is_finite(scale) || scale <= 0){
            return false;
        }
        rl::environments::hyperdrone::Rotorcraft fitted{};
        fitted.num_props = props.size();
        std::vector<bool> matched(num_rotors, false);
        T maximum_distance_squared = 0;
        for(size_t prop_i = 0; prop_i < props.size(); prop_i++){
            TI nearest = 0;
            T nearest_distance = 0;
            for(TI rotor_i = 0; rotor_i < num_rotors; rotor_i++){
                const T dx = (T)props[prop_i].hub[0] / (T)asset_wheelbase - rotor_positions[rotor_i][0] / dynamics_wheelbase;
                const T dy = (T)props[prop_i].hub[1] / (T)asset_wheelbase - rotor_positions[rotor_i][1] / dynamics_wheelbase;
                const T distance = dx * dx + dy * dy;
                if(rotor_i == 0 || distance < nearest_distance){
                    nearest = rotor_i;
                    nearest_distance = distance;
                }
            }
            const T torque = rotor_torque_directions[nearest][2];
            if(matched[nearest] || !hyperlux::is_finite(rotor_torque_directions[nearest], 3) || torque == 0){
                return false;
            }
            matched[nearest] = true;
            maximum_distance_squared = nearest_distance > maximum_distance_squared ? nearest_distance : maximum_distance_squared;
            fitted.prop_rotor[prop_i] = (size_t)nearest;
            fitted.prop_spin[prop_i] = torque > 0 ? -1.0f : 1.0f;
        }
        rotorcraft = fitted;
        rotorcraft.wheelbase = asset_wheelbase;
        rotorcraft.scale = scale;
        rotorcraft.rotor_position_deviation = (float)math::sqrt(device.math, maximum_distance_squared);
        constexpr float threshold = rl::environments::hyperdrone::Rotorcraft::ROTOR_POSITION_WARNING_THRESHOLD;
        if(rotorcraft.rotor_position_deviation > threshold){
            log(device, device.logger, devices::logging::Verbosity::warning, "hyperdrone::rotorcraft: GLB motor positions differ from dynamics by up to ", 100 * rotorcraft.rotor_position_deviation, "% of wheelbase (", 1000 * rotorcraft.rotor_position_deviation * (float)dynamics_wheelbase, " mm horizontally after scaling; warning threshold ", 100 * threshold, "%).");
        }
        return true;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
