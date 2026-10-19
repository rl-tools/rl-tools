#include "../../../../version.h"
#include "../../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_RIG_OPERATIONS_CPU_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_RIG_OPERATIONS_CPU_H

#include "rig.h"
#include "operations_generic.h"
#include "../../../../rendering/scene.h"
#include "../../../../rendering/transforms.h"

#include <vector>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {
    namespace rl::environments::hyperdrone::rig::detail {
        inline bool part_bounds(const rendering::Object& object, float low[3], float high[3]){
            bool first = true;
            for(const auto& mesh : object.meshes){
                for(size_t vertex_i = 0; vertex_i + 2 < mesh.vertices.size(); vertex_i += 3){
                    for(unsigned dim = 0; dim < 3; dim++){
                        const float value = mesh.vertices[vertex_i + dim];
                        if(!rendering::is_finite(value)){
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

        // validates the body/prop_* convention (the first part is "body", a prop_* part has geometry)
        inline bool collect_props(const rendering::ObjectAssembly& assembly, std::vector<Prop>& props){
            props.clear();
            if(assembly.parts.empty() || assembly.objects[assembly.parts[0].object].name != "body"){
                return false;
            }
            for(size_t part_i = 0; part_i < assembly.parts.size(); part_i++){
                const auto& part = assembly.parts[part_i];
                if(!rendering::is_finite(part.transform, 12)){
                    return false;
                }
                const auto& object = assembly.objects[part.object];
                if(object.name.rfind("prop_", 0) == 0){
                    float low[3] = {0, 0, 0}, high[3] = {0, 0, 0};
                    if(!part_bounds(object, low, high)){
                        return false;
                    }
                    Prop prop{part_i, {(low[0] + high[0]) / 2, (low[1] + high[1]) / 2, 0}, {}};
                    rendering::transform_point(part.transform, prop.pivot, prop.hub);
                    if(!rendering::is_finite(prop.hub, 3)){
                        return false;
                    }
                    props.push_back(prop);
                }
                else if(object.name == "body" && part_i != 0){
                    return false;
                }
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

        // prop p turns about its hub along the part's z with the sign spin[p] as coordinate
        // coordinate_of_prop[p]; every other part is fixed
        inline void articulate(rendering::ObjectAssembly& assembly, const std::vector<Prop>& props, const size_t* coordinate_of_prop, const float* spin){
            for(auto& part : assembly.parts){
                part.joint = rendering::Joint{};
            }
            for(size_t prop_i = 0; prop_i < props.size(); prop_i++){
                auto& joint = assembly.parts[props[prop_i].part].joint;
                joint.type = rendering::Joint::Type::REVOLUTE;
                joint.axis[0] = 0;
                joint.axis[1] = 0;
                joint.axis[2] = spin[prop_i];
                for(unsigned dim = 0; dim < 3; dim++){
                    joint.pivot[dim] = props[prop_i].pivot[dim];
                }
                joint.coordinate = coordinate_of_prop[prop_i];
            }
        }
    }

    // writes the rotorcraft joints onto a body/prop_* assembly by the Quad-X convention; false on
    // convention violations (fail-fast for rigged-drone datasets), leaving the assembly unchanged
    template <typename DEVICE>
    bool init(DEVICE& device, rl::environments::hyperdrone::rig::Rotorcraft& rotorcraft, rendering::ObjectAssembly& assembly){
        std::vector<rl::environments::hyperdrone::rig::detail::Prop> props;
        if(!rl::environments::hyperdrone::rig::detail::collect_props(assembly, props)){
            return false;
        }
        std::vector<size_t> coordinate_of_prop(props.size());
        std::vector<float> spin(props.size());
        for(size_t prop_i = 0; prop_i < props.size(); prop_i++){
            coordinate_of_prop[prop_i] = prop_i;
            spin[prop_i] = props[prop_i].hub[0] * props[prop_i].hub[1] < 0 ? 1.0f : -1.0f;
        }
        rl::environments::hyperdrone::rig::detail::articulate(assembly, props, coordinate_of_prop.data(), spin.data());
        rotorcraft = {};
        rotorcraft.wheelbase = rl::environments::hyperdrone::rig::detail::hub_wheelbase(device, props);
        return true;
    }

    // the same convention fitted to a dynamics model's rotors: each hub pairs with the nearest rotor
    // in the XY plane after normalizing both layouts by their wheelbase (a bijection, else false),
    // coordinate i is the phase of rotor i and the prop spins against the rotor's reaction torque
    template <typename DEVICE, typename T, typename TI>
    bool init(DEVICE& device, rl::environments::hyperdrone::rig::Rotorcraft& rotorcraft, rendering::ObjectAssembly& assembly, const T rotor_positions[][3], const T rotor_torque_directions[][3], TI num_rotors){
        std::vector<rl::environments::hyperdrone::rig::detail::Prop> props;
        if(!rl::environments::hyperdrone::rig::detail::collect_props(assembly, props) || props.size() != (size_t)num_rotors){
            return false;
        }
        const float asset_wheelbase = rl::environments::hyperdrone::rig::detail::hub_wheelbase(device, props);
        const T dynamics_wheelbase = rotor_wheelbase(device, rotor_positions, num_rotors);
        if(!(asset_wheelbase > 0) || !(dynamics_wheelbase > 0)){
            return false;
        }
        const float scale = (float)(dynamics_wheelbase / (T)asset_wheelbase);
        if(!rendering::is_finite(scale) || scale <= 0){
            return false;
        }
        std::vector<size_t> coordinate_of_prop(props.size());
        std::vector<float> spin(props.size());
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
            if(matched[nearest] || !rendering::is_finite(rotor_torque_directions[nearest], 3) || torque == 0){
                return false;
            }
            matched[nearest] = true;
            maximum_distance_squared = nearest_distance > maximum_distance_squared ? nearest_distance : maximum_distance_squared;
            coordinate_of_prop[prop_i] = (size_t)nearest;
            spin[prop_i] = torque > 0 ? -1.0f : 1.0f;
        }
        rl::environments::hyperdrone::rig::detail::articulate(assembly, props, coordinate_of_prop.data(), spin.data());
        rotorcraft.wheelbase = asset_wheelbase;
        rotorcraft.scale = scale;
        rotorcraft.rotor_position_deviation = (float)math::sqrt(device.math, maximum_distance_squared);
        constexpr float threshold = rl::environments::hyperdrone::rig::ROTOR_POSITION_WARNING_THRESHOLD;
        if(rotorcraft.rotor_position_deviation > threshold){
            log(device, device.logger, devices::logging::Verbosity::warning, "hyperdrone::rig: GLB motor positions differ from dynamics by up to ", 100 * rotorcraft.rotor_position_deviation, "% of wheelbase (", 1000 * rotorcraft.rotor_position_deviation * (float)dynamics_wheelbase, " mm horizontally after scaling; warning threshold ", 100 * threshold, "%).");
        }
        return true;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
