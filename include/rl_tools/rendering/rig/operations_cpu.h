#include "../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RENDERING_RIG_OPERATIONS_CPU_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RENDERING_RIG_OPERATIONS_CPU_H

#include "operations_generic.h"
#include "../scene.h"

#include <cstring>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {
    // validates the declaration (parents before children, finite frames, nonzero axes),
    // normalizes the axes and assigns the coordinate offsets
    template <typename DEVICE, typename SPEC>
    bool init(DEVICE& device, rendering::rig::Model<SPEC>& model){
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        using Joint = rendering::rig::Joint;
        if(model.num_links == 0 || model.num_links > SPEC::MAX_LINKS || model.num_parts > SPEC::MAX_PARTS){
            return false;
        }
        model.num_coordinates = 0;
        for(TI link_i = 0; link_i < model.num_links; link_i++){
            auto& link = model.links[link_i];
            if((link.parent != SPEC::BASE && link.parent >= link_i) || !rendering::rig::detail::is_finite(link.origin, 12)){
                return false;
            }
            link.coordinate = model.num_coordinates;
            if(link.joint == Joint::FIXED){
                continue;
            }
            if(link.joint == Joint::SPHERICAL){
                model.num_coordinates += 4;
                continue;
            }
            if(link.joint != Joint::REVOLUTE && link.joint != Joint::PRISMATIC){
                return false;
            }
            const T norm = math::sqrt(device.math, link.axis[0]*link.axis[0] + link.axis[1]*link.axis[1] + link.axis[2]*link.axis[2]);
            if(!rendering::rig::detail::is_finite(norm) || norm <= 0){
                return false;
            }
            for(auto& component : link.axis){
                component /= norm;
            }
            model.num_coordinates += 1;
        }
        for(TI part_i = 0; part_i < model.num_parts; part_i++){
            const auto& part = model.parts[part_i];
            if(part.link >= model.num_links || !rendering::rig::detail::is_finite(part.offset, 12)){
                return false;
            }
        }
        return true;
    }

    // one fixed link per assembly part at the part's rest frame (relative to its parent part when
    // the loader recorded one): evaluated at the default coordinates it reproduces the placement
    // set_transform renders; joints are declared by editing the links afterwards
    template <typename DEVICE, typename SPEC>
    bool init(DEVICE& device, rendering::rig::Model<SPEC>& model, const rendering::ObjectAssembly& assembly){
        using TI = typename SPEC::TI;
        if(assembly.parts.size() > SPEC::MAX_LINKS || assembly.parts.size() > SPEC::MAX_PARTS){
            return false;
        }
        model.num_links = (TI)assembly.parts.size();
        model.num_parts = (TI)assembly.parts.size();
        for(TI part_i = 0; part_i < model.num_parts; part_i++){
            const auto& part = assembly.parts[part_i];
            model.links[part_i] = rendering::rig::Link<SPEC>{};
            model.parts[part_i] = rendering::rig::Part<SPEC>{};
            model.parts[part_i].link = part_i;
            float origin[12];
            if(part.parent == rendering::ObjectAssembly::Part::NO_PARENT){
                std::memcpy(origin, part.transform, sizeof(origin));
            }
            else{
                if(part.parent >= part_i){
                    return false;
                }
                model.links[part_i].parent = (TI)part.parent;
                float parent_inverse[12];
                rendering::invert_transform(assembly.parts[part.parent].transform, parent_inverse);
                rendering::compose_transforms(parent_inverse, part.transform, origin);
            }
            for(unsigned element = 0; element < 12; element++){
                model.links[part_i].origin[element] = origin[element];
            }
        }
        return init(device, model);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
