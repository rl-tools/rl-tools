#pragma once
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "verbosity.h"
#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/rendering/rig/operations_cpu.h>

#include "bindings.h"

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

// RigModel is the one Python-facing rig type; this header lets a second translation unit
// (rotorcraft_bindings.cpp) construct it from a convention-built model
namespace hyperdrone::rig_bindings {
    namespace nb = nanobind;
    namespace rlt = rl_tools;

    using IndexArray = nb::ndarray<const int64_t, nb::c_contig, nb::device::cpu>;
    using OutputArray = nb::ndarray<float, nb::c_contig, nb::device::cpu>;
    using RIG_SPEC = rlt::rendering::rig::Specification<float, size_t, 64, 64>;
    using Joint = rlt::rendering::rig::Joint;
    using DEVICE = rlt::devices::DEVICE_FACTORY<hyperdrone::CPUSpecification>;

    inline DEVICE& device(){
        static DEVICE instance;
        return instance;
    }

    inline Joint parse_joint(const std::string& name){
        if(name == "fixed"){
            return Joint::FIXED;
        }
        if(name == "revolute"){
            return Joint::REVOLUTE;
        }
        if(name == "prismatic"){
            return Joint::PRISMATIC;
        }
        if(name == "spherical"){
            return Joint::SPHERICAL;
        }
        throw std::invalid_argument("hyperdrone: unknown joint type \"" + name + "\" (fixed, revolute, prismatic, spherical)");
    }

    inline const char* joint_name(Joint joint){
        switch(joint){
            case Joint::FIXED: return "fixed";
            case Joint::REVOLUTE: return "revolute";
            case Joint::PRISMATIC: return "prismatic";
            case Joint::SPHERICAL: return "spherical";
        }
        throw std::logic_error("hyperdrone: joint type without a name");
    }

    template <typename ARRAY>
    void check_rows(const ARRAY& values, size_t count, const char* what){
        if(values.ndim() != 3 || values.shape(0) != count || values.shape(1) != 3 || values.shape(2) != 4){
            throw std::invalid_argument(std::string("hyperdrone: ") + what + " must have shape (" + std::to_string(count) + ", 3, 4) and dtype float32");
        }
    }

    struct RigModel {
        rlt::rendering::rig::Model<RIG_SPEC> model;
        rlt::rendering::rig::Buffer<RIG_SPEC> buffer;

        explicit RigModel(const rlt::rendering::rig::Model<RIG_SPEC>& model): model(model) {}

        RigModel(IndexArray parents, const std::vector<std::string>& joints, FloatArray axes, FloatArray origins, IndexArray part_links, FloatArray offsets){
            const size_t num_links = parents.size();
            const size_t num_parts = part_links.size();
            if(num_links == 0 || num_links > RIG_SPEC::MAX_LINKS || num_parts > RIG_SPEC::MAX_PARTS){
                throw std::invalid_argument("hyperdrone: a rig has between 1 and 64 links and at most 64 parts");
            }
            if(parents.ndim() != 1 || joints.size() != num_links || axes.ndim() != 2 || axes.shape(0) != num_links || axes.shape(1) != 3 || part_links.ndim() != 1){
                throw std::invalid_argument("hyperdrone: rig arrays must have shapes parents (L,), joints (L,), axes (L, 3), part_links (K,)");
            }
            check_rows(origins, num_links, "origins");
            check_rows(offsets, num_parts, "offsets");
            model.num_links = num_links;
            model.num_parts = num_parts;
            for(size_t link_i = 0; link_i < num_links; link_i++){
                auto& link = model.links[link_i];
                link = rlt::rendering::rig::Link<RIG_SPEC>{};
                const int64_t parent = parents.data()[link_i];
                link.parent = parent < 0 ? RIG_SPEC::BASE : (size_t)parent;
                link.joint = parse_joint(joints[link_i]);
                std::memcpy(link.axis, axes.data() + link_i * 3, sizeof(link.axis));
                std::memcpy(link.origin, origins.data() + link_i * 12, sizeof(link.origin));
            }
            for(size_t part_i = 0; part_i < num_parts; part_i++){
                auto& part = model.parts[part_i];
                part = rlt::rendering::rig::Part<RIG_SPEC>{};
                const int64_t link = part_links.data()[part_i];
                if(link < 0 || (size_t)link >= num_links){
                    throw std::invalid_argument("hyperdrone: part_links must index the rig's links");
                }
                part.link = (size_t)link;
                std::memcpy(part.offset, offsets.data() + part_i * 12, sizeof(part.offset));
            }
            if(!rlt::init(device(), model)){
                throw std::invalid_argument("hyperdrone: invalid rig: links must be declared parent before child with finite frames and nonzero joint axes");
            }
        }

        void check_coordinates(const FloatArray& coordinates, size_t batch) const {
            const bool valid = batch == 1 ? (coordinates.ndim() == 1 && coordinates.shape(0) == model.num_coordinates) || (coordinates.ndim() == 2 && coordinates.shape(0) == 1 && coordinates.shape(1) == model.num_coordinates)
                                          : coordinates.ndim() == 2 && coordinates.shape(0) == batch && coordinates.shape(1) == model.num_coordinates;
            if(!valid){
                throw std::invalid_argument("hyperdrone: joint coordinates must have shape (num_coordinates,) or (batch, num_coordinates) matching the base poses");
            }
            for(size_t i = 0; i < coordinates.size(); i++){
                if(!rlt::rendering::rig::detail::is_finite(coordinates.data()[i])){
                    throw std::invalid_argument("hyperdrone: joint coordinates must be finite");
                }
            }
            for(size_t sample = 0; sample < batch; sample++){
                const float* q = coordinates.data() + sample * model.num_coordinates;
                for(size_t link_i = 0; link_i < model.num_links; link_i++){
                    const auto& link = model.links[link_i];
                    if(link.joint == Joint::SPHERICAL){
                        const float* quaternion = q + link.coordinate;
                        if(quaternion[0]*quaternion[0] + quaternion[1]*quaternion[1] + quaternion[2]*quaternion[2] + quaternion[3]*quaternion[3] <= 0){
                            throw std::invalid_argument("hyperdrone: a spherical joint's quaternion must have nonzero norm");
                        }
                    }
                }
            }
        }

        static size_t batch_size(const FloatArray& base){
            if(base.ndim() == 2 && base.shape(0) == 3 && base.shape(1) == 4){
                return 1;
            }
            if(base.ndim() == 3 && base.shape(1) == 3 && base.shape(2) == 4){
                return base.shape(0);
            }
            throw std::invalid_argument("hyperdrone: base poses must have shape (3, 4) or (batch, 3, 4) and dtype float32");
        }

        static void check_finite(const FloatArray& values, const char* what){
            for(size_t i = 0; i < values.size(); i++){
                if(!rlt::rendering::rig::detail::is_finite(values.data()[i])){
                    throw std::invalid_argument(std::string("hyperdrone: ") + what + " must be finite");
                }
            }
        }
    };
}
