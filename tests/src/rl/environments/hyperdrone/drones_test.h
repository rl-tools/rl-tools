#pragma once

#include <rl_tools/rl/environments/hyperdrone/presets.h>

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace rlt = rl_tools;
namespace l2f = rlt::rl::environments::l2f;

using DEVICE = rlt::devices::DEVICE_FACTORY<>;
using RNG = DEVICE::SPEC::RANDOM::ENGINE<>;
using T = float;
using TI = typename DEVICE::index_t;

#ifdef RL_TOOLS_TEST_DATA_PATH
static const std::string DATA_PATH = std::string(RL_TOOLS_MACRO_TO_STR(RL_TOOLS_TEST_DATA_PATH));
#else
static const std::string DATA_PATH = "";
#endif
static const std::string SCENE_PATH = DATA_PATH.empty() ? "" : DATA_PATH + "/ProcTHOR-Train-1.glb";

namespace test_drones {
    struct DYNAMICS_STATIC_PARAMETERS: rlt::rl::environments::hyperdrone::presets::x500_fpv::DynamicsStaticParameters<T, TI> {
        static constexpr auto dynamics = l2f::parameters::dynamics::registry<l2f::parameters::dynamics::REGISTRY::crazyflie, PARAMETERS_SPEC>;
        static constexpr PARAMETERS_TYPE PARAMETER_VALUES = {{dynamics, integration, mdp}, disturbances};
    };
    using PARAMETERS_SPEC = DYNAMICS_STATIC_PARAMETERS::PARAMETERS_SPEC;
    struct WORLD_SPEC: rlt::rl::environments::hyperdrone::Specification<T, TI, DYNAMICS_STATIC_PARAMETERS> {
        static constexpr TI N_CAMERA_AGENTS = 1;
        static constexpr TI INSTANCES_PER_ENVIRONMENT = 4;
        static constexpr TI CAM_WIDTH = 32;
        static constexpr TI CAM_HEIGHT = 32;
        using SHADING = rlt::rendering::raytracing::Low;
        static constexpr TI MAX_ENTITY_SLOTS_PER_INSTANCE = 8;
        static constexpr bool SELF_VISIBLE = true;
        static constexpr bool OUTPUT_SEGMENTATION = true;
        static constexpr TI MAX_DRONES = 8;
    };
    using WORLD = rlt::rl::environments::hyperdrone::World<WORLD_SPEC>;
    using DRONE = rlt::rl::environments::hyperdrone::Drone<WORLD_SPEC>;
    using REGISTRY = l2f::parameters::dynamics::REGISTRY;
    constexpr TI INSTANCES = WORLD::INSTANCES;
    constexpr TI CAM_PIXELS = WORLD_SPEC::CAM_WIDTH * WORLD_SPEC::CAM_HEIGHT;
    constexpr TI MAX_SLOTS = WORLD_SPEC::MAX_ENTITY_SLOTS_PER_INSTANCE;
    constexpr uint32_t MISS = 0xFFFFFFFFu;

    template <typename VALUE>
    using InstanceTensor = rlt::Tensor<rlt::tensor::Specification<VALUE, TI, rlt::tensor::Shape<TI, INSTANCES>>>;
    using PARAMETERS_TENSOR = InstanceTensor<WORLD::Parameters>;
    using STATES_TENSOR = InstanceTensor<WORLD::State>;
    using MASK_TENSOR = InstanceTensor<bool>;

    template <REGISTRY MODEL>
    DRONE drone(const char* asset, T fov){
        DRONE entry;
        entry.asset = DATA_PATH + "/" + asset;
        entry.parameters.dynamics.dynamics = l2f::parameters::dynamics::registry<MODEL, PARAMETERS_SPEC>;
        entry.parameters.cameras[0].fov = fov;
        return entry;
    }
    struct Table {
        std::vector<DRONE> drones;
        std::vector<TI> asset;
    };
    inline Table one_asset_two_dynamics(){
        return {{drone<REGISTRY::x500_real>("x500.glb", 60), drone<REGISTRY::x500_sim>("x500.glb", 61)}, {0, 0}};
    }
    inline Table one_asset_per_dynamics(){
        return {{drone<REGISTRY::crazyflie>("crazyflie.glb", 60), drone<REGISTRY::arpl>("arpl.glb", 61), drone<REGISTRY::soft>("soft.glb", 62), drone<REGISTRY::x500_real>("x500.glb", 63)}, {0, 1, 2, 3}};
    }
    inline Table many_to_many(){
        return {{drone<REGISTRY::crazyflie>("crazyflie_brushless.glb", 60), drone<REGISTRY::crazyflie_openmv>("crazyflie_brushless.glb", 61), drone<REGISTRY::x500>("x500.glb", 62), drone<REGISTRY::x500_sim>("x500.glb", 63), drone<REGISTRY::crazyflie>("crazyflie.glb", 64), drone<REGISTRY::arpl>("arpl.glb", 65), drone<REGISTRY::soft>("soft.glb", 66)}, {0, 0, 1, 1, 2, 3, 4}};
    }
    // every instance flies its drone entry at its nominal dynamics, as a reset without
    // randomization leaves it, in front of its camera
    inline void arrange(DEVICE& device, PARAMETERS_TENSOR& parameters, STATES_TENSOR& states, const std::vector<DRONE>& table, const std::vector<TI>& drones, T phase = 0){
        for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
            auto& instance_parameters = rlt::get_ref(device, parameters, instance_i);
            auto& state = rlt::get_ref(device, states, instance_i);
            instance_parameters.agents[0].drone = drones[instance_i];
            instance_parameters.agents[0].dynamics.dynamics = table[drones[instance_i]].parameters.dynamics.dynamics;
            instance_parameters.scene_yaw_cos = 1;
            instance_parameters.scene_yaw_sin = 0;
            instance_parameters.scene_translation[0] = 0;
            instance_parameters.scene_translation[1] = 0;
            instance_parameters.scene_translation[2] = 100;
            using rlt::rotor_wheelbase;
            const T wheelbase = rotor_wheelbase(device, instance_parameters.agents[0].dynamics.dynamics.rotor_positions, TI(4));
            instance_parameters.agents[0].cameras[0].mount.pose.position[0] = (T)-2 * wheelbase;
            instance_parameters.agents[0].cameras[0].mount.pose.position[1] = 0;
            instance_parameters.agents[0].cameras[0].mount.pose.position[2] = (T)0.75 * wheelbase;
            const T half_pitch = std::atan2((T)0.75, (T)2) / 2;
            instance_parameters.agents[0].cameras[0].mount.pose.orientation[0] = std::cos(half_pitch);
            instance_parameters.agents[0].cameras[0].mount.pose.orientation[1] = 0;
            instance_parameters.agents[0].cameras[0].mount.pose.orientation[2] = std::sin(half_pitch);
            instance_parameters.agents[0].cameras[0].mount.pose.orientation[3] = 0;
            instance_parameters.agents[0].cameras[0].fov = 60;
            state.orientation[0] = 1;
            state.orientation[1] = 0;
            state.orientation[2] = 0;
            state.orientation[3] = 0;
            state.position[0] = 0;
            state.position[1] = 0;
            state.position[2] = 0;
            for(TI rotor_i = 0; rotor_i < 4; rotor_i++){
                const auto& position = instance_parameters.agents[0].dynamics.dynamics.rotor_positions[rotor_i];
                const TI quadrant = (position[0] > 0 ? 1 : 0) + (position[1] > 0 ? 2 : 0);
                state.rotor_phase[rotor_i] = phase * (T)quadrant;
            }
        }
    }
    template <typename RENDER_DEVICE>
    std::vector<std::vector<uint32_t>> own_masks(DEVICE& device, RENDER_DEVICE& render_device, WORLD& world, const WORLD::SharedContext& shared){
        const uint32_t scene_instances = (uint32_t)shared.library.scenes.front().instances.size();
        std::vector<uint32_t> segmentation(INSTANCES * CAM_PIXELS);
        rlt::Tensor<rlt::tensor::Specification<uint32_t, TI, rlt::tensor::Shape<TI, INSTANCES, WORLD_SPEC::CAM_HEIGHT, WORLD_SPEC::CAM_WIDTH>>> alias;
        alias._data = segmentation.data();
        auto& renderer = rlt::get_renderer(render_device, world);
        rlt::copy(renderer.device, device, rlt::segmentation_buffer(render_device, renderer), alias);
        std::vector<std::vector<uint32_t>> masks(INSTANCES, std::vector<uint32_t>(CAM_PIXELS, MISS));
        for(TI camera_i = 0; camera_i < INSTANCES; camera_i++){
            const uint32_t own_first = scene_instances + (uint32_t)(camera_i * MAX_SLOTS);
            for(TI pixel_i = 0; pixel_i < CAM_PIXELS; pixel_i++){
                const uint32_t id = segmentation[camera_i * CAM_PIXELS + pixel_i];
                if(id == MISS || id < scene_instances){
                    continue;
                }
                EXPECT_GE(id, own_first) << "camera " << camera_i;
                EXPECT_LT(id, own_first + (uint32_t)MAX_SLOTS) << "camera " << camera_i;
                masks[camera_i][pixel_i] = id - own_first;
            }
        }
        return masks;
    }
    // the drone mesh an instance's region holds in a hot slot's host truth (a CPU World's
    // renderer), drone_meshes.size() for none
    template <typename TEST_WORLD>
    TI rendered_mesh(const TEST_WORLD& world, TI slot_i, TI instance_i){
        const auto& renderer = world.slots[slot_i].renderer;
        const auto object = renderer.overlays[instance_i].slots[world.drone_slot_offset].object;
        for(TI mesh_i = 0; mesh_i < (TI)world.drone_meshes.size(); mesh_i++){
            if(renderer.asset_part_objects[renderer.assets[world.drone_meshes[mesh_i].index].first_part] == object){
                return mesh_i;
            }
        }
        return (TI)world.drone_meshes.size();
    }
    inline void check_placements(const WORLD& world, const WORLD::SharedContext& shared, const std::vector<TI>& drones){
        const auto& renderer = world.slots[world.active_slot].renderer;
        for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
            EXPECT_EQ(rendered_mesh(world, world.active_slot, instance_i), shared.drone_bodies[drones[instance_i]].mesh) << "instance " << instance_i;
            for(TI slot_i = 0; slot_i < MAX_SLOTS; slot_i++){
                EXPECT_EQ(renderer.overlays[instance_i].slots[slot_i].active, slot_i >= world.drone_slot_offset && slot_i < world.drone_slot_offset + world.drone_parts) << "instance " << instance_i << " slot " << slot_i;
            }
        }
    }
    inline bool seen(const std::vector<uint32_t>& mask){
        for(uint32_t value : mask){
            if(value != MISS){
                return true;
            }
        }
        return false;
    }
}
