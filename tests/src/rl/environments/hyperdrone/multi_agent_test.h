#pragma once

#include <gtest/gtest.h>
#include <vector>

namespace test_agent_parameters {
    namespace rlt = rl_tools;
    template <typename WORLD, typename DEVICE, typename PARAMETER_SPEC, typename COUNTS>
    bool check_samples(DEVICE& device, const typename WORLD::SharedContext& shared, const rlt::Tensor<PARAMETER_SPEC>& parameters, const typename WORLD::Parameters& untouched, COUNTS& counts){
        using TI = typename WORLD::TI;
        const auto& sampled = rlt::get_ref(device, parameters, (TI)0);
        const auto& preserved = rlt::get_ref(device, parameters, (TI)1);
        for(TI agent_i = 0; agent_i < WORLD::N_AGENTS; agent_i++){
            const auto& agent = sampled.agents[agent_i];
            EXPECT_LT(agent.drone, shared.drones.size());
            if(agent.drone < shared.drones.size()){
                counts[agent_i][agent.drone]++;
                const auto& nominal = shared.drones[agent.drone].parameters;
                EXPECT_EQ(agent.dynamics.dynamics.mass, nominal.dynamics.dynamics.mass);
                EXPECT_EQ(agent.cameras[0].fov, nominal.cameras[0].fov);
            }
            EXPECT_EQ(preserved.agents[agent_i].drone, untouched.agents[agent_i].drone);
            EXPECT_EQ(preserved.agents[agent_i].cameras[0].fov, untouched.agents[agent_i].cameras[0].fov);
            EXPECT_EQ(preserved.agents[agent_i].dynamics.dynamics.mass, untouched.agents[agent_i].dynamics.dynamics.mass);
        }
        return sampled.agents[0].drone != sampled.agents[1].drone;
    }

    template <typename DEVICE, typename WORLD, typename TENSORS>
    void check_individual_steps(DEVICE& device, WORLD& world, TENSORS& tensors){
        using T = typename WORLD::T;
        using TI = typename WORLD::TI;
        typename DEVICE::SPEC::RANDOM::template ENGINE<> rng;
        rlt::malloc(device, rng);
        rlt::init(device, rng, 42);
        for(TI instance_i = 0; instance_i < WORLD::INSTANCES; instance_i++){
            auto& parameters = rlt::get_ref(device, tensors.parameters, instance_i);
            const auto& before = rlt::get_ref(device, tensors.states, instance_i);
            const auto& after = rlt::get_ref(device, tensors.next_states, instance_i);
            EXPECT_NE(parameters.agents[0].dynamics.dynamics.mass, parameters.agents[1].dynamics.dynamics.mass);
            EXPECT_NE(after.agent_states[0].linear_velocity[2], after.agent_states[1].linear_velocity[2]);
            for(TI agent_i = 0; agent_i < WORLD::N_AGENTS; agent_i++){
                rlt::Matrix<rlt::matrix::Specification<T, TI, 1, WORLD::DYNAMICS_ENV::ACTION_DIM, false>> action;
                rlt::set_all(device, action, (T)0);
                auto expected = before.agent_states[agent_i];
                rlt::step(device, world.dynamics, parameters.agents[agent_i].dynamics, before.agent_states[agent_i], action, expected, rng);
                const auto& actual = after.agent_states[agent_i];
                for(TI dim = 0; dim < 3; dim++){
                    EXPECT_NEAR(actual.position[dim], expected.position[dim], 1e-5);
                    EXPECT_NEAR(actual.linear_velocity[dim], expected.linear_velocity[dim], 1e-5);
                    EXPECT_NEAR(actual.angular_velocity[dim], expected.angular_velocity[dim], 1e-4);
                }
                for(TI rotor_i = 0; rotor_i < WORLD::DYNAMICS_ENV::ACTION_DIM; rotor_i++){
                    EXPECT_NEAR(actual.rpm[rotor_i], expected.rpm[rotor_i], 1e-3);
                    EXPECT_NEAR(actual.rotor_phase[rotor_i], expected.rotor_phase[rotor_i], 1e-4);
                }
            }
        }
        rlt::free(device, rng);
    }

    template <typename DEVICE, typename RENDER_DEVICE, typename WORLD, typename TENSORS>
    void check_rigs(DEVICE& device, RENDER_DEVICE& render_device, WORLD& world, const typename WORLD::SharedContext& shared, TENSORS& tensors){
        using TI = typename WORLD::TI;
        using SPEC = typename WORLD::SPEC;
        auto& renderer = rlt::get_renderer(render_device, world);
        std::vector<float> rows(WORLD::INSTANCES * SPEC::MAX_ENTITY_SLOTS_PER_INSTANCE * 12);
        rlt::Tensor<typename decltype(renderer.transforms)::SPEC> alias;
        alias._data = rows.data();
        rlt::copy(renderer.device, device, rlt::transforms(render_device, renderer), alias);
        std::vector<rlt::rendering::Camera<typename WORLD::T>> cameras(WORLD::INSTANCES * WORLD::N_AGENTS);
        rlt::Tensor<typename decltype(renderer.cameras)::SPEC> camera_alias;
        camera_alias._data = cameras.data();
        rlt::copy(renderer.device, device, rlt::cameras(render_device, renderer), camera_alias);
        for(TI instance_i = 0; instance_i < WORLD::INSTANCES; instance_i++){
            const auto& parameters = rlt::get_ref(device, tensors.parameters, instance_i);
            const auto& state = rlt::get_ref(device, tensors.states, instance_i);
            for(TI agent_i = 0; agent_i < WORLD::N_AGENTS; agent_i++){
                const auto& body = shared.drone_bodies[parameters.agents[agent_i].drone];
                const auto& agent = state.agent_states[agent_i];
                for(TI dim = 0; dim < 3; dim++){
                    EXPECT_NEAR(cameras[instance_i * WORLD::N_AGENTS + agent_i].pos[dim], parameters.scene_translation[dim] + agent.position[dim] + parameters.agents[agent_i].cameras[0].mount.pose.position[dim], 1e-5);
                }
                using RIG_SPEC = typename decltype(body.rig)::SPEC;
                float base[12];
                rlt::rl::environments::hyperdrone::rig::make_body_transform(device, agent.orientation, agent.position, parameters.scene_translation, parameters.scene_yaw_cos, parameters.scene_yaw_sin, base);
                float coordinates[WORLD::DYNAMICS_ENV::ACTION_DIM];
                for(TI rotor_i = 0; rotor_i < WORLD::DYNAMICS_ENV::ACTION_DIM; rotor_i++){
                    coordinates[rotor_i] = agent.rotor_phase[rotor_i];
                }
                rlt::rendering::rig::Buffer<RIG_SPEC> buffer;
                float poses[RIG_SPEC::MAX_PARTS][12];
                rlt::evaluate(device, body.rig, base, coordinates, buffer, &poses[0][0]);
                for(TI part_i = 0; part_i < body.rig.num_parts; part_i++){
                    const TI slot = instance_i * SPEC::MAX_ENTITY_SLOTS_PER_INSTANCE + world.drone_slot_offset + agent_i * world.drone_parts + part_i;
                    for(TI element = 0; element < 12; element++){
                        EXPECT_NEAR(rows[slot * 12 + element], poses[part_i][element], 1e-4) << "instance " << instance_i << " agent " << agent_i << " part " << part_i;
                    }
                }
            }
        }
    }
}
