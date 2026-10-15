#pragma once

#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <vector>
#include "multi_agent_test.h"

namespace test_multicamera {
    using namespace rl_tools;
    using rl::environments::hyperdrone::_agent_state;

    template <typename BASE, typename BASE::TI AGENTS, typename BASE::TI CAMERA_AGENTS>
    struct Specification: BASE {
        static constexpr typename BASE::TI N_AGENTS = AGENTS;
        static constexpr typename BASE::TI N_CAMERA_AGENTS = CAMERA_AGENTS;
        static constexpr typename BASE::TI HISTORY_LENGTH = 2;
        static constexpr bool ENABLE_MOTION_BLUR = true;
        static constexpr typename BASE::TI MOTION_BLUR_SAMPLES = 2;
        using CAMERA_PROFILE = typename rendering::raytracing::outputs::Concat<typename BASE::CAMERA_PROFILE, rendering::raytracing::outputs::List<rendering::raytracing::outputs::Flow<>, rendering::raytracing::outputs::FlowStatus<>>>::type;
        using CAMERA_OUTPUTS = rl::environments::hyperdrone::CameraOutputs<CAMERA_PROFILE, CAMERA_PROFILE>;
        static constexpr typename BASE::T CAMERA_FOV_RANDOMIZATION_RANGE = 2;
    };

    template <typename WORLD>
    struct Tensors {
        using TI = typename WORLD::TI;
        Tensor<tensor::Specification<typename WORLD::Parameters, TI, tensor::Shape<TI, WORLD::INSTANCES>>> parameters;
        Tensor<tensor::Specification<typename WORLD::State, TI, tensor::Shape<TI, WORLD::INSTANCES>>> states;
        Tensor<tensor::Specification<typename WORLD::State, TI, tensor::Shape<TI, WORLD::INSTANCES>>> next_states;
        Tensor<tensor::Specification<float, TI, tensor::Shape<TI, WORLD::INSTANCES, WORLD::ACTION_DIM>>> actions;
        Tensor<tensor::Specification<bool, TI, tensor::Shape<TI, WORLD::INSTANCES>>> reset;
        Tensor<tensor::Specification<float, TI, tensor::Shape<TI, WORLD::INSTANCES, WORLD::OBSERVATION_DIM>>> observations;
        Tensor<tensor::Specification<float, TI, tensor::Shape<TI, WORLD::INSTANCES, WORLD::OBSERVATION_DIM_PRIVILEGED>>> observations_privileged;
        Tensor<tensor::Specification<float, TI, tensor::Shape<TI, WORLD::INSTANCES>>> rewards;
        Tensor<tensor::Specification<bool, TI, tensor::Shape<TI, WORLD::INSTANCES>>> terminated_flags;
        template <typename DEVICE>
        void allocate(DEVICE& device) {
            malloc(device, parameters);
            malloc(device, states);
            malloc(device, next_states);
            malloc(device, actions);
            set_all(device, actions, 0);
            malloc(device, reset);
            malloc(device, observations);
            malloc(device, observations_privileged);
            malloc(device, rewards);
            malloc(device, terminated_flags);
        }
        template <typename DEVICE>
        void deallocate(DEVICE& device) {
            free(device, parameters);
            free(device, states);
            free(device, next_states);
            free(device, actions);
            free(device, reset);
            free(device, observations);
            free(device, observations_privileged);
            free(device, rewards);
            free(device, terminated_flags);
        }
    };

    template <typename T>
    void check_camera(const rendering::Camera<T>& actual, const rendering::Camera<T>& expected) {
        for (unsigned dim = 0; dim < 3; dim++) {
            EXPECT_NEAR(actual.pos[dim], expected.pos[dim], 1e-5);
            EXPECT_NEAR(actual.dir_00[dim], expected.dir_00[dim], 1e-5);
            EXPECT_NEAR(actual.dir_du[dim], expected.dir_du[dim], 1e-5);
            EXPECT_NEAR(actual.dir_dv[dim], expected.dir_dv[dim], 1e-5);
        }
    }

    template <typename WORLD, typename HOST, typename DEVICE>
    void run(HOST& host, DEVICE& device, const std::string& scene_path, const std::string& drone_path) {
        using T = typename WORLD::T;
        using TI = typename WORLD::TI;
        using SPEC = typename WORLD::SPEC;
        constexpr TI NUM_CAMERAS = WORLD::INSTANCES * WORLD::N_VIEWS;
        constexpr TI PIXELS = SPEC::CAM_WIDTH * SPEC::CAM_HEIGHT;
        static_assert(WORLD::N_VIEWS == WORLD::N_CAMERA_AGENTS * 2);
        static_assert(WORLD::OBSERVATION_DIM == WORLD::N_VIEWS * WORLD::FRAME_DIM);
        static_assert(WORLD::ACTION_DIM == WORLD::N_AGENTS * WORLD::DYNAMICS_ENV::ACTION_DIM);
        static_assert(WORLD::OBSERVATION_DIM_PRIVILEGED == WORLD::N_AGENTS * WORLD::DYNAMICS_ENV::ObservationPrivileged::DIM);
        WORLD world;
        typename WORLD::SharedContext shared;
        malloc(host, shared.library);
        malloc(device, world);
        shared.drones.push_back({drone_path, {}});
        using PARAMETERS_SPEC = typename WORLD::DYNAMICS_ENV::Parameters::BASE_SPEC;
        shared.drones[0].parameters.dynamics.dynamics = rl::environments::l2f::parameters::dynamics::registry<rl::environments::l2f::parameters::dynamics::REGISTRY::x500, PARAMETERS_SPEC>;
        shared.drones[0].parameters.cameras[0].fov = 60;
        shared.drones[0].parameters.cameras[1].fov = 85;
        rendering::datasets::glb::Dataset dataset{{}, {scene_path}};
        typename decltype(dataset)::Corpus corpus;
        enumerate(host, dataset, corpus);
        init(device, world, shared, dataset, corpus, 0, 1, 0);
        typename DEVICE::SPEC::RANDOM::template ENGINE<> rng;
        malloc(device, rng);
        init(device, rng, 211);
        typename HOST::SPEC::RANDOM::template ENGINE<> reference_rng;
        malloc(host, reference_rng);
        init(host, reference_rng, 211);
        Tensors<WORLD> resident, staging;
        resident.allocate(device);
        staging.allocate(host);
        for (TI instance_i = 0; instance_i < WORLD::INSTANCES; instance_i++) {
            for (TI agent_i = 0; agent_i < WORLD::N_AGENTS; agent_i++) {
                for (TI motor_i = 0; motor_i < WORLD::DYNAMICS_ENV::ACTION_DIM; motor_i++) {
                    set(host, staging.actions, (T)0.2 * agent_i, instance_i, agent_i * WORLD::DYNAMICS_ENV::ACTION_DIM + motor_i);
                }
            }
        }
        copy(host, device, staging.actions, resident.actions);
        set_all(device, resident.reset, true);
        sample_initial_parameters(device, world, resident.parameters, resident.reset, rng);
        sample_initial_state(device, world, resident.parameters, resident.states, resident.reset, rng);
        copy(device, host, resident.parameters, staging.parameters);
        copy(device, host, resident.states, staging.states);
        for (TI instance_i = 0; instance_i < WORLD::INSTANCES; instance_i++) {
            auto& parameters = get_ref(host, staging.parameters, instance_i);
            auto& state = get_ref(host, staging.states, instance_i);
            // ProcTHOR-Train-1 living-room free space used by the raytracing overlay goldens.
            parameters.scene_translation[0] = (T)-5.7;
            parameters.scene_translation[1] = (T)-5.2;
            parameters.scene_translation[2] = (T)1.5;
            parameters.scene_yaw_cos = 1;
            parameters.scene_yaw_sin = 0;
            parameters.shutter_fraction = (T)0.5;
            for (TI agent_i = 0; agent_i < WORLD::N_AGENTS; agent_i++) {
                auto& agent = _agent_state<SPEC>(state, agent_i);
                parameters.agents[agent_i].dynamics.mdp.reward.constant = (T)(agent_i + 1);
                for (TI dim = 0; dim < 3; dim++) {
                    agent.position[dim] = dim == 0 ? (T)0.7 * agent_i : (T)0;
                    agent.orientation[dim + 1] = 0;
                }
                agent.orientation[0] = 1;
                if (agent_i > 0 && agent_i + 1 < WORLD::N_AGENTS) {
                    agent.position[1] = (T)-0.8;
                }
                for (TI camera_i = 0; camera_i < WORLD::N_CAMERAS_PER_AGENT; camera_i++) {
                    auto& camera = parameters.agents[agent_i].cameras[camera_i];
                    const auto& nominal = shared.drones[0].parameters.cameras[camera_i];
                    EXPECT_LE(std::abs(camera.fov - nominal.fov), SPEC::CAMERA_FOV_RANDOMIZATION_RANGE);
                    EXPECT_NE(camera.fov, nominal.fov);
                    EXPECT_EQ(world.parameters.agents[agent_i].cameras[camera_i].fov, nominal.fov);
                    camera.mount.pose.position[0] = camera_i == 0 ? (T)0.2 : (T)-0.6;
                    camera.mount.pose.position[1] = camera_i == 0 ? (T)0 : (T)0.12;
                    camera.mount.pose.position[2] = camera_i == 0 ? (T)0.2 : (T)0.3;
                }
            }
        }
        auto& renderer = get_renderer(device, world);
        for (TI instance_i = 0; instance_i < WORLD::INSTANCES; instance_i++) {
            TI active_parts = 0;
            for (TI slot = 0; slot < SPEC::MAX_ENTITY_SLOTS_PER_INSTANCE; slot++) {
                active_parts += renderer.overlays[instance_i].slots[slot].active ? 1 : 0;
            }
            EXPECT_EQ(active_parts, WORLD::N_AGENTS * world.drone_parts);
            for (TI view = 0; view < WORLD::N_VIEWS; view++) {
                EXPECT_EQ(renderer.attachments[instance_i * WORLD::N_VIEWS + view], instance_i);
            }
        }
        std::vector<rendering::Camera<T>> previous(NUM_CAMERAS), current(NUM_CAMERAS), opened(NUM_CAMERAS);
        Tensor<typename WORLD::PREV_CAMERAS_SPEC> current_alias, opened_alias;
        current_alias._data = current.data();
        opened_alias._data = opened.data();
        std::vector<float> rgb(NUM_CAMERAS * WORLD::FRAME_DIM);
        Tensor<rl_tools::rendering::raytracing::outputs::TensorSpecification<typename WORLD::RENDERER_SPEC, rl_tools::rendering::raytracing::outputs::RGB<rl_tools::rendering::raytracing::outputs::format::Float32>>> rgb_alias;
        rgb_alias._data = rgb.data();
        std::vector<float> history(SPEC::HISTORY_LENGTH * NUM_CAMERAS * WORLD::FRAME_DIM);
        Tensor<typename WORLD::HISTORY_SPEC> history_alias;
        history_alias._data = history.data();
        std::vector<float> flow(NUM_CAMERAS * PIXELS * 2);
        Tensor<rl_tools::rendering::raytracing::outputs::TensorSpecification<typename WORLD::RENDERER_SPEC, rl_tools::rendering::raytracing::outputs::Flow<>>> flow_alias;
        flow_alias._data = flow.data();
        std::vector<uint32_t> segmentation(NUM_CAMERAS * PIXELS);
        Tensor<rl_tools::rendering::raytracing::outputs::TensorSpecification<typename WORLD::RENDERER_SPEC, rl_tools::rendering::raytracing::outputs::Segmentation<>>> segmentation_alias;
        segmentation_alias._data = segmentation.data();
        const uint32_t scene_instances = (uint32_t)shared.library.scenes.front().instances.size();
        for (TI frame = 0; frame < 3; frame++) {
            if (frame > 0) {
                step(device, world, resident.parameters, resident.states, resident.actions, resident.next_states, rng);
                reward(device, world, resident.parameters, resident.states, resident.actions, resident.next_states, resident.rewards, rng);
                copy(device, host, resident.next_states, staging.next_states);
                copy(device, host, resident.rewards, staging.rewards);
                for (TI instance_i = 0; instance_i < WORLD::INSTANCES; instance_i++) {
                    const auto& parameters = get_ref(host, staging.parameters, instance_i);
                    EXPECT_NEAR(get(host, staging.rewards, instance_i), (T)(WORLD::N_AGENTS + 1) / 2, 1e-6);
                    for (TI agent_i = 0; agent_i < WORLD::N_AGENTS; agent_i++) {
                        const auto& before = _agent_state<SPEC>(get_ref(host, staging.states, instance_i), agent_i);
                        const auto& actual = _agent_state<SPEC>(get_ref(host, staging.next_states, instance_i), agent_i);
                        auto expected = before;
                        Matrix<matrix::Specification<T, TI, 1, WORLD::DYNAMICS_ENV::ACTION_DIM, false>> action;
                        for (TI motor_i = 0; motor_i < WORLD::DYNAMICS_ENV::ACTION_DIM; motor_i++) {
                            set(action, 0, motor_i, get(host, staging.actions, instance_i, agent_i * WORLD::DYNAMICS_ENV::ACTION_DIM + motor_i));
                        }
                        step(host, world.dynamics, parameters.agents[agent_i].dynamics, before, action, expected, reference_rng);
                        for (TI dim = 0; dim < 3; dim++) {
                            EXPECT_NEAR(actual.position[dim], expected.position[dim], 1e-5);
                            EXPECT_NEAR(actual.linear_velocity[dim], expected.linear_velocity[dim], 1e-5);
                            EXPECT_NEAR(actual.angular_velocity[dim], expected.angular_velocity[dim], 1e-4);
                        }
                        for (TI motor_i = 0; motor_i < WORLD::DYNAMICS_ENV::ACTION_DIM; motor_i++) {
                            EXPECT_NEAR(actual.rpm[motor_i], expected.rpm[motor_i], 1e-3);
                            EXPECT_NEAR(actual.rotor_phase[motor_i], expected.rotor_phase[motor_i], 1e-4);
                        }
                    }
                }
            }
            set_all(host, staging.reset, frame != 1);
            if (frame == 1) {
                set(host, staging.reset, true, (TI)0);
            }
            if (frame == 2) {
                select_scene(device, world, (TI)0);
            }
            for (TI instance_i = 0; instance_i < WORLD::INSTANCES; instance_i++) {
                if (frame > 0) {
                    auto& state = get_ref(host, staging.states, instance_i);
                    for (TI agent_i = 0; agent_i < WORLD::N_AGENTS; agent_i++) {
                        auto& agent = _agent_state<SPEC>(state, agent_i);
                        agent.position[1] += (T)0.06;
                    }
                }
            }
            copy(host, device, staging.parameters, resident.parameters);
            copy(host, device, staging.states, resident.states);
            copy(host, device, staging.reset, resident.reset);
            render(device, world, resident.parameters, resident.states, resident.reset);
            observe(device, world, resident.parameters, resident.states, typename WORLD::Observation{}, resident.observations, rng);
            observe(device, world, resident.parameters, resident.states, typename WORLD::ObservationPrivileged{}, resident.observations_privileged, rng);
            copy(device, host, resident.observations, staging.observations);
            copy(device, host, resident.observations_privileged, staging.observations_privileged);
            copy(renderer.device, host, cameras(device, renderer), current_alias);
            copy(renderer.device, host, cameras_open(device, renderer), opened_alias);
            copy(renderer.device, host, output(device, renderer, rl_tools::rendering::raytracing::outputs::RGB<rl_tools::rendering::raytracing::outputs::format::Float32>{}), rgb_alias);
            copy(device, host, world.history, history_alias);
            copy(renderer.device, host, output(device, renderer, rl_tools::rendering::raytracing::outputs::Flow<>{}), flow_alias);
            copy(renderer.device, host, output(device, renderer, rl_tools::rendering::raytracing::outputs::Segmentation<>{}), segmentation_alias);
            test_agent_parameters::check_rigs(host, device, world, shared, staging);
            test_agent_parameters::check_semantics(host, world, shared);
            for (TI instance_i = 0; instance_i < WORLD::INSTANCES; instance_i++) {
                const auto& parameters = get_ref(host, staging.parameters, instance_i);
                const auto& state = get_ref(host, staging.states, instance_i);
                for (TI agent_i = 0; agent_i < WORLD::N_AGENTS; agent_i++) {
                    const auto& agent = _agent_state<SPEC>(state, agent_i);
                    for (TI dim = 0; dim < 3; dim++) {
                        EXPECT_NEAR(get(host, staging.observations_privileged, instance_i, agent_i * WORLD::DYNAMICS_ENV::ObservationPrivileged::DIM + dim), agent.position[dim], 1e-6);
                    }
                }
                for (TI agent_i = 0; agent_i < WORLD::N_CAMERA_AGENTS; agent_i++) {
                    const auto& agent = _agent_state<SPEC>(state, agent_i);
                    for (TI camera_i = 0; camera_i < WORLD::N_CAMERAS_PER_AGENT; camera_i++) {
                        const TI view = agent_i * WORLD::N_CAMERAS_PER_AGENT + camera_i;
                        const TI index = instance_i * WORLD::N_VIEWS + view;
                        const auto expected = rl::environments::hyperdrone::make_camera<HOST, T>(host, parameters.agents[agent_i].cameras[camera_i], agent.orientation, agent.position, (T)SPEC::CAM_WIDTH / SPEC::CAM_HEIGHT, parameters.scene_translation, parameters.scene_yaw_cos, parameters.scene_yaw_sin);
                        check_camera(current[index], expected);
                        const bool reset = get(host, staging.reset, instance_i);
                        check_camera(opened[index], reset ? expected : rl::environments::hyperdrone::interpolate_camera(expected, previous[index], parameters.shutter_fraction));
                        T max_flow = 0;
                        TI max_flow_pixel = 0;
                        bool drone_seen = false;
                        for (TI pixel = 0; pixel < PIXELS; pixel++) {
                            const uint32_t id = segmentation[index * PIXELS + pixel];
                            if (id != 0xFFFFFFFFu && id >= scene_instances) {
                                const uint32_t first = scene_instances + instance_i * SPEC::MAX_ENTITY_SLOTS_PER_INSTANCE;
                                EXPECT_GE(id, first);
                                EXPECT_LT(id, first + WORLD::N_AGENTS * world.drone_parts);
                                drone_seen = drone_seen || id >= first + (WORLD::N_AGENTS - 1) * world.drone_parts;
                            }
                            for (TI axis = 0; axis < 2; axis++) {
                                const T magnitude = std::abs(flow[(index * PIXELS + pixel) * 2 + axis]);
                                if (magnitude > max_flow) {
                                    max_flow = magnitude;
                                    max_flow_pixel = pixel;
                                }
                            }
                        }
                        if (WORLD::N_AGENTS > 1 && agent_i == 0 && camera_i == 0) {
                            EXPECT_TRUE(drone_seen);
                        }
                        if (reset) {
                            EXPECT_LT(max_flow, (T)0.001) << "agents=" << WORLD::N_AGENTS << " frame=" << frame << " instance=" << instance_i << " agent=" << agent_i << " camera=" << camera_i << " pixel=" << max_flow_pixel << " object=" << segmentation[index * PIXELS + max_flow_pixel] << " scene_instances=" << scene_instances << " camera_position=" << current[index].pos[0] << "," << current[index].pos[1] << "," << current[index].pos[2];
                        } else {
                            EXPECT_GT(max_flow, (T)0.01);
                        }
                        for (TI pixel_channel = 0; pixel_channel < WORLD::FRAME_DIM; pixel_channel++) {
                            const TI flat = index * WORLD::FRAME_DIM + pixel_channel;
                            const T expected_value = std::min((T)1, std::max((T)0, rgb[flat]));
                            EXPECT_NEAR(get(host, staging.observations, instance_i, view * WORLD::FRAME_DIM + pixel_channel), expected_value, 1e-6);
                            EXPECT_NEAR(history[(frame % SPEC::HISTORY_LENGTH) * NUM_CAMERAS * WORLD::FRAME_DIM + flat], expected_value, 1e-6);
                        }
                    }
                }
            }
            previous = current;
        }
        if constexpr (WORLD::N_CAMERA_AGENTS < WORLD::N_AGENTS) {
            terminated(device, world, resident.parameters, resident.states, resident.terminated_flags, rng);
            copy(device, host, resident.terminated_flags, staging.terminated_flags);
            for (TI instance_i = 0; instance_i < WORLD::INSTANCES; instance_i++) {
                EXPECT_FALSE(get(host, staging.terminated_flags, instance_i));
            }
            auto& state = get_ref(host, staging.states, (TI)0);
            auto& last_agent = _agent_state<SPEC>(state, WORLD::N_AGENTS - 1);
            auto& termination = get_ref(host, staging.parameters, (TI)0).agents[WORLD::N_AGENTS - 1].dynamics.mdp.termination;
            termination.enabled = true;
            termination.position_threshold = 100;
            termination.linear_velocity_threshold = 1;
            termination.angular_velocity_threshold = 100;
            last_agent.linear_velocity[0] = 2;
            copy(host, device, staging.parameters, resident.parameters);
            copy(host, device, staging.states, resident.states);
            terminated(device, world, resident.parameters, resident.states, resident.terminated_flags, rng);
            copy(device, host, resident.terminated_flags, staging.terminated_flags);
            EXPECT_TRUE(get(host, staging.terminated_flags, (TI)0));
            EXPECT_FALSE(get(host, staging.terminated_flags, (TI)1));
            termination.enabled = false;
            for (TI dim = 0; dim < 3; dim++) {
                last_agent.position[dim] = _agent_state<SPEC>(state, (TI)0).position[dim];
            }
            copy(host, device, staging.parameters, resident.parameters);
            copy(host, device, staging.states, resident.states);
            terminated(device, world, resident.parameters, resident.states, resident.terminated_flags, rng);
            copy(device, host, resident.terminated_flags, staging.terminated_flags);
            EXPECT_TRUE(get(host, staging.terminated_flags, (TI)0));
            EXPECT_FALSE(get(host, staging.terminated_flags, (TI)1));
        }
        resident.deallocate(device);
        staging.deallocate(host);
        free(device, rng);
        free(host, reference_rng);
        free(device, world);
        free(host, shared.library);
    }
}
