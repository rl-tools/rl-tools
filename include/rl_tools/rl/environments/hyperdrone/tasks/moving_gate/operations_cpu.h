#include "../../../../../version.h"
#include "../../../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_TASKS_MOVING_GATE_OPERATIONS_CPU_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_TASKS_MOVING_GATE_OPERATIONS_CPU_H

#include "moving_gate.h"
#include "../../operations_cpu.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {

    namespace rl::environments::hyperdrone::tasks::moving_gate {
        template <typename DEVICE, typename TASK_SPEC>
        RL_TOOLS_FUNCTION_PLACEMENT void drone_scene_position(DEVICE&, const typename World<TASK_SPEC>::Parameters& parameters, const typename World<TASK_SPEC>::State& state, typename TASK_SPEC::T out[3]){
            using T = typename TASK_SPEC::T;
            out[0] = parameters.scene_yaw_cos * state.position[0] - parameters.scene_yaw_sin * state.position[1] + parameters.scene_translation[0];
            out[1] = parameters.scene_yaw_sin * state.position[0] + parameters.scene_yaw_cos * state.position[1] + parameters.scene_translation[1];
            out[2] = state.position[2] + parameters.scene_translation[2];
        }
        // signed distance to the (moving) gate plane and radial offset within it
        template <typename DEVICE, typename TASK_SPEC>
        RL_TOOLS_FUNCTION_PLACEMENT void gate_plane_coordinates(DEVICE& device, const typename World<TASK_SPEC>::Parameters& parameters, typename TASK_SPEC::T phase, const typename TASK_SPEC::T position[3], typename TASK_SPEC::T& signed_distance, typename TASK_SPEC::T& radial_distance){
            using T = typename TASK_SPEC::T;
            T R[3][3];
            rl::environments::l2f::quaternion_to_rotation_matrix<DEVICE, T>(parameters.gate_orientation, R);
            T offset = parameters.gate_amplitude * math::sin(device.math, phase);
            T relative[3];
            for(unsigned dim = 0; dim < 3; dim++){
                relative[dim] = position[dim] - (parameters.gate_center[dim] + offset * parameters.gate_axis[dim]);
            }
            // gate local axes are the columns of R; the plane normal is local +X
            signed_distance = R[0][0] * relative[0] + R[1][0] * relative[1] + R[2][0] * relative[2];
            T in_plane_y = R[0][1] * relative[0] + R[1][1] * relative[1] + R[2][1] * relative[2];
            T in_plane_z = R[0][2] * relative[0] + R[1][2] * relative[1] + R[2][2] * relative[2];
            radial_distance = math::sqrt(device.math, in_plane_y * in_plane_y + in_plane_z * in_plane_z);
        }
        // the gate's motion state and interaction flags, advanced after the dynamics step
        template <typename DEVICE, typename TASK_SPEC>
        RL_TOOLS_FUNCTION_PLACEMENT void _gate_step(DEVICE& device, const typename World<TASK_SPEC>::Parameters& parameters, const typename World<TASK_SPEC>::State& state, typename World<TASK_SPEC>::State& next_state, typename TASK_SPEC::T dt){
            using T = typename TASK_SPEC::T;
            constexpr T TWO_PI = (T)2 * math::PI<T>;
            T phase = state.gate_phase + TWO_PI * parameters.gate_frequency * dt;
            phase = phase - math::floor(device.math, phase / TWO_PI) * TWO_PI;
            next_state.gate_phase = phase;
            next_state.gate_passed = state.gate_passed;
            next_state.gate_crashed = state.gate_crashed;
            T position_before[3], position_after[3];
            drone_scene_position<DEVICE, TASK_SPEC>(device, parameters, state, position_before);
            drone_scene_position<DEVICE, TASK_SPEC>(device, parameters, next_state, position_after);
            T signed_before, radial_before, signed_after, radial_after;
            gate_plane_coordinates<DEVICE, TASK_SPEC>(device, parameters, state.gate_phase, position_before, signed_before, radial_before);
            gate_plane_coordinates<DEVICE, TASK_SPEC>(device, parameters, phase, position_after, signed_after, radial_after);
            if((signed_before <= 0) != (signed_after <= 0)){
                if(radial_after <= parameters.gate_aperture_radius){
                    next_state.gate_passed = true;
                } else {
                    next_state.gate_crashed = true;
                }
            }
        }
        template <typename DEVICE, typename TASK_SPEC, typename RNG>
        RL_TOOLS_FUNCTION_PLACEMENT void _sample_gate(DEVICE& device, const typename World<TASK_SPEC>::NEXT_WORLD::SCENE& scene, typename World<TASK_SPEC>::Parameters& parameters, typename World<TASK_SPEC>::State& state, RNG& rng){
            using T = typename TASK_SPEC::T;
            constexpr T TWO_PI = (T)2 * math::PI<T>;
            auto gate_position = rendering::raytracing::scene::procthor::sample_indoor_position(device, scene, rng);
            parameters.gate_center[0] = gate_position.position[0];
            parameters.gate_center[1] = gate_position.position[1];
            parameters.gate_center[2] = gate_position.position[2];
            parameters.gate_axis[0] = 0;
            parameters.gate_axis[1] = 0;
            parameters.gate_axis[2] = 1;
            T facing = gate_position.yaw;
            parameters.gate_orientation[0] = math::cos(device.math, facing / (T)2);
            parameters.gate_orientation[1] = 0;
            parameters.gate_orientation[2] = 0;
            parameters.gate_orientation[3] = math::sin(device.math, facing / (T)2);
            parameters.gate_amplitude = random::uniform_real_distribution(device.random, TASK_SPEC::GATE_AMPLITUDE_MIN, TASK_SPEC::GATE_AMPLITUDE_MAX, rng);
            parameters.gate_frequency = random::uniform_real_distribution(device.random, TASK_SPEC::GATE_FREQUENCY_MIN, TASK_SPEC::GATE_FREQUENCY_MAX, rng);
            parameters.gate_aperture_radius = TASK_SPEC::GATE_APERTURE_RADIUS;
            state.gate_phase = random::uniform_real_distribution(device.random, (T)0, TWO_PI, rng);
            state.gate_passed = false;
            state.gate_crashed = false;
        }
    }

    template <typename DEVICE, typename TASK_SPEC>
    void init(DEVICE& device, rl::environments::hyperdrone::tasks::moving_gate::World<TASK_SPEC>& world, typename TASK_SPEC::NEXT_WORLD::SharedContext& shared, typename TASK_SPEC::TI first_scene, typename TASK_SPEC::TI num_scenes, typename TASK_SPEC::TI member_index) {
        using NEXT_WORLD = typename TASK_SPEC::NEXT_WORLD;
        using BASE_SPEC = typename NEXT_WORLD::SPEC;
        utils::assert_exit(device, !world.gate_asset_path.empty(), "hyperdrone::tasks::moving_gate: gate_asset_path must be set before init");
        auto asset = register_pool_asset<DEVICE, typename NEXT_WORLD::SharedContext, typename BASE_SPEC::SHADING, BASE_SPEC::OUTPUT_RGB>(device, shared, world.gate_asset_path);
        world.entity_kind_index = (typename TASK_SPEC::TI)world.entity_kinds.size();
        world.entity_kinds.push_back({asset, 1, 0});
        init(device, static_cast<NEXT_WORLD&>(world), shared, first_scene, num_scenes, member_index);
        malloc(world.renderer.device, world.gate_pose_staging);
    }
    template <typename DEVICE, typename TASK_SPEC>
    void free(DEVICE& device, rl::environments::hyperdrone::tasks::moving_gate::World<TASK_SPEC>& world) {
        using NEXT_WORLD = typename TASK_SPEC::NEXT_WORLD;
        if (!world.slots.empty()) {
            free(world.renderer.device, world.gate_pose_staging);
        }
        free(device, static_cast<NEXT_WORLD&>(world));
    }

    template <typename DEVICE, typename TASK_SPEC, typename RNG>
    void sample_initial_parameters(DEVICE& device, rl::environments::hyperdrone::tasks::moving_gate::World<TASK_SPEC>& world, typename rl::environments::hyperdrone::tasks::moving_gate::World<TASK_SPEC>::Parameters& parameters, RNG& rng) {
        using NEXT_WORLD = typename TASK_SPEC::NEXT_WORLD;
        sample_initial_parameters(device, static_cast<NEXT_WORLD&>(world), static_cast<typename NEXT_WORLD::Parameters&>(parameters), rng);
    }
    // gate trajectory sampling needs the scene (free-space placement), so it rides the state
    // sampling like the base's scene placement does
    template <typename DEVICE, typename TASK_SPEC, typename RNG>
    void sample_initial_state(DEVICE& device, rl::environments::hyperdrone::tasks::moving_gate::World<TASK_SPEC>& world, typename rl::environments::hyperdrone::tasks::moving_gate::World<TASK_SPEC>::Parameters& parameters, typename rl::environments::hyperdrone::tasks::moving_gate::World<TASK_SPEC>::State& state, RNG& rng) {
        using NEXT_WORLD = typename TASK_SPEC::NEXT_WORLD;
        sample_initial_state(device, static_cast<NEXT_WORLD&>(world), static_cast<typename NEXT_WORLD::Parameters&>(parameters), static_cast<typename NEXT_WORLD::State&>(state), rng);
        rl::environments::hyperdrone::tasks::moving_gate::_sample_gate<DEVICE, TASK_SPEC>(device, world.slots[world.active_slot].scene, parameters, state, rng);
    }
    template <typename DEVICE, typename TASK_SPEC, typename PARAMETER_SPEC, typename STATE_SPEC, typename RESET_SPEC, typename RNG>
    void sample_initial_state(DEVICE& device, rl::environments::hyperdrone::tasks::moving_gate::World<TASK_SPEC>& world, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, const Tensor<RESET_SPEC>& reset_mask, RNG& rng) {
        using TI = typename TASK_SPEC::TI;
        using WORLD = rl::environments::hyperdrone::tasks::moving_gate::World<TASK_SPEC>;
        static_assert(utils::typing::is_same_v<typename STATE_SPEC::T, typename WORLD::State>);
        for (TI instance_i = 0; instance_i < WORLD::INSTANCES; instance_i++) {
            if (get(device, reset_mask, instance_i)) {
                sample_initial_state(device, world, get_ref(device, parameters, instance_i), get_ref(device, states, instance_i), rng);
            }
        }
    }

    // pose delivery: write this task's gate pose for every instance into the active slot, then
    // forward to the inner render
    template <typename DEVICE, typename TASK_SPEC, typename PARAMETER_SPEC, typename STATE_SPEC, typename RESET_SPEC>
    void render(DEVICE& device, rl::environments::hyperdrone::tasks::moving_gate::World<TASK_SPEC>& world, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, const Tensor<RESET_SPEC>& reset_mask) {
        using T = typename TASK_SPEC::T;
        using TI = typename TASK_SPEC::TI;
        using WORLD = rl::environments::hyperdrone::tasks::moving_gate::World<TASK_SPEC>;
        using NEXT_WORLD = typename TASK_SPEC::NEXT_WORLD;
        using BASE_SPEC = typename NEXT_WORLD::SPEC;
        auto& slot = world.slots[world.active_slot];
        const TI kinds = (TI)world.entity_kinds.size();
        for (TI instance_i = 0; instance_i < WORLD::INSTANCES; instance_i++) {
            const auto& instance_parameters = get_ref(device, parameters, instance_i);
            const auto& state = get_ref(device, states, instance_i);
            float pose[12];
            rl::environments::hyperdrone::tasks::moving_gate::gate_pose(device, instance_parameters, state.gate_phase, pose);
            const auto& placement = slot.entity_placements[instance_i * kinds + world.entity_kind_index];
            if constexpr (NEXT_WORLD::RENDERER_CONFIG::ENABLE_DYNAMIC_MOTION_BLUR) {
                float pose_open[12];
                const T dt = instance_parameters.dynamics.integration.dt;
                constexpr T TWO_PI = (T)2 * math::PI<T>;
                T phase_open = state.gate_phase - TWO_PI * instance_parameters.gate_frequency * dt * instance_parameters.shutter_fraction;
                rl::environments::hyperdrone::tasks::moving_gate::gate_pose(device, instance_parameters, phase_open, pose_open);
                set_transform_pair(device, world.renderer, rendering::raytracing::OverlayIndex{instance_i}, placement, pose_open, pose);
            } else {
                set_transform(device, world.renderer, rendering::raytracing::OverlayIndex{instance_i}, placement, pose);
            }
        }
        render(device, static_cast<NEXT_WORLD&>(world), parameters, states, reset_mask);
    }

    // the gate's motion and interaction flags advance with the environment step
    template <typename DEVICE, typename TASK_SPEC, typename PARAMETER_SPEC, typename STATE_SPEC, typename ACTION_SPEC, typename NEXT_STATE_SPEC, typename RNG>
    void step(DEVICE& device, rl::environments::hyperdrone::tasks::moving_gate::World<TASK_SPEC>& world, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, const Tensor<ACTION_SPEC>& actions, Tensor<NEXT_STATE_SPEC>& next_states, RNG& rng) {
        using T = typename TASK_SPEC::T;
        using TI = typename TASK_SPEC::TI;
        using WORLD = rl::environments::hyperdrone::tasks::moving_gate::World<TASK_SPEC>;
        using NEXT_WORLD = typename TASK_SPEC::NEXT_WORLD;
        step(device, static_cast<NEXT_WORLD&>(world), parameters, states, actions, next_states, rng);
        for (TI instance_i = 0; instance_i < WORLD::INSTANCES; instance_i++) {
            const auto& instance_parameters = get_ref(device, parameters, instance_i);
            rl::environments::hyperdrone::tasks::moving_gate::_gate_step<DEVICE, TASK_SPEC>(device, instance_parameters, get_ref(device, states, instance_i), get_ref(device, next_states, instance_i), instance_parameters.dynamics.integration.dt);
        }
    }

    template <typename DEVICE, typename TASK_SPEC, typename PARAMETER_SPEC, typename STATE_SPEC, typename ACTION_SPEC, typename NEXT_STATE_SPEC, typename REWARD_SPEC, typename RNG>
    void reward(DEVICE& device, rl::environments::hyperdrone::tasks::moving_gate::World<TASK_SPEC>& world, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, const Tensor<ACTION_SPEC>& actions, Tensor<NEXT_STATE_SPEC>& next_states, Tensor<REWARD_SPEC>& rewards, RNG& rng) {
        using T = typename TASK_SPEC::T;
        using TI = typename TASK_SPEC::TI;
        using WORLD = rl::environments::hyperdrone::tasks::moving_gate::World<TASK_SPEC>;
        using NEXT_WORLD = typename TASK_SPEC::NEXT_WORLD;
        reward(device, static_cast<NEXT_WORLD&>(world), parameters, states, actions, next_states, rewards, rng);
        for (TI instance_i = 0; instance_i < WORLD::INSTANCES; instance_i++) {
            const auto& state = get_ref(device, states, instance_i);
            const auto& next_state = get_ref(device, next_states, instance_i);
            if (next_state.gate_passed && !state.gate_passed) {
                set(device, rewards, get(device, rewards, instance_i) + TASK_SPEC::GATE_PASS_REWARD, instance_i);
            }
        }
    }

    template <typename DEVICE, typename TASK_SPEC, typename PARAMETER_SPEC, typename STATE_SPEC, typename TERMINATED_SPEC, typename RNG>
    void terminated(DEVICE& device, rl::environments::hyperdrone::tasks::moving_gate::World<TASK_SPEC>& world, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, Tensor<TERMINATED_SPEC>& terminated_flags, RNG& rng) {
        using TI = typename TASK_SPEC::TI;
        using WORLD = rl::environments::hyperdrone::tasks::moving_gate::World<TASK_SPEC>;
        using NEXT_WORLD = typename TASK_SPEC::NEXT_WORLD;
        terminated(device, static_cast<NEXT_WORLD&>(world), parameters, states, terminated_flags, rng);
        for (TI instance_i = 0; instance_i < WORLD::INSTANCES; instance_i++) {
            if (get_ref(device, states, instance_i).gate_crashed) {
                set(device, terminated_flags, true, instance_i);
            }
        }
    }

    // privileged contribution for the asymmetric critic: [dynamics observation | gate state]
    template <typename DEVICE, typename TASK_SPEC, typename PARAMETER_SPEC, typename STATE_SPEC, typename OBSERVATION_SPEC, typename RNG>
    void observe(DEVICE& device, rl::environments::hyperdrone::tasks::moving_gate::World<TASK_SPEC>& world, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, typename rl::environments::hyperdrone::tasks::moving_gate::World<TASK_SPEC>::ObservationPrivileged, Tensor<OBSERVATION_SPEC>& observations, RNG& rng) {
        using T = typename TASK_SPEC::T;
        using TI = typename TASK_SPEC::TI;
        using WORLD = rl::environments::hyperdrone::tasks::moving_gate::World<TASK_SPEC>;
        using NEXT_WORLD = typename TASK_SPEC::NEXT_WORLD;
        static_assert(get<1>(typename OBSERVATION_SPEC::SHAPE{}) == WORLD::OBSERVATION_DIM_PRIVILEGED);
        constexpr TI BASE_DIM = NEXT_WORLD::OBSERVATION_DIM_PRIVILEGED;
        auto base_observations = view_range(device, observations, 0, tensor::ViewSpec<1, BASE_DIM>{});
        observe(device, static_cast<NEXT_WORLD&>(world), parameters, states, typename NEXT_WORLD::ObservationPrivileged{}, base_observations, rng);
        for (TI instance_i = 0; instance_i < WORLD::INSTANCES; instance_i++) {
            const auto& instance_parameters = get_ref(device, parameters, instance_i);
            const auto& state = get_ref(device, states, instance_i);
            T position[3];
            rl::environments::hyperdrone::tasks::moving_gate::drone_scene_position<DEVICE, TASK_SPEC>(device, instance_parameters, state, position);
            T offset = instance_parameters.gate_amplitude * math::sin(device.math, state.gate_phase);
            T gate_velocity_magnitude = instance_parameters.gate_amplitude * math::cos(device.math, state.gate_phase) * (T)2 * math::PI<T> * instance_parameters.gate_frequency;
            for (TI dim = 0; dim < 3; dim++) {
                T gate_position = instance_parameters.gate_center[dim] + offset * instance_parameters.gate_axis[dim];
                set(device, observations, gate_position - position[dim], instance_i, BASE_DIM + dim);
                set(device, observations, gate_velocity_magnitude * instance_parameters.gate_axis[dim], instance_i, BASE_DIM + 3 + dim);
            }
            set(device, observations, math::sin(device.math, state.gate_phase), instance_i, BASE_DIM + 6);
            set(device, observations, math::cos(device.math, state.gate_phase), instance_i, BASE_DIM + 7);
        }
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

// ADL entry points: generic composite code dispatches member lifecycle verbs without Tensor
// arguments, so the task's overloads must be reachable through the member type's namespace
RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::environments::hyperdrone::tasks::moving_gate {
    template <typename DEVICE, typename TASK_SPEC>
    void init(DEVICE& device, World<TASK_SPEC>& world, typename TASK_SPEC::NEXT_WORLD::SharedContext& shared, typename TASK_SPEC::TI first_scene, typename TASK_SPEC::TI num_scenes, typename TASK_SPEC::TI member_index){
        ::rl_tools::init(device, world, shared, first_scene, num_scenes, member_index);
    }
    template <typename DEVICE, typename TASK_SPEC>
    void free(DEVICE& device, World<TASK_SPEC>& world){
        ::rl_tools::free(device, world);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
