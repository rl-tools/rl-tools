#include "../../../../../version.h"
#include "../../../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_TASKS_VISUAL_INERTIAL_LOCALIZATION_OPERATIONS_CPU_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_TASKS_VISUAL_INERTIAL_LOCALIZATION_OPERATIONS_CPU_H

#include "visual_inertial_localization.h"
#include "../../operations_cpu.h"
#include "../../../../../nn/layers/dense/operations_generic.h"
#include "../../../../../nn/layers/gru/operations_generic.h"
#include "../../../../../nn_models/sequential/operations_generic.h"
#include "../../../../../persist/backends/tar/operations_cpu.h"
#include "../../../../../nn/layers/dense/persist.h"
#include "../../../../../nn/layers/gru/persist.h"
#include "../../../../../nn_models/sequential/persist.h"

#include <string>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {

    namespace rl::environments::hyperdrone::tasks::visual_inertial_localization {
        template <typename DEVICE, typename TASK_SPEC>
        RL_TOOLS_FUNCTION_PLACEMENT void drone_scene_pose(DEVICE& device, const typename World<TASK_SPEC>::Parameters& parameters, const typename World<TASK_SPEC>::State& state, typename TASK_SPEC::T position_out[3], typename TASK_SPEC::T orientation_out[4]){
            using T = typename TASK_SPEC::T;
            position_out[0] = parameters.scene_yaw_cos * state.position[0] - parameters.scene_yaw_sin * state.position[1] + parameters.scene_translation[0];
            position_out[1] = parameters.scene_yaw_sin * state.position[0] + parameters.scene_yaw_cos * state.position[1] + parameters.scene_translation[1];
            position_out[2] = state.position[2] + parameters.scene_translation[2];
            T yaw = math::atan2(device.math, parameters.scene_yaw_sin, parameters.scene_yaw_cos);
            T yaw_half_cos = math::cos(device.math, yaw / (T)2);
            T yaw_half_sin = math::sin(device.math, yaw / (T)2);
            orientation_out[0] = yaw_half_cos * state.orientation[0] - yaw_half_sin * state.orientation[3];
            orientation_out[1] = yaw_half_cos * state.orientation[1] - yaw_half_sin * state.orientation[2];
            orientation_out[2] = yaw_half_cos * state.orientation[2] + yaw_half_sin * state.orientation[1];
            orientation_out[3] = yaw_half_cos * state.orientation[3] + yaw_half_sin * state.orientation[0];
        }
        template <typename DEVICE, typename TASK_SPEC>
        RL_TOOLS_FUNCTION_PLACEMENT void scene_to_dynamics_position(DEVICE&, const typename World<TASK_SPEC>::Parameters& parameters, const typename TASK_SPEC::T in[3], typename TASK_SPEC::T out[3]){
            using T = typename TASK_SPEC::T;
            T dx = in[0] - parameters.scene_translation[0];
            T dy = in[1] - parameters.scene_translation[1];
            out[0] =  parameters.scene_yaw_cos * dx + parameters.scene_yaw_sin * dy;
            out[1] = -parameters.scene_yaw_sin * dx + parameters.scene_yaw_cos * dy;
            out[2] = in[2] - parameters.scene_translation[2];
        }
        // the base world samples a uniformly random scene yaw, which can leave the camera
        // staring at a blank wall; visual estimators need features from the first frame, so the
        // task re-orients the spawn along the anchor annotation's free-space facing direction
        template <typename DEVICE, typename TASK_SPEC>
        void _face_free_space(DEVICE& device, const typename World<TASK_SPEC>::NEXT_WORLD::ANNOTATIONS& annotations, typename World<TASK_SPEC>::Parameters& parameters){
            using T = typename TASK_SPEC::T;
            using TI = typename TASK_SPEC::TI;
            const TI count = annotations.num_positions;
            if (count == 0) {
                return;
            }
            TI anchor_index = 0;
            T anchor_distance_squared = 0;
            for (TI candidate_i = 0; candidate_i < count; candidate_i++) {
                T distance_squared = 0;
                for (TI dim_i = 0; dim_i < 3; dim_i++) {
                    T delta = (T)annotations.positions[candidate_i].position[dim_i] - parameters.scene_translation[dim_i];
                    distance_squared += delta * delta;
                }
                if (candidate_i == 0 || distance_squared < anchor_distance_squared) {
                    anchor_distance_squared = distance_squared;
                    anchor_index = candidate_i;
                }
            }
            T yaw = (T)annotations.positions[anchor_index].yaw;
            parameters.scene_yaw_cos = math::cos(device.math, yaw);
            parameters.scene_yaw_sin = math::sin(device.math, yaw);
        }
        // greedy nearest-unused-neighbor route over the scene's free-space positions, anchored
        // at the initial placement; stored in the dynamics frame
        template <typename DEVICE, typename TASK_SPEC, typename RNG>
        void _sample_route(DEVICE& device, const typename World<TASK_SPEC>::NEXT_WORLD::ANNOTATIONS& annotations, typename World<TASK_SPEC>::Parameters& parameters, typename World<TASK_SPEC>::State& state, RNG& rng){
            using T = typename TASK_SPEC::T;
            using TI = typename TASK_SPEC::TI;
            using PARAMETERS = typename World<TASK_SPEC>::Parameters;
            using ANNOTATIONS = typename World<TASK_SPEC>::NEXT_WORLD::ANNOTATIONS;
            constexpr TI NUM_WAYPOINTS = PARAMETERS::NUM_WAYPOINTS;
            constexpr T MIN_DISTANCE_SQUARED = TASK_SPEC::WAYPOINT_MIN_DISTANCE * TASK_SPEC::WAYPOINT_MIN_DISTANCE;
            for (TI dim_i = 0; dim_i < 3; dim_i++) {
                parameters.waypoints[0][dim_i] = 0;
            }
            const TI count = annotations.num_positions;
            bool used[ANNOTATIONS::SPEC::MAX_POSITIONS] = {};
            T current[3] = {parameters.scene_translation[0], parameters.scene_translation[1], parameters.scene_translation[2]};
            for (TI waypoint_i = 1; waypoint_i < NUM_WAYPOINTS; waypoint_i++) {
                if (count == 0) {
                    for (TI dim_i = 0; dim_i < 3; dim_i++) {
                        parameters.waypoints[waypoint_i][dim_i] = 0;
                    }
                    continue;
                }
                TI best_index = count;
                T best_distance_squared = 0;
                for (TI pass = 0; pass < 2 && best_index == count; pass++) {
                    for (TI candidate_i = 0; candidate_i < count; candidate_i++) {
                        if (used[candidate_i]) {
                            continue;
                        }
                        const auto& candidate = annotations.positions[candidate_i].position;
                        T distance_squared = 0;
                        for (TI dim_i = 0; dim_i < 3; dim_i++) {
                            T delta = (T)candidate[dim_i] - current[dim_i];
                            distance_squared += delta * delta;
                        }
                        // first pass enforces the minimum spacing, second pass takes what is left
                        if (pass == 0 && distance_squared < MIN_DISTANCE_SQUARED) {
                            continue;
                        }
                        if (best_index == count || distance_squared < best_distance_squared) {
                            best_index = candidate_i;
                            best_distance_squared = distance_squared;
                        }
                    }
                }
                if (best_index == count) { // every position used: recycle with a random restart
                    for (TI candidate_i = 0; candidate_i < count; candidate_i++) {
                        used[candidate_i] = false;
                    }
                    best_index = random::uniform_int_distribution(device.random, (TI)0, (TI)(count - 1), rng);
                }
                used[best_index] = true;
                T scene_position[3];
                for (TI dim_i = 0; dim_i < 3; dim_i++) {
                    scene_position[dim_i] = (T)annotations.positions[best_index].position[dim_i];
                    current[dim_i] = scene_position[dim_i];
                }
                scene_to_dynamics_position<DEVICE, TASK_SPEC>(device, parameters, scene_position, parameters.waypoints[waypoint_i]);
            }
            state.current_waypoint = (TASK_SPEC::INITIALIZATION_HOLD_STEPS > 0 || NUM_WAYPOINTS == 1) ? 0 : 1;
        }
        template <typename DEVICE, typename TASK_SPEC>
        void _load_autopilot(DEVICE& device, World<TASK_SPEC>& world){
            using TI = typename TASK_SPEC::TI;
            std::string path;
            utils::assert_exit(device, rendering::datasets::resolve_reference(device, world.autopilot_policy_path, path), "hyperdrone::tasks::visual_inertial_localization: cannot resolve the autopilot checkpoint");
            persist::backends::tar::File<TI> checkpoint(path, persist::backends::tar::Mode::READ);
            auto actor = get_group(device, checkpoint, "actor");
            utils::assert_exit(device, load(device, world.autopilot.model, actor), "hyperdrone::tasks::visual_inertial_localization: the autopilot checkpoint does not match the RAPTOR architecture");
        }
        template <typename DEVICE, typename TASK_SPEC, typename PARAMETER_SPEC, typename STATE_SPEC, typename RNG>
        void _control(DEVICE& device, World<TASK_SPEC>& world, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, RNG& rng){
            using WORLD = World<TASK_SPEC>;
            using T = typename WORLD::T;
            using TI = typename WORLD::TI;
            using AUTOPILOT = typename WORLD::AUTOPILOT;
            for (TI instance_i = 0; instance_i < WORLD::INSTANCES; instance_i++){
                auto& instance_parameters = get_ref(device, parameters, instance_i);
                const auto& state = get_ref(device, states, instance_i);
                Matrix<matrix::Specification<T, TI, 1, AUTOPILOT::OBSERVATION_DIM, true, matrix::layouts::RowMajorAlignment<TI, 1>>> observation_row;
                observation_row._data = data(world.autopilot.observations) + instance_i * AUTOPILOT::OBSERVATION_DIM;
                observe(device, world.dynamics, instance_parameters.dynamics, state, typename AUTOPILOT::OBSERVATION_TYPE{}, observation_row, rng);
                for (TI dim_i = 0; dim_i < 3; dim_i++){
                    T error = state.position[dim_i] - instance_parameters.waypoints[state.current_waypoint][dim_i];
                    error = math::clamp(device.math, error, -TASK_SPEC::TARGET_POSITION_ERROR_CLIP, TASK_SPEC::TARGET_POSITION_ERROR_CLIP);
                    set(observation_row, 0, dim_i, error);
                }
            }
            Mode<nn::layers::gru::NoAutoResetMode<mode::Default<>>> no_auto_reset_mode;
            evaluate_step(device, world.autopilot.model, world.autopilot.observations, world.autopilot.state, world.autopilot.actions, world.autopilot.buffer, rng, no_auto_reset_mode);
        }
        template <typename DEVICE, typename TASK_SPEC>
        RL_TOOLS_FUNCTION_PLACEMENT void _waypoint_step(DEVICE& device, const typename World<TASK_SPEC>::Parameters& parameters, const typename World<TASK_SPEC>::State& state, typename World<TASK_SPEC>::State& next_state){
            using T = typename TASK_SPEC::T;
            using TI = typename TASK_SPEC::TI;
            using PARAMETERS = typename World<TASK_SPEC>::Parameters;
            next_state.current_waypoint = state.current_waypoint;
            T distance_squared = 0;
            for (TI dim_i = 0; dim_i < 3; dim_i++) {
                T delta = next_state.position[dim_i] - parameters.waypoints[state.current_waypoint][dim_i];
                distance_squared += delta * delta;
            }
            if (distance_squared < TASK_SPEC::WAYPOINT_ACCEPTANCE_RADIUS * TASK_SPEC::WAYPOINT_ACCEPTANCE_RADIUS) {
                next_state.current_waypoint = (state.current_waypoint + 1) % PARAMETERS::NUM_WAYPOINTS;
            }
        }
    }

    template <typename DEVICE, typename TASK_SPEC>
    void malloc(DEVICE& device, rl::environments::hyperdrone::tasks::visual_inertial_localization::World<TASK_SPEC>& world) {
        using NEXT_WORLD = typename TASK_SPEC::NEXT_WORLD;
        malloc(device, static_cast<NEXT_WORLD&>(world));
        malloc(device, world.autopilot.model);
        malloc(device, world.autopilot.state);
        malloc(device, world.autopilot.buffer);
        malloc(device, world.autopilot.observations);
        malloc(device, world.autopilot.actions);
    }
    template <typename DEVICE, typename TASK_SPEC, typename DATASET>
    void init(DEVICE& device, rl::environments::hyperdrone::tasks::visual_inertial_localization::World<TASK_SPEC>& world, typename TASK_SPEC::NEXT_WORLD::SharedContext& shared, const DATASET& dataset, const typename DATASET::Corpus& corpus, typename TASK_SPEC::TI first_scene, typename TASK_SPEC::TI num_scenes, typename TASK_SPEC::TI member_index) {
        using NEXT_WORLD = typename TASK_SPEC::NEXT_WORLD;
        init(device, static_cast<NEXT_WORLD&>(world), shared, dataset, corpus, first_scene, num_scenes, member_index);
        rl::environments::hyperdrone::tasks::visual_inertial_localization::_load_autopilot<DEVICE, TASK_SPEC>(device, world);
        world.task_step = 0;
    }
    template <typename DEVICE, typename TASK_SPEC>
    void free(DEVICE& device, rl::environments::hyperdrone::tasks::visual_inertial_localization::World<TASK_SPEC>& world) {
        using NEXT_WORLD = typename TASK_SPEC::NEXT_WORLD;
        free(device, world.autopilot.model);
        free(device, world.autopilot.state);
        free(device, world.autopilot.buffer);
        free(device, world.autopilot.observations);
        free(device, world.autopilot.actions);
        free(device, static_cast<NEXT_WORLD&>(world));
    }

    template <typename DEVICE, typename TASK_SPEC, typename RNG>
    void sample_initial_parameters(DEVICE& device, rl::environments::hyperdrone::tasks::visual_inertial_localization::World<TASK_SPEC>& world, typename rl::environments::hyperdrone::tasks::visual_inertial_localization::World<TASK_SPEC>::Parameters& parameters, RNG& rng) {
        using NEXT_WORLD = typename TASK_SPEC::NEXT_WORLD;
        sample_initial_parameters(device, static_cast<NEXT_WORLD&>(world), static_cast<typename NEXT_WORLD::Parameters&>(parameters), rng);
    }
    // route sampling needs the scene (free-space positions) and the sampled scene placement, so
    // it rides the state sampling like the base's scene placement does
    template <typename DEVICE, typename TASK_SPEC, typename RNG>
    void sample_initial_state(DEVICE& device, rl::environments::hyperdrone::tasks::visual_inertial_localization::World<TASK_SPEC>& world, typename rl::environments::hyperdrone::tasks::visual_inertial_localization::World<TASK_SPEC>::Parameters& parameters, typename rl::environments::hyperdrone::tasks::visual_inertial_localization::World<TASK_SPEC>::State& state, RNG& rng) {
        using NEXT_WORLD = typename TASK_SPEC::NEXT_WORLD;
        sample_initial_state(device, static_cast<NEXT_WORLD&>(world), static_cast<typename NEXT_WORLD::Parameters&>(parameters), static_cast<typename NEXT_WORLD::State&>(state), rng);
        rl::environments::hyperdrone::tasks::visual_inertial_localization::_face_free_space<DEVICE, TASK_SPEC>(device, world.slots[world.active_slot].annotations, parameters);
        rl::environments::hyperdrone::tasks::visual_inertial_localization::_sample_route<DEVICE, TASK_SPEC>(device, world.slots[world.active_slot].annotations, parameters, state, rng);
    }
    template <typename DEVICE, typename TASK_SPEC, typename PARAMETER_SPEC, typename STATE_SPEC, typename RESET_SPEC, typename RNG>
    void sample_initial_state(DEVICE& device, rl::environments::hyperdrone::tasks::visual_inertial_localization::World<TASK_SPEC>& world, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, const Tensor<RESET_SPEC>& reset_mask, RNG& rng) {
        using TI = typename TASK_SPEC::TI;
        using WORLD = rl::environments::hyperdrone::tasks::visual_inertial_localization::World<TASK_SPEC>;
        static_assert(utils::typing::is_same_v<typename STATE_SPEC::T, typename WORLD::State>);
        bool any_reset = false;
        bool all_reset = true;
        for (TI instance_i = 0; instance_i < WORLD::INSTANCES; instance_i++) {
            bool reset = get(device, reset_mask, instance_i);
            any_reset = any_reset || reset;
            all_reset = all_reset && reset;
            if (reset) {
                sample_initial_state(device, world, get_ref(device, parameters, instance_i), get_ref(device, states, instance_i), rng);
            }
        }
        utils::assert_exit(device, !any_reset || all_reset, "hyperdrone::tasks::visual_inertial_localization: resets must be synchronized (fixed-length episodes)");
        if (all_reset) {
            reset(device, world.autopilot.model, world.autopilot.state, rng);
        }
        request_render(device, world, reset_mask);
    }

    // camera stride: the base render (and with it history/episode_start/shutter bookkeeping)
    // runs only on frame boundaries; the shutter pair then spans the frame interval. Resets
    // re-anchor the phase, so the reset step always produces a fresh frame
    template <typename DEVICE, typename TASK_SPEC, typename PARAMETER_SPEC, typename STATE_SPEC, typename RESET_SPEC>
    void render(DEVICE& device, rl::environments::hyperdrone::tasks::visual_inertial_localization::World<TASK_SPEC>& world, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, const Tensor<RESET_SPEC>& reset_mask) {
        using TI = typename TASK_SPEC::TI;
        using WORLD = rl::environments::hyperdrone::tasks::visual_inertial_localization::World<TASK_SPEC>;
        using NEXT_WORLD = typename TASK_SPEC::NEXT_WORLD;
        bool any_reset = false;
        bool all_reset = true;
        for (TI instance_i = 0; instance_i < WORLD::INSTANCES; instance_i++) {
            bool reset = get(device, reset_mask, instance_i);
            any_reset = any_reset || reset;
            all_reset = all_reset && reset;
        }
        utils::assert_exit(device, !any_reset || all_reset, "hyperdrone::tasks::visual_inertial_localization: resets must be synchronized (fixed-length episodes)");
        if (all_reset) {
            world.task_step = 0;
        }
        if (world.task_step % WORLD::FRAME_STRIDE == 0) {
            render(device, static_cast<NEXT_WORLD&>(world), parameters, states, reset_mask);
        }
        world.task_step++;
        world.render_pending = false;
    }

    // the autopilot produces the motor commands, the base step integrates them on the l2f
    // sub-state, and the waypoint index is carried and advanced here
    template <typename DEVICE, typename TASK_SPEC, typename PARAMETER_SPEC, typename STATE_SPEC, typename ACTION_SPEC, typename NEXT_STATE_SPEC, typename RNG>
    void step(DEVICE& device, rl::environments::hyperdrone::tasks::visual_inertial_localization::World<TASK_SPEC>& world, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, const Tensor<ACTION_SPEC>&, Tensor<NEXT_STATE_SPEC>& next_states, RNG& rng) {
        using TI = typename TASK_SPEC::TI;
        using WORLD = rl::environments::hyperdrone::tasks::visual_inertial_localization::World<TASK_SPEC>;
        using NEXT_WORLD = typename TASK_SPEC::NEXT_WORLD;
        static_assert(get<1>(typename ACTION_SPEC::SHAPE{}) == WORLD::ACTION_DIM, "the visual-inertial localization task is autonomous: its action tensor is empty");
        rl::environments::hyperdrone::tasks::visual_inertial_localization::_control<DEVICE, TASK_SPEC>(device, world, parameters, states, rng);
        step(device, static_cast<NEXT_WORLD&>(world), parameters, states, world.autopilot.actions, next_states, rng);
        const bool hold = WORLD::INITIALIZATION_HOLD_STEPS > 0 && world.task_step <= WORLD::INITIALIZATION_HOLD_STEPS;
        for (TI instance_i = 0; instance_i < WORLD::INSTANCES; instance_i++) {
            if (hold) {
                get_ref(device, next_states, instance_i).current_waypoint = get_ref(device, states, instance_i).current_waypoint;
            } else {
                rl::environments::hyperdrone::tasks::visual_inertial_localization::_waypoint_step<DEVICE, TASK_SPEC>(device, get_ref(device, parameters, instance_i), get_ref(device, states, instance_i), get_ref(device, next_states, instance_i));
            }
        }
    }

    template <typename DEVICE, typename TASK_SPEC, typename PARAMETER_SPEC, typename STATE_SPEC, typename ACTION_SPEC, typename NEXT_STATE_SPEC, typename REWARD_SPEC, typename RNG>
    void reward(DEVICE& device, rl::environments::hyperdrone::tasks::visual_inertial_localization::World<TASK_SPEC>& world, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, const Tensor<ACTION_SPEC>&, Tensor<NEXT_STATE_SPEC>& next_states, Tensor<REWARD_SPEC>& rewards, RNG& rng) {
        using NEXT_WORLD = typename TASK_SPEC::NEXT_WORLD;
        reward(device, static_cast<NEXT_WORLD&>(world), parameters, states, world.autopilot.actions, next_states, rewards, rng);
    }

    template <typename DEVICE, typename TASK_SPEC, typename PARAMETER_SPEC, typename STATE_SPEC, typename OBSERVATION_SPEC, typename RNG>
    void observe(DEVICE& device, rl::environments::hyperdrone::tasks::visual_inertial_localization::World<TASK_SPEC>& world, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, typename TASK_SPEC::NEXT_WORLD::Observation observation_type, Tensor<OBSERVATION_SPEC>& observations, RNG& rng) {
        using NEXT_WORLD = typename TASK_SPEC::NEXT_WORLD;
        if(world.render_pending){
            render(device, world, parameters, states, world.render_reset);
        }
        observe(device, static_cast<NEXT_WORLD&>(world), parameters, states, observation_type, observations, rng);
    }

    template <typename DEVICE, typename TASK_SPEC, typename PARAMETER_SPEC, typename STATE_SPEC, typename OBSERVATION_SPEC, typename RNG>
    void observe(DEVICE& device, rl::environments::hyperdrone::tasks::visual_inertial_localization::World<TASK_SPEC>& world, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, typename rl::environments::hyperdrone::tasks::visual_inertial_localization::World<TASK_SPEC>::ObservationIMU, Tensor<OBSERVATION_SPEC>& observations, RNG& rng) {
        using T = typename TASK_SPEC::T;
        using TI = typename TASK_SPEC::TI;
        using WORLD = rl::environments::hyperdrone::tasks::visual_inertial_localization::World<TASK_SPEC>;
        static_assert(get<0>(typename OBSERVATION_SPEC::SHAPE{}) == WORLD::INSTANCES);
        static_assert(get<1>(typename OBSERVATION_SPEC::SHAPE{}) == WORLD::ObservationIMU::DIM);
        utils::assert_exit(device, world.task_step > 0, "hyperdrone::tasks::visual_inertial_localization: render must be called before observe");
        const TI frame_age = (world.task_step - 1) % WORLD::FRAME_STRIDE;
        for (TI instance_i = 0; instance_i < WORLD::INSTANCES; instance_i++) {
            const auto& state = get_ref(device, states, instance_i);
            for (TI dim_i = 0; dim_i < 3; dim_i++) {
                set(device, observations, (T)state.accelerometer[dim_i], instance_i, dim_i);
                set(device, observations, (T)state.gyro[dim_i], instance_i, 3 + dim_i);
            }
            set(device, observations, (T)frame_age / (T)WORLD::FRAME_STRIDE, instance_i, 6);
            set(device, observations, frame_age == 0 ? (T)1 : (T)0, instance_i, 7);
        }
    }

    template <typename DEVICE, typename TASK_SPEC, typename PARAMETER_SPEC, typename STATE_SPEC, typename OBSERVATION_SPEC, typename RNG>
    void observe(DEVICE& device, rl::environments::hyperdrone::tasks::visual_inertial_localization::World<TASK_SPEC>& world, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, typename rl::environments::hyperdrone::tasks::visual_inertial_localization::World<TASK_SPEC>::ObservationGroundTruthPose, Tensor<OBSERVATION_SPEC>& observations, RNG& rng) {
        using T = typename TASK_SPEC::T;
        using TI = typename TASK_SPEC::TI;
        using WORLD = rl::environments::hyperdrone::tasks::visual_inertial_localization::World<TASK_SPEC>;
        static_assert(get<0>(typename OBSERVATION_SPEC::SHAPE{}) == WORLD::INSTANCES);
        static_assert(get<1>(typename OBSERVATION_SPEC::SHAPE{}) == WORLD::ObservationGroundTruthPose::DIM);
        for (TI instance_i = 0; instance_i < WORLD::INSTANCES; instance_i++) {
            const auto& instance_parameters = get_ref(device, parameters, instance_i);
            const auto& state = get_ref(device, states, instance_i);
            T position[3];
            T orientation[4];
            rl::environments::hyperdrone::tasks::visual_inertial_localization::drone_scene_pose<DEVICE, TASK_SPEC>(device, instance_parameters, state, position, orientation);
            for (TI dim_i = 0; dim_i < 3; dim_i++) {
                set(device, observations, position[dim_i], instance_i, dim_i);
            }
            for (TI dim_i = 0; dim_i < 4; dim_i++) {
                set(device, observations, orientation[dim_i], instance_i, 3 + dim_i);
            }
        }
    }

    template <typename DEVICE, typename TASK_SPEC, typename PARAMETER_SPEC, typename STATE_SPEC, typename OBSERVATION_SPEC, typename RNG>
    void observe(DEVICE& device, rl::environments::hyperdrone::tasks::visual_inertial_localization::World<TASK_SPEC>& world, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, typename rl::environments::hyperdrone::tasks::visual_inertial_localization::World<TASK_SPEC>::ObservationPrivileged, Tensor<OBSERVATION_SPEC>& observations, RNG& rng) {
        using T = typename TASK_SPEC::T;
        using TI = typename TASK_SPEC::TI;
        using WORLD = rl::environments::hyperdrone::tasks::visual_inertial_localization::World<TASK_SPEC>;
        using NEXT_WORLD = typename TASK_SPEC::NEXT_WORLD;
        static_assert(get<1>(typename OBSERVATION_SPEC::SHAPE{}) == WORLD::OBSERVATION_DIM_PRIVILEGED);
        constexpr TI BASE_DIM = NEXT_WORLD::OBSERVATION_DIM_PRIVILEGED;
        auto base_observations = view_range(device, observations, 0, tensor::ViewSpec<1, BASE_DIM>{});
        observe(device, static_cast<NEXT_WORLD&>(world), parameters, states, typename NEXT_WORLD::ObservationPrivileged{}, base_observations, rng);
        for (TI instance_i = 0; instance_i < WORLD::INSTANCES; instance_i++) {
            const auto& instance_parameters = get_ref(device, parameters, instance_i);
            const auto& state = get_ref(device, states, instance_i);
            for (TI dim_i = 0; dim_i < WORLD::WAYPOINT_DIM; dim_i++) {
                set(device, observations, (T)instance_parameters.waypoints[state.current_waypoint][dim_i], instance_i, BASE_DIM + dim_i);
            }
        }
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

// ADL entry points: generic composite code dispatches member lifecycle verbs without Tensor
// arguments, so the task's overloads must be reachable through the member type's namespace
RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::environments::hyperdrone::tasks::visual_inertial_localization {
    template <typename DEVICE, typename TASK_SPEC>
    void malloc(DEVICE& device, World<TASK_SPEC>& world){
        ::rl_tools::malloc(device, world);
    }
    template <typename DEVICE, typename TASK_SPEC, typename DATASET>
    void init(DEVICE& device, World<TASK_SPEC>& world, typename TASK_SPEC::NEXT_WORLD::SharedContext& shared, const DATASET& dataset, const typename DATASET::Corpus& corpus, typename TASK_SPEC::TI first_scene, typename TASK_SPEC::TI num_scenes, typename TASK_SPEC::TI member_index){
        ::rl_tools::init(device, world, shared, dataset, corpus, first_scene, num_scenes, member_index);
    }
    template <typename DEVICE, typename TASK_SPEC>
    void free(DEVICE& device, World<TASK_SPEC>& world){
        ::rl_tools::free(device, world);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
