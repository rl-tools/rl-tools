#include "../../../../../version.h"
#include "../../../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_TASKS_VISUAL_INERTIAL_LOCALIZATION_TRACKING_OPERATIONS_CPU_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_TASKS_VISUAL_INERTIAL_LOCALIZATION_TRACKING_OPERATIONS_CPU_H

#include "planning_operations_cpu.h"
#include "../../controllers/raptor/operations_cpu.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {
    template <typename DEVICE, typename TASK_SPEC>
    bool sample_trajectory_reference(DEVICE& device, const rl::environments::hyperdrone::tasks::visual_inertial_localization::World<TASK_SPEC>& world, const typename rl::environments::hyperdrone::tasks::visual_inertial_localization::World<TASK_SPEC>::Parameters& parameters, const typename rl::environments::hyperdrone::tasks::visual_inertial_localization::World<TASK_SPEC>::State& state, rl::environments::hyperdrone::tasks::visual_inertial_localization::trajectory::Sample<typename rl::environments::hyperdrone::tasks::visual_inertial_localization::World<TASK_SPEC>::TRAJECTORY_SPEC>& sample) {
        using WORLD = rl::environments::hyperdrone::tasks::visual_inertial_localization::World<TASK_SPEC>;
        sample = {};
        if (!parameters.route_valid || state.collision) {
            for (unsigned axis = 0; axis < 3; axis++) {
                sample.position[axis] = state.position[axis];
            }
            return false;
        }
        if (state.current_waypoint == 0) {
            for (unsigned axis = 0; axis < 3; axis++) {
                sample.position[axis] = parameters.waypoints[0][axis];
            }
            return true;
        }
        rl::environments::hyperdrone::tasks::visual_inertial_localization::trajectory::Segment<typename WORLD::TRAJECTORY_SPEC> segment;
        return trajectory_segment(device, world, parameters, state.current_waypoint, segment) && evaluate(device, segment, state.segment_time, sample);
    }
    template <typename DEVICE, typename TASK_SPEC, typename PARAMETER_SPEC, typename STATE_SPEC, typename RNG>
    void control_trajectory(DEVICE& device, rl::environments::hyperdrone::tasks::visual_inertial_localization::World<TASK_SPEC>& world, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, RNG& rng) {
        using WORLD = rl::environments::hyperdrone::tasks::visual_inertial_localization::World<TASK_SPEC>;
        using T = typename WORLD::T;
        using TI = typename WORLD::TI;
        for (TI instance = 0; instance < WORLD::INSTANCES; instance++) {
            auto& state = get_ref(device, states, instance);
            const auto& instance_parameters = get_ref(device, parameters, instance);
            rl::environments::hyperdrone::tasks::visual_inertial_localization::trajectory::Sample<typename WORLD::TRAJECTORY_SPEC> sample;
            const bool valid = sample_trajectory_reference(device, world, instance_parameters, state, sample);
            rl::environments::hyperdrone::controllers::raptor::Target<T> target;
            for (TI axis = 0; axis < 3; axis++) {
                target.position[axis] = state.position[axis] + math::clamp(device.math, sample.position[axis] - state.position[axis], -TASK_SPEC::TARGET_POSITION_ERROR_CLIP, TASK_SPEC::TARGET_POSITION_ERROR_CLIP);
                target.linear_velocity[axis] = sample.velocity[axis] * state.phase_rate;
            }
            target.yaw = state.target_yaw;
            const T dt = instance_parameters.agents[0].dynamics.integration.dt;
            if (valid && state.current_waypoint > 0) {
                T forward[2] = {sample.velocity[0], sample.velocity[1]};
                if (forward[0] * forward[0] + forward[1] * forward[1] < (T)1e-6) {
                    for (TI axis = 0; axis < 2; axis++) {
                        forward[axis] = instance_parameters.waypoints[state.current_waypoint][axis] - sample.position[axis];
                    }
                }
                if (forward[0] * forward[0] + forward[1] * forward[1] > (T)1e-8) {
                    const T desired_yaw = math::atan2(device.math, forward[1], forward[0]);
                    const T delta = math::atan2(device.math, math::sin(device.math, desired_yaw - target.yaw), math::cos(device.math, desired_yaw - target.yaw));
                    update_yaw_target(device, world.controller, state, target, delta, dt, TASK_SPEC::MAX_YAW_RATE);
                }
            }
            state.target_yaw = target.yaw;
            observe(device, world.controller, world.controller_state, world.controller_buffer, state, target, instance);
        }
        evaluate_step(device, world.controller, world.controller_state, world.controller_buffer, rng);
    }
    template <typename DEVICE, typename TASK_SPEC, typename RNG>
    void advance_trajectory(DEVICE& device, rl::environments::hyperdrone::tasks::visual_inertial_localization::World<TASK_SPEC>& world, typename rl::environments::hyperdrone::tasks::visual_inertial_localization::World<TASK_SPEC>::Parameters& parameters, const typename rl::environments::hyperdrone::tasks::visual_inertial_localization::World<TASK_SPEC>::State& state, typename rl::environments::hyperdrone::tasks::visual_inertial_localization::World<TASK_SPEC>::State& next, RNG& rng) {
        using WORLD = rl::environments::hyperdrone::tasks::visual_inertial_localization::World<TASK_SPEC>;
        using T = typename WORLD::T;
        using TI = typename WORLD::TI;
        next.current_waypoint = state.current_waypoint;
        next.segment_time = state.segment_time;
        next.target_yaw = state.target_yaw;
        next.phase_rate = state.phase_rate;
        next.phase_acceleration = state.phase_acceleration;
        next.completed_routes = state.completed_routes;
        next.collision = state.collision;
        T from[3], to[3];
        trajectory_scene_position(device, parameters, state.position, from);
        trajectory_scene_position(device, parameters, next.position, to);
        bool finite = true;
        for (TI axis = 0; axis < 3; axis++) {
            finite = finite && finite_trajectory_value(device, next.linear_velocity[axis]) && finite_trajectory_value(device, next.angular_velocity[axis]);
        }
        for (TI axis = 0; axis < 4; axis++) {
            finite = finite && finite_trajectory_value(device, next.orientation[axis]);
        }
        if (state.collision || !parameters.route_valid || !finite || !collision_free(device, world.navigation[world.active_slot].geometry, from, to, TASK_SPEC::COLLISION_RADIUS + (T)0.0001)) {
            next = state;
            next.collision = true;
            return;
        }
        if constexpr (TASK_SPEC::NUM_WAYPOINTS == 1) {
            return;
        }
        if (world.task_step <= TASK_SPEC::INITIALIZATION_HOLD_STEPS) {
            return;
        }
        if (next.current_waypoint == 0) {
            next.current_waypoint = 1;
        }
        const T dt = parameters.agents[0].dynamics.integration.dt;
        rl::environments::hyperdrone::tasks::visual_inertial_localization::trajectory::Sample<typename WORLD::TRAJECTORY_SPEC> sample;
        if (!sample_trajectory_reference(device, world, parameters, next, sample)) {
            next.collision = true;
            return;
        }
        const T error = trajectory_distance(device, sample.position, next.position);
        T pace = math::clamp(device.math, ((T)0.9 * TASK_SPEC::TRACKING_MARGIN - error) / ((T)0.4 * TASK_SPEC::TRACKING_MARGIN), (T)0, (T)1);
        T forward[2] = {sample.velocity[0], sample.velocity[1]};
        if (forward[0] * forward[0] + forward[1] * forward[1] < (T)1e-6) {
            for (TI axis = 0; axis < 2; axis++) {
                forward[axis] = parameters.waypoints[next.current_waypoint][axis] - sample.position[axis];
            }
        }
        if (forward[0] * forward[0] + forward[1] * forward[1] > (T)1e-8) {
            const auto& q = next.orientation;
            const T yaw = math::atan2(device.math, (T)2 * (q[0] * q[3] + q[1] * q[2]), (T)1 - (T)2 * (q[2] * q[2] + q[3] * q[3]));
            const T heading = math::atan2(device.math, forward[1], forward[0]);
            const T alignment = math::max(device.math, (T)0, math::cos(device.math, heading - yaw));
            pace *= alignment * alignment;
        }
        rl::environments::hyperdrone::tasks::visual_inertial_localization::trajectory::Phase<typename WORLD::TRAJECTORY_SPEC> phase{next.phase_rate, next.phase_acceleration};
        const T phase_acceleration = math::min(device.math, TASK_SPEC::PHASE_ACCELERATION, math::min(device.math, (T)0.35 * TASK_SPEC::MAX_ACCELERATION / TASK_SPEC::CRUISE_SPEED_MAX, (T)0.25 * TASK_SPEC::MAX_JERK / ((T)3 * (T)0.65 * TASK_SPEC::MAX_ACCELERATION)));
        const T phase_jerk = math::min(device.math, TASK_SPEC::PHASE_JERK, (T)0.25 * TASK_SPEC::MAX_JERK / TASK_SPEC::CRUISE_SPEED_MAX);
        next.segment_time += advance(device, phase, pace, dt, phase_acceleration, phase_jerk);
        next.phase_rate = phase.rate;
        next.phase_acceleration = phase.acceleration;
        if (next.segment_time >= parameters.durations[next.current_waypoint]) {
            next.segment_time -= parameters.durations[next.current_waypoint];
            if (next.current_waypoint + 1 < TASK_SPEC::NUM_WAYPOINTS) {
                next.current_waypoint++;
            } else {
                next.completed_routes++;
                next.collision = !sample_trajectory(device, world, parameters, next, rng, false);
            }
        }
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
