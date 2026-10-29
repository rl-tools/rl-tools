#include "../../../../../version.h"
#include "../../../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_TASKS_VISUAL_INERTIAL_LOCALIZATION_PLANNING_OPERATIONS_CPU_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_TASKS_VISUAL_INERTIAL_LOCALIZATION_PLANNING_OPERATIONS_CPU_H

#include "visual_inertial_localization.h"
#include "trajectory_operations_generic.h"
#include "collision_operations_cpu.h"

#include <algorithm>
#include <limits>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {
    template <typename DEVICE, typename T>
    T trajectory_distance(DEVICE& device, const T* a, const T* b) {
        T squared = 0;
        for (unsigned axis = 0; axis < 3; axis++) {
            squared += (a[axis] - b[axis]) * (a[axis] - b[axis]);
        }
        return math::sqrt(device.math, squared);
    }
    template <typename DEVICE, typename PARAMETERS, typename T>
    void trajectory_scene_position(DEVICE&, const PARAMETERS& parameters, const T* in, T* out) {
        out[0] = parameters.scene_yaw_cos * in[0] - parameters.scene_yaw_sin * in[1] + parameters.scene_translation[0];
        out[1] = parameters.scene_yaw_sin * in[0] + parameters.scene_yaw_cos * in[1] + parameters.scene_translation[1];
        out[2] = in[2] + parameters.scene_translation[2];
    }
    template <typename DEVICE, typename PARAMETERS, typename T>
    void trajectory_dynamics_position(DEVICE&, const PARAMETERS& parameters, const T* in, T* out) {
        const T dx = in[0] - parameters.scene_translation[0];
        const T dy = in[1] - parameters.scene_translation[1];
        out[0] = parameters.scene_yaw_cos * dx + parameters.scene_yaw_sin * dy;
        out[1] = -parameters.scene_yaw_sin * dx + parameters.scene_yaw_cos * dy;
        out[2] = in[2] - parameters.scene_translation[2];
    }
    template <typename T, typename TI>
    T trajectory_radical_inverse(TI index, TI base) {
        T value = 0;
        T fraction = 1;
        while (index > 0) {
            fraction /= static_cast<T>(base);
            value += fraction * static_cast<T>(index % base);
            index /= base;
        }
        return value;
    }
    template <typename DEVICE, typename TASK_SPEC>
    void build_navigation(DEVICE& device, rl::environments::hyperdrone::tasks::visual_inertial_localization::World<TASK_SPEC>& world, const typename TASK_SPEC::NEXT_WORLD::SharedContext& shared) {
        using T = typename TASK_SPEC::T;
        using TI = typename TASK_SPEC::TI;
        const T clearance = TASK_SPEC::COLLISION_RADIUS + TASK_SPEC::TRACKING_MARGIN;
        world.navigation.resize(world.slots.size());
        for (TI slot_index = 0; slot_index < world.slots.size(); slot_index++) {
            auto& navigation = world.navigation[slot_index];
            const auto& slot = world.slots[slot_index];
            utils::assert_exit(device, init(device, navigation.geometry, shared.library.scenes[slot.scene_id]), "visual_inertial_localization: invalid collision geometry");
            navigation.positions.clear();
            for (TI index = 0; index < slot.annotations.num_positions && navigation.positions.size() < TASK_SPEC::ROADMAP_NODES; index++) {
                const auto& position = slot.annotations.positions[index].position;
                if (collision_free(device, navigation.geometry, position, position, clearance)) {
                    navigation.positions.push_back({position[0], position[1], position[2]});
                }
            }
            if (navigation.positions.empty()) {
                navigation.neighbors.clear();
                continue;
            }
            auto lower = navigation.positions.front();
            auto upper = lower;
            for (const auto& position : navigation.positions) {
                for (TI axis = 0; axis < 3; axis++) {
                    lower[axis] = math::min(device.math, lower[axis], position[axis]);
                    upper[axis] = math::max(device.math, upper[axis], position[axis]);
                }
            }
            lower[2] -= (T)0.3;
            upper[2] += (T)0.3;
            for (TI sample = 1; sample <= TASK_SPEC::ROADMAP_NODES * 32 && navigation.positions.size() < TASK_SPEC::ROADMAP_NODES; sample++) {
                const TI bases[3] = {2, 3, 5};
                std::array<T, 3> candidate;
                for (TI axis = 0; axis < 3; axis++) {
                    candidate[axis] = lower[axis] + (upper[axis] - lower[axis]) * trajectory_radical_inverse<T>(sample, bases[axis]);
                }
                TI nearest = 0;
                T distance = std::numeric_limits<T>::max();
                for (TI index = 0; index < navigation.positions.size(); index++) {
                    const T current = trajectory_distance(device, candidate.data(), navigation.positions[index].data());
                    if (current < distance) {
                        distance = current;
                        nearest = index;
                    }
                }
                if (distance >= TASK_SPEC::ROADMAP_SPACING && distance <= TASK_SPEC::CONNECTION_DISTANCE && collision_free(device, navigation.geometry, candidate.data(), navigation.positions[nearest].data(), clearance)) {
                    navigation.positions.push_back(candidate);
                }
            }
            navigation.neighbors.assign(navigation.positions.size(), {});
            for (TI first = 0; first < navigation.positions.size(); first++) {
                for (TI second = first + 1; second < navigation.positions.size(); second++) {
                    const T distance = trajectory_distance(device, navigation.positions[first].data(), navigation.positions[second].data());
                    if (distance >= TASK_SPEC::ROADMAP_SPACING / (T)2 && distance <= TASK_SPEC::CONNECTION_DISTANCE && collision_free(device, navigation.geometry, navigation.positions[first].data(), navigation.positions[second].data(), clearance)) {
                        navigation.neighbors[first].push_back(second);
                        navigation.neighbors[second].push_back(first);
                    }
                }
            }
            std::vector<TI> components(navigation.positions.size(), navigation.positions.size());
            std::vector<T> extents;
            for (TI index = 0; index < navigation.positions.size(); index++) {
                if (components[index] != navigation.positions.size()) {
                    continue;
                }
                const TI component = extents.size();
                std::vector<TI> pending{index};
                components[index] = component;
                auto minimum = navigation.positions[index];
                auto maximum = minimum;
                for (TI cursor = 0; cursor < pending.size(); cursor++) {
                    const TI node = pending[cursor];
                    for (TI axis = 0; axis < 3; axis++) {
                        minimum[axis] = math::min(device.math, minimum[axis], navigation.positions[node][axis]);
                        maximum[axis] = math::max(device.math, maximum[axis], navigation.positions[node][axis]);
                    }
                    for (const TI neighbor : navigation.neighbors[node]) {
                        if (components[neighbor] == navigation.positions.size()) {
                            components[neighbor] = component;
                            pending.push_back(neighbor);
                        }
                    }
                }
                extents.push_back(trajectory_distance(device, minimum.data(), maximum.data()));
            }
            const T largest = *std::max_element(extents.begin(), extents.end());
            navigation.spawn_nodes.clear();
            for (TI index = 0; index < navigation.positions.size(); index++) {
                if (!navigation.neighbors[index].empty() && extents[components[index]] >= largest * (T)0.5) {
                    navigation.spawn_nodes.push_back(index);
                }
            }
        }
    }
    template <typename DEVICE, typename TASK_SPEC>
    bool trajectory_segment(DEVICE& device, const rl::environments::hyperdrone::tasks::visual_inertial_localization::World<TASK_SPEC>&, const typename rl::environments::hyperdrone::tasks::visual_inertial_localization::World<TASK_SPEC>::Parameters& parameters, typename TASK_SPEC::TI index, rl::environments::hyperdrone::tasks::visual_inertial_localization::trajectory::Segment<typename rl::environments::hyperdrone::tasks::visual_inertial_localization::World<TASK_SPEC>::TRAJECTORY_SPEC>& segment) {
        return index > 0 && index < TASK_SPEC::NUM_WAYPOINTS && init(device, segment, parameters.waypoints[index - 1], parameters.waypoints[index], parameters.velocities[index - 1], parameters.velocities[index], parameters.durations[index]);
    }
    template <typename DEVICE, typename TASK_SPEC>
    bool fit_trajectory(DEVICE& device, const rl::environments::hyperdrone::tasks::visual_inertial_localization::World<TASK_SPEC>& world, typename rl::environments::hyperdrone::tasks::visual_inertial_localization::World<TASK_SPEC>::Parameters& parameters) {
        using T = typename TASK_SPEC::T;
        using TI = typename TASK_SPEC::TI;
        using WORLD = rl::environments::hyperdrone::tasks::visual_inertial_localization::World<TASK_SPEC>;
        const auto& geometry = world.navigation[world.active_slot].geometry;
        for (TI index = 0; index < TASK_SPEC::NUM_WAYPOINTS; index++) {
            for (TI axis = 0; axis < 3; axis++) {
                parameters.velocities[index][axis] = 0;
            }
            if (index > 0) {
                parameters.durations[index] = estimate_duration(device, parameters.waypoints[index - 1], parameters.waypoints[index], parameters.cruise_speed, TASK_SPEC::MAX_ACCELERATION * (T)0.65, TASK_SPEC::MAX_JERK * (T)0.5);
                parameters.durations[index] = math::max(device.math, parameters.durations[index], (T)0.5);
            }
        }
        for (TI index = 1; index + 1 < TASK_SPEC::NUM_WAYPOINTS; index++) {
            const T before = trajectory_distance(device, parameters.waypoints[index - 1], parameters.waypoints[index]);
            const T after = trajectory_distance(device, parameters.waypoints[index], parameters.waypoints[index + 1]);
            if (before < (T)1e-5 || after < (T)1e-5) {
                continue;
            }
            T tangent[3];
            T alignment = 0;
            T norm = 0;
            for (TI axis = 0; axis < 3; axis++) {
                const T incoming = (parameters.waypoints[index][axis] - parameters.waypoints[index - 1][axis]) / before;
                const T outgoing = (parameters.waypoints[index + 1][axis] - parameters.waypoints[index][axis]) / after;
                alignment += incoming * outgoing;
                tangent[axis] = incoming + outgoing;
                norm += tangent[axis] * tangent[axis];
            }
            if (alignment <= (T)-0.5 || norm < (T)1e-6) {
                continue;
            }
            const T speed = math::min(device.math, parameters.cruise_speed * (T)0.7, math::min(device.math, before / parameters.durations[index], after / parameters.durations[index + 1])) * ((T)1 + alignment) / (T)2;
            for (TI axis = 0; axis < 3; axis++) {
                parameters.velocities[index][axis] = tangent[axis] * speed / math::sqrt(device.math, norm);
            }
        }
        // Velocity reductions affect both adjacent segments, so certify the whole horizon again.
        for (TI pass = 0; pass < 24; pass++) {
            bool complete = true;
            for (TI index = 1; index < TASK_SPEC::NUM_WAYPOINTS; index++) {
                rl::environments::hyperdrone::tasks::visual_inertial_localization::trajectory::Segment<typename WORLD::TRAJECTORY_SPEC> segment;
                rl::environments::hyperdrone::tasks::visual_inertial_localization::trajectory::Bounds<typename WORLD::TRAJECTORY_SPEC> bounds;
                if (!trajectory_segment(device, world, parameters, index, segment) || !derivative_bounds(device, segment, bounds)) {
                    return false;
                }
                const T stretch = math::max(device.math, bounds.speed / parameters.cruise_speed, math::max(device.math, math::sqrt(device.math, bounds.acceleration / (TASK_SPEC::MAX_ACCELERATION * (T)0.65)), math::cbrt(device.math, bounds.jerk / (TASK_SPEC::MAX_JERK * (T)0.5))));
                if (stretch > (T)1.001) {
                    parameters.durations[index] *= stretch * (T)1.01;
                    complete = false;
                }
                T controls[6][3];
                for (TI control = 0; control < 6; control++) {
                    trajectory_scene_position(device, parameters, segment.control[control], controls[control]);
                }
                if (!bezier_collision_free(device, geometry, controls, TASK_SPEC::COLLISION_RADIUS + TASK_SPEC::TRACKING_MARGIN)) {
                    for (TI axis = 0; axis < 3; axis++) {
                        parameters.velocities[index - 1][axis] *= (T)0.5;
                        parameters.velocities[index][axis] *= (T)0.5;
                    }
                    complete = false;
                }
            }
            if (complete) {
                return true;
            }
        }
        return false;
    }
    template <typename DEVICE, typename TASK_SPEC, typename RNG>
    void sample_trajectory_placement(DEVICE& device, const rl::environments::hyperdrone::tasks::visual_inertial_localization::World<TASK_SPEC>& world, typename rl::environments::hyperdrone::tasks::visual_inertial_localization::World<TASK_SPEC>::Parameters& parameters, RNG& rng) {
        using T = typename TASK_SPEC::T;
        using TI = typename TASK_SPEC::TI;
        const auto& navigation = world.navigation[world.active_slot];
        if (navigation.spawn_nodes.empty()) {
            return;
        }
        const TI start = navigation.spawn_nodes[random::uniform_int_distribution(device.random, (TI)0, (TI)(navigation.spawn_nodes.size() - 1), rng)];
        const auto& neighbors = navigation.neighbors[start];
        const TI next = neighbors[random::uniform_int_distribution(device.random, (TI)0, (TI)(neighbors.size() - 1), rng)];
        const auto& anchor = navigation.positions[start];
        const auto& target = navigation.positions[next];
        const auto& q = parameters.agents[0].dynamics.mdp.init.orientation;
        const T body_yaw = math::atan2(device.math, (T)2 * (q[0] * q[3] + q[1] * q[2]), (T)1 - (T)2 * (q[2] * q[2] + q[3] * q[3]));
        const T yaw = math::atan2(device.math, target[1] - anchor[1], target[0] - anchor[0]) - body_yaw;
        parameters.scene_yaw_cos = math::cos(device.math, yaw);
        parameters.scene_yaw_sin = math::sin(device.math, yaw);
        const auto& center = parameters.agents[0].dynamics.mdp.init.position;
        parameters.scene_translation[0] = anchor[0] - parameters.scene_yaw_cos * center[0] + parameters.scene_yaw_sin * center[1];
        parameters.scene_translation[1] = anchor[1] - parameters.scene_yaw_sin * center[0] - parameters.scene_yaw_cos * center[1];
        parameters.scene_translation[2] = anchor[2] - center[2];
    }
    template <typename DEVICE, typename TASK_SPEC, typename RNG>
    bool sample_trajectory(DEVICE& device, rl::environments::hyperdrone::tasks::visual_inertial_localization::World<TASK_SPEC>& world, typename rl::environments::hyperdrone::tasks::visual_inertial_localization::World<TASK_SPEC>::Parameters& parameters, typename rl::environments::hyperdrone::tasks::visual_inertial_localization::World<TASK_SPEC>::State& state, RNG& rng, bool initial) {
        using T = typename TASK_SPEC::T;
        using TI = typename TASK_SPEC::TI;
        const auto& navigation = world.navigation[world.active_slot];
        parameters.route_valid = false;
        if (initial) {
            state.current_waypoint = 0;
            state.segment_time = 0;
            state.phase_rate = 0;
            state.phase_acceleration = 0;
            for (TI index = 0; index < TASK_SPEC::NUM_WAYPOINTS; index++) {
                for (TI axis = 0; axis < 3; axis++) {
                    parameters.waypoints[index][axis] = state.position[axis];
                }
            }
            for (TI index = 0; index < TASK_SPEC::ROADMAP_NODES; index++) {
                parameters.visits[index] = 0;
            }
            parameters.cruise_speed = random::uniform_real_distribution(device.random, TASK_SPEC::CRUISE_SPEED_MIN, TASK_SPEC::CRUISE_SPEED_MAX, rng);
        }
        std::vector<TI> candidates;
        for (TI index = 0; index < navigation.positions.size(); index++) {
            if (!navigation.neighbors[index].empty()) {
                candidates.push_back(index);
            }
        }
        if (candidates.empty()) {
            return false;
        }
        T anchor[3];
        if (initial) {
            trajectory_scene_position(device, parameters, state.position, anchor);
        } else {
            trajectory_scene_position(device, parameters, parameters.waypoints[TASK_SPEC::NUM_WAYPOINTS - 1], anchor);
        }
        TI start = navigation.positions.size();
        T nearest = std::numeric_limits<T>::max();
        for (const TI candidate : candidates) {
            const T distance = trajectory_distance(device, anchor, navigation.positions[candidate].data());
            if (distance < nearest && collision_free(device, navigation.geometry, anchor, navigation.positions[candidate].data(), TASK_SPEC::COLLISION_RADIUS + TASK_SPEC::TRACKING_MARGIN)) {
                start = candidate;
                nearest = distance;
            }
        }
        if (start == navigation.positions.size()) {
            return false;
        }
        std::vector<std::array<T, 3>> points(TASK_SPEC::NUM_WAYPOINTS);
        std::copy(anchor, anchor + 3, points[0].begin());
        parameters.route_nodes[0] = start;
        parameters.visits[start]++;
        TI previous = navigation.positions.size();
        TI current = start;
        const auto& q = state.orientation;
        const T body_yaw = math::atan2(device.math, (T)2 * (q[0] * q[3] + q[1] * q[2]), (T)1 - (T)2 * (q[2] * q[2] + q[3] * q[3]));
        const T forward_cos = math::cos(device.math, body_yaw);
        const T forward_sin = math::sin(device.math, body_yaw);
        T direction[3] = {parameters.scene_yaw_cos * forward_cos - parameters.scene_yaw_sin * forward_sin, parameters.scene_yaw_sin * forward_cos + parameters.scene_yaw_cos * forward_sin, 0};
        if constexpr (TASK_SPEC::NUM_WAYPOINTS > 1) {
            if (!initial) {
                previous = parameters.route_nodes[TASK_SPEC::NUM_WAYPOINTS - 2];
                T incoming[3];
                trajectory_scene_position(device, parameters, parameters.waypoints[TASK_SPEC::NUM_WAYPOINTS - 2], incoming);
                const T length = trajectory_distance(device, anchor, incoming);
                if (length > (T)1e-5) {
                    for (TI axis = 0; axis < 3; axis++) {
                        direction[axis] = (anchor[axis] - incoming[axis]) / length;
                    }
                }
            }
        }
        for (TI index = 1; index < TASK_SPEC::NUM_WAYPOINTS; index++) {
            const auto& neighbors = navigation.neighbors[current];
            std::vector<T> weights(neighbors.size());
            T total = 0;
            for (TI neighbor = 0; neighbor < neighbors.size(); neighbor++) {
                const TI next = neighbors[neighbor];
                const T distance = trajectory_distance(device, points[index - 1].data(), navigation.positions[next].data());
                T alignment = 0;
                for (TI axis = 0; axis < 3; axis++) {
                    alignment += direction[axis] * (navigation.positions[next][axis] - points[index - 1][axis]) / math::max(device.math, distance, (T)1e-5);
                }
                const T novelty = (T)1 / ((T)1 + (T)parameters.visits[next]);
                weights[neighbor] = next == previous && neighbors.size() > 1 ? 0 : math::exp(device.math, (T)3 * alignment) * novelty * novelty * distance / ((T)1 + distance * distance * (T)0.2);
                if (index == 1 && !collision_free(device, navigation.geometry, anchor, navigation.positions[next].data(), TASK_SPEC::COLLISION_RADIUS + TASK_SPEC::TRACKING_MARGIN)) {
                    weights[neighbor] = 0;
                }
                total += weights[neighbor];
            }
            if (total <= 0) {
                return false;
            }
            T draw = random::uniform_real_distribution(device.random, (T)0, total, rng);
            TI choice = 0;
            for (TI neighbor = 0; neighbor < neighbors.size(); neighbor++) {
                if (weights[neighbor] > 0) {
                    choice = neighbor;
                    draw -= weights[neighbor];
                    if (draw <= 0) {
                        break;
                    }
                }
            }
            previous = current;
            current = neighbors[choice];
            points[index] = navigation.positions[current];
            parameters.route_nodes[index] = current;
            parameters.visits[current]++;
            const T length = trajectory_distance(device, points[index].data(), points[index - 1].data());
            for (TI axis = 0; axis < 3; axis++) {
                direction[axis] = (points[index][axis] - points[index - 1][axis]) / math::max(device.math, length, (T)1e-5);
            }
        }
        for (TI index = 0; index < TASK_SPEC::NUM_WAYPOINTS; index++) {
            trajectory_dynamics_position(device, parameters, points[index].data(), parameters.waypoints[index]);
        }
        parameters.route_valid = fit_trajectory(device, world, parameters);
        state.current_waypoint = initial && TASK_SPEC::INITIALIZATION_HOLD_STEPS > 0 ? 0 : (TASK_SPEC::NUM_WAYPOINTS > 1 ? 1 : 0);
        state.segment_time = 0;
        if (initial) {
            state.phase_rate = 0;
            state.phase_acceleration = 0;
        }
        return parameters.route_valid;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
