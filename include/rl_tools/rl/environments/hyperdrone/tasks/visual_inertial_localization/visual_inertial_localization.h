#include "../../../../../version.h"
#include "../../../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_TASKS_VISUAL_INERTIAL_LOCALIZATION_VISUAL_INERTIAL_LOCALIZATION_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_TASKS_VISUAL_INERTIAL_LOCALIZATION_VISUAL_INERTIAL_LOCALIZATION_H

#include "../../world.h"
#include "../../controllers/raptor/raptor.h"
#include "collision.h"
#include "trajectory.h"

#include <array>
#include <string>
#include <vector>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::environments::hyperdrone::tasks::visual_inertial_localization {

    template <typename T_T, typename T_TI, T_TI T_NUM_WAYPOINTS, T_TI T_ROADMAP_NODES, typename T_NEXT_COMPONENT>
    struct ParametersSpecification {
        using T = T_T;
        using TI = T_TI;
        static constexpr TI NUM_WAYPOINTS = T_NUM_WAYPOINTS;
        static constexpr TI ROADMAP_NODES = T_ROADMAP_NODES;
        using NEXT_COMPONENT = T_NEXT_COMPONENT;
    };
    // The horizon is renewed at its end; all positions and derivatives use the dynamics frame.
    template <typename T_SPEC>
    struct ParametersVisualInertialLocalization: T_SPEC::NEXT_COMPONENT {
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        using NEXT_COMPONENT = typename SPEC::NEXT_COMPONENT;
        static constexpr TI NUM_WAYPOINTS = SPEC::NUM_WAYPOINTS;
        T waypoints[NUM_WAYPOINTS][3] = {};
        T velocities[NUM_WAYPOINTS][3] = {};
        T durations[NUM_WAYPOINTS] = {};
        TI route_nodes[NUM_WAYPOINTS] = {};
        TI visits[SPEC::ROADMAP_NODES] = {};
        T cruise_speed = 0;
        bool route_valid = false;
    };
    template <typename T_T, typename T_TI, typename T_NEXT_COMPONENT>
    struct ComponentSpecification {
        using T = T_T;
        using TI = T_TI;
        using NEXT_COMPONENT = T_NEXT_COMPONENT;
    };
    template <typename T_SPEC>
    struct StateVisualInertialLocalization: T_SPEC::NEXT_COMPONENT {
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        using NEXT_COMPONENT = typename SPEC::NEXT_COMPONENT;
        static constexpr bool REQUIRES_INTEGRATION = false;
        static constexpr TI DIM = 7 + NEXT_COMPONENT::DIM;
        TI current_waypoint = 0;
        T segment_time = 0;
        T target_yaw = 0;
        T phase_rate = 0;
        T phase_acceleration = 0;
        TI completed_routes = 0;
        bool collision = false;
    };

    template <typename STATE, typename = void>
    struct HasIMUMeasurement {
        static constexpr bool VALUE = false;
    };
    template <typename STATE>
    struct HasIMUMeasurement<STATE, utils::typing::void_t<decltype(STATE{}.accelerometer), decltype(STATE{}.gyro)>> {
        static constexpr bool VALUE = true;
    };

    template <typename T_NEXT_WORLD>
    struct Specification {
        using NEXT_WORLD = T_NEXT_WORLD;
        using T = typename NEXT_WORLD::T;
        using TI = typename NEXT_WORLD::TI;
        // derive-and-shadow task knobs
        static constexpr TI FRAME_STRIDE = 4; // env/IMU steps per camera frame
        static constexpr TI NUM_WAYPOINTS = 48;
        static constexpr T TARGET_POSITION_ERROR_CLIP = 1.0; // autopilot position-error observation clamp
        static constexpr TI ROADMAP_NODES = 160;
        static constexpr T ROADMAP_SPACING = 0.65;
        static constexpr T CONNECTION_DISTANCE = 4.0;
        static constexpr T COLLISION_RADIUS = 0.45; // enclosing body/propeller sphere, including all attitudes
        static constexpr T TRACKING_MARGIN = 0.25;
        static constexpr T CRUISE_SPEED_MIN = 0.45;
        static constexpr T CRUISE_SPEED_MAX = 0.8;
        static constexpr T MAX_ACCELERATION = 0.7;
        static constexpr T MAX_JERK = 2.0;
        static constexpr T MAX_YAW_RATE = 0.8;
        static constexpr T PHASE_ACCELERATION = 0.2;
        static constexpr T PHASE_JERK = 0.5;
        // hover at waypoint 0 (the final spawn) for this many env steps before the route starts:
        // gives estimators a stationary window, and the launch toward waypoint 1 provides the
        // acceleration jerk that static visual-inertial initializers wait for
        static constexpr TI INITIALIZATION_HOLD_STEPS = 0;
        // the autopilot checkpoint (RAPTOR/policy.tar in the conta store): a path or a conta:HASH reference
        using CONTROLLER = controllers::raptor::Config;
        static constexpr const char* AUTOPILOT_POLICY = controllers::raptor::CHECKPOINT;
    };

    // Episodes reset together because camera cadence is shared; controller invalidation may be per row.
    // The autonomous task generates motor commands internally and exposes no external actions.
    template <typename T_TASK_SPEC>
    struct World: T_TASK_SPEC::NEXT_WORLD {
        using TASK_SPEC = T_TASK_SPEC;
        using NEXT_WORLD = typename TASK_SPEC::NEXT_WORLD;
        using T = typename NEXT_WORLD::T;
        using TI = typename NEXT_WORLD::TI;
        static_assert(HasIMUMeasurement<typename NEXT_WORLD::State>::VALUE, "the visual-inertial localization task needs l2f::StateIMU in the dynamics state chain (e.g. presets::X500FPVIMU)");
        static_assert(NEXT_WORLD::SPEC::N_AGENTS == 1, "the visual-inertial localization task is single-agent");
        static_assert(NEXT_WORLD::SPEC::HISTORY_LENGTH == 1, "the visual-inertial localization task holds one camera frame");

        static_assert(TASK_SPEC::NUM_WAYPOINTS > 0 && TASK_SPEC::ROADMAP_NODES > 1);
        static_assert(TASK_SPEC::COLLISION_RADIUS > 0 && TASK_SPEC::TRACKING_MARGIN > 0);
        static_assert(TASK_SPEC::CRUISE_SPEED_MIN > 0 && TASK_SPEC::CRUISE_SPEED_MAX >= TASK_SPEC::CRUISE_SPEED_MIN);
        static_assert(TASK_SPEC::MAX_ACCELERATION > 0 && TASK_SPEC::MAX_JERK > 0 && TASK_SPEC::MAX_YAW_RATE > 0);
        static_assert(TASK_SPEC::ROADMAP_SPACING > 0 && TASK_SPEC::CONNECTION_DISTANCE > TASK_SPEC::ROADMAP_SPACING);
        using Parameters = ParametersVisualInertialLocalization<ParametersSpecification<T, TI, TASK_SPEC::NUM_WAYPOINTS, TASK_SPEC::ROADMAP_NODES, typename NEXT_WORLD::Parameters>>;
        using State = StateVisualInertialLocalization<ComponentSpecification<T, TI, typename NEXT_WORLD::State>>;
        static constexpr TI FRAME_STRIDE = TASK_SPEC::FRAME_STRIDE;
        static constexpr bool SYNCHRONIZED_RESETS = true;
        static constexpr TI INITIALIZATION_HOLD_STEPS = TASK_SPEC::INITIALIZATION_HOLD_STEPS;
        static constexpr TI ACTION_DIM = 0;
        using CONTROLLER_CONFIG = typename TASK_SPEC::CONTROLLER;
        using CONTROLLER = typename CONTROLLER_CONFIG::template Bind<numeric_types::Policy<T>, TI, NEXT_WORLD::INSTANCES>;
        static_assert(CONTROLLER::ACTION_DIM == NEXT_WORLD::ACTION_DIM);
        static_assert(NEXT_WORLD::CONTROLLER::SUPPORTS_SCALAR && NEXT_WORLD::CONTROLLER::ACTION_INTERFACE == l2f::parameters::ActionInterface::DIRECT_MOTOR, "the autonomous task requires a direct motor base World");

        // estimator inputs beyond the frames: [accelerometer(3) | gyroscope(3) | frame_age/FRAME_STRIDE | new_frame]
        struct ObservationIMU {
            static constexpr TI DIM = 8;
            using SHAPE = tensor::Shape<TI, DIM>;
        };
        // scene-frame drone pose: position(3) + orientation quaternion wxyz(4); evaluation-only ground truth
        struct ObservationGroundTruthPose {
            static constexpr TI DIM = 7;
            using SHAPE = tensor::Shape<TI, DIM>;
        };
        // privileged contribution: the current waypoint in the dynamics frame, so an external
        // autopilot can be fed the same target as the in-tree one
        static constexpr TI WAYPOINT_DIM = 3;
        struct ObservationPrivileged {
            static constexpr TI DIM = NEXT_WORLD::OBSERVATION_DIM_PRIVILEGED + WAYPOINT_DIM;
            using SHAPE = tensor::Shape<TI, DIM>;
        };
        static constexpr TI OBSERVATION_DIM_PRIVILEGED = ObservationPrivileged::DIM;

        using TRAJECTORY_SPEC = trajectory::Specification<T, TI>;
        struct Navigation {
            collision::Scene<collision::Specification<double, TI>> geometry;
            std::vector<std::array<T, 3>> positions;
            std::vector<std::vector<TI>> neighbors;
            std::vector<TI> spawn_nodes;
        };
        std::vector<Navigation> navigation;

        CONTROLLER controller;
        typename CONTROLLER::State controller_state;
        typename CONTROLLER::Buffer controller_buffer;
        std::string autopilot_policy_path = TASK_SPEC::AUTOPILOT_POLICY; // override before init
        TI task_step = 0;
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
