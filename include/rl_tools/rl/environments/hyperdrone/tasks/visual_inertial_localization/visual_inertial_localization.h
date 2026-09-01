#include "../../../../../version.h"
#include "../../../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_TASKS_VISUAL_INERTIAL_LOCALIZATION_VISUAL_INERTIAL_LOCALIZATION_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_TASKS_VISUAL_INERTIAL_LOCALIZATION_VISUAL_INERTIAL_LOCALIZATION_H

#include "../../world.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::environments::hyperdrone::tasks::visual_inertial_localization {

    template <typename T_T, typename T_TI, T_TI T_NUM_WAYPOINTS, typename T_NEXT_COMPONENT>
    struct ParametersSpecification {
        using T = T_T;
        using TI = T_TI;
        static constexpr TI NUM_WAYPOINTS = T_NUM_WAYPOINTS;
        using NEXT_COMPONENT = T_NEXT_COMPONENT;
    };
    // per-episode waypoint route in the dynamics frame (waypoint 0 = origin, the initial free-space anchor)
    template <typename T_SPEC>
    struct ParametersVisualInertialLocalization: T_SPEC::NEXT_COMPONENT {
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        using NEXT_COMPONENT = typename SPEC::NEXT_COMPONENT;
        static constexpr TI NUM_WAYPOINTS = SPEC::NUM_WAYPOINTS;
        T waypoints[NUM_WAYPOINTS][3] = {};
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
        static constexpr TI DIM = 1 + NEXT_COMPONENT::DIM;
        TI current_waypoint = 0;
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
        static constexpr TI NUM_WAYPOINTS = 8;
        static constexpr T WAYPOINT_MIN_DISTANCE = 1.0;
        static constexpr T WAYPOINT_ACCEPTANCE_RADIUS = 0.35;
        static constexpr T TARGET_POSITION_ERROR_CLIP = 1.0; // autopilot position-error observation clamp
        // hover at waypoint 0 (the origin) for this many env steps before the route starts:
        // gives estimators a stationary window, and the launch toward waypoint 1 provides the
        // acceleration jerk that static visual-inertial initializers wait for
        static constexpr TI INITIALIZATION_HOLD_STEPS = 0;
    };

    // benchmark protocol: fixed-length synchronized episodes only — the camera stride phase is
    // global, so the reset mask must be all-or-none (asserted in render)
    template <typename T_TASK_SPEC>
    struct World: T_TASK_SPEC::NEXT_WORLD {
        using TASK_SPEC = T_TASK_SPEC;
        using NEXT_WORLD = typename TASK_SPEC::NEXT_WORLD;
        using T = typename NEXT_WORLD::T;
        using TI = typename NEXT_WORLD::TI;
        static_assert(HasIMUMeasurement<typename NEXT_WORLD::State>::VALUE, "the visual-inertial localization task needs l2f::StateIMU in the dynamics state chain (e.g. presets::X500FPVIMU)");
        static_assert(NEXT_WORLD::SPEC::N_AGENTS == 1, "the visual-inertial localization task is single-agent");

        using Parameters = ParametersVisualInertialLocalization<ParametersSpecification<T, TI, TASK_SPEC::NUM_WAYPOINTS, typename NEXT_WORLD::Parameters>>;
        using State = StateVisualInertialLocalization<ComponentSpecification<T, TI, typename NEXT_WORLD::State>>;
        static constexpr TI FRAME_STRIDE = TASK_SPEC::FRAME_STRIDE;
        static constexpr TI INITIALIZATION_HOLD_STEPS = TASK_SPEC::INITIALIZATION_HOLD_STEPS;

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

        TI task_step = 0; // env-step counter driving the camera stride phase (history_step only counts rendered frames)
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
