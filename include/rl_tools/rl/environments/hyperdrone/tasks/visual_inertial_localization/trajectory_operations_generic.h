#include "../../../../../version.h"
#include "../../../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_TASKS_VISUAL_INERTIAL_LOCALIZATION_TRAJECTORY_OPERATIONS_GENERIC_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_TASKS_VISUAL_INERTIAL_LOCALIZATION_TRAJECTORY_OPERATIONS_GENERIC_H

#include "trajectory.h"
#include "../../../../../utils/generic/typing.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {
    // Volatile byte reads preserve validation of both inputs and arithmetic overflow under fast-math.
    template <typename DEVICE, typename T>
    RL_TOOLS_FUNCTION_PLACEMENT bool finite_trajectory_value(DEVICE&, const T& value) {
        static_assert(sizeof(T) == 4 || sizeof(T) == 8);
        using BITS = utils::typing::conditional_t<sizeof(T) == 4, unsigned int, unsigned long long>;
        static_assert(sizeof(BITS) == sizeof(T));
        BITS bits;
        auto* destination = reinterpret_cast<unsigned char*>(&bits);
        const auto* source = reinterpret_cast<const volatile unsigned char*>(&value);
        for (unsigned byte = 0; byte < sizeof(T); byte++) {
            destination[byte] = source[byte];
        }
        constexpr BITS EXPONENT = sizeof(T) == 4 ? (BITS)0x7f800000u : (BITS)0x7ff0000000000000ull;
        return (bits & EXPONENT) != EXPONENT;
    }

    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT bool valid(DEVICE& device, const rl::environments::hyperdrone::tasks::visual_inertial_localization::trajectory::Segment<SPEC>& segment) {
        if (!finite_trajectory_value(device, segment.duration) || !(segment.duration > 0)) {
            return false;
        }
        for (typename SPEC::TI point = 0; point < 6; point++) {
            for (typename SPEC::TI axis = 0; axis < 3; axis++) {
                if (!finite_trajectory_value(device, segment.control[point][axis])) {
                    return false;
                }
            }
        }
        return true;
    }

    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT bool init(DEVICE& device, rl::environments::hyperdrone::tasks::visual_inertial_localization::trajectory::Segment<SPEC>& segment, const typename SPEC::T from[3], const typename SPEC::T to[3], const typename SPEC::T start_velocity[3], const typename SPEC::T end_velocity[3], const typename SPEC::T& duration) {
        using T = typename SPEC::T;
        segment = {};
        if (!finite_trajectory_value(device, duration) || !(duration > 0)) {
            return false;
        }
        rl::environments::hyperdrone::tasks::visual_inertial_localization::trajectory::Segment<SPEC> candidate;
        candidate.duration = duration;
        for (typename SPEC::TI axis = 0; axis < 3; axis++) {
            if (!finite_trajectory_value(device, from[axis]) || !finite_trajectory_value(device, to[axis]) || !finite_trajectory_value(device, start_velocity[axis]) || !finite_trajectory_value(device, end_velocity[axis])) {
                return false;
            }
            candidate.control[0][axis] = from[axis];
            candidate.control[1][axis] = from[axis] + start_velocity[axis] * (duration / (T)5);
            candidate.control[2][axis] = from[axis] + start_velocity[axis] * ((T)2 * duration / (T)5);
            candidate.control[3][axis] = to[axis] - end_velocity[axis] * ((T)2 * duration / (T)5);
            candidate.control[4][axis] = to[axis] - end_velocity[axis] * (duration / (T)5);
            candidate.control[5][axis] = to[axis];
        }
        if (!valid(device, candidate)) {
            return false;
        }
        segment = candidate;
        return true;
    }

    template <typename DEVICE, typename T, typename TI, TI N>
    RL_TOOLS_FUNCTION_PLACEMENT void evaluate_trajectory_polynomial(DEVICE&, const T control[N][3], T fraction, T result[3]) {
        T work[N][3];
        for (TI point = 0; point < N; point++) {
            for (TI axis = 0; axis < 3; axis++) {
                work[point][axis] = control[point][axis];
            }
        }
        for (TI remaining = N - 1; remaining > 0; remaining--) {
            for (TI point = 0; point < remaining; point++) {
                for (TI axis = 0; axis < 3; axis++) {
                    work[point][axis] = ((T)1 - fraction) * work[point][axis] + fraction * work[point + 1][axis];
                }
            }
        }
        for (TI axis = 0; axis < 3; axis++) {
            result[axis] = work[0][axis];
        }
    }

    template <typename DEVICE, typename T, typename TI, TI N>
    RL_TOOLS_FUNCTION_PLACEMENT void trajectory_derivative(DEVICE&, const T control[N][3], T duration, T derivative[N - 1][3]) {
        const T scale = (T)(N - 1) / duration;
        for (TI point = 0; point + 1 < N; point++) {
            for (TI axis = 0; axis < 3; axis++) {
                derivative[point][axis] = (control[point + 1][axis] - control[point][axis]) * scale;
            }
        }
    }

    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT bool evaluate(DEVICE& device, const rl::environments::hyperdrone::tasks::visual_inertial_localization::trajectory::Segment<SPEC>& segment, const typename SPEC::T& elapsed, rl::environments::hyperdrone::tasks::visual_inertial_localization::trajectory::Sample<SPEC>& sample) {
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        sample = {};
        if (!valid(device, segment) || !finite_trajectory_value(device, elapsed)) {
            return false;
        }
        const T inverse_duration = (T)1 / segment.duration;
        const T inverse_duration_cubed = inverse_duration * inverse_duration * inverse_duration;
        if (!finite_trajectory_value(device, inverse_duration_cubed)) {
            return false;
        }
        const T fraction = math::clamp(device.math, elapsed / segment.duration, (T)0, (T)1);
        T velocity[5][3];
        T acceleration[4][3];
        T jerk[3][3];
        trajectory_derivative<DEVICE, T, TI, 6>(device, segment.control, segment.duration, velocity);
        trajectory_derivative<DEVICE, T, TI, 5>(device, velocity, segment.duration, acceleration);
        trajectory_derivative<DEVICE, T, TI, 4>(device, acceleration, segment.duration, jerk);
        evaluate_trajectory_polynomial<DEVICE, T, TI, 6>(device, segment.control, fraction, sample.position);
        evaluate_trajectory_polynomial<DEVICE, T, TI, 5>(device, velocity, fraction, sample.velocity);
        evaluate_trajectory_polynomial<DEVICE, T, TI, 4>(device, acceleration, fraction, sample.acceleration);
        evaluate_trajectory_polynomial<DEVICE, T, TI, 3>(device, jerk, fraction, sample.jerk);
        for (TI axis = 0; axis < 3; axis++) {
            if (!finite_trajectory_value(device, sample.position[axis]) || !finite_trajectory_value(device, sample.velocity[axis]) || !finite_trajectory_value(device, sample.acceleration[axis]) || !finite_trajectory_value(device, sample.jerk[axis])) {
                sample = {};
                return false;
            }
        }
        return true;
    }

    template <typename DEVICE, typename T, typename TI, TI N>
    RL_TOOLS_FUNCTION_PLACEMENT T trajectory_polynomial_bound(DEVICE& device, const T control[N][3], TI subdivisions) {
        if (subdivisions == 0) {
            T maximum = 0;
            for (TI point = 0; point < N; point++) {
                T squared_norm = 0;
                for (TI axis = 0; axis < 3; axis++) {
                    squared_norm += control[point][axis] * control[point][axis];
                }
                maximum = math::max(device.math, maximum, squared_norm);
            }
            return math::sqrt(device.math, maximum);
        }
        T work[N][3];
        T left[N][3];
        T right[N][3];
        for (TI point = 0; point < N; point++) {
            for (TI axis = 0; axis < 3; axis++) {
                work[point][axis] = control[point][axis];
            }
        }
        for (TI level = 0; level < N; level++) {
            for (TI axis = 0; axis < 3; axis++) {
                left[level][axis] = work[0][axis];
                right[N - 1 - level][axis] = work[N - 1 - level][axis];
            }
            for (TI point = 0; point + level + 1 < N; point++) {
                for (TI axis = 0; axis < 3; axis++) {
                    work[point][axis] = (T)0.5 * work[point][axis] + (T)0.5 * work[point + 1][axis];
                }
            }
        }
        return math::max(device.math, trajectory_polynomial_bound<DEVICE, T, TI, N>(device, left, subdivisions - 1), trajectory_polynomial_bound<DEVICE, T, TI, N>(device, right, subdivisions - 1));
    }

    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT bool derivative_bounds(DEVICE& device, const rl::environments::hyperdrone::tasks::visual_inertial_localization::trajectory::Segment<SPEC>& segment, rl::environments::hyperdrone::tasks::visual_inertial_localization::trajectory::Bounds<SPEC>& bounds) {
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        bounds = {};
        if (!valid(device, segment)) {
            return false;
        }
        T velocity[5][3];
        T acceleration[4][3];
        T jerk[3][3];
        trajectory_derivative<DEVICE, T, TI, 6>(device, segment.control, segment.duration, velocity);
        trajectory_derivative<DEVICE, T, TI, 5>(device, velocity, segment.duration, acceleration);
        trajectory_derivative<DEVICE, T, TI, 4>(device, acceleration, segment.duration, jerk);
        for (TI axis = 0; axis < 3; axis++) {
            for (TI point = 0; point < 5; point++) {
                if (!finite_trajectory_value(device, velocity[point][axis]) || (point < 4 && !finite_trajectory_value(device, acceleration[point][axis])) || (point < 3 && !finite_trajectory_value(device, jerk[point][axis]))) {
                    return false;
                }
            }
        }
        // Every derivative lies in its Bezier control hull; subdivision tightens that certificate without sampling gaps.
        bounds.speed = trajectory_polynomial_bound<DEVICE, T, TI, 5>(device, velocity, (TI)4);
        bounds.acceleration = trajectory_polynomial_bound<DEVICE, T, TI, 4>(device, acceleration, (TI)4);
        bounds.jerk = trajectory_polynomial_bound<DEVICE, T, TI, 3>(device, jerk, (TI)4);
        if (!finite_trajectory_value(device, bounds.speed) || !finite_trajectory_value(device, bounds.acceleration) || !finite_trajectory_value(device, bounds.jerk)) {
            bounds = {};
            return false;
        }
        return true;
    }

    template <typename DEVICE, typename T>
    RL_TOOLS_FUNCTION_PLACEMENT T estimate_duration(DEVICE& device, const T from[3], const T to[3], const T& max_speed, const T& max_acceleration, const T& max_jerk) {
        if (!finite_trajectory_value(device, max_speed) || !finite_trajectory_value(device, max_acceleration) || !finite_trajectory_value(device, max_jerk) || !(max_speed > 0) || !(max_acceleration > 0) || !(max_jerk > 0)) {
            return 0;
        }
        T squared_distance = 0;
        for (typename DEVICE::index_t axis = 0; axis < 3; axis++) {
            if (!finite_trajectory_value(device, from[axis]) || !finite_trajectory_value(device, to[axis])) {
                return 0;
            }
            const T delta = to[axis] - from[axis];
            squared_distance += delta * delta;
        }
        const T distance = math::sqrt(device.math, squared_distance);
        const T duration = math::max(device.math, (T)1.875 * distance / max_speed, math::max(device.math, math::sqrt(device.math, (T)6 * distance / max_acceleration), math::cbrt(device.math, (T)60 * distance / max_jerk)));
        return finite_trajectory_value(device, duration) ? duration : (T)0;
    }

    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void trajectory_phase_extents(DEVICE& device, const rl::environments::hyperdrone::tasks::visual_inertial_localization::trajectory::Phase<SPEC>& phase, typename SPEC::T jerk, typename SPEC::T dt, typename SPEC::T max_jerk, typename SPEC::T& lower, typename SPEC::T& upper) {
        using T = typename SPEC::T;
        const T next_rate = phase.rate + phase.acceleration * dt + (T)0.5 * jerk * dt * dt;
        const T next_acceleration = phase.acceleration + jerk * dt;
        const T stopping_distance = next_acceleration * next_acceleration / ((T)2 * max_jerk);
        lower = next_rate - (next_acceleration < 0 ? stopping_distance : (T)0);
        upper = next_rate + (next_acceleration > 0 ? stopping_distance : (T)0);
        if ((phase.acceleration > 0 && next_acceleration < 0) || (phase.acceleration < 0 && next_acceleration > 0)) {
            const T extremum = phase.rate - phase.acceleration * phase.acceleration / ((T)2 * jerk);
            lower = math::min(device.math, lower, extremum);
            upper = math::max(device.math, upper, extremum);
        }
    }

    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT typename SPEC::T advance(DEVICE& device, rl::environments::hyperdrone::tasks::visual_inertial_localization::trajectory::Phase<SPEC>& phase, const typename SPEC::T& desired_rate, const typename SPEC::T& dt, const typename SPEC::T& max_acceleration, const typename SPEC::T& max_jerk) {
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        if (!finite_trajectory_value(device, phase.rate) || !finite_trajectory_value(device, phase.acceleration) || !finite_trajectory_value(device, desired_rate) || !finite_trajectory_value(device, dt) || !finite_trajectory_value(device, max_acceleration) || !finite_trajectory_value(device, max_jerk) || !(dt > 0) || !(max_acceleration > 0) || !(max_jerk > 0) || phase.rate < 0 || phase.rate > 1 || math::abs(device.math, phase.acceleration) > max_acceleration) {
            return 0;
        }
        T lower_jerk = math::max(device.math, -max_jerk, (-max_acceleration - phase.acceleration) / dt);
        T upper_jerk = math::min(device.math, max_jerk, (max_acceleration - phase.acceleration) / dt);
        T lower, upper;
        trajectory_phase_extents(device, phase, lower_jerk, dt, max_jerk, lower, upper);
        if (lower < 0) {
            T invalid = lower_jerk, admissible = upper_jerk;
            trajectory_phase_extents(device, phase, admissible, dt, max_jerk, lower, upper);
            if (lower < 0) {
                return 0;
            }
            for (TI iteration = 0; iteration < 32; iteration++) {
                const T middle = (T)0.5 * invalid + (T)0.5 * admissible;
                trajectory_phase_extents(device, phase, middle, dt, max_jerk, lower, upper);
                if (lower < 0) {
                    invalid = middle;
                } else {
                    admissible = middle;
                }
            }
            lower_jerk = admissible;
        }
        trajectory_phase_extents(device, phase, upper_jerk, dt, max_jerk, lower, upper);
        if (upper > 1) {
            T admissible = lower_jerk, invalid = upper_jerk;
            trajectory_phase_extents(device, phase, admissible, dt, max_jerk, lower, upper);
            if (upper > 1) {
                return 0;
            }
            for (TI iteration = 0; iteration < 32; iteration++) {
                const T middle = (T)0.5 * admissible + (T)0.5 * invalid;
                trajectory_phase_extents(device, phase, middle, dt, max_jerk, lower, upper);
                if (upper > 1) {
                    invalid = middle;
                } else {
                    admissible = middle;
                }
            }
            upper_jerk = admissible;
        }
        const T target = math::clamp(device.math, desired_rate, (T)0, (T)1);
        const T requested_jerk = (T)16 * (target - phase.rate) - (T)8 * phase.acceleration;
        const T jerk = math::clamp(device.math, requested_jerk, lower_jerk, upper_jerk);
        const T increment = phase.rate * dt + (T)0.5 * phase.acceleration * dt * dt + jerk * dt * dt * dt / (T)6;
        const T next_rate = phase.rate + phase.acceleration * dt + (T)0.5 * jerk * dt * dt;
        const T next_acceleration = phase.acceleration + jerk * dt;
        if (!finite_trajectory_value(device, increment) || !finite_trajectory_value(device, next_rate) || !finite_trajectory_value(device, next_acceleration)) {
            return 0;
        }
        phase.rate = math::clamp(device.math, next_rate, (T)0, (T)1);
        phase.acceleration = math::clamp(device.math, next_acceleration, -max_acceleration, max_acceleration);
        return math::clamp(device.math, increment, (T)0, dt);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
