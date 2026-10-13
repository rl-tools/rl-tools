#pragma once

#include <rl_tools/rl/environments/l2f/operations_generic.h>
#include <rl_tools/rendering/camera.h>

#include "simulator_matrix_physics.h"

#include <cmath>

namespace rl_tools::rendering::raytracing::benchmark::paper {
    using T = float;
    using TI = int;
    namespace l2f = rl::environments::l2f;
    using REWARD_FUNCTION = l2f::parameters::reward_functions::Squared<T>;
    constexpr TI EPISODE_STEP_LIMIT = 10;
    using PARAMETERS_SPEC = l2f::ParametersBaseSpecification<T, TI, 4, EPISODE_STEP_LIMIT, REWARD_FUNCTION>;
    using PARAMETERS = l2f::ParametersDisturbances<l2f::ParametersSpecification<T, TI, l2f::ParametersBase<PARAMETERS_SPEC>>>;

    // Freeze the May 25/28 workload even if the Crazyflie registry or simulator defaults change.
    constexpr PARAMETERS::Dynamics DYNAMICS = {
        {{0.028, -0.028, 0}, {-0.028, -0.028, 0}, {-0.028, 0.028, 0}, {0.028, 0.028, 0}},
        {{0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}},
        {{0, 0, -1}, {0, 0, 1}, {0, 0, -1}, {0, 0, 1}},
        {{0.00352526, 0.01437313, 0.09223048}, {0.00352526, 0.01437313, 0.09223048},
         {0.00352526, 0.01437313, 0.09223048}, {0.00352526, 0.01437313, 0.09223048}},
        {4.665e-3, 4.665e-3, 4.665e-3, 4.665e-3},
        {0.05545454545454546, 0.05545454545454546, 0.05545454545454546, 0.05545454545454546},
        {0.24939393939393945, 0.24939393939393945, 0.24939393939393945, 0.24939393939393945},
        0.027 + 0.0017 + 0.0003 + 0.0016,
        {0, 0, -9.81},
        {{9.416556729130406e-06, 0, 0}, {0, 9.644051701582312e-06, 0}, {0, 0, 1.745951732253285e-05}},
        {{106195.93007988465, 0, 0}, {0, 103690.85846314249, 0}, {0, 0, 57275.35197719487}},
        0.7261389721508553,
        {0, 1},
        10000 // max_rpm: visual convention; the frozen physics never reads it
    };
    constexpr PARAMETERS NOMINAL_PARAMETERS = {
        {DYNAMICS, {T(1) / T(100)},
         {{0, 0, 0, 0, 0, true, -1, 1},
          {false, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, {0, 0, 0, 0}, {0, 0, 0, 0}, 0},
          {}, {}, {false, 100000, 0, 100000, 100000, 100000, 100000}}},
        {{0, 0}, {0, 0}}
    };

    struct StaticParameters {
        static constexpr auto ACTION_INTERFACE = l2f::parameters::ActionInterface::DIRECT_MOTOR;
        static constexpr TI N_SUBSTEPS = 1;
        static constexpr bool CLOSED_FORM = false;
        static constexpr TI EPISODE_STEP_LIMIT = paper::EPISODE_STEP_LIMIT;
        using STATE_BASE = l2f::StateBase<l2f::StateSpecification<T, TI>>;
        using STATE_TYPE = l2f::StateRotors<l2f::StateRotorsSpecification<T, TI, CLOSED_FORM, l2f::StateRandomForce<l2f::StateSpecification<T, TI, STATE_BASE>>>>;
        using OBSERVATION_TYPE = l2f::observation::NONE<TI>;
        using OBSERVATION_TYPE_PRIVILEGED = l2f::observation::NONE<TI>;
        static constexpr bool PRIVILEGED_OBSERVATION_NOISE = false;
        using PARAMETERS = paper::PARAMETERS;
        static constexpr PARAMETERS PARAMETER_VALUES = NOMINAL_PARAMETERS;
        static constexpr T STATE_LIMIT_POSITION_X = 100000;
        static constexpr T STATE_LIMIT_POSITION_Y = 100000;
        static constexpr T STATE_LIMIT_POSITION_Z = 100000;
        static constexpr T STATE_LIMIT_VELOCITY_X = 100000;
        static constexpr T STATE_LIMIT_VELOCITY_Y = 100000;
        static constexpr T STATE_LIMIT_VELOCITY_Z = 100000;
        static constexpr T STATE_LIMIT_ANGULAR_VELOCITY_X = 100000;
        static constexpr T STATE_LIMIT_ANGULAR_VELOCITY_Y = 100000;
        static constexpr T STATE_LIMIT_ANGULAR_VELOCITY_Z = 100000;
    };

    using ENVIRONMENT = rl::environments::Multirotor<l2f::Specification<T, TI, StaticParameters>>;
    using STATE = ENVIRONMENT::State;
}

namespace rl_tools {
    template <typename DEVICE>
    rendering::raytracing::benchmark::paper::STATE paper_benchmark_initial_state(DEVICE&, const float position[3], const rendering::raytracing::benchmark::PhysicsCameraPose& pose) {
        using namespace rendering::raytracing::benchmark::paper;
        T x[3] = {pose.direction[0], pose.direction[1], pose.direction[2]};
        T z[3] = {pose.up[0], pose.up[1], pose.up[2]};
        auto normalize = [](T vector[3]) {
            const T inverse_norm = T(1) / std::sqrt(vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]);
            for(TI axis = 0; axis < 3; axis++) {
                vector[axis] *= inverse_norm;
            }
        };
        normalize(x);
        normalize(z);
        T y[3];
        rendering::vec3::cross(z, x, y);
        normalize(y);
        rendering::vec3::cross(x, y, z);
        normalize(z);

        STATE state{};
        T* q = state.orientation;
        const T trace = x[0] + y[1] + z[2];
        if(trace > 0) {
            const T s = std::sqrt(trace + T(1)) * T(2);
            q[0] = T(0.25) * s;
            q[1] = (y[2] - z[1]) / s;
            q[2] = (z[0] - x[2]) / s;
            q[3] = (x[1] - y[0]) / s;
        }
        else if(x[0] > y[1] && x[0] > z[2]) {
            const T s = std::sqrt(T(1) + x[0] - y[1] - z[2]) * T(2);
            q[0] = (y[2] - z[1]) / s;
            q[1] = T(0.25) * s;
            q[2] = (y[0] + x[1]) / s;
            q[3] = (z[0] + x[2]) / s;
        }
        else if(y[1] > z[2]) {
            const T s = std::sqrt(T(1) + y[1] - x[0] - z[2]) * T(2);
            q[0] = (z[0] - x[2]) / s;
            q[1] = (y[0] + x[1]) / s;
            q[2] = T(0.25) * s;
            q[3] = (z[1] + y[2]) / s;
        }
        else {
            const T s = std::sqrt(T(1) + z[2] - x[0] - y[1]) * T(2);
            q[0] = (x[1] - y[0]) / s;
            q[1] = (z[0] + x[2]) / s;
            q[2] = (z[1] + y[2]) / s;
            q[3] = T(0.25) * s;
        }
        const T norm = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
        for(TI component = 0; component < 4; component++) {
            q[component] /= norm;
            state.rpm[component] = DYNAMICS.hovering_throttle_relative * (DYNAMICS.action_limit.max - DYNAMICS.action_limit.min) + DYNAMICS.action_limit.min;
        }
        for(TI axis = 0; axis < 3; axis++) {
            state.position[axis] = position[axis];
        }
        return state;
    }

    template <typename DEVICE, typename ACTION_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void paper_benchmark_action(DEVICE& device, const rendering::raytracing::benchmark::paper::PARAMETERS& parameters, int env_i, int iteration, Matrix<ACTION_SPEC>& action) {
        using T = rendering::raytracing::benchmark::paper::T;
        const T hover = parameters.dynamics.hovering_throttle_relative * T(2) - T(1);
        const T phase_a = T(0.013) * T(iteration) + T(0.0017) * T(env_i);
        const T phase_b = T(0.009) * T(iteration) + T(0.0023) * T(env_i);
        const T roll_mix = T(0.010) * math::sin(device.math, phase_a);
        const T pitch_mix = T(0.010) * math::cos(device.math, phase_b);
        set(action, 0, 0, math::clamp(device.math, hover + roll_mix, T(-1), T(1)));
        set(action, 0, 1, math::clamp(device.math, hover - roll_mix, T(-1), T(1)));
        set(action, 0, 2, math::clamp(device.math, hover + pitch_mix, T(-1), T(1)));
        set(action, 0, 3, math::clamp(device.math, hover - pitch_mix, T(-1), T(1)));
    }

    template <typename DEVICE, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void paper_benchmark_step(DEVICE& device, rendering::raytracing::benchmark::paper::ENVIRONMENT& environment, rendering::raytracing::benchmark::paper::PARAMETERS& parameters, rendering::raytracing::benchmark::paper::STATE& state, const rendering::raytracing::benchmark::paper::STATE& initial_state, int env_i, int iteration, RNG& rng) {
        using namespace rendering::raytracing::benchmark::paper;
        if(iteration > 0 && iteration % EPISODE_STEP_LIMIT == 0) {
            state = initial_state;
        }
        Matrix<matrix::Specification<T, TI, 1, 4, false>> action;
        paper_benchmark_action(device, parameters, env_i, iteration, action);
        STATE next_state;
        step(device, environment, parameters, state, action, next_state, rng);
        const T dx = next_state.position[0] - initial_state.position[0];
        const T dy = next_state.position[1] - initial_state.position[1];
        const T dz = next_state.position[2] - initial_state.position[2];
        if(dx * dx + dy * dy + dz * dz > T(16) || math::abs(device.math, dz) > T(3)) {
            next_state = initial_state;
        }
        state = next_state;
    }

    template <typename DEVICE>
    RL_TOOLS_FUNCTION_PLACEMENT rendering::Camera<float> paper_benchmark_camera(DEVICE&, const rendering::raytracing::benchmark::paper::STATE& state, float fov_degrees, float aspect) {
        using namespace rl::environments::l2f;
        const float forward_body[3] = {1, 0, 0};
        const float up_body[3] = {0, 0, 1};
        float forward[3], up[3], look_at[3];
        rotate_vector_by_quaternion<DEVICE, float>(state.orientation, forward_body, forward);
        rotate_vector_by_quaternion<DEVICE, float>(state.orientation, up_body, up);
        for(int axis = 0; axis < 3; axis++) {
            look_at[axis] = state.position[axis] + forward[axis];
        }
        return make_camera_data(state.position, look_at, up, fov_degrees, aspect);
    }
}
