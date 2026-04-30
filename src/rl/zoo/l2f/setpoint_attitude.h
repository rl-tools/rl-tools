#include <rl_tools/version.h>
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ZOO_L2F_SETPOINT_ATTITUDE_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ZOO_L2F_SETPOINT_ATTITUDE_H

#include <rl_tools/rl/environments/l2f/multirotor.h>
#include <rl_tools/rl/environments/l2f/operations_generic/05_state_is_nan.h>
#include <rl_tools/rl/environments/l2f/operations_generic/10_sample_initial_parameters.h>
#include <rl_tools/rl/environments/l2f/operations_generic/20_initial_state.h>
#include <rl_tools/rl/environments/l2f/operations_generic/30_sample_initial_state.h>
#include <rl_tools/rl/environments/l2f/operations_generic/40_observe.h>
#include <rl_tools/rl/environments/l2f/operations_generic/70_post_integration.h>
#include <rl_tools/math/operations_generic.h>
#include <rl_tools/random/operations_generic.h>
#include <rl_tools/utils/generic/typing.h>

#include <string>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::environments::l2f{
    namespace parameters{
        template <typename T, typename TI>
        struct AttitudeSetpointSampling{
            T max_tilt_angle; // roll/pitch commands are sampled inside this tilt cone
            T max_yaw_rate;   // absolute body yaw-rate command limit
            T thrust_min_g;   // commanded collective acceleration along body z, in g
            T thrust_max_g;
            TI hold_steps_min; // command hold duration at the environment control rate
            TI hold_steps_max;
        };
    }

    template <typename T_T, typename T_TI, typename T_NEXT_COMPONENT>
    struct ParametersAttitudeSetpointSpecification{
        using T = T_T;
        using TI = T_TI;
        using NEXT_COMPONENT = T_NEXT_COMPONENT;
    };
    template <typename T_SPEC>
    struct ParametersAttitudeSetpoint: T_SPEC::NEXT_COMPONENT{
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        using NEXT_COMPONENT = typename SPEC::NEXT_COMPONENT;
        static constexpr TI N = NEXT_COMPONENT::N;
        using AttitudeSetpointSampling = parameters::AttitudeSetpointSampling<T, TI>;
        AttitudeSetpointSampling attitude_setpoint_sampling;
    };

    template <typename T_T, typename T_TI, typename T_NEXT_COMPONENT>
    struct StateAttitudeSetpointSpecification{
        using T = T_T;
        using TI = T_TI;
        using NEXT_COMPONENT = T_NEXT_COMPONENT;
    };
    template <typename T_SPEC>
    struct StateAttitudeSetpoint: T_SPEC::NEXT_COMPONENT{
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        using NEXT_COMPONENT = typename SPEC::NEXT_COMPONENT;
        static constexpr bool REQUIRES_INTEGRATION = false;
        static constexpr TI DIM = 4 + NEXT_COMPONENT::DIM;
        T target_roll; // policy-facing command
        T target_pitch;
        T target_yaw_rate;
        T target_thrust_g;
        TI target_steps_remaining;
    };

    namespace observation{
        template <typename T_T, typename T_TI, typename T_NEXT_COMPONENT = LastComponent<T_TI>>
        struct AttitudeSetpointSpecification{
            using T = T_T;
            using TI = T_TI;
            using NEXT_COMPONENT = T_NEXT_COMPONENT;
            static constexpr bool PRIVILEGED = false;
        };
        template <typename T_T, typename T_TI, typename T_NEXT_COMPONENT = LastComponent<T_TI>>
        struct AttitudeSetpointSpecificationPrivileged: AttitudeSetpointSpecification<T_T, T_TI, T_NEXT_COMPONENT>{
            static constexpr bool PRIVILEGED = true;
        };
        template <typename SPEC>
        struct AttitudeSetpoint{
            using T = typename SPEC::T;
            using TI = typename SPEC::TI;
            using NEXT_COMPONENT = typename SPEC::NEXT_COMPONENT;
            static constexpr bool PRIVILEGED = SPEC::PRIVILEGED;
            static constexpr TI CURRENT_DIM = 4;
            static constexpr TI DIM = NEXT_COMPONENT::DIM + CURRENT_DIM;
            using SHAPE = tensor::Shape<TI, DIM>;
        };
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template <typename DEVICE, typename SPEC, typename OBS_SPEC>
    std::string string(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, const rl::environments::l2f::observation::AttitudeSetpoint<OBS_SPEC>& obs, bool first = true){
        return std::string(first ? "" : ".") + "AttitudeSetpoint";
    }

    namespace rl::environments::l2f{
        template <typename DEVICE, typename T, typename TI, typename RNG>
        RL_TOOLS_FUNCTION_PLACEMENT void sample_attitude_setpoint(DEVICE& device, const parameters::AttitudeSetpointSampling<T, TI>& sampling, T& roll, T& pitch, T& yaw_rate, T& thrust_g, TI& steps_remaining, RNG& rng){
            T cos_tilt_min = math::cos(device.math, sampling.max_tilt_angle);
            T cos_tilt = random::uniform_real_distribution(device.random, cos_tilt_min, (T)1, rng);
            T sin_tilt = math::sqrt(device.math, (T)1 - cos_tilt*cos_tilt);
            T azimuth = random::uniform_real_distribution(device.random, (T)0, (T)2 * math::PI<T>, rng);
            T world_z_body_x = sin_tilt * math::cos(device.math, azimuth);
            T world_z_body_y = sin_tilt * math::sin(device.math, azimuth);
            roll = math::atan2(device.math, world_z_body_y, cos_tilt);
            pitch = math::atan2(device.math, -world_z_body_x, math::sqrt(device.math, world_z_body_y*world_z_body_y + cos_tilt*cos_tilt));
            yaw_rate = random::uniform_real_distribution(device.random, -sampling.max_yaw_rate, sampling.max_yaw_rate, rng);
            thrust_g = random::uniform_real_distribution(device.random, sampling.thrust_min_g, sampling.thrust_max_g, rng);
            steps_remaining = random::uniform_int_distribution(device.random, sampling.hold_steps_min, sampling.hold_steps_max, rng);
        }

        template <typename DEVICE, typename T>
        RL_TOOLS_FUNCTION_PLACEMENT void roll_pitch_to_world_z_body(DEVICE& device, T roll, T pitch, T world_z_body[3]){
            T sin_roll = math::sin(device.math, roll);
            T cos_roll = math::cos(device.math, roll);
            T sin_pitch = math::sin(device.math, pitch);
            T cos_pitch = math::cos(device.math, pitch);
            world_z_body[0] = -sin_pitch;
            world_z_body[1] = cos_pitch * sin_roll;
            world_z_body[2] = cos_pitch * cos_roll;
        }

        template<typename DEVICE, typename SPEC, typename PARAMETER_SPEC, typename RNG>
        RL_TOOLS_FUNCTION_PLACEMENT static void _sample_initial_parameters(DEVICE& device, Multirotor<SPEC>& env, ParametersAttitudeSetpoint<PARAMETER_SPEC>& parameters, RNG& rng){
            sample_initial_parameters(device, env, static_cast<typename PARAMETER_SPEC::NEXT_COMPONENT&>(parameters), rng);
            parameters.attitude_setpoint_sampling = env.parameters.attitude_setpoint_sampling;
        }

        template<typename DEVICE, typename SPEC, typename PARAMETERS, typename STATE_SPEC>
        RL_TOOLS_FUNCTION_PLACEMENT static void _initial_state(DEVICE& device, Multirotor<SPEC>& env, PARAMETERS& parameters, StateAttitudeSetpoint<STATE_SPEC>& state){
            initial_state(device, env, parameters, static_cast<typename STATE_SPEC::NEXT_COMPONENT&>(state));
            state.target_roll = 0;
            state.target_pitch = 0;
            state.target_yaw_rate = 0;
            state.target_thrust_g = 1;
            state.target_steps_remaining = parameters.attitude_setpoint_sampling.hold_steps_min;
        }

        template<typename DEVICE, typename SPEC, typename PARAMETERS, typename STATE_SPEC, typename RNG>
        RL_TOOLS_FUNCTION_PLACEMENT static void _sample_initial_state(DEVICE& device, Multirotor<SPEC>& env, PARAMETERS& parameters, StateAttitudeSetpoint<STATE_SPEC>& state, RNG& rng){
            sample_initial_state(device, env, parameters, static_cast<typename STATE_SPEC::NEXT_COMPONENT&>(state), rng);
            sample_attitude_setpoint(device, parameters.attitude_setpoint_sampling, state.target_roll, state.target_pitch, state.target_yaw_rate, state.target_thrust_g, state.target_steps_remaining, rng);
        }

        template<typename DEVICE, typename STATE_SPEC>
        RL_TOOLS_FUNCTION_PLACEMENT static bool _is_nan(DEVICE& device, StateAttitudeSetpoint<STATE_SPEC>& state){
            bool nan = _is_nan(device, static_cast<typename STATE_SPEC::NEXT_COMPONENT&>(state));
            nan = nan || math::is_nan(device.math, state.target_roll);
            nan = nan || math::is_nan(device.math, state.target_pitch);
            nan = nan || math::is_nan(device.math, state.target_yaw_rate);
            nan = nan || math::is_nan(device.math, state.target_thrust_g);
            return nan;
        }

        template<typename DEVICE, typename SPEC, typename PARAMETERS, typename STATE_SPEC, typename ACTION_SPEC, typename RNG>
        RL_TOOLS_FUNCTION_PLACEMENT void post_integration(DEVICE& device, const Multirotor<SPEC>& env, PARAMETERS& parameters, const StateAttitudeSetpoint<STATE_SPEC>& state, const Matrix<ACTION_SPEC>& action, StateAttitudeSetpoint<STATE_SPEC>& next_state, RNG& rng){
            post_integration(device, env, parameters, static_cast<const typename STATE_SPEC::NEXT_COMPONENT&>(state), action, static_cast<typename STATE_SPEC::NEXT_COMPONENT&>(next_state), rng);
            if(state.target_steps_remaining <= 1){
                sample_attitude_setpoint(device, parameters.attitude_setpoint_sampling, next_state.target_roll, next_state.target_pitch, next_state.target_yaw_rate, next_state.target_thrust_g, next_state.target_steps_remaining, rng);
            }
            else{
                next_state.target_roll = state.target_roll;
                next_state.target_pitch = state.target_pitch;
                next_state.target_yaw_rate = state.target_yaw_rate;
                next_state.target_thrust_g = state.target_thrust_g;
                next_state.target_steps_remaining = state.target_steps_remaining - 1;
            }
        }

        template<typename DEVICE, typename SPEC, typename PARAMETERS, typename STATE, typename OBSERVATION_SPEC, typename OBS_SPEC, typename RNG>
        RL_TOOLS_FUNCTION_PLACEMENT static void _observe(DEVICE& device, const Multirotor<SPEC>& env, PARAMETERS& parameters, const STATE& state, observation::AttitudeSetpoint<OBSERVATION_SPEC>, Matrix<OBS_SPEC>& observation, RNG& rng){
            using OBSERVATION = observation::AttitudeSetpoint<OBSERVATION_SPEC>;
            static_assert(OBS_SPEC::COLS >= OBSERVATION::CURRENT_DIM);
            static_assert(OBS_SPEC::ROWS == 1);
            set(observation, 0, 0, state.target_roll);
            set(observation, 0, 1, state.target_pitch);
            set(observation, 0, 2, state.target_yaw_rate);
            set(observation, 0, 3, state.target_thrust_g);
            auto next_observation = view(device, observation, matrix::ViewSpec<1, OBS_SPEC::COLS - OBSERVATION::CURRENT_DIM>{}, 0, OBSERVATION::CURRENT_DIM);
            observe(device, env, parameters, state, typename OBSERVATION::NEXT_COMPONENT{}, next_observation, rng);
        }
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
