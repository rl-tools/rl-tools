#include "../../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_L2F_OPERATIONS_GENERIC_POST_INTEGRATION_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_L2F_OPERATIONS_GENERIC_POST_INTEGRATION_H

#include "../multirotor.h"

#include <rl_tools/utils/generic/vector_operations.h>
#include "../quaternion_helper.h"

#include <rl_tools/utils/generic/typing.h>

#include <rl_tools/rl/environments/operations_generic.h>

// This file contains functions for parts of the state that do not evolve through integration but through discrete steps or for refinement operations after integration (e.g. normalizing the quaternion)

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::environments::l2f{
    namespace detail{
        template<typename DEVICE, typename T>
        RL_TOOLS_FUNCTION_PLACEMENT void project_to_tangent(DEVICE&, const T z[3], T v[3]) {
            T dot = v[0]*z[0] + v[1]*z[1] + v[2]*z[2];
            for(typename DEVICE::index_t i = 0; i < 3; i++){
                v[i] -= dot * z[i];
            }
        }
    }
    template<typename DEVICE, typename SPEC, typename PARAMETERS, typename STATE_SPEC, typename ACTION_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void post_integration(DEVICE& device, const Multirotor<SPEC>& env, PARAMETERS& parameters, const StateBase<STATE_SPEC>& state, const Matrix<ACTION_SPEC>& action, StateBase<STATE_SPEC>& next_state, RNG& rng) {
        using T = typename STATE_SPEC::T;
        using TI = typename DEVICE::index_t;
        T quaternion_norm = 0;
        for(TI state_i = 0; state_i < 4; state_i++){
            quaternion_norm += next_state.orientation[state_i] * next_state.orientation[state_i];
        }
        quaternion_norm = math::sqrt(device.math, quaternion_norm);
        for(TI state_i = 0; state_i < 4; state_i++){
            next_state.orientation[state_i] /= quaternion_norm;
        }
        for(TI dim_i=0; dim_i < 3; dim_i++){
            using STATIC_PARAMETERS = typename SPEC::STATIC_PARAMETERS;
            T position_limit = dim_i == 0 ? STATIC_PARAMETERS::STATE_LIMIT_POSITION_X : (dim_i == 1 ? STATIC_PARAMETERS::STATE_LIMIT_POSITION_Y : STATIC_PARAMETERS::STATE_LIMIT_POSITION_Z);
            T velocity_limit = dim_i == 0 ? STATIC_PARAMETERS::STATE_LIMIT_VELOCITY_X : (dim_i == 1 ? STATIC_PARAMETERS::STATE_LIMIT_VELOCITY_Y : STATIC_PARAMETERS::STATE_LIMIT_VELOCITY_Z);
            T angular_velocity_limit = dim_i == 0 ? STATIC_PARAMETERS::STATE_LIMIT_ANGULAR_VELOCITY_X : (dim_i == 1 ? STATIC_PARAMETERS::STATE_LIMIT_ANGULAR_VELOCITY_Y : STATIC_PARAMETERS::STATE_LIMIT_ANGULAR_VELOCITY_Z);
            next_state.position[dim_i]         = math::clamp(device.math, next_state.position[dim_i]       , -position_limit, position_limit);
            next_state.linear_velocity[dim_i]  = math::clamp(device.math, next_state.linear_velocity[dim_i], -velocity_limit, velocity_limit);
            next_state.angular_velocity[dim_i] = math::clamp(device.math, next_state.angular_velocity[dim_i], -angular_velocity_limit, angular_velocity_limit);
        }

    }
    template<typename DEVICE, typename SPEC, typename PARAMETERS, typename STATE_SPEC, typename ACTION_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void post_integration(DEVICE& device, const Multirotor<SPEC>& env, PARAMETERS& parameters, const StateLastAction<STATE_SPEC>& state, const Matrix<ACTION_SPEC>& action, StateLastAction<STATE_SPEC>& next_state, RNG& rng) {
        using MULTIROTOR = Multirotor<SPEC>;
        using TI = typename DEVICE::index_t;
        static_assert(ACTION_SPEC::COLS == MULTIROTOR::ACTION_DIM);
        post_integration(device, env, parameters, static_cast<const typename STATE_SPEC::NEXT_COMPONENT&>(state), action, static_cast<typename STATE_SPEC::NEXT_COMPONENT&>(next_state), rng);
        for(TI action_i = 0; action_i < MULTIROTOR::ACTION_DIM; action_i++){
            next_state.last_action[action_i] = get(action, 0, action_i);
        }
    }
    template<typename DEVICE, typename SPEC, typename PARAMETERS, typename STATE_SPEC, typename ACTION_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void post_integration(DEVICE& device, const Multirotor<SPEC>& env, PARAMETERS& parameters, const StateLinearAcceleration<STATE_SPEC>& state, const Matrix<ACTION_SPEC>& action, StateLinearAcceleration<STATE_SPEC>& next_state, RNG& rng) {
        using T = typename STATE_SPEC::T;
        using TI = typename DEVICE::index_t;
        post_integration(device, env, parameters, static_cast<const typename STATE_SPEC::NEXT_COMPONENT&>(state), action, static_cast<typename STATE_SPEC::NEXT_COMPONENT&>(next_state), rng);
        for(TI state_i = 0; state_i < 3; state_i++){
            next_state.linear_acceleration[state_i] = (next_state.linear_velocity[state_i] - state.linear_velocity[state_i])/parameters.integration.dt;
        }
    }
    template<typename DEVICE, typename SPEC, typename PARAMETERS, typename STATE_SPEC, typename ACTION_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void post_integration(DEVICE& device, const Multirotor<SPEC>& env, PARAMETERS& parameters, const StateGyroBias<STATE_SPEC>& state, const Matrix<ACTION_SPEC>& action, StateGyroBias<STATE_SPEC>& next_state, RNG& rng) {
        typename DEVICE::SPEC::RANDOM random_dev;
        using T = typename STATE_SPEC::T;
        using TI = typename DEVICE::index_t;
        post_integration(device, env, parameters, static_cast<const typename STATE_SPEC::NEXT_COMPONENT&>(state), action, static_cast<typename STATE_SPEC::NEXT_COMPONENT&>(next_state), rng);
        T tau = parameters.imu.gyro_bias.tau;
        T sigma = parameters.imu.gyro_bias.sigma;
        T dt = parameters.integration.dt;
        if(tau > 0){
            T alpha = math::exp(device.math, -dt / tau);
            T sigma_d = sigma * math::sqrt(device.math, (T)1 - alpha * alpha);
            for(TI i = 0; i < 3; i++){
                T noise = random::normal_distribution::sample(random_dev, (T)0, (T)1, rng);
                next_state.gyro_bias[i] = alpha * state.gyro_bias[i] + sigma_d * noise;
            }
        }
        else{
            for(TI i = 0; i < 3; i++){
                next_state.gyro_bias[i] = state.gyro_bias[i];
            }
        }
    }
    template<typename DEVICE, typename SPEC, typename PARAMETERS, typename STATE_SPEC, typename ACTION_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void post_integration(DEVICE& device, const Multirotor<SPEC>& env, PARAMETERS& parameters, const StateMahony<STATE_SPEC>& state, const Matrix<ACTION_SPEC>& action, StateMahony<STATE_SPEC>& next_state, RNG& rng) {
        typename DEVICE::SPEC::RANDOM random_dev;
        using T = typename STATE_SPEC::T;
        using TI = typename DEVICE::index_t;
        using STATE = StateMahony<STATE_SPEC>;
        post_integration(device, env, parameters, static_cast<const typename STATE_SPEC::NEXT_COMPONENT&>(state), action, static_cast<typename STATE_SPEC::NEXT_COMPONENT&>(next_state), rng);
        T dt = parameters.integration.dt;
        T gyro_meas[3];
        for(TI i = 0; i < 3; i++){
            T noise = random::normal_distribution::sample(random_dev, (T)0, parameters.mdp.observation_noise.angular_velocity, rng);
            gyro_meas[i] = next_state.angular_velocity[i] + next_state.gyro_bias[i] + noise;
        }
        T conjugate_orientation[4];
        conjugate_orientation[0] =  next_state.orientation[0];
        conjugate_orientation[1] = -next_state.orientation[1];
        conjugate_orientation[2] = -next_state.orientation[2];
        conjugate_orientation[3] = -next_state.orientation[3];
        T accel_global[3];
        accel_global[0] = next_state.linear_acceleration[0] - parameters.dynamics.gravity[0];
        accel_global[1] = next_state.linear_acceleration[1] - parameters.dynamics.gravity[1];
        accel_global[2] = next_state.linear_acceleration[2] - parameters.dynamics.gravity[2];
        T accel_body[3];
        rotate_vector_by_quaternion<DEVICE, T>(conjugate_orientation, accel_global, accel_body);
        for(TI i = 0; i < 3; i++){
            T noise = random::normal_distribution::sample(random_dev, (T)0, parameters.mdp.observation_noise.imu_acceleration, rng);
            accel_body[i] += noise;
        }
        T accel_norm = math::sqrt(device.math, accel_body[0]*accel_body[0] + accel_body[1]*accel_body[1] + accel_body[2]*accel_body[2]);
        T z_estimate[3];
        T z_norm = math::sqrt(device.math,
            state.world_z_body_estimate[0]*state.world_z_body_estimate[0] +
            state.world_z_body_estimate[1]*state.world_z_body_estimate[1] +
            state.world_z_body_estimate[2]*state.world_z_body_estimate[2]);
        if(z_norm > 0){
            for(TI i = 0; i < 3; i++){
                z_estimate[i] = state.world_z_body_estimate[i] / z_norm;
            }
        }
        else{
            z_estimate[0] = 0;
            z_estimate[1] = 0;
            z_estimate[2] = 1;
        }
        T error[3] = {0, 0, 0};
        if(accel_norm > 0){
            // Gating: only trust the accelerometer-as-gravity assumption when |accel| is close to g.
            // During aggressive maneuvers |accel| deviates from g and the cross-product correction would
            // pull the estimate toward an arbitrary direction. Real flight controllers do this.
            T g_mag = math::sqrt(device.math, parameters.dynamics.gravity[0]*parameters.dynamics.gravity[0] + parameters.dynamics.gravity[1]*parameters.dynamics.gravity[1] + parameters.dynamics.gravity[2]*parameters.dynamics.gravity[2]);
            T ratio = g_mag > 0 ? accel_norm / g_mag : (T)0;
            T deviation = math::abs(device.math, ratio - (T)1);
            T gate = (T)1 - (T)3 * deviation;
            gate = gate < (T)0 ? (T)0 : (gate > (T)1 ? (T)1 : gate);
            T accel_unit[3];
            for(TI i = 0; i < 3; i++){
                accel_unit[i] = accel_body[i] / accel_norm;
            }
            error[0] = gate * (accel_unit[1]*z_estimate[2] - accel_unit[2]*z_estimate[1]);
            error[1] = gate * (accel_unit[2]*z_estimate[0] - accel_unit[0]*z_estimate[2]);
            error[2] = gate * (accel_unit[0]*z_estimate[1] - accel_unit[1]*z_estimate[0]);
        }
        T gyro_bias_tangent[3];
        T omega_corr[3];
        for(TI i = 0; i < 3; i++){
            gyro_bias_tangent[i] = state.gyro_bias_tangent[i] - STATE::KI * error[i] * dt;
        }
        detail::project_to_tangent(device, z_estimate, gyro_bias_tangent);
        for(TI i = 0; i < 3; i++){
            omega_corr[i] = gyro_meas[i] - gyro_bias_tangent[i] + STATE::KP * error[i];
        }
        T z_dot[3];
        z_dot[0] = z_estimate[1]*omega_corr[2] - z_estimate[2]*omega_corr[1];
        z_dot[1] = z_estimate[2]*omega_corr[0] - z_estimate[0]*omega_corr[2];
        z_dot[2] = z_estimate[0]*omega_corr[1] - z_estimate[1]*omega_corr[0];
        T z_new[3];
        T z_new_norm = 0;
        for(TI i = 0; i < 3; i++){
            z_new[i] = z_estimate[i] + z_dot[i] * dt;
            z_new_norm += z_new[i] * z_new[i];
        }
        z_new_norm = math::sqrt(device.math, z_new_norm);
        if(z_new_norm > 0){
            for(TI i = 0; i < 3; i++){
                next_state.world_z_body_estimate[i] = z_new[i] / z_new_norm;
            }
        }
        else{
            next_state.world_z_body_estimate[0] = 0;
            next_state.world_z_body_estimate[1] = 0;
            next_state.world_z_body_estimate[2] = 1;
        }
        detail::project_to_tangent(device, next_state.world_z_body_estimate, gyro_bias_tangent);
        for(TI i = 0; i < 3; i++){
            next_state.gyro_bias_tangent[i] = gyro_bias_tangent[i];
        }
    }
    template<typename DEVICE, typename SPEC, typename PARAMETERS, typename STATE_SPEC, typename ACTION_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void post_integration(DEVICE& device, const Multirotor<SPEC>& env, PARAMETERS& parameters, const StateLinearAccelerationHistory<STATE_SPEC>& state, const Matrix<ACTION_SPEC>& action, StateLinearAccelerationHistory<STATE_SPEC>& next_state, RNG& rng) {
        using T = typename STATE_SPEC::T;
        using TI = typename DEVICE::index_t;
        using STATE = StateLinearAccelerationHistory<STATE_SPEC>;
        post_integration(device, env, parameters, static_cast<const typename STATE_SPEC::NEXT_COMPONENT&>(state), action, static_cast<typename STATE_SPEC::NEXT_COMPONENT&>(next_state), rng);
        if constexpr(STATE_SPEC::HISTORY_LENGTH > 0){
            T conjugate_orientation[4];
            conjugate_orientation[0] =  next_state.orientation[0];
            conjugate_orientation[1] = -next_state.orientation[1];
            conjugate_orientation[2] = -next_state.orientation[2];
            conjugate_orientation[3] = -next_state.orientation[3];
            T acceleration_global[3];
            for(TI dim_i = 0; dim_i < 3; dim_i++){
                acceleration_global[dim_i] = (next_state.linear_velocity[dim_i] - state.linear_velocity[dim_i]) / parameters.integration.dt - parameters.dynamics.gravity[dim_i];
            }
            T acceleration_body[3];
            rotate_vector_by_quaternion<DEVICE, T>(conjugate_orientation, acceleration_global, acceleration_body);
            TI current_step = state.acceleration_history_step;
            for(TI dim_i = 0; dim_i < STATE::ACCELERATION_DIM; dim_i++){
                next_state.linear_acceleration_body_history[current_step][dim_i] = acceleration_body[dim_i];
            }
            next_state.acceleration_history_step = (state.acceleration_history_step + 1) % STATE_SPEC::HISTORY_LENGTH;
        }
    }
    template<typename DEVICE, typename SPEC, typename PARAMETERS, typename STATE_SPEC, typename ACTION_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void post_integration(DEVICE& device, const Multirotor<SPEC>& env, PARAMETERS& parameters, const StateAngularVelocityDelay<STATE_SPEC>& state, const Matrix<ACTION_SPEC>& action, StateAngularVelocityDelay<STATE_SPEC>& next_state, RNG& rng) {
        using TI = typename DEVICE::index_t;
        post_integration(device, env, parameters, static_cast<const typename STATE_SPEC::NEXT_COMPONENT&>(state), action, static_cast<typename STATE_SPEC::NEXT_COMPONENT&>(next_state), rng);

        if constexpr (STATE_SPEC::HISTORY_LENGTH == 0){
            for(TI dim_i = 0; dim_i < 3; dim_i++){
                next_state.angular_velocity_history[0][dim_i] = next_state.angular_velocity[dim_i];
            }
        }
        else
        {
            for(TI step_i = 0; step_i < STATE_SPEC::HISTORY_LENGTH; step_i++){
                for(TI dim_i = 0; dim_i < 3; dim_i++){
                    if (step_i == (STATE_SPEC::HISTORY_LENGTH - 1)){
                        next_state.angular_velocity_history[STATE_SPEC::HISTORY_LENGTH-1][dim_i] = state.angular_velocity[dim_i];
                    }
                    else{
                        next_state.angular_velocity_history[step_i][dim_i] = state.angular_velocity_history[step_i+1][dim_i];
                    }
                }
            }
        }
    }
    template<typename DEVICE, typename SPEC, typename PARAMETERS, typename STATE_SPEC, typename ACTION_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void post_integration(DEVICE& device, const Multirotor<SPEC>& env, PARAMETERS& parameters, const StateLinearVelocityDelay<STATE_SPEC>& state, const Matrix<ACTION_SPEC>& action, StateLinearVelocityDelay<STATE_SPEC>& next_state, RNG& rng) {
        using TI = typename DEVICE::index_t;
        post_integration(device, env, parameters, static_cast<const typename STATE_SPEC::NEXT_COMPONENT&>(state), action, static_cast<typename STATE_SPEC::NEXT_COMPONENT&>(next_state), rng);

        if constexpr (STATE_SPEC::HISTORY_LENGTH == 0){
            for(TI dim_i = 0; dim_i < 3; dim_i++){
                next_state.linear_velocity_history[0][dim_i] = next_state.linear_velocity[dim_i];
            }
        }
        else
        {
            for(TI step_i = 0; step_i < STATE_SPEC::HISTORY_LENGTH; step_i++){
                for(TI dim_i = 0; dim_i < 3; dim_i++){
                    if (step_i == (STATE_SPEC::HISTORY_LENGTH - 1)){
                        next_state.linear_velocity_history[STATE_SPEC::HISTORY_LENGTH-1][dim_i] = state.linear_velocity[dim_i];
                    }
                    else{
                        next_state.linear_velocity_history[step_i][dim_i] = state.linear_velocity_history[step_i+1][dim_i];
                    }
                }
            }
        }
    }
    //    template<typename DEVICE, typename SPEC, typename T, typename TI, typename NEXT_COMPONENT>
//    RL_TOOLS_FUNCTION_PLACEMENT void post_integration(DEVICE& device, const Multirotor<SPEC>& env, StateRotors<STATE_SPEC>& state) {
    template<typename DEVICE, typename SPEC, typename PARAMETERS, typename STATE_SPEC, typename ACTION_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void post_integration(DEVICE& device, const Multirotor<SPEC>& env, PARAMETERS& parameters, const StateRotors<STATE_SPEC>& state, const Matrix<ACTION_SPEC>& action, StateRotors<STATE_SPEC>& next_state, RNG& rng) {
        post_integration(device, env, parameters, static_cast<const typename STATE_SPEC::NEXT_COMPONENT&>(state), action, static_cast<typename STATE_SPEC::NEXT_COMPONENT&>(next_state), rng);
        using MULTIROTOR = Multirotor<SPEC>;
        using T = typename STATE_SPEC::T;
        for(typename DEVICE::index_t rpm_i = 0; rpm_i < MULTIROTOR::ACTION_DIM; rpm_i++){
            if constexpr(STATE_SPEC::CLOSED_FORM) {
                T setpoint_clamped = math::clamp(typename DEVICE::SPEC::MATH{}, get(action, 0, rpm_i), parameters.dynamics.action_limit.min, parameters.dynamics.action_limit.max);
                T tau = setpoint_clamped >= state.rpm[rpm_i] ? parameters.dynamics.rotor_time_constants_rising[rpm_i] : parameters.dynamics.rotor_time_constants_falling[rpm_i] ;
                T alpha = math::exp(device.math, - parameters.integration.dt / tau);
                next_state.rpm[rpm_i] = alpha * state.rpm[rpm_i] + (1 - alpha) * setpoint_clamped;
            }
            else {
                next_state.rpm[rpm_i] = math::clamp(typename DEVICE::SPEC::MATH{}, next_state.rpm[rpm_i], parameters.dynamics.action_limit.min, parameters.dynamics.action_limit.max);
            }
        }
    }
    template<typename DEVICE, typename SPEC, typename PARAMETERS, typename STATE_SPEC, typename ACTION_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void post_integration(DEVICE& device, const Multirotor<SPEC>& env, PARAMETERS& parameters, const StateRandomForce<STATE_SPEC>& state, const Matrix<ACTION_SPEC>& action, StateRandomForce<STATE_SPEC>& next_state, RNG& rng) {
        post_integration(device, env, parameters, static_cast<const typename STATE_SPEC::NEXT_COMPONENT&>(state), action, static_cast<typename STATE_SPEC::NEXT_COMPONENT&>(next_state), rng);
        next_state.force[0] = state.force[0];
        next_state.force[1] = state.force[1];
        next_state.force[2] = state.force[2];
        next_state.torque[0] = state.torque[0];
        next_state.torque[1] = state.torque[1];
        next_state.torque[2] = state.torque[2];
    }
    template<typename DEVICE, typename SPEC, typename PARAMETERS, typename STATE_SPEC, typename ACTION_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void post_integration(DEVICE& device, const Multirotor<SPEC>& env, PARAMETERS& parameters, const StateRotorsHistory<STATE_SPEC>& state, const Matrix<ACTION_SPEC>& action, StateRotorsHistory<STATE_SPEC>& next_state, RNG& rng) {
        using MULTIROTOR = Multirotor<SPEC>;
        using TI = typename DEVICE::index_t;
        using STATE = StateRotorsHistory<STATE_SPEC>;
        static_assert(ACTION_SPEC::COLS == MULTIROTOR::ACTION_DIM);
        post_integration(device, env, parameters, static_cast<const typename STATE::NEXT_COMPONENT&>(state), action, static_cast<typename STATE::NEXT_COMPONENT&>(next_state), rng);
        if constexpr(STATE_SPEC::HISTORY_LENGTH > 0){
            TI current_step = state.rotor_history_step;
            for(TI action_i = 0; action_i < MULTIROTOR::ACTION_DIM; action_i++){
                next_state.action_history[current_step][action_i] = get(action, 0, action_i);
            }
            next_state.rotor_history_step = (state.rotor_history_step + 1) % STATE_SPEC::HISTORY_LENGTH;
        }
    }
    template<typename DEVICE, typename SPEC, typename PARAMETERS, typename STATE_SPEC, typename ACTION_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void post_integration(DEVICE& device, const Multirotor<SPEC>& env, PARAMETERS& parameters, const StateTrajectory<STATE_SPEC>& state, const Matrix<ACTION_SPEC>& action, StateTrajectory<STATE_SPEC>& next_state, RNG& rng) {
        using MULTIROTOR = Multirotor<SPEC>;
        using TI = typename DEVICE::index_t;
        using T = typename STATE_SPEC::T;
        using STATE = StateTrajectory<STATE_SPEC>;
        static_assert(ACTION_SPEC::COLS == MULTIROTOR::ACTION_DIM);
        post_integration(device, env, parameters, static_cast<const typename STATE::NEXT_COMPONENT&>(state), action, static_cast<typename STATE::NEXT_COMPONENT&>(next_state), rng);
        next_state.trajectory_step = state.trajectory_step + 1;
    }

}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
