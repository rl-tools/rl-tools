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
        using T = typename STATE_SPEC::T;
        using TI = typename DEVICE::index_t;
        static_assert(ACTION_SPEC::COLS == MULTIROTOR::ACTION_DIM);
        post_integration(device, env, parameters, static_cast<const typename STATE_SPEC::NEXT_COMPONENT&>(state), action, static_cast<typename STATE_SPEC::NEXT_COMPONENT&>(next_state), rng);
        for(TI action_i = 0; action_i < MULTIROTOR::ACTION_DIM; action_i++){
            next_state.last_action[action_i] = math::clamp(device.math, get(action, 0, action_i), (T)-1, (T)1);
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
        T q0 = state.q_estimate[0];
        T q1 = state.q_estimate[1];
        T q2 = state.q_estimate[2];
        T q3 = state.q_estimate[3];
        T bx = state.bias_estimate[0];
        T by = state.bias_estimate[1];
        T bz = state.bias_estimate[2];

        T accel_norm = math::sqrt(device.math, accel_body[0]*accel_body[0] + accel_body[1]*accel_body[1] + accel_body[2]*accel_body[2]);
        T accel_gate_lo = STATE::ACCEL_GATE_LO_G * STATE::G_REF;
        T accel_gate_hi = STATE::ACCEL_GATE_HI_G * STATE::G_REF;
        bool use_accel = accel_norm > (T)1e-6 && accel_gate_lo <= accel_norm && accel_norm <= accel_gate_hi;

        T wx;
        T wy;
        T wz;
        if(use_accel){
            T inv_accel_norm = (T)1 / accel_norm;
            T ahx = accel_body[0] * inv_accel_norm;
            T ahy = accel_body[1] * inv_accel_norm;
            T ahz = accel_body[2] * inv_accel_norm;

            T vx = (T)2 * (q1 * q3 - q0 * q2);
            T vy = (T)2 * (q2 * q3 + q0 * q1);
            T vz_est = q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3;

            T ex = ahy * vz_est - ahz * vy;
            T ey = ahz * vx - ahx * vz_est;
            T ez = ahx * vy - ahy * vx;

            bx -= STATE::KI * ex * dt;
            by -= STATE::KI * ey * dt;
            bz -= STATE::KI * ez * dt;
            bx = math::clamp(device.math, bx, -STATE::MAX_BIAS, STATE::MAX_BIAS);
            by = math::clamp(device.math, by, -STATE::MAX_BIAS, STATE::MAX_BIAS);
            bz = math::clamp(device.math, bz, -STATE::MAX_BIAS, STATE::MAX_BIAS);

            wx = gyro_meas[0] - bx + STATE::KP * ex;
            wy = gyro_meas[1] - by + STATE::KP * ey;
            wz = gyro_meas[2] - bz + STATE::KP * ez;
        }
        else{
            wx = gyro_meas[0] - bx;
            wy = gyro_meas[1] - by;
            wz = gyro_meas[2] - bz;
        }

        T dq0 = (T)0.5 * (-q1 * wx - q2 * wy - q3 * wz);
        T dq1 = (T)0.5 * ( q0 * wx + q2 * wz - q3 * wy);
        T dq2 = (T)0.5 * ( q0 * wy - q1 * wz + q3 * wx);
        T dq3 = (T)0.5 * ( q0 * wz + q1 * wy - q2 * wx);

        q0 += dq0 * dt;
        q1 += dq1 * dt;
        q2 += dq2 * dt;
        q3 += dq3 * dt;

        T q_norm = math::sqrt(device.math, q0*q0 + q1*q1 + q2*q2 + q3*q3);
        if(q_norm > (T)1e-12){
            T inv_q_norm = (T)1 / q_norm;
            next_state.q_estimate[0] = q0 * inv_q_norm;
            next_state.q_estimate[1] = q1 * inv_q_norm;
            next_state.q_estimate[2] = q2 * inv_q_norm;
            next_state.q_estimate[3] = q3 * inv_q_norm;
        }
        else{
            next_state.q_estimate[0] = 1;
            next_state.q_estimate[1] = 0;
            next_state.q_estimate[2] = 0;
            next_state.q_estimate[3] = 0;
        }
        next_state.bias_estimate[0] = bx;
        next_state.bias_estimate[1] = by;
        next_state.bias_estimate[2] = bz;
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
    RL_TOOLS_FUNCTION_PLACEMENT void post_integration(DEVICE& device, const Multirotor<SPEC>& env, PARAMETERS& parameters, const StateCTBRController<STATE_SPEC>& state, const Matrix<ACTION_SPEC>& action, StateCTBRController<STATE_SPEC>& next_state, RNG& rng) {
        using TI = typename DEVICE::index_t;
        post_integration(device, env, parameters, static_cast<const typename STATE_SPEC::NEXT_COMPONENT&>(state), action, static_cast<typename STATE_SPEC::NEXT_COMPONENT&>(next_state), rng);
        if constexpr(SPEC::STATIC_PARAMETERS::ACTION_INTERFACE != parameters::ActionInterface::CTBR || SPEC::STATIC_PARAMETERS::N_SUBSTEPS == 1){
            for(TI i = 0; i < 3; i++){
                next_state.previous_angular_velocity[i] = state.angular_velocity[i];
            }
        }
    }
    template<typename DEVICE, typename SPEC, typename PARAMETERS, typename STATE_SPEC, typename ACTION_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void post_integration(DEVICE& device, const Multirotor<SPEC>& env, PARAMETERS& parameters, const StateRotorsHistory<STATE_SPEC>& state, const Matrix<ACTION_SPEC>& action, StateRotorsHistory<STATE_SPEC>& next_state, RNG& rng) {
        using MULTIROTOR = Multirotor<SPEC>;
        using T = typename STATE_SPEC::T;
        using TI = typename DEVICE::index_t;
        using STATE = StateRotorsHistory<STATE_SPEC>;
        static_assert(ACTION_SPEC::COLS == MULTIROTOR::ACTION_DIM);
        post_integration(device, env, parameters, static_cast<const typename STATE::NEXT_COMPONENT&>(state), action, static_cast<typename STATE::NEXT_COMPONENT&>(next_state), rng);
        if constexpr(STATE_SPEC::HISTORY_LENGTH > 0){
            TI current_step = state.rotor_history_step;
            for(TI action_i = 0; action_i < MULTIROTOR::ACTION_DIM; action_i++){
                next_state.action_history[current_step][action_i] = math::clamp(device.math, get(action, 0, action_i), (T)-1, (T)1);
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
    template<typename DEVICE, typename SPEC, typename PARAMETERS, typename STATE_SPEC, typename ACTION_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void post_integration(DEVICE& device, const Multirotor<SPEC>& env, PARAMETERS& parameters, const StateRenderRotorPhase<STATE_SPEC>& state, const Matrix<ACTION_SPEC>& action, StateRenderRotorPhase<STATE_SPEC>& next_state, RNG& rng) {
        using TI = typename DEVICE::index_t;
        using T = typename STATE_SPEC::T;
        using STATE = StateRenderRotorPhase<STATE_SPEC>;
        post_integration(device, env, parameters, static_cast<const typename STATE::NEXT_COMPONENT&>(state), action, static_cast<typename STATE::NEXT_COMPONENT&>(next_state), rng);
        constexpr T TWO_PI = (T)2 * math::PI<T>;
        for(TI rotor_i = 0; rotor_i < STATE::ACTION_DIM; rotor_i++){
            T phase = state.rotor_phase[rotor_i] + next_state.rpm[rotor_i] * (TWO_PI / (T)60) * parameters.integration.dt;
            phase = phase - math::floor(device.math, phase / TWO_PI) * TWO_PI;
            next_state.rotor_phase[rotor_i] = phase;
        }
    }

}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
