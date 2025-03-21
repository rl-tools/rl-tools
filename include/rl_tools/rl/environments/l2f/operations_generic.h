#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_L2F_OPERATIONS_GENERIC_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_L2F_OPERATIONS_GENERIC_H

#include "multirotor.h"

#include <rl_tools/utils/generic/vector_operations.h>
#include "quaternion_helper.h"

#include <rl_tools/utils/generic/typing.h>

#include <rl_tools/rl/environments/operations_generic.h>

#ifndef RL_TOOLS_FUNCTION_PLACEMENT
#define RL_TOOLS_FUNCTION_PLACEMENT
#endif

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#include <rl_tools/utils/generic/integrators.h>


// Since L2F is quite flexible in the way states and observations are composed, the operations might need to call each other in arbitrary order (depending on the definition). Hence we implement a dispatch scheme, were a dispatch function is forward declared such that it can be called from all specialized functions (_xxx). This dispatch function serves as the public interface xxx as well.
#include "operations_generic/05_state_is_nan.h"
#include "operations_generic/10_sample_initial_parameters.h"
#include "operations_generic/20_initial_state.h"
#include "operations_generic/30_sample_initial_state.h"
#include "operations_generic/40_observe.h"
#include "operations_generic/50_state_algebra.h"
RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::environments::l2f{
    template<typename DEVICE, typename STATE_SPEC, typename PARAMETERS, typename T>
    RL_TOOLS_FUNCTION_PLACEMENT void multirotor_dynamics(DEVICE& device, const PARAMETERS& params, const StateBase<STATE_SPEC>& state, const T* action, StateBase<STATE_SPEC>& state_change) {
        using STATE = StateBase<STATE_SPEC>;

        T thrust[3];
        T torque[3];
        thrust[0] = 0;
        thrust[1] = 0;
        thrust[2] = 0;
        torque[0] = 0;
        torque[1] = 0;
        torque[2] = 0;
        // flops: N*23 => 4 * 23 = 92
        for(typename DEVICE::index_t i_rotor = 0; i_rotor < 4; i_rotor++){
            // flops: 3 + 1 + 3 + 3 + 3 + 4 + 6 = 23
            T rpm = action[i_rotor];
            T thrust_magnitude = params.dynamics.rotor_thrust_coefficients[i_rotor][0] + params.dynamics.rotor_thrust_coefficients[i_rotor][1] * rpm + params.dynamics.rotor_thrust_coefficients[i_rotor][2] * rpm * rpm;
            T rotor_thrust[3];
            rl_tools::utils::vector_operations::scalar_multiply<DEVICE, T, 3>(params.dynamics.rotor_thrust_directions[i_rotor], thrust_magnitude, rotor_thrust);
            rl_tools::utils::vector_operations::add_accumulate<DEVICE, T, 3>(rotor_thrust, thrust);

            rl_tools::utils::vector_operations::scalar_multiply_accumulate<DEVICE, T, 3>(params.dynamics.rotor_torque_directions[i_rotor], thrust_magnitude * params.dynamics.rotor_torque_constants[i_rotor], torque);
            rl_tools::utils::vector_operations::cross_product_accumulate<DEVICE, T>(params.dynamics.rotor_positions[i_rotor], rotor_thrust, torque);
        }

        // linear_velocity_global
        state_change.position[0] = state.linear_velocity[0];
        state_change.position[1] = state.linear_velocity[1];
        state_change.position[2] = state.linear_velocity[2];

        // angular_velocity_global
        // flops: 16
        quaternion_derivative<DEVICE, T>(state.orientation, state.angular_velocity, state_change.orientation);

        // linear_acceleration_global
        // flops: 21
        rotate_vector_by_quaternion<DEVICE, T>(state.orientation, thrust, state_change.linear_velocity);
        // flops: 4
        rl_tools::utils::vector_operations::scalar_multiply<DEVICE, T, 3>(state_change.linear_velocity, 1 / params.dynamics.mass);
        rl_tools::utils::vector_operations::add_accumulate<DEVICE, T, 3>(params.dynamics.gravity, state_change.linear_velocity);

        T vector[3];
        T vector2[3];

        // angular_acceleration_local
        // flops: 9
        rl_tools::utils::vector_operations::matrix_vector_product<DEVICE, T, 3, 3>(params.dynamics.J, state.angular_velocity, vector);
        // flops: 6
        rl_tools::utils::vector_operations::cross_product<DEVICE, T>(state.angular_velocity, vector, vector2);
        rl_tools::utils::vector_operations::sub<DEVICE, T, 3>(torque, vector2, vector);
        // flops: 9
        rl_tools::utils::vector_operations::matrix_vector_product<DEVICE, T, 3, 3>(params.dynamics.J_inv, vector, state_change.angular_velocity);
        // total flops: (quadrotor): 92 + 16 + 21 + 4 + 9 + 6 + 9 = 157
//        multirotor_dynamics<DEVICE, T, TI, PARAMETERS>(device, params, (const typename STATE::LATENT_STATE&)state, action, state_change);
//        multirotor_dynamics(device, params, (const typename STATE::LATENT_STATE&)state, action, state_change);
    }
    template<typename DEVICE, typename PARAMETERS, typename STATE_SPEC, typename T>
    RL_TOOLS_FUNCTION_PLACEMENT void multirotor_dynamics(DEVICE& device, const PARAMETERS& params, const StatePoseErrorIntegral<STATE_SPEC>& state, const T* action, StatePoseErrorIntegral<STATE_SPEC>& state_change){
        multirotor_dynamics(device, params, static_cast<const typename STATE_SPEC::NEXT_COMPONENT&>(state), action, static_cast<typename STATE_SPEC::NEXT_COMPONENT&>(state_change));
        T position_error = state.position[0] * state.position[0] + state.position[1] * state.position[1] + state.position[2] * state.position[2];
        position_error = math::sqrt(device.math, position_error);
        T w_clamped = math::clamp(device.math, state.orientation[0], (T)-1, (T)1);
        T orientation_error = 2*math::acos(device.math, w_clamped);
        state_change.position_integral = position_error;
        state_change.orientation_integral = orientation_error;
    }
    template<typename DEVICE, typename PARAMETERS, typename STATE_SPEC, typename T>
    RL_TOOLS_FUNCTION_PLACEMENT void multirotor_dynamics(DEVICE& device, const PARAMETERS& params, const StateRandomForce<STATE_SPEC>& state, const T* action, StateRandomForce<STATE_SPEC>& state_change){
        multirotor_dynamics(device, params, static_cast<const typename STATE_SPEC::NEXT_COMPONENT&>(state), action, static_cast<typename STATE_SPEC::NEXT_COMPONENT&>(state_change));

        state_change.linear_velocity[0] += state.force[0] / params.dynamics.mass;
        state_change.linear_velocity[1] += state.force[1] / params.dynamics.mass;
        state_change.linear_velocity[2] += state.force[2] / params.dynamics.mass;

        T angular_acceleration[3];

        rl_tools::utils::vector_operations::matrix_vector_product<DEVICE, T, 3, 3>(params.dynamics.J_inv, state.torque, angular_acceleration);
        rl_tools::utils::vector_operations::add_accumulate<DEVICE, T, 3>(angular_acceleration, state_change.angular_velocity);
    }
    template<typename DEVICE, typename PARAMETERS, typename STATE_SPEC, typename T>
    RL_TOOLS_FUNCTION_PLACEMENT void multirotor_dynamics(DEVICE& device, const PARAMETERS& params, const StateRotors<STATE_SPEC>& state, const T* action, StateRotors<STATE_SPEC>& state_change) {
        multirotor_dynamics(device, params, static_cast<const typename STATE_SPEC::NEXT_COMPONENT&>(state), state.rpm, static_cast<typename STATE_SPEC::NEXT_COMPONENT&>(state_change));

        if constexpr(!STATE_SPEC::CLOSED_FORM) {
            for(typename DEVICE::index_t i_rotor = 0; i_rotor < 4; i_rotor++){
                T tau = action[i_rotor] >= state.rpm[i_rotor] ? params.dynamics.rotor_time_constants_rising[i_rotor] : params.dynamics.rotor_time_constants_falling[i_rotor] ;
                state_change.rpm[i_rotor] = (action[i_rotor] - state.rpm[i_rotor]) * 1/tau;
            }
        }

    }
    template<typename DEVICE, typename PARAMETERS, typename STATE_SPEC, typename T>
    RL_TOOLS_FUNCTION_PLACEMENT void multirotor_dynamics(DEVICE& device, const PARAMETERS& params, const StateRotorsHistory<STATE_SPEC>& state, const T* action, StateRotorsHistory<STATE_SPEC>& state_change){
        using STATE = StateRotorsHistory<STATE_SPEC>;
        multirotor_dynamics(device, params, static_cast<const typename STATE::NEXT_COMPONENT&>(state), action, static_cast<typename STATE::NEXT_COMPONENT&>(state_change));
    }
    template<typename DEVICE, typename T, typename PARAMETERS, typename STATE>
    RL_TOOLS_FUNCTION_PLACEMENT void multirotor_dynamics_dispatch(DEVICE& device, const PARAMETERS& params, const STATE& state, const T* action, STATE& state_change) {
        // this dispatch function is required to pass the multirotor dynamics function to the integrator (euler, rk4) as a template parameter (so that it can be inlined/optimized at compile time)
        // If we would try to pass the multirotor_dynamics function directly the state type-based overloading would make the inference of the auto template parameter for the dynamics function in the integrator function impossible
//        multirotor_dynamics<DEVICE, T, typename DEVICE::index_t, typename STATE::LATENT_STATE, PARAMETERS>(device, params, state, action, state_change);
        multirotor_dynamics(device, params, state, action, state_change);
    }

}
RL_TOOLS_NAMESPACE_WRAPPER_END


RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools
{
    template<typename DEVICE, typename SPEC>
    void malloc(DEVICE&, rl::environments::Multirotor<SPEC>& env){
    }
    template<typename DEVICE, typename SPEC>
    void free(DEVICE&, rl::environments::Multirotor<SPEC>&){ }
    template<typename DEVICE, typename SPEC>
    void init(DEVICE&, rl::environments::Multirotor<SPEC>& env){
        env.parameters = SPEC::STATIC_PARAMETERS::PARAMETER_VALUES;
    }
    template<typename DEVICE, typename STATE>
    static bool is_nan(DEVICE& device, STATE& state){
        return rl::environments::l2f::_is_nan(device, state);
    }
    template<typename DEVICE, typename SPEC, typename PARAMETERS>
    static void initial_parameters(DEVICE& device, rl::environments::Multirotor<SPEC>& env, PARAMETERS& parameters){
        parameters = env.parameters;
        //        parameters = SPEC::STATIC_PARAMETERS::PARAMETER_VALUES;
    }
    template<typename DEVICE, typename SPEC, typename PARAMETERS, typename RNG>
    static void sample_initial_parameters(DEVICE& device, rl::environments::Multirotor<SPEC>& env, PARAMETERS& parameters, RNG& rng){
        // to allow out of declaration order dispatch
        rl::environments::l2f::_sample_initial_parameters(device, env, parameters, rng);
    }
    template<typename DEVICE, typename SPEC, typename PARAMETERS, typename STATE>
    static void initial_state(DEVICE& device, rl::environments::Multirotor<SPEC>& env, PARAMETERS& parameters, STATE& state){
        rl::environments::l2f::_initial_state(device, env, parameters, state);
    }
    template<typename DEVICE, typename SPEC, typename PARAMETERS, typename STATE, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT static void sample_initial_state(DEVICE& device, rl::environments::Multirotor<SPEC>& env, PARAMETERS& parameters, STATE& state, RNG& rng){
        rl::environments::l2f::_sample_initial_state(device, env, parameters, state, rng);
    }
    template<typename DEVICE, typename SPEC, typename PARAMETERS, typename STATE, typename OBSERVATION, typename OBS_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT static void observe(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, PARAMETERS& parameters, const STATE& state, const OBSERVATION& observation_type, Matrix<OBS_SPEC>& observation, RNG& rng){
        static_assert(OBS_SPEC::COLS == OBSERVATION::DIM);
        static_assert(OBS_SPEC::ROWS == 1);
        rl::environments::l2f::_observe(device, env, parameters, state, observation_type, observation, rng);
    }
    template<typename DEVICE, typename SPEC, typename PARAMETERS, typename STATE_SPEC, typename ACTION_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void post_integration(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, PARAMETERS& parameters, const rl::environments::l2f::StateBase<STATE_SPEC>& state, const Matrix<ACTION_SPEC>& action, rl::environments::l2f::StateBase<STATE_SPEC>& next_state, RNG& rng) {
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
            next_state.position[dim_i]         = math::clamp(device.math, next_state.position[dim_i]       , -STATIC_PARAMETERS::STATE_LIMIT_POSITION, STATIC_PARAMETERS::STATE_LIMIT_POSITION);
            next_state.linear_velocity[dim_i]  = math::clamp(device.math, next_state.linear_velocity[dim_i], -STATIC_PARAMETERS::STATE_LIMIT_VELOCITY, STATIC_PARAMETERS::STATE_LIMIT_VELOCITY);
            next_state.angular_velocity[dim_i] = math::clamp(device.math, next_state.angular_velocity[dim_i], -STATIC_PARAMETERS::STATE_LIMIT_ANGULAR_VELOCITY, STATIC_PARAMETERS::STATE_LIMIT_ANGULAR_VELOCITY);
        }

    }
    template<typename DEVICE, typename SPEC, typename PARAMETERS, typename STATE_SPEC, typename ACTION_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void post_integration(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, PARAMETERS& parameters, const rl::environments::l2f::StateLastAction<STATE_SPEC>& state, const Matrix<ACTION_SPEC>& action, rl::environments::l2f::StateLastAction<STATE_SPEC>& next_state, RNG& rng) {
        using MULTIROTOR = rl::environments::Multirotor<SPEC>;
        using TI = typename DEVICE::index_t;
        static_assert(ACTION_SPEC::COLS == MULTIROTOR::ACTION_DIM);
        post_integration(device, env, parameters, static_cast<const typename STATE_SPEC::NEXT_COMPONENT&>(state), action, static_cast<typename STATE_SPEC::NEXT_COMPONENT&>(next_state), rng);
        for(TI action_i = 0; action_i < MULTIROTOR::ACTION_DIM; action_i++){
            next_state.last_action[action_i] = get(action, 0, action_i);
        }
    }
    template<typename DEVICE, typename SPEC, typename PARAMETERS, typename STATE_SPEC, typename ACTION_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void post_integration(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, PARAMETERS& parameters, const rl::environments::l2f::StateLinearAcceleration<STATE_SPEC>& state, const Matrix<ACTION_SPEC>& action, rl::environments::l2f::StateLinearAcceleration<STATE_SPEC>& next_state, RNG& rng) {
        using T = typename STATE_SPEC::T;
        using TI = typename DEVICE::index_t;
        post_integration(device, env, parameters, static_cast<const typename STATE_SPEC::NEXT_COMPONENT&>(state), action, static_cast<typename STATE_SPEC::NEXT_COMPONENT&>(next_state), rng);
        for(TI state_i = 0; state_i < 3; state_i++){
            next_state.linear_acceleration[state_i] = (next_state.linear_velocity[state_i] - state.linear_velocity[state_i])/parameters.integration.dt;
        }
    }
    template<typename DEVICE, typename SPEC, typename PARAMETERS, typename STATE_SPEC, typename ACTION_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void post_integration(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, PARAMETERS& parameters, const rl::environments::l2f::StateAngularVelocityDelay<STATE_SPEC>& state, const Matrix<ACTION_SPEC>& action, rl::environments::l2f::StateAngularVelocityDelay<STATE_SPEC>& next_state, RNG& rng) {
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
//    template<typename DEVICE, typename SPEC, typename T, typename TI, typename NEXT_COMPONENT>
//    RL_TOOLS_FUNCTION_PLACEMENT void post_integration(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, rl::environments::l2f::StateRotors<STATE_SPEC>& state) {
    template<typename DEVICE, typename SPEC, typename PARAMETERS, typename STATE_SPEC, typename ACTION_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void post_integration(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, PARAMETERS& parameters, const rl::environments::l2f::StateRotors<STATE_SPEC>& state, const Matrix<ACTION_SPEC>& action, rl::environments::l2f::StateRotors<STATE_SPEC>& next_state, RNG& rng) {
        post_integration(device, env, parameters, static_cast<const typename STATE_SPEC::NEXT_COMPONENT&>(state), action, static_cast<typename STATE_SPEC::NEXT_COMPONENT&>(next_state), rng);
        using MULTIROTOR = rl::environments::Multirotor<SPEC>;
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
    RL_TOOLS_FUNCTION_PLACEMENT void post_integration(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, PARAMETERS& parameters, const rl::environments::l2f::StateRandomForce<STATE_SPEC>& state, const Matrix<ACTION_SPEC>& action, rl::environments::l2f::StateRandomForce<STATE_SPEC>& next_state, RNG& rng) {
        post_integration(device, env, parameters, static_cast<const typename STATE_SPEC::NEXT_COMPONENT&>(state), action, static_cast<typename STATE_SPEC::NEXT_COMPONENT&>(next_state), rng);
        next_state.force[0] = state.force[0];
        next_state.force[1] = state.force[1];
        next_state.force[2] = state.force[2];
        next_state.torque[0] = state.torque[0];
        next_state.torque[1] = state.torque[1];
        next_state.torque[2] = state.torque[2];
    }
    template<typename DEVICE, typename SPEC, typename PARAMETERS, typename STATE_SPEC, typename ACTION_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void post_integration(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, PARAMETERS& parameters, const rl::environments::l2f::StateRotorsHistory<STATE_SPEC>& state, const Matrix<ACTION_SPEC>& action, rl::environments::l2f::StateRotorsHistory<STATE_SPEC>& next_state, RNG& rng) {
        using MULTIROTOR = rl::environments::Multirotor<SPEC>;
        using TI = typename DEVICE::index_t;
        using STATE = rl::environments::l2f::StateRotorsHistory<STATE_SPEC>;
        static_assert(ACTION_SPEC::COLS == MULTIROTOR::ACTION_DIM);
        post_integration(device, env, parameters, static_cast<const typename STATE::NEXT_COMPONENT&>(state), action, static_cast<typename STATE::NEXT_COMPONENT&>(next_state), rng);
        if constexpr(STATE_SPEC::HISTORY_LENGTH > 0){
            TI current_step = state.current_step;
            for(TI action_i = 0; action_i < MULTIROTOR::ACTION_DIM; action_i++){
                next_state.action_history[current_step][action_i] = get(action, 0, action_i);
            }
            next_state.current_step = (state.current_step + 1) % STATE_SPEC::HISTORY_LENGTH;
        }
    }
    // todo: make state const again
    template<typename DEVICE, typename SPEC, typename PARAMETERS, typename STATE, typename ACTION_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT static typename SPEC::T step(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, PARAMETERS& parameters, const STATE& state, const Matrix<ACTION_SPEC>& action, STATE& next_state, RNG& rng) {
        using T = typename SPEC::T;
        using TI = typename DEVICE::index_t;
        constexpr auto STATE_DIM = STATE::DIM;
        constexpr auto ACTION_DIM = rl::environments::Multirotor<SPEC>::ACTION_DIM;
        static_assert(ACTION_SPEC::ROWS == 1);
        static_assert(ACTION_SPEC::COLS == ACTION_DIM);
        T action_scaled[ACTION_DIM];

        for(TI action_i = 0; action_i < ACTION_DIM; action_i++){
            T half_range = (parameters.dynamics.action_limit.max - parameters.dynamics.action_limit.min) / 2;
            T action_noisy = get(action, 0, action_i);
            action_noisy += random::normal_distribution::sample(typename DEVICE::SPEC::RANDOM(), (T)0, parameters.mdp.action_noise.normalized_rpm, rng);
            action_noisy = math::clamp(device.math, action_noisy, -(T)1, (T)1);
            action_scaled[action_i] = action_noisy * half_range + parameters.dynamics.action_limit.min + half_range;
//            state.rpm[action_i] = action_scaled[action_i];
        }
        if constexpr(SPEC::STATIC_PARAMETERS::N_SUBSTEPS == 1){
            utils::integrators::rk4  <DEVICE, typename SPEC::T, typename SPEC::PARAMETERS, STATE, ACTION_DIM, rl::environments::l2f::multirotor_dynamics_dispatch<DEVICE, typename SPEC::T, typename SPEC::PARAMETERS, STATE>>(device, parameters, state, action_scaled, parameters.integration.dt, next_state);
    //        utils::integrators::euler<DEVICE, typename SPEC::T, typename SPEC::PARAMETERS, STATE, ACTION_DIM, rl::environments::l2f::multirotor_dynamics_dispatch<DEVICE, typename SPEC::T, typename SPEC::PARAMETERS, STATE>>(device, parameters, state, action_scaled, parameters.integration.dt, next_state);
        }
        else{
            auto substep_state = state;
            auto substep_next_state = state;
            T substep_dt = parameters.integration.dt / SPEC::STATIC_PARAMETERS::N_SUBSTEPS;
            for (TI substep_i=0; substep_i < SPEC::STATIC_PARAMETERS::N_SUBSTEPS; substep_i++){
                utils::integrators::rk4  <DEVICE, typename SPEC::T, typename SPEC::PARAMETERS, STATE, ACTION_DIM, rl::environments::l2f::multirotor_dynamics_dispatch<DEVICE, typename SPEC::T, typename SPEC::PARAMETERS, STATE>>(device, parameters, substep_state, action_scaled, substep_dt, substep_next_state);
        //        utils::integrators::euler<DEVICE, typename SPEC::T, typename SPEC::PARAMETERS, STATE, ACTION_DIM, rl::environments::l2f::multirotor_dynamics_dispatch<DEVICE, typename SPEC::T, typename SPEC::PARAMETERS, STATE>>(device, parameters, substep_state, action_scaled, substep_dt, substep_next_state);
                substep_state = substep_next_state;
            }
            next_state = substep_next_state;
        }

        post_integration(device, env, parameters, state, action, next_state, rng);

//        utils::assert_exit(device, !math::is_nan(device.math, next_state.position_integral), "pi nan");
//        utils::assert_exit(device, !math::is_nan(device.math, next_state.orientation_integral), "oi nan");
//        utils::assert_exit(device, !is_nan(device, action), "action nan");
//        utils::assert_exit(device, !is_nan(device, next_state), "nan");
        return parameters.integration.dt;
    }

    template<typename DEVICE, typename SPEC, typename PARAMETERS, typename STATE, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT static bool terminated(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, const PARAMETERS& parameters, const STATE& state, RNG& rng){
        using T = typename SPEC::T;
        using TI = typename DEVICE::index_t;
        if(parameters.mdp.termination.enabled){
            for(TI i = 0; i < 3; i++){
                if(
                    math::abs(device.math, state.position[i]) > parameters.mdp.termination.position_threshold ||
                    math::abs(device.math, state.linear_velocity[i]) > parameters.mdp.termination.linear_velocity_threshold ||
                    math::abs(device.math, state.angular_velocity[i]) > parameters.mdp.termination.angular_velocity_threshold
                ){
                    return true;
                }
            }
        }
//        if(state.position_integral > parameters.mdp.termination.position_integral_threshold){
//            return true;
//        }
//        if(state.orientation_integral > parameters.mdp.termination.orientation_integral_threshold){
//            return true;
//        }
        return false;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#include "parameters/reward_functions/reward_functions.h"
RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template<typename DEVICE, typename SPEC, typename PARAMETERS, typename STATE, typename ACTION_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT static typename SPEC::T reward(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, PARAMETERS& parameters, const STATE& state, const Matrix<ACTION_SPEC>& action, const STATE& next_state, RNG& rng) {
        return rl::environments::l2f::parameters::reward_functions::reward(device, env, parameters, parameters.mdp.reward, state, action, next_state, rng);
    }
    template<typename DEVICE, typename SPEC, typename PARAMETERS, typename STATE, typename ACTION_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void log_reward(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, PARAMETERS& parameters, const STATE& state, const Matrix<ACTION_SPEC>& action, const STATE& next_state, RNG& rng, typename DEVICE::index_t cadence = 1) {
        rl::environments::l2f::parameters::reward_functions::log_reward(device, env, parameters, parameters.mdp.reward, state, action, next_state, rng, cadence);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#include "parameters/default.h"

#endif