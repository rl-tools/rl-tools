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
#include "operations_generic/60_dynamics.h"
RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::environments::l2f{
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