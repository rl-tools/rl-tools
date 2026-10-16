#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_L2F_OPERATIONS_GENERIC_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_L2F_OPERATIONS_GENERIC_H

#include "multirotor.h"

#include <rl_tools/utils/generic/vector_operations.h>
#include "quaternion_helper.h"

#include <rl_tools/rl/environments/operations_generic.h>

#include "../../../rl_tools.h"

#include <rl_tools/utils/generic/integrators.h>


// Since L2F is quite flexible in the way states and observations are composed, the operations might need to call each other in arbitrary order (depending on the definition). Hence we implement a dispatch scheme, were a dispatch function is forward declared such that it can be called from all specialized functions (_xxx). This dispatch function serves as the public interface xxx as well.
#include "operations_generic/00_actuation.h"
#include "operations_generic/05_state_is_nan.h"
#include "operations_generic/10_sample_initial_parameters.h"
#include "operations_generic/20_initial_state.h"
#include "operations_generic/30_sample_initial_state.h"
#include "operations_generic/35_get_desired_state.h"
#include "operations_generic/40_observe.h"
#include "operations_generic/50_state_algebra.h"
#include "operations_generic/60_dynamics.h"
#include "operations_generic/70_post_integration.h"
#include "operations_generic/80_abs_diff.h"


RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools
{
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE&, rl::environments::Multirotor<SPEC>& env){
    }
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE&, rl::environments::Multirotor<SPEC>&){ }
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void init(DEVICE&, rl::environments::Multirotor<SPEC>& env){
        env.parameters = SPEC::STATIC_PARAMETERS::PARAMETER_VALUES;
    }
    template<typename DEVICE, typename T, typename TI, TI N>
    RL_TOOLS_FUNCTION_PLACEMENT bool is_canonical_quadrotor(DEVICE&, const rl::environments::l2f::parameters::Dynamics<T, TI, N>& dynamics){
        if constexpr(N != 4){
            return false;
        }
        else{
            for(TI rotor = 0; rotor < 4; rotor++){
                T x_sign = (rotor == 0 || rotor == 3) ? (T)1 : (T)-1;
                T y_sign = rotor < 2 ? (T)-1 : (T)1;
                T yaw_sign = rotor % 2 == 0 ? (T)-1 : (T)1;
                if(!(x_sign * dynamics.rotor_positions[rotor][0] > 0 && y_sign * dynamics.rotor_positions[rotor][1] > 0 && dynamics.rotor_torque_constants[rotor] > 0)){
                    return false;
                }
                for(TI axis = 0; axis < 3; axis++){
                    if(dynamics.rotor_thrust_directions[rotor][axis] != (axis == 2 ? (T)1 : (T)0) || dynamics.rotor_torque_directions[rotor][axis] != (axis == 2 ? yaw_sign : (T)0)){
                        return false;
                    }
                }
            }
            return true;
        }
    }
    template<typename DEVICE, typename SPEC, typename PARAMETERS>
    RL_TOOLS_FUNCTION_PLACEMENT static void initial_parameters(DEVICE& device, rl::environments::Multirotor<SPEC>& env, PARAMETERS& parameters){
        parameters = env.parameters;
        //        parameters = SPEC::STATIC_PARAMETERS::PARAMETER_VALUES;
    }
    namespace rl::environments::l2f{
        template<typename DEVICE, typename T>
        RL_TOOLS_FUNCTION_PLACEMENT bool solve_4x4(DEVICE& device, T A[4][4], T b[4], T x[4]){
            T aug[4][5];
            for(typename DEVICE::index_t i = 0; i < 4; i++){
                for(typename DEVICE::index_t j = 0; j < 4; j++){
                    aug[i][j] = A[i][j];
                }
                aug[i][4] = b[i];
            }
            for(typename DEVICE::index_t col = 0; col < 4; col++){
                typename DEVICE::index_t pivot = col;
                T pivot_abs = math::abs(device.math, aug[pivot][col]);
                for(typename DEVICE::index_t row = col + 1; row < 4; row++){
                    T row_abs = math::abs(device.math, aug[row][col]);
                    if(row_abs > pivot_abs){
                        pivot = row;
                        pivot_abs = row_abs;
                    }
                }
                if(pivot_abs < (T)1e-12){
                    return false;
                }
                if(pivot != col){
                    for(typename DEVICE::index_t j = col; j < 5; j++){
                        T temp = aug[col][j];
                        aug[col][j] = aug[pivot][j];
                        aug[pivot][j] = temp;
                    }
                }
                T inv_pivot = (T)1 / aug[col][col];
                for(typename DEVICE::index_t j = col; j < 5; j++){
                    aug[col][j] *= inv_pivot;
                }
                for(typename DEVICE::index_t row = 0; row < 4; row++){
                    if(row != col){
                        T factor = aug[row][col];
                        for(typename DEVICE::index_t j = col; j < 5; j++){
                            aug[row][j] -= factor * aug[col][j];
                        }
                    }
                }
            }
            for(typename DEVICE::index_t i = 0; i < 4; i++){
                x[i] = aug[i][4];
            }
            return true;
        }
        template<typename DEVICE, typename PARAMETERS, typename STATE, typename ACTION_SPEC, typename RNG>
        RL_TOOLS_FUNCTION_PLACEMENT void direct_motor_action_to_motor_commands(DEVICE& device, const PARAMETERS& parameters, const STATE&, const Matrix<ACTION_SPEC>& action, typename PARAMETERS::T motor_commands[4], RNG& rng){
            using T = typename PARAMETERS::T;
            using TI = typename DEVICE::index_t;
            for(TI action_i = 0; action_i < 4; action_i++){
                T half_range = (parameters.dynamics.action_limit.max - parameters.dynamics.action_limit.min) / (T)2;
                T action_noisy = get(action, 0, action_i);
                action_noisy += random::normal_distribution::sample(typename DEVICE::SPEC::RANDOM(), (T)0, parameters.mdp.action_noise.normalized_rpm, rng);
                action_noisy = math::clamp(device.math, action_noisy, -(T)1, (T)1);
                motor_commands[action_i] = action_noisy * half_range + parameters.dynamics.action_limit.min + half_range;
            }
        }
        template<typename DEVICE, typename PARAMETERS, typename STATE, typename ACTION_SPEC, typename RNG>
        RL_TOOLS_FUNCTION_PLACEMENT void ctbr_action_to_motor_commands(DEVICE& device, const PARAMETERS& parameters, const STATE& state, const Matrix<ACTION_SPEC>& action, typename PARAMETERS::T motor_commands[4], RNG& rng, typename PARAMETERS::T controller_dt){
            using T = typename PARAMETERS::T;
            using TI = typename DEVICE::index_t;
            static_assert(PARAMETERS::N == 4);
            T normalized[4];
            for(TI action_i = 0; action_i < 4; action_i++){
                normalized[action_i] = math::clamp(device.math, get(action, 0, action_i), -(T)1, (T)1);
                normalized[action_i] += random::normal_distribution::sample(typename DEVICE::SPEC::RANDOM(), (T)0, parameters.mdp.action_noise.normalized_rpm, rng);
                normalized[action_i] = math::clamp(device.math, normalized[action_i], -(T)1, (T)1);
            }
            T collective = (normalized[0] + (T)1) / (T)2;
            collective = parameters.ctbr_controller.thrust_min + collective * (parameters.ctbr_controller.thrust_max - parameters.ctbr_controller.thrust_min);
            collective = math::clamp(device.math, collective, (T)0, (T)1);
            T collective_command = parameters.dynamics.action_limit.min + collective * (parameters.dynamics.action_limit.max - parameters.dynamics.action_limit.min);
            T total_thrust = 0;
            for(TI rotor_i = 0; rotor_i < 4; rotor_i++){
                total_thrust += parameters.dynamics.rotor_thrust_directions[rotor_i][2] * rotor_thrust_from_command(device, parameters, rotor_i, collective_command);
            }
            T rate_error[3];
            T angular_acceleration_measured[3];
            T desired_angular_acceleration[3];
            for(TI axis_i = 0; axis_i < 3; axis_i++){
                T rate_setpoint = normalized[axis_i + 1] * parameters.ctbr_controller.rate_limit[axis_i];
                rate_error[axis_i] = rate_setpoint - state.angular_velocity[axis_i];
                angular_acceleration_measured[axis_i] = (state.angular_velocity[axis_i] - state.previous_angular_velocity[axis_i]) / controller_dt;
                desired_angular_acceleration[axis_i] = parameters.ctbr_controller.kp[axis_i] * rate_error[axis_i] - parameters.ctbr_controller.kd[axis_i] * angular_acceleration_measured[axis_i];
            }
            T torque[3];
            rl_tools::utils::vector_operations::matrix_vector_product<DEVICE, T, 3, 3>(parameters.dynamics.J, desired_angular_acceleration, torque);
            for(TI axis_i = 0; axis_i < 3; axis_i++){
                T limit = parameters.ctbr_controller.torque_limit[axis_i];
                if(limit > 0){
                    torque[axis_i] = math::clamp(device.math, torque[axis_i], -limit, limit);
                }
            }
            T A[4][4];
            for(TI rotor_i = 0; rotor_i < 4; rotor_i++){
                T thrust_direction[3] = {
                    parameters.dynamics.rotor_thrust_directions[rotor_i][0],
                    parameters.dynamics.rotor_thrust_directions[rotor_i][1],
                    parameters.dynamics.rotor_thrust_directions[rotor_i][2]
                };
                T torque_per_thrust[3];
                torque_per_thrust[0] = parameters.dynamics.rotor_torque_directions[rotor_i][0] * parameters.dynamics.rotor_torque_constants[rotor_i] + parameters.dynamics.rotor_positions[rotor_i][1] * thrust_direction[2] - parameters.dynamics.rotor_positions[rotor_i][2] * thrust_direction[1];
                torque_per_thrust[1] = parameters.dynamics.rotor_torque_directions[rotor_i][1] * parameters.dynamics.rotor_torque_constants[rotor_i] + parameters.dynamics.rotor_positions[rotor_i][2] * thrust_direction[0] - parameters.dynamics.rotor_positions[rotor_i][0] * thrust_direction[2];
                torque_per_thrust[2] = parameters.dynamics.rotor_torque_directions[rotor_i][2] * parameters.dynamics.rotor_torque_constants[rotor_i] + parameters.dynamics.rotor_positions[rotor_i][0] * thrust_direction[1] - parameters.dynamics.rotor_positions[rotor_i][1] * thrust_direction[0];
                A[0][rotor_i] = thrust_direction[2];
                A[1][rotor_i] = torque_per_thrust[0];
                A[2][rotor_i] = torque_per_thrust[1];
                A[3][rotor_i] = torque_per_thrust[2];
            }
            T b[4] = {total_thrust, torque[0], torque[1], torque[2]};
            T rotor_thrusts[4];
            bool solved = solve_4x4(device, A, b, rotor_thrusts);
            rl_tools::utils::assert_exit(device, solved, "allocation matrix not solved");
            for(TI rotor_i = 0; rotor_i < 4; rotor_i++){
                motor_commands[rotor_i] = solved ? rotor_command_from_thrust(device, parameters, rotor_i, rotor_thrusts[rotor_i]) : collective_command;
            }
        }
        // normalized [thrust, rates] into collective acceleration and body rates; the rates are written into the state because the ideal model tracks them instantly
        template<typename DEVICE, typename SPEC, typename PARAMETERS, typename STATE, typename ACTION_SPEC, typename RNG>
        RL_TOOLS_FUNCTION_PLACEMENT void ideal_rates_actuation(DEVICE& device, const Multirotor<SPEC>&, const PARAMETERS& parameters, STATE& state, const Matrix<ACTION_SPEC>& action, typename SPEC::T actuation[4], RNG& rng){
            using T = typename SPEC::T;
            using TI = typename DEVICE::index_t;
            T normalized[4];
            for(TI action_i = 0; action_i < 4; action_i++){
                normalized[action_i] = math::clamp(device.math, get(action, 0, action_i), -(T)1, (T)1);
                normalized[action_i] += random::normal_distribution::sample(typename DEVICE::SPEC::RANDOM(), (T)0, parameters.mdp.action_noise.normalized_rpm, rng);
                normalized[action_i] = math::clamp(device.math, normalized[action_i], -(T)1, (T)1);
            }
            actuation[0] = ideal_rates_collective_acceleration(device, parameters, normalized[0]);
            for(TI axis_i = 0; axis_i < 3; axis_i++){
                actuation[axis_i + 1] = normalized[axis_i + 1] * parameters.ideal_rates.rate_limit[axis_i];
                state.angular_velocity[axis_i] = actuation[axis_i + 1];
            }
        }
        template<typename DEVICE, typename SPEC, typename PARAMETERS, typename STATE>
        RL_TOOLS_FUNCTION_PLACEMENT void integrate(DEVICE& device, const Multirotor<SPEC>&, const PARAMETERS& parameters, const STATE& state, const typename SPEC::T actuation[4], STATE& next_state){
            using MULTIROTOR = Multirotor<SPEC>;
            using T = typename SPEC::T;
            using TI = typename DEVICE::index_t;
            constexpr TI N_SUBSTEPS = SPEC::STATIC_PARAMETERS::N_SUBSTEPS;
            if constexpr(N_SUBSTEPS == 1){
                utils::integrators::rk4<DEVICE, T, typename SPEC::PARAMETERS, STATE, MULTIROTOR::ACTION_DIM, multirotor_dynamics_dispatch<DEVICE, T, typename SPEC::PARAMETERS, STATE, typename MULTIROTOR::ACTUATION_MODEL>>(device, parameters, state, actuation, parameters.integration.dt, next_state);
            }
            else{
                T substep_dt = parameters.integration.dt / N_SUBSTEPS;
                auto substep_state = state;
                for(TI substep_i = 0; substep_i < N_SUBSTEPS; substep_i++){
                    utils::integrators::rk4<DEVICE, T, typename SPEC::PARAMETERS, STATE, MULTIROTOR::ACTION_DIM, multirotor_dynamics_dispatch<DEVICE, T, typename SPEC::PARAMETERS, STATE, typename MULTIROTOR::ACTUATION_MODEL>>(device, parameters, substep_state, actuation, substep_dt, next_state);
                    substep_state = next_state;
                }
            }
        }
    }
    template<typename DEVICE, typename SPEC, typename PARAMETERS, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT static void sample_initial_parameters(DEVICE& device, rl::environments::Multirotor<SPEC>& env, PARAMETERS& parameters, RNG& rng){
        // to allow out of declaration order dispatch
        rl::environments::l2f::_sample_initial_parameters(device, env, parameters, rng);
    }
    template<typename DEVICE, typename SPEC, typename PARAMETERS, typename STATE>
    RL_TOOLS_FUNCTION_PLACEMENT static void initial_state(DEVICE& device, rl::environments::Multirotor<SPEC>& env, PARAMETERS& parameters, STATE& state){
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
    template<typename DEVICE, typename SPEC, typename PARAMETERS, typename STATE, typename ACTION_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT static typename SPEC::T step(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, PARAMETERS& parameters, const STATE& state, const Matrix<ACTION_SPEC>& action, STATE& next_state, RNG& rng) {
        using MULTIROTOR = rl::environments::Multirotor<SPEC>;
        using T = typename SPEC::T;
        using TI = typename DEVICE::index_t;
        static_assert(ACTION_SPEC::ROWS == 1);
        static_assert(ACTION_SPEC::COLS == MULTIROTOR::ACTION_DIM);
        T actuation[MULTIROTOR::ACTION_DIM];
        if constexpr(MULTIROTOR::RATE_CONTROLLER){
            constexpr TI N_SUBSTEPS = SPEC::STATIC_PARAMETERS::N_SUBSTEPS;
            T substep_dt = parameters.integration.dt / N_SUBSTEPS;
            auto substep_state = state;
            for(TI substep_i = 0; substep_i < N_SUBSTEPS; substep_i++){
                rl::environments::l2f::ctbr_action_to_motor_commands(device, parameters, substep_state, action, actuation, rng, substep_dt);
                utils::integrators::rk4<DEVICE, T, typename SPEC::PARAMETERS, STATE, MULTIROTOR::ACTION_DIM, rl::environments::l2f::multirotor_dynamics_dispatch<DEVICE, T, typename SPEC::PARAMETERS, STATE, rl::environments::l2f::actuation::Rotors>>(device, parameters, substep_state, actuation, substep_dt, next_state);
                for(TI axis_i = 0; axis_i < 3; axis_i++){
                    next_state.previous_angular_velocity[axis_i] = substep_state.angular_velocity[axis_i];
                }
                substep_state = next_state;
            }
        }
        else if constexpr(utils::typing::is_same_v<typename MULTIROTOR::ACTUATION_MODEL, rl::environments::l2f::actuation::IdealRates>){
            auto commanded = state;
            rl::environments::l2f::ideal_rates_actuation(device, env, parameters, commanded, action, actuation, rng);
            rl::environments::l2f::integrate(device, env, parameters, commanded, actuation, next_state);
        }
        else{
            rl::environments::l2f::direct_motor_action_to_motor_commands(device, parameters, state, action, actuation, rng);
            rl::environments::l2f::integrate(device, env, parameters, state, actuation, next_state);
        }
        post_integration(device, env, parameters, state, action, next_state, rng);
        return parameters.integration.dt;
    }
    namespace rl::environments::l2f{
        template<typename STATE>
        RL_TOOLS_FUNCTION_PLACEMENT constexpr bool is_pose_error_integral(const STATE&){
            return false;
        }
        template<typename SPEC>
        RL_TOOLS_FUNCTION_PLACEMENT constexpr bool is_pose_error_integral(const StatePoseErrorIntegral<SPEC>&){
            return true;
        }
    }

    template<typename DEVICE, typename SPEC, typename PARAMETERS, typename STATE, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT static bool terminated(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, const PARAMETERS& parameters, const STATE& state, RNG& rng){
        using T = typename SPEC::T;
        using TI = typename DEVICE::index_t;
        if(parameters.mdp.termination.enabled){
            STATE desired_state;
            get_desired_state(device, env, parameters, state, desired_state, rng);
            for(TI i = 0; i < 3; i++){
                if(
                    math::abs(device.math, state.position[i] - desired_state.position[i]) > parameters.mdp.termination.position_threshold ||
                    math::abs(device.math, state.linear_velocity[i] - desired_state.linear_velocity[i]) > parameters.mdp.termination.linear_velocity_threshold ||
                    math::abs(device.math, state.angular_velocity[i]) > parameters.mdp.termination.angular_velocity_threshold
                ){
                    return true;
                }
            }
            if(parameters.mdp.termination.angle_threshold > 0){
                T w = math::clamp(device.math, math::abs(device.math, state.orientation[0]), (T)0, (T)1);
                T angle = 2 * math::acos(device.math, w);
                if(angle > parameters.mdp.termination.angle_threshold){
                    return true;
                }
            }
        }
        if constexpr(rl::environments::l2f::is_pose_error_integral(STATE{})){
            // if(state.position_integral > parameters.mdp.termination.position_integral_threshold){
            //     return true;
            // }
            // if(state.orientation_integral > parameters.mdp.termination.orientation_integral_threshold){
            //     return true;
            // }
        }
        return false;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#include "parameters/reward_functions/reward_functions.h"
#include "parameters/reward_functions/squared/operations_generic.h" // such that terminated can be called from rwd functions
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
