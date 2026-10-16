#include "../../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_L2F_OPERATIONS_GENERIC_DYNAMICS_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_L2F_OPERATIONS_GENERIC_DYNAMICS_H

#include "../multirotor.h"

#include <rl_tools/utils/generic/vector_operations.h>
#include "../quaternion_helper.h"

#include <rl_tools/utils/generic/typing.h>

#include <rl_tools/rl/environments/operations_generic.h>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::environments::l2f{
    template<typename DEVICE, typename PARAMETERS, typename T>
    RL_TOOLS_FUNCTION_PLACEMENT void angular_acceleration(DEVICE&, actuation::Rotors, const PARAMETERS& params, const T torque[3], T acceleration[3]){
        rl_tools::utils::vector_operations::matrix_vector_product<DEVICE, T, 3, 3>(params.dynamics.J_inv, torque, acceleration);
    }
    // ideal rate tracking rejects every torque, rotor-generated or disturbance
    template<typename DEVICE, typename PARAMETERS, typename T>
    RL_TOOLS_FUNCTION_PLACEMENT void angular_acceleration(DEVICE&, actuation::IdealRates, const PARAMETERS&, const T[3], T acceleration[3]){
        acceleration[0] = 0;
        acceleration[1] = 0;
        acceleration[2] = 0;
    }
    // The two StateBase overloads repeat the translational and attitude kinematics on purpose: a shared helper stays out of line and costs about 1.5 ns per RK4 stage (7% of a motor step).
    template<typename DEVICE, typename STATE_SPEC, typename PARAMETERS, typename T>
    RL_TOOLS_FUNCTION_PLACEMENT void multirotor_dynamics(DEVICE& device, const PARAMETERS& params, const StateBase<STATE_SPEC>& state, const T* action, StateBase<STATE_SPEC>& state_change, actuation::Rotors actuation) {
        T thrust[3] = {0, 0, 0};
        T torque[3] = {0, 0, 0};
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

        state_change.position[0] = state.linear_velocity[0];
        state_change.position[1] = state.linear_velocity[1];
        state_change.position[2] = state.linear_velocity[2];

        // flops: 16
        quaternion_derivative<DEVICE, T>(state.orientation, state.angular_velocity, state_change.orientation);

        // flops: 21 + 4
        rotate_vector_by_quaternion<DEVICE, T>(state.orientation, thrust, state_change.linear_velocity);
        rl_tools::utils::vector_operations::scalar_multiply<DEVICE, T, 3>(state_change.linear_velocity, 1 / params.dynamics.mass);
        rl_tools::utils::vector_operations::add_accumulate<DEVICE, T, 3>(params.dynamics.gravity, state_change.linear_velocity);

        T vector[3];
        T vector2[3];
        // angular_acceleration_local: J_inv (torque - omega x J omega)
        // flops: 9 + 6 + 3 + 9
        rl_tools::utils::vector_operations::matrix_vector_product<DEVICE, T, 3, 3>(params.dynamics.J, state.angular_velocity, vector);
        rl_tools::utils::vector_operations::cross_product<DEVICE, T>(state.angular_velocity, vector, vector2);
        rl_tools::utils::vector_operations::sub<DEVICE, T, 3>(torque, vector2, vector);
        angular_acceleration(device, actuation, params, vector, state_change.angular_velocity);
        // total flops (quadrotor): 92 + 16 + 25 + 27 = 160
    }
    // action[0]: collective acceleration along body z; the prescribed body rates are already in the state
    template<typename DEVICE, typename STATE_SPEC, typename PARAMETERS, typename T>
    RL_TOOLS_FUNCTION_PLACEMENT void multirotor_dynamics(DEVICE& device, const PARAMETERS& params, const StateBase<STATE_SPEC>& state, const T* action, StateBase<STATE_SPEC>& state_change, actuation::IdealRates) {
        T acceleration[3] = {0, 0, action[0]};
        state_change.position[0] = state.linear_velocity[0];
        state_change.position[1] = state.linear_velocity[1];
        state_change.position[2] = state.linear_velocity[2];
        quaternion_derivative<DEVICE, T>(state.orientation, state.angular_velocity, state_change.orientation);
        rotate_vector_by_quaternion<DEVICE, T>(state.orientation, acceleration, state_change.linear_velocity);
        rl_tools::utils::vector_operations::add_accumulate<DEVICE, T, 3>(params.dynamics.gravity, state_change.linear_velocity);
        state_change.angular_velocity[0] = 0;
        state_change.angular_velocity[1] = 0;
        state_change.angular_velocity[2] = 0;
    }
    template<typename DEVICE, typename PARAMETERS, typename STATE_SPEC, typename T, typename ACTUATION>
    RL_TOOLS_FUNCTION_PLACEMENT void multirotor_dynamics(DEVICE& device, const PARAMETERS& params, const StatePoseErrorIntegral<STATE_SPEC>& state, const T* action, StatePoseErrorIntegral<STATE_SPEC>& state_change, ACTUATION actuation){
        using TI = typename DEVICE::index_t;
        multirotor_dynamics(device, params, static_cast<const typename STATE_SPEC::NEXT_COMPONENT&>(state), action, static_cast<typename STATE_SPEC::NEXT_COMPONENT&>(state_change), actuation);
        for (TI dim_i=0; dim_i < 3; dim_i++){
            state_change.position_integral[dim_i] = state.position[dim_i];
        }
        T w_clamped = math::clamp(device.math, state.orientation[0], (T)-1, (T)1);
        T orientation_error = 2*math::acos(device.math, w_clamped);
        state_change.orientation_integral = orientation_error;
    }
    template<typename DEVICE, typename PARAMETERS, typename STATE_SPEC, typename T, typename ACTUATION>
    RL_TOOLS_FUNCTION_PLACEMENT void multirotor_dynamics(DEVICE& device, const PARAMETERS& params, const StateRandomForce<STATE_SPEC>& state, const T* action, StateRandomForce<STATE_SPEC>& state_change, ACTUATION actuation){
        multirotor_dynamics(device, params, static_cast<const typename STATE_SPEC::NEXT_COMPONENT&>(state), action, static_cast<typename STATE_SPEC::NEXT_COMPONENT&>(state_change), actuation);

        state_change.linear_velocity[0] += state.force[0] / params.dynamics.mass;
        state_change.linear_velocity[1] += state.force[1] / params.dynamics.mass;
        state_change.linear_velocity[2] += state.force[2] / params.dynamics.mass;

        T disturbance[3];
        angular_acceleration(device, actuation, params, state.torque, disturbance);
        rl_tools::utils::vector_operations::add_accumulate<DEVICE, T, 3>(disturbance, state_change.angular_velocity);
    }
    template<typename DEVICE, typename PARAMETERS, typename STATE_SPEC, typename T>
    RL_TOOLS_FUNCTION_PLACEMENT void multirotor_dynamics(DEVICE& device, const PARAMETERS& params, const StateRotors<STATE_SPEC>& state, const T* action, StateRotors<STATE_SPEC>& state_change, actuation::Rotors actuation) {
        multirotor_dynamics(device, params, static_cast<const typename STATE_SPEC::NEXT_COMPONENT&>(state), state.rpm, static_cast<typename STATE_SPEC::NEXT_COMPONENT&>(state_change), actuation);

        if constexpr(!STATE_SPEC::CLOSED_FORM) {
            for(typename DEVICE::index_t i_rotor = 0; i_rotor < 4; i_rotor++){
                T tau = action[i_rotor] >= state.rpm[i_rotor] ? params.dynamics.rotor_time_constants_rising[i_rotor] : params.dynamics.rotor_time_constants_falling[i_rotor] ;
                state_change.rpm[i_rotor] = (action[i_rotor] - state.rpm[i_rotor]) * 1/tau;
            }
        }
    }
    // rotor state is synthesized from the collective after integration (post_integration), not integrated
    template<typename DEVICE, typename PARAMETERS, typename STATE_SPEC, typename T>
    RL_TOOLS_FUNCTION_PLACEMENT void multirotor_dynamics(DEVICE& device, const PARAMETERS& params, const StateRotors<STATE_SPEC>& state, const T* action, StateRotors<STATE_SPEC>& state_change, actuation::IdealRates actuation) {
        multirotor_dynamics(device, params, static_cast<const typename STATE_SPEC::NEXT_COMPONENT&>(state), action, static_cast<typename STATE_SPEC::NEXT_COMPONENT&>(state_change), actuation);
        for(typename DEVICE::index_t i_rotor = 0; i_rotor < 4; i_rotor++){
            state_change.rpm[i_rotor] = 0;
        }
    }
    template<typename DEVICE, typename PARAMETERS, typename STATE, typename T, typename ACTUATION>
    RL_TOOLS_FUNCTION_PLACEMENT void multirotor_dynamics(DEVICE& device, const PARAMETERS& params, const STATE& state, const T* action, STATE& state_change, ACTUATION actuation){
        static_assert(!STATE::REQUIRES_INTEGRATION, "integrated state component without a multirotor_dynamics overload");
        multirotor_dynamics(device, params, static_cast<const typename STATE::NEXT_COMPONENT&>(state), action, static_cast<typename STATE::NEXT_COMPONENT&>(state_change), actuation);
    }
    template<typename DEVICE, typename T, typename PARAMETERS, typename STATE, typename ACTUATION>
    RL_TOOLS_FUNCTION_PLACEMENT void multirotor_dynamics_dispatch(DEVICE& device, const PARAMETERS& params, const STATE& state, const T* action, STATE& state_change) {
        // the integrator takes the integrand as a template parameter, which an overload set cannot be
        multirotor_dynamics(device, params, state, action, state_change, ACTUATION{});
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
