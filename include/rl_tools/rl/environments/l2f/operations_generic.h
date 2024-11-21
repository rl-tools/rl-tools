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
    // State arithmetic for RK4 integration
    // scalar multiply
    template<typename DEVICE, typename T, typename TI, typename T2>
    RL_TOOLS_FUNCTION_PLACEMENT static void scalar_multiply(DEVICE& device, const typename rl::environments::l2f::StateBase<T, TI>& state, T2 scalar, typename rl::environments::l2f::StateBase<T, TI>& out){
        for(int i = 0; i < 3; ++i){
            out.position[i]         = scalar * state.position[i]        ;
            out.orientation[i]      = scalar * state.orientation[i]     ;
            out.linear_velocity[i]  = scalar * state.linear_velocity[i] ;
            out.angular_velocity[i] = scalar * state.angular_velocity[i];
        }
        out.orientation[3] = scalar * state.orientation[3];
    }
    template<typename DEVICE, typename T, typename TI, typename NEXT_COMPONENT, typename T2>
    RL_TOOLS_FUNCTION_PLACEMENT static void scalar_multiply(DEVICE& device, const typename rl::environments::l2f::StatePoseErrorIntegral<T, TI, NEXT_COMPONENT>& state, T2 scalar, typename rl::environments::l2f::StatePoseErrorIntegral<T, TI, NEXT_COMPONENT>& out){
        scalar_multiply(device, (const NEXT_COMPONENT&)state, scalar, (NEXT_COMPONENT&)out);
        out.position_integral = scalar * out.position_integral;
        out.orientation_integral = scalar * out.orientation_integral;
    }
    template<typename DEVICE, typename T, typename TI, typename T2, bool T_CLOSED_FORM, typename NEXT_COMPONENT>
    RL_TOOLS_FUNCTION_PLACEMENT static void scalar_multiply(DEVICE& device, const typename rl::environments::l2f::StateRotors<T, TI, T_CLOSED_FORM, NEXT_COMPONENT>& state, T2 scalar, typename rl::environments::l2f::StateRotors<T, TI, T_CLOSED_FORM, NEXT_COMPONENT>& out){
        scalar_multiply(device, (const NEXT_COMPONENT&)state, scalar, (NEXT_COMPONENT&)out);
        if constexpr(!T_CLOSED_FORM){
            for(int i = 0; i < 4; ++i){
                out.rpm[i] = scalar * state.rpm[i];
            }
        }
    }
    template<typename DEVICE, typename STATE, typename T2>
    RL_TOOLS_FUNCTION_PLACEMENT static void scalar_multiply(DEVICE& device, const STATE& state, T2 scalar, STATE& out, utils::typing::enable_if_t<!STATE::REQUIRES_INTEGRATION, bool> disable = false){
        static_assert(!STATE::REQUIRES_INTEGRATION);
        scalar_multiply(device, (const typename STATE::NEXT_COMPONENT&)state, scalar, (typename STATE::NEXT_COMPONENT&)out);
    }
    // scalar multiply in place
    template<typename DEVICE, typename T, typename TI, typename T2>
    RL_TOOLS_FUNCTION_PLACEMENT static void scalar_multiply(DEVICE& device, typename rl::environments::l2f::StateBase<T, TI>& state, T2 scalar){
        scalar_multiply(device, state, scalar, state);
    }
    template<typename DEVICE, typename T, typename TI, typename NEXT_COMPONENT, typename T2>
    RL_TOOLS_FUNCTION_PLACEMENT static void scalar_multiply(DEVICE& device, typename rl::environments::l2f::StatePoseErrorIntegral<T, TI, NEXT_COMPONENT>& state, T2 scalar){
        scalar_multiply(device, state, scalar, state);
    }
    template<typename DEVICE, typename T, typename TI, typename T2, bool T_CLOSED_FORM, typename NEXT_COMPONENT>
    RL_TOOLS_FUNCTION_PLACEMENT static void scalar_multiply(DEVICE& device, typename rl::environments::l2f::StateRotors<T, TI, T_CLOSED_FORM, NEXT_COMPONENT>& state, T2 scalar){
        scalar_multiply(device, state, scalar, state);
    }
    template<typename DEVICE, typename STATE, typename T2>
    RL_TOOLS_FUNCTION_PLACEMENT static void scalar_multiply(DEVICE& device, STATE& state, T2 scalar, utils::typing::enable_if_t<!STATE::REQUIRES_INTEGRATION, bool> disable = false){
        static_assert(!STATE::REQUIRES_INTEGRATION);
        scalar_multiply(device, (typename STATE::NEXT_COMPONENT&)state, scalar);
    }

    template<typename DEVICE, typename T, typename TI, typename T2>
    RL_TOOLS_FUNCTION_PLACEMENT static void scalar_multiply_accumulate(DEVICE& device, const typename rl::environments::l2f::StateBase<T, TI>& state, T2 scalar, typename rl::environments::l2f::StateBase<T, TI>& out){
        for(int i = 0; i < 3; ++i){
            out.position[i]         += scalar * state.position[i]        ;
            out.orientation[i]      += scalar * state.orientation[i]     ;
            out.linear_velocity[i]  += scalar * state.linear_velocity[i] ;
            out.angular_velocity[i] += scalar * state.angular_velocity[i];
        }
        out.orientation[3] += scalar * state.orientation[3];
    }
    template<typename DEVICE, typename T, typename TI, typename T2, typename NEXT_COMPONENT>
    RL_TOOLS_FUNCTION_PLACEMENT static void scalar_multiply_accumulate(DEVICE& device, const typename rl::environments::l2f::StatePoseErrorIntegral<T, TI, NEXT_COMPONENT>& state, T2 scalar, typename rl::environments::l2f::StatePoseErrorIntegral<T, TI, NEXT_COMPONENT>& out){
        scalar_multiply_accumulate(device, static_cast<const NEXT_COMPONENT&>(state), scalar, static_cast<NEXT_COMPONENT&>(out));
        out.position_integral += scalar * out.position_integral;
        out.orientation_integral += scalar * out.orientation_integral;
    }
    template<typename DEVICE, typename T, typename TI, typename T2, bool T_CLOSED_FORM, typename NEXT_COMPONENT>
    RL_TOOLS_FUNCTION_PLACEMENT static void scalar_multiply_accumulate(DEVICE& device, const typename rl::environments::l2f::StateRotors<T, TI, T_CLOSED_FORM, NEXT_COMPONENT>& state, T2 scalar, typename rl::environments::l2f::StateRotors<T, TI, T_CLOSED_FORM, NEXT_COMPONENT>& out){
        scalar_multiply_accumulate(device, static_cast<const NEXT_COMPONENT&>(state), scalar, static_cast<NEXT_COMPONENT&>(out));
        if constexpr(!T_CLOSED_FORM) {
            for(int i = 0; i < 4; ++i){
                out.rpm[i] += scalar * state.rpm[i];
            }
        }
    }
    template<typename DEVICE, typename STATE, typename T2>
    RL_TOOLS_FUNCTION_PLACEMENT static void scalar_multiply_accumulate(DEVICE& device, const STATE& state, T2 scalar, STATE& out, utils::typing::enable_if_t<!STATE::REQUIRES_INTEGRATION, bool> disable = false){
        static_assert(!STATE::REQUIRES_INTEGRATION);
        scalar_multiply_accumulate(device, static_cast<const typename STATE::NEXT_COMPONENT&>(state), scalar, static_cast<typename STATE::NEXT_COMPONENT&>(out));
    }

    template<typename DEVICE, typename T, typename TI>
    RL_TOOLS_FUNCTION_PLACEMENT static void add_accumulate(DEVICE& device, const typename rl::environments::l2f::StateBase<T, TI>& s1, const typename rl::environments::l2f::StateBase<T, TI>& s2, typename rl::environments::l2f::StateBase<T, TI>& out){
        for(int i = 0; i < 3; ++i){
            out.position[i]         = s1.position[i] + s2.position[i];
            out.orientation[i]      = s1.orientation[i] + s2.orientation[i];
            out.linear_velocity[i]  = s1.linear_velocity[i] + s2.linear_velocity[i];
            out.angular_velocity[i] = s1.angular_velocity[i] + s2.angular_velocity[i];
        }
        out.orientation[3] = s1.orientation[3] + s2.orientation[3];
    }
    template<typename DEVICE, typename T, typename TI, typename NEXT_COMPONENT>
    RL_TOOLS_FUNCTION_PLACEMENT static void add_accumulate(DEVICE& device, const typename rl::environments::l2f::StatePoseErrorIntegral<T, TI, NEXT_COMPONENT>& s1, const typename rl::environments::l2f::StatePoseErrorIntegral<T, TI, NEXT_COMPONENT>& s2, typename rl::environments::l2f::StatePoseErrorIntegral<T, TI, NEXT_COMPONENT>& out){
        add_accumulate(device, static_cast<const NEXT_COMPONENT&>(s1), static_cast<const NEXT_COMPONENT&>(s2), static_cast<NEXT_COMPONENT&>(out));
        out.position_integral = s1.position_integral + s2.position_integral;
        out.orientation_integral = s1.orientation_integral + s2.orientation_integral;
    }
    template<typename DEVICE, typename T, typename TI, bool T_CLOSED_FORM, typename NEXT_COMPONENT>
    RL_TOOLS_FUNCTION_PLACEMENT static void add_accumulate(DEVICE& device, const typename rl::environments::l2f::StateRotors<T, TI, T_CLOSED_FORM, NEXT_COMPONENT>& s1, const typename rl::environments::l2f::StateRotors<T, TI, T_CLOSED_FORM, NEXT_COMPONENT>& s2, typename rl::environments::l2f::StateRotors<T, TI, T_CLOSED_FORM, NEXT_COMPONENT>& out){
        add_accumulate(device, static_cast<const NEXT_COMPONENT&>(s1), static_cast<const NEXT_COMPONENT&>(s2), static_cast<NEXT_COMPONENT&>(out));
        if constexpr(!T_CLOSED_FORM) {
            for(int i = 0; i < 4; ++i){
                out.rpm[i] = s1.rpm[i] + s2.rpm[i];
            }
        }
    }
    template<typename DEVICE, typename STATE>
    RL_TOOLS_FUNCTION_PLACEMENT static void add_accumulate(DEVICE& device, const STATE& s1, const STATE& s2, STATE& out, utils::typing::enable_if_t<!STATE::REQUIRES_INTEGRATION, bool> disable = false){
        static_assert(!STATE::REQUIRES_INTEGRATION);
        add_accumulate(device, static_cast<const typename STATE::NEXT_COMPONENT&>(s1), static_cast<const typename STATE::NEXT_COMPONENT&>(s2), static_cast<typename STATE::NEXT_COMPONENT&>(out));
    }
    template<typename DEVICE, typename T, typename TI>
    RL_TOOLS_FUNCTION_PLACEMENT static void add_accumulate(DEVICE& device, const typename rl::environments::l2f::StateBase<T, TI>& s, typename rl::environments::l2f::StateBase<T, TI>& out){
        add_accumulate(device, s, out, out);
    }
    template<typename DEVICE, typename T, typename TI, typename NEXT_COMPONENT>
    RL_TOOLS_FUNCTION_PLACEMENT static void add_accumulate(DEVICE& device, const typename rl::environments::l2f::StatePoseErrorIntegral<T, TI, NEXT_COMPONENT>& s, typename rl::environments::l2f::StatePoseErrorIntegral<T, TI, NEXT_COMPONENT>& out){
        add_accumulate(device, s, out, out);
    }
    template<typename DEVICE, typename T, typename TI, bool T_CLOSED_FORM, typename NEXT_COMPONENT>
    RL_TOOLS_FUNCTION_PLACEMENT static void add_accumulate(DEVICE& device, const typename rl::environments::l2f::StateRotors<T, TI, T_CLOSED_FORM, NEXT_COMPONENT>& s, typename rl::environments::l2f::StateRotors<T, TI, T_CLOSED_FORM, NEXT_COMPONENT>& out){
        add_accumulate(device, s, out, out);
    }
    template<typename DEVICE, typename STATE>
    RL_TOOLS_FUNCTION_PLACEMENT static void add_accumulate(DEVICE& device, const STATE& s, STATE& out, utils::typing::enable_if_t<!STATE::REQUIRES_INTEGRATION, bool> disable = false){
        static_assert(!STATE::REQUIRES_INTEGRATION);
        add_accumulate(device, static_cast<const typename STATE::NEXT_COMPONENT&>(s), static_cast<const typename STATE::NEXT_COMPONENT&>(out), static_cast<typename STATE::NEXT_COMPONENT&>(out));
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#include <rl_tools/utils/generic/integrators.h>


RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::environments::l2f {
    template<typename DEVICE, typename T, typename TI, typename PARAMETERS>
    RL_TOOLS_FUNCTION_PLACEMENT void multirotor_dynamics(DEVICE& device, const PARAMETERS& params, const StateBase<T, TI>& state, const T* action, StateBase<T, TI>& state_change) {
        using STATE = StateBase<T, TI>;

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
            T thrust_magnitude = params.dynamics.rotor_thrust_coefficients[0] + params.dynamics.rotor_thrust_coefficients[1] * rpm + params.dynamics.rotor_thrust_coefficients[2] * rpm * rpm;
            T rotor_thrust[3];
            rl_tools::utils::vector_operations::scalar_multiply<DEVICE, T, 3>(params.dynamics.rotor_thrust_directions[i_rotor], thrust_magnitude, rotor_thrust);
            rl_tools::utils::vector_operations::add_accumulate<DEVICE, T, 3>(rotor_thrust, thrust);

            rl_tools::utils::vector_operations::scalar_multiply_accumulate<DEVICE, T, 3>(params.dynamics.rotor_torque_directions[i_rotor], thrust_magnitude * params.dynamics.rotor_torque_constant, torque);
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
    template<typename DEVICE, typename T, typename TI, typename PARAMETERS, typename NEXT_COMPONENT>
    RL_TOOLS_FUNCTION_PLACEMENT void multirotor_dynamics(DEVICE& device, const PARAMETERS& params, const StatePoseErrorIntegral<T, TI, NEXT_COMPONENT>& state, const T* action, StatePoseErrorIntegral<T, TI, NEXT_COMPONENT>& state_change){
        multirotor_dynamics(device, params, static_cast<const NEXT_COMPONENT&>(state), action, static_cast<NEXT_COMPONENT&>(state_change));
        T position_error = state.position[0] * state.position[0] + state.position[1] * state.position[1] + state.position[2] * state.position[2];
        position_error = math::sqrt(device.math, position_error);
        T w_clamped = math::clamp(device.math, state.orientation[0], (T)-1, (T)1);
        T orientation_error = 2*math::acos(device.math, w_clamped);
        state_change.position_integral = position_error;
        state_change.orientation_integral = orientation_error;
    }
    template<typename DEVICE, typename T, typename TI, typename PARAMETERS, typename NEXT_COMPONENT>
    RL_TOOLS_FUNCTION_PLACEMENT void multirotor_dynamics(DEVICE& device, const PARAMETERS& params, const StateRandomForce<T, TI, NEXT_COMPONENT>& state, const T* action, StateRandomForce<T, TI, NEXT_COMPONENT>& state_change){
        multirotor_dynamics(device, params, static_cast<const NEXT_COMPONENT&>(state), action, static_cast<NEXT_COMPONENT&>(state_change));

        state_change.linear_velocity[0] += state.force[0] / params.dynamics.mass;
        state_change.linear_velocity[1] += state.force[1] / params.dynamics.mass;
        state_change.linear_velocity[2] += state.force[2] / params.dynamics.mass;

        T angular_acceleration[3];

        rl_tools::utils::vector_operations::matrix_vector_product<DEVICE, T, 3, 3>(params.dynamics.J_inv, state.torque, angular_acceleration);
        rl_tools::utils::vector_operations::add_accumulate<DEVICE, T, 3>(angular_acceleration, state_change.angular_velocity);
    }
    template<typename DEVICE, typename T, typename TI, bool T_CLOSED_FORM, typename NEXT_COMPONENT, typename PARAMETERS>
    RL_TOOLS_FUNCTION_PLACEMENT void multirotor_dynamics(DEVICE& device, const PARAMETERS& params, const StateRotors<T, TI, T_CLOSED_FORM, NEXT_COMPONENT>& state, const T* action, StateRotors<T, TI, T_CLOSED_FORM, NEXT_COMPONENT>& state_change) {
        multirotor_dynamics(device, params, static_cast<const NEXT_COMPONENT&>(state), state.rpm, static_cast<NEXT_COMPONENT&>(state_change));

        if constexpr(!T_CLOSED_FORM) {
            for(typename DEVICE::index_t i_rotor = 0; i_rotor < 4; i_rotor++){
                state_change.rpm[i_rotor] = (action[i_rotor] - state.rpm[i_rotor]) * 1/params.dynamics.motor_time_constant;
            }
        }

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
namespace rl_tools{
    template<typename DEVICE, typename SPEC>
    void malloc(DEVICE&, rl::environments::Multirotor<SPEC>& env){
    }
    template<typename DEVICE, typename SPEC>
    void free(DEVICE&, rl::environments::Multirotor<SPEC>&){ }
    template<typename DEVICE, typename SPEC>
    void init(DEVICE&, rl::environments::Multirotor<SPEC>& env){
        env.parameters = SPEC::STATIC_PARAMETERS::PARAMETER_VALUES;
    }
//    template<typename DEVICE, typename SPEC>
//    void init(DEVICE&, rl::environments::Multirotor<SPEC>& env, typename rl::environments::Multirotor<SPEC>::Parameters& parameters){
//        env.parameters = parameters;
//    }
    template<typename DEVICE, typename SPEC>
    static void initial_parameters(DEVICE& device, rl::environments::Multirotor<SPEC>& env, typename rl::environments::Multirotor<SPEC>::Parameters& parameters){
        parameters = env.parameters;
//        parameters = SPEC::STATIC_PARAMETERS::PARAMETER_VALUES;
    }
    namespace rl::environments::l2f {
        template <typename T, typename DEVICE, typename RNG>
        T sample_domain_randomization_factor(DEVICE& device, T range, RNG& rng) {
            T factor = random::normal_distribution::sample(device.random, -range, range, rng);
            factor = factor < 0 ? 1/(1-factor) : 1+factor; // reciprocal scale, note 1-factor because factor is negative in that case anyways
            return factor;
        }
    }
    template<typename DEVICE, typename SPEC, typename RNG>
    static void sample_initial_parameters(DEVICE& device, rl::environments::Multirotor<SPEC>& env, typename rl::environments::Multirotor<SPEC>::Parameters& parameters, RNG& rng, bool reset = true){
        using T = typename SPEC::T;
        using TI = typename DEVICE::index_t;
        if(reset){
            initial_parameters(device, env, parameters);
        }
        /*
         *  Strategy:
         *  1. Sample Thrust to Weight
         *  2. Sample Mass
         *  3. Calculate resulting thrust curve (based on max input)
         *  4. Get torque to inertia based on the scale (based on the sampled mass)
         *  5. Sample new torque_to_inertia around calculated one
         *  6. Sample a new size based on the scale (based on the sampled mass)
         *      a. Adjust the rotor positions
         *  7. Adjust inertia to fit the sampled torque to inertia ratio
         */
        T max_action = parameters.dynamics.action_limit.max;
        T max_thrust_nominal = parameters.dynamics.rotor_thrust_coefficients[0] + parameters.dynamics.rotor_thrust_coefficients[1] * max_action + parameters.dynamics.rotor_thrust_coefficients[2] * max_action * max_action;
        T gravity_norm = math::sqrt(device.math, parameters.dynamics.gravity[0] * parameters.dynamics.gravity[0] + parameters.dynamics.gravity[1] * parameters.dynamics.gravity[1] + parameters.dynamics.gravity[2] * parameters.dynamics.gravity[2]);
        T thrust_to_weight_nominal = 4 * max_thrust_nominal / (parameters.dynamics.mass * gravity_norm); // this assumes all the rotors are pointing into the same direction

        T thrust_to_weight = thrust_to_weight_nominal;
        T factor_thrust_to_weight = 1;
        if(parameters.domain_randomization.thrust_to_weight_min != 0){
            thrust_to_weight = random::uniform_real_distribution(device.random, parameters.domain_randomization.thrust_to_weight_min, parameters.domain_randomization.thrust_to_weight_max, rng);
            factor_thrust_to_weight = thrust_to_weight / thrust_to_weight_nominal;
        }

        T factor_mass = 1;
        T scale_absolute = 1;
        T scale_relative = 1;
        if(parameters.domain_randomization.mass_min != 0) {
            T mass_new = random::uniform_real_distribution(device.random, parameters.domain_randomization.mass_min, parameters.domain_randomization.mass_max, rng);
            scale_relative = math::cbrt(device.math, mass_new/parameters.dynamics.mass);
            scale_absolute = math::cbrt(device.math, mass_new); // thrust_to_weight_by_torque_to_inertia is defined wrt. to the crazyflie
            factor_mass = mass_new / parameters.dynamics.mass;
            parameters.dynamics.mass = mass_new;
        }
        T factor_thrust_coefficients = factor_thrust_to_weight * factor_mass;
        for(TI order_i = 0; order_i < 3; order_i++){
            parameters.dynamics.rotor_thrust_coefficients[order_i] *= factor_thrust_coefficients;
        }
        T max_thrust = parameters.dynamics.rotor_thrust_coefficients[0] + parameters.dynamics.rotor_thrust_coefficients[1] * max_action + parameters.dynamics.rotor_thrust_coefficients[2] * max_action * max_action;
        T first_rotor_distance_nominal = math::sqrt(device.math, parameters.dynamics.rotor_positions[0][0] * parameters.dynamics.rotor_positions[0][0] + parameters.dynamics.rotor_positions[0][1] * parameters.dynamics.rotor_positions[0][1] + parameters.dynamics.rotor_positions[0][2] * parameters.dynamics.rotor_positions[0][2]);
        T max_torque = first_rotor_distance_nominal * 1.414213562373095 * max_thrust; // 2/sqrt(2) = sqrt(2): max thrust assuming all rotors have equal angles and the same distance to the center two rotors active
        T x_inertia = parameters.dynamics.J[0][0];
        T torque_to_inertia_nominal = max_torque / x_inertia;
        // T thrust_to_weight_by_torque_to_inertia_upper = 0.7  * scale_absolute;
        // T thrust_to_weight_by_torque_to_inertia_lower = 0.300 * (scale_absolute-1.0);

        T torque_to_inertia_factor = 1;
        if(parameters.domain_randomization.thrust_to_weight_by_torque_to_inertia_min != 0) {
            T thrust_to_weight_by_torque_to_inertia = random::uniform_real_distribution(device.random, parameters.domain_randomization.thrust_to_weight_by_torque_to_inertia_min, parameters.domain_randomization.thrust_to_weight_by_torque_to_inertia_max, rng);
            T torque_to_inertia = thrust_to_weight / thrust_to_weight_by_torque_to_inertia;
            torque_to_inertia_factor = torque_to_inertia / torque_to_inertia_nominal;
        }

        T rotor_distance_factor = 1;
        if(parameters.domain_randomization.mass_size_deviation != 0) {
            T size_factor = rl::environments::l2f::sample_domain_randomization_factor(device, parameters.domain_randomization.mass_size_deviation, rng);
            rotor_distance_factor = scale_relative * size_factor;
            parameters.mdp.termination.position_threshold *= size_factor * size_factor;
            if(parameters.mdp.termination.position_threshold < parameters.mdp.init.max_position * 2){
                parameters.mdp.termination.position_threshold = parameters.mdp.init.max_position * 2;
            }
        }
        T inertia_factor = torque_to_inertia_factor/rotor_distance_factor;

        for(TI axis_i = 0; axis_i < 3; axis_i++){
            parameters.dynamics.J[axis_i][axis_i] /= inertia_factor;
            parameters.dynamics.J_inv[axis_i][axis_i] *= inertia_factor;
            // todo sample I_yy and I_zz, I_xx is random already through the torque_to_inertia mechanism
        }
        for(TI rotor_i = 0; rotor_i < 4; rotor_i++){
            for(TI axis_i = 0; axis_i < 3; axis_i++){
                parameters.dynamics.rotor_positions[rotor_i][axis_i] *= rotor_distance_factor;
            }
        }
    }
    template<typename DEVICE, typename T, typename STATE_TI, typename SPEC>
    static void initial_state(DEVICE& device, rl::environments::Multirotor<SPEC>& env, typename rl::environments::Multirotor<SPEC>::Parameters& parameters, typename rl::environments::l2f::StateBase<T, STATE_TI>& state){
        using TI = typename DEVICE::index_t;
        using STATE = typename rl::environments::Multirotor<SPEC>::State;
        for(TI i = 0; i < 3; i++){
            state.position[i] = 0;
        }
        state.orientation[0] = 1;
        for(TI i = 1; i < 4; i++){
            state.orientation[i] = 0;
        }
        for(TI i = 0; i < 3; i++){
            state.linear_velocity[i] = 0;
        }
        for(TI i = 0; i < 3; i++){
            state.angular_velocity[i] = 0;
        }
    }
    template<typename DEVICE, typename T, typename TI, typename NEXT_COMPONENT, typename SPEC>
    static void initial_state(DEVICE& device, rl::environments::Multirotor<SPEC>& env, typename rl::environments::Multirotor<SPEC>::Parameters& parameters, typename rl::environments::l2f::StateLinearAcceleration<T, TI, NEXT_COMPONENT>& state){
        initial_state(device, env, parameters, static_cast<NEXT_COMPONENT&>(state));
        for(TI i = 0; i < 3; i++){
            state.linear_acceleration[i] = 0;
        }
    }
    template<typename DEVICE, typename T, typename TI, typename NEXT_COMPONENT, typename SPEC>
    static void initial_state(DEVICE& device, rl::environments::Multirotor<SPEC>& env, typename rl::environments::Multirotor<SPEC>::Parameters& parameters, typename rl::environments::l2f::StatePoseErrorIntegral<T, TI, NEXT_COMPONENT>& state){
        initial_state(device, env, parameters, static_cast<NEXT_COMPONENT&>(state));
        state.position_integral = 0;
        state.orientation_integral = 0;
    }
    template<typename DEVICE, typename T, typename TI, typename SPEC, typename NEXT_COMPONENT>
    static void initial_state(DEVICE& device, rl::environments::Multirotor<SPEC>& env, typename rl::environments::Multirotor<SPEC>::Parameters& parameters, typename rl::environments::l2f::StateRandomForce<T, TI, NEXT_COMPONENT>& state){
        initial_state(device, env, parameters, static_cast<NEXT_COMPONENT&>(state));
        state.force[0] = 0;
        state.force[1] = 0;
        state.force[2] = 0;
        state.torque[0] = 0;
        state.torque[1] = 0;
        state.torque[2] = 0;
    }
    template<typename DEVICE, typename T, typename TI, typename SPEC, bool T_CLOSED_FORM, typename NEXT_COMPONENT>
    static void initial_state(DEVICE& device, rl::environments::Multirotor<SPEC>& env, typename rl::environments::Multirotor<SPEC>::Parameters& parameters, typename rl::environments::l2f::StateRotors<T, TI, T_CLOSED_FORM, NEXT_COMPONENT>& state){
        initial_state(device, env, parameters, static_cast<NEXT_COMPONENT&>(state));
        for(typename DEVICE::index_t i = 0; i < 4; i++){
//            state.rpm[i] = (parameters.dynamics.action_limit.max - parameters.dynamics.action_limit.min) / 2 + parameters.dynamics.action_limit.min;
            state.rpm[i] = parameters.dynamics.hovering_throttle_relative * (parameters.dynamics.action_limit.max - parameters.dynamics.action_limit.min) + parameters.dynamics.action_limit.min;
        }
    }
    template<typename DEVICE, typename T, typename TI_H, TI_H HISTORY_LENGTH, typename SPEC, bool T_CLOSED_FORM, typename NEXT_COMPONENT>
    static void initial_state(DEVICE& device, rl::environments::Multirotor<SPEC>& env, typename rl::environments::Multirotor<SPEC>::Parameters& parameters, typename rl::environments::l2f::StateRotorsHistory<T, TI_H, HISTORY_LENGTH, T_CLOSED_FORM, NEXT_COMPONENT>& state){
        using TI = typename DEVICE::index_t;
        using MULTIROTOR = rl::environments::Multirotor<SPEC>;
        initial_state(device, env, parameters, static_cast<rl::environments::l2f::StateRotors<T, TI, T_CLOSED_FORM, NEXT_COMPONENT>&>(state));
        state.current_step = 0;
        for(TI step_i = 0; step_i < HISTORY_LENGTH; step_i++){
            for(TI action_i = 0; action_i < MULTIROTOR::ACTION_DIM; action_i++){
                state.action_history[step_i][action_i] = (state.rpm[action_i] - parameters.dynamics.action_limit.min) / (parameters.dynamics.action_limit.max - parameters.dynamics.action_limit.min) * 2 - 1;
            }
        }
    }
    template<typename DEVICE, typename T, typename TI>
    static bool is_nan(DEVICE& device, typename rl::environments::l2f::StateBase<T, TI>& state){
        bool nan = false;
        for(typename DEVICE::index_t i = 0; i < 3; i++){
            nan = nan || math::is_nan(device.math, state.position[i]);
        }
        for(typename DEVICE::index_t i = 0; i < 4; i++){
            nan = nan || math::is_nan(device.math, state.orientation[i]);
        }
        for(typename DEVICE::index_t i = 0; i < 3; i++){
            nan = nan || math::is_nan(device.math, state.linear_velocity[i]);
        }
        for(typename DEVICE::index_t i = 0; i < 3; i++){
            nan = nan || math::is_nan(device.math, state.angular_velocity[i]);
        }
        return nan;
    }
    template<typename DEVICE, typename T, typename TI, typename NEXT_COMPONENT>
    static bool is_nan(DEVICE& device, typename rl::environments::l2f::StatePoseErrorIntegral<T, TI, NEXT_COMPONENT>& state){
        is_nan(device, static_cast<NEXT_COMPONENT&>(state));
        bool nan = false;
        nan = nan || math::is_nan(device.math, state.position_integral);
        nan = nan || math::is_nan(device.math, state.orientation_integral);
        return nan;
    }
    template<typename DEVICE, typename T, typename TI, typename NEXT_COMPONENT>
    static bool is_nan(DEVICE& device, typename rl::environments::l2f::StateRandomForce<T, TI, NEXT_COMPONENT>& state){
        is_nan(device, static_cast<NEXT_COMPONENT&>(state));
        bool nan = false;
        nan = nan || math::is_nan(device.math, state.force[0]);
        nan = nan || math::is_nan(device.math, state.force[1]);
        nan = nan || math::is_nan(device.math, state.force[2]);
        nan = nan || math::is_nan(device.math, state.torque[0]);
        nan = nan || math::is_nan(device.math, state.torque[1]);
        nan = nan || math::is_nan(device.math, state.torque[2]);
        return nan;
    }
    template<typename DEVICE, typename T, typename TI, bool T_CLOSED_FORM, typename NEXT_COMPONENT>
    static bool is_nan(DEVICE& device, typename rl::environments::l2f::StateRotors<T, TI, T_CLOSED_FORM, NEXT_COMPONENT>& state){
        is_nan(device, static_cast<NEXT_COMPONENT&>(state));
        bool nan = false;
        for(typename DEVICE::index_t i = 0; i < 4; i++){
            nan = nan || math::is_nan(device.math, state.rpm[2]);
        }
        return nan;
    }
    template<typename DEVICE, typename T, typename TI_H, TI_H HISTORY_LENGTH, bool T_CLOSED_FORM, typename NEXT_COMPONENT>
    static bool is_nan(DEVICE& device, typename rl::environments::l2f::StateRotorsHistory<T, TI_H, HISTORY_LENGTH, T_CLOSED_FORM, NEXT_COMPONENT>& state){
        using STATE = typename rl::environments::l2f::StateRotorsHistory<T, TI_H, HISTORY_LENGTH, T_CLOSED_FORM, NEXT_COMPONENT>;
        using TI = typename DEVICE::index_t;
        is_nan(device, static_cast<rl::environments::l2f::StateRotors<T, TI, T_CLOSED_FORM, NEXT_COMPONENT>&>(state));
        bool nan = false;
        for(TI step_i = 0; step_i < HISTORY_LENGTH; step_i++){
            for(TI action_i = 0; action_i < STATE::ACTION_DIM; action_i++){
                nan = nan || math::is_nan(device.math, state.action_history[step_i][action_i]);
            }
        }
        return nan;
    }
    template<typename DEVICE, typename T, typename TI, typename SPEC, typename RNG, bool INHERIT_GUIDANCE = false>
    RL_TOOLS_FUNCTION_PLACEMENT static void sample_initial_state(DEVICE& device, rl::environments::Multirotor<SPEC>& env, typename rl::environments::Multirotor<SPEC>::Parameters& parameters, typename rl::environments::l2f::StateBase<T, TI>& state, RNG& rng, bool inherited_guidance = false){
        typename DEVICE::SPEC::MATH math_dev;
        typename DEVICE::SPEC::RANDOM random_dev;
        using STATE = typename rl::environments::l2f::StateBase<T, TI>;
        bool guidance;
        guidance = random::uniform_real_distribution(random_dev, (T)0, (T)1, rng) < parameters.mdp.init.guidance;
        if(!guidance){
            for(TI i = 0; i < 3; i++){
                state.position[i] = random::uniform_real_distribution(random_dev, -parameters.mdp.init.max_position, parameters.mdp.init.max_position, rng);
            }
        }
        else{
            for(TI i = 0; i < 3; i++){
                state.position[i] = 0;
            }
        }
        if(parameters.mdp.init.max_angle > 0 && !guidance){
            // https://web.archive.org/web/20181126051029/http://planning.cs.uiuc.edu/node198.html
            do{
                T u[3];
                for(TI i = 0; i < 3; i++){
                    u[i] = random::uniform_real_distribution(random_dev, (T)0, (T)1, rng);
                }
                state.orientation[0] = math::sqrt(math_dev, 1-u[0]) * math::sin(math_dev, 2*math::PI<T>*u[1]);
                state.orientation[1] = math::sqrt(math_dev, 1-u[0]) * math::cos(math_dev, 2*math::PI<T>*u[1]);
                state.orientation[2] = math::sqrt(math_dev,   u[0]) * math::sin(math_dev, 2*math::PI<T>*u[2]);
                state.orientation[3] = math::sqrt(math_dev,   u[0]) * math::cos(math_dev, 2*math::PI<T>*u[2]);
            } while(math::abs(math_dev, 2*math::acos(math_dev, math::abs(math_dev, state.orientation[0]))) > parameters.mdp.init.max_angle);
        }
        else{
            state.orientation[0] = 1;
            state.orientation[1] = 0;
            state.orientation[2] = 0;
            state.orientation[3] = 0;
        }
        if(!guidance) {
            for(TI i = 0; i < 3; i++){
                state.linear_velocity[i] = random::uniform_real_distribution(random_dev, -parameters.mdp.init.max_linear_velocity, parameters.mdp.init.max_linear_velocity, rng);
            }
            for(TI i = 0; i < 3; i++){
                state.angular_velocity[i] = random::uniform_real_distribution(random_dev, -parameters.mdp.init.max_angular_velocity, parameters.mdp.init.max_angular_velocity, rng);
            }
        }
        else{
            for(TI i = 0; i < 3; i++){
                state.linear_velocity[i] = 0;
            }
            for(TI i = 0; i < 3; i++){
                state.angular_velocity[i] = 0;
            }
        }
    }
    template<typename DEVICE, typename T_S, typename TI_S, typename SPEC, typename NEXT_COMPONENT, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT static void sample_initial_state(DEVICE& device, rl::environments::Multirotor<SPEC>& env, typename rl::environments::Multirotor<SPEC>::Parameters& parameters, typename rl::environments::l2f::StateLinearAcceleration<T_S, TI_S, NEXT_COMPONENT>& state, RNG& rng){
        using TI = typename DEVICE::index_t;
        sample_initial_state(device, env, parameters, static_cast<NEXT_COMPONENT&>(state), rng);
        for(TI i = 0; i < 3; i++){
            state.linear_acceleration[i] = 0;
        }
    }
    template<typename DEVICE, typename T_S, typename TI_S, typename SPEC, typename NEXT_COMPONENT, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT static void sample_initial_state(DEVICE& device, rl::environments::Multirotor<SPEC>& env, typename rl::environments::Multirotor<SPEC>::Parameters& parameters, typename rl::environments::l2f::StatePoseErrorIntegral<T_S, TI_S, NEXT_COMPONENT>& state, RNG& rng){
        sample_initial_state(device, env, parameters, static_cast<NEXT_COMPONENT&>(state), rng);
        state.position_integral = 0;
        state.orientation_integral = 0;
    }
    template<typename DEVICE, typename T_S, typename TI_S, typename SPEC, typename NEXT_COMPONENT, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT static void sample_initial_state(DEVICE& device, rl::environments::Multirotor<SPEC>& env, typename rl::environments::Multirotor<SPEC>::Parameters& parameters, typename rl::environments::l2f::StateRandomForce<T_S, TI_S, NEXT_COMPONENT>& state, RNG& rng){
        typename DEVICE::SPEC::RANDOM random_dev;
        using T = typename SPEC::T;
//        bool guidance = random::uniform_real_distribution(random_dev, (T)0, (T)1, rng) < parameters.mdp.init.guidance;
        sample_initial_state(device, env, parameters, static_cast<NEXT_COMPONENT&>(state), rng);
//        if(!guidance){
        {
            auto distribution = parameters.disturbances.random_force;
            state.force[0] = random::normal_distribution::sample(random_dev, (T)distribution.mean, (T)distribution.std, rng);
            state.force[1] = random::normal_distribution::sample(random_dev, (T)distribution.mean, (T)distribution.std, rng);
            state.force[2] = random::normal_distribution::sample(random_dev, (T)distribution.mean, (T)distribution.std, rng);
        }
        {
            auto distribution = parameters.disturbances.random_torque;
            state.torque[0] = random::normal_distribution::sample(random_dev, (T)distribution.mean, (T)distribution.std, rng);
            state.torque[1] = random::normal_distribution::sample(random_dev, (T)distribution.mean, (T)distribution.std, rng);
            state.torque[2] = random::normal_distribution::sample(random_dev, (T)distribution.mean, (T)distribution.std/100, rng);
        }
//        }
//        else{
//            state.force[0] = 0;
//            state.force[1] = 0;
//            state.force[2] = 0;
//            state.torque[0] = 0;
//            state.torque[1] = 0;
//            state.torque[2] = 0;
//        }

    }
    template<typename DEVICE, typename T, typename TI, typename SPEC, bool T_CLOSED_FORM, typename NEXT_COMPONENT, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT static void sample_initial_state(DEVICE& device, rl::environments::Multirotor<SPEC>& env, typename rl::environments::Multirotor<SPEC>::Parameters& parameters, typename rl::environments::l2f::StateRotors<T, TI, T_CLOSED_FORM, NEXT_COMPONENT>& state, RNG& rng){
        sample_initial_state(device, env, parameters, static_cast<NEXT_COMPONENT&>(state), rng);
        T min_rpm, max_rpm;
        if(parameters.mdp.init.relative_rpm){
            min_rpm = (parameters.mdp.init.min_rpm + 1)/2 * (parameters.dynamics.action_limit.max - parameters.dynamics.action_limit.min) + parameters.dynamics.action_limit.min;
            max_rpm = (parameters.mdp.init.max_rpm + 1)/2 * (parameters.dynamics.action_limit.max - parameters.dynamics.action_limit.min) + parameters.dynamics.action_limit.min;
        }
        else{
            min_rpm = parameters.mdp.init.min_rpm < 0 ? parameters.dynamics.action_limit.min : parameters.mdp.init.min_rpm;
            max_rpm = parameters.mdp.init.max_rpm < 0 ? parameters.dynamics.action_limit.max : parameters.mdp.init.max_rpm;
            if(max_rpm > parameters.dynamics.action_limit.max){
                max_rpm = parameters.dynamics.action_limit.max;
            }
            if(min_rpm > max_rpm){
                min_rpm = max_rpm;
            }
        }
        for(TI i = 0; i < 4; i++){
            state.rpm[i] = random::uniform_real_distribution(typename DEVICE::SPEC::RANDOM(), min_rpm, max_rpm, rng);
        }
    }
    template<typename DEVICE, typename T_S, typename TI_S, TI_S HISTORY_LENGTH, bool T_CLOSED_FORM, typename NEXT_COMPONENT, typename SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT static void sample_initial_state(DEVICE& device, rl::environments::Multirotor<SPEC>& env, typename rl::environments::Multirotor<SPEC>::Parameters& parameters, typename rl::environments::l2f::StateRotorsHistory<T_S, TI_S, HISTORY_LENGTH, T_CLOSED_FORM, NEXT_COMPONENT>& state, RNG& rng){
        using MULTIROTOR = rl::environments::Multirotor<SPEC>;
        using TI = typename DEVICE::index_t;
        sample_initial_state(device, env, parameters, static_cast<typename rl::environments::l2f::StateRotors<T_S, TI_S, T_CLOSED_FORM, NEXT_COMPONENT>&>(state), rng);
        state.current_step = 0;
        for(TI step_i = 0; step_i < HISTORY_LENGTH; step_i++){
            for(TI action_i = 0; action_i < MULTIROTOR::ACTION_DIM; action_i++){
                state.action_history[step_i][action_i] = (state.rpm[action_i] - parameters.dynamics.action_limit.min) / (parameters.dynamics.action_limit.max - parameters.dynamics.action_limit.min) * 2 - 1;
            }
        }
    }
    namespace rl::environments::l2f::observations{
        template<typename DEVICE, typename SPEC, typename STATE, typename OBSERVATION_TI, typename OBS_SPEC, typename RNG>
        RL_TOOLS_FUNCTION_PLACEMENT static void observe(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, typename rl::environments::Multirotor<SPEC>::Parameters& parameters, const STATE& state, rl::environments::l2f::observation::LastComponent<OBSERVATION_TI>, Matrix<OBS_SPEC>& observation, RNG& rng){
            static_assert(OBS_SPEC::COLS == 0);
            static_assert(OBS_SPEC::ROWS == 1);
        }
        template<typename DEVICE, typename SPEC, typename OBSERVATION_SPEC, typename OBS_SPEC, typename RNG>
        RL_TOOLS_FUNCTION_PLACEMENT static void observe(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, typename rl::environments::Multirotor<SPEC>::Parameters& parameters, const typename rl::environments::Multirotor<SPEC>::State& state, rl::environments::l2f::observation::PoseIntegral<OBSERVATION_SPEC>, Matrix<OBS_SPEC>& observation, RNG& rng){
            using T = typename SPEC::T;
            using TI = typename DEVICE::index_t;
            using OBSERVATION = rl::environments::l2f::observation::PoseIntegral<OBSERVATION_SPEC>;
            static_assert(OBS_SPEC::COLS >= OBSERVATION::CURRENT_DIM);
            static_assert(OBS_SPEC::ROWS == 1);
            set(observation, 0, 0, state.position_integral);
            set(observation, 0, 1, state.orientation_integral);
            auto current_observation = view(device, observation, matrix::ViewSpec<1, OBSERVATION::CURRENT_DIM>{}, 0, 0);
            auto next_observation = view(device, observation, matrix::ViewSpec<1, OBS_SPEC::COLS - OBSERVATION::CURRENT_DIM>{}, 0, OBSERVATION::CURRENT_DIM);
            observe(device, env, parameters, state, typename OBSERVATION::NEXT_COMPONENT{}, next_observation, rng);
        }
        template<typename DEVICE, typename SPEC, typename OBSERVATION_SPEC, typename OBS_SPEC, typename RNG>
        RL_TOOLS_FUNCTION_PLACEMENT static void observe(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, typename rl::environments::Multirotor<SPEC>::Parameters& parameters, const typename rl::environments::Multirotor<SPEC>::State& state, rl::environments::l2f::observation::Position<OBSERVATION_SPEC>, Matrix<OBS_SPEC>& observation, RNG& rng){
            using OBSERVATION = rl::environments::l2f::observation::Position<OBSERVATION_SPEC>;
            static_assert(OBS_SPEC::COLS >= OBSERVATION::CURRENT_DIM);
            static_assert(OBS_SPEC::ROWS == 1);
            using T = typename SPEC::T;
            using TI = typename DEVICE::index_t;

            for(TI i = 0; i < 3; i++){
                if constexpr(OBSERVATION_SPEC::PRIVILEGED && !SPEC::STATIC_PARAMETERS::PRIVILEGED_OBSERVATION_NOISE){
                    set(observation, 0, i, state.position[i]);
                }
                else{
                    T noise = random::normal_distribution::sample(typename DEVICE::SPEC::RANDOM{}, (T)0, parameters.mdp.observation_noise.position, rng);
                    set(observation, 0, i, state.position[i] + noise);
                }
            }
            auto next_observation = view(device, observation, matrix::ViewSpec<1, OBS_SPEC::COLS - OBSERVATION::CURRENT_DIM>{}, 0, OBSERVATION::CURRENT_DIM);
            observe(device, env, parameters, state, typename OBSERVATION::NEXT_COMPONENT{}, next_observation, rng);
        }
        template<typename DEVICE, typename SPEC, typename OBSERVATION_SPEC, typename OBS_SPEC, typename RNG>
        RL_TOOLS_FUNCTION_PLACEMENT static void observe(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, typename rl::environments::Multirotor<SPEC>::Parameters& parameters, const typename rl::environments::Multirotor<SPEC>::State& state, rl::environments::l2f::observation::OrientationQuaternion<OBSERVATION_SPEC>, Matrix<OBS_SPEC>& observation, RNG& rng){
            using T = typename SPEC::T;
            using TI = typename DEVICE::index_t;
            using OBSERVATION = rl::environments::l2f::observation::OrientationQuaternion<OBSERVATION_SPEC>;
            static_assert(OBS_SPEC::COLS >= OBSERVATION::CURRENT_DIM);
            static_assert(OBS_SPEC::ROWS == 1);
            for(TI i = 0; i < OBSERVATION::CURRENT_DIM; i++){
                if constexpr(OBSERVATION_SPEC::PRIVILEGED && !SPEC::STATIC_PARAMETERS::PRIVILEGED_OBSERVATION_NOISE){
                    set(observation, 0, i, state.orientation[i]);
                }
                else{
                    T noise = random::normal_distribution::sample(typename DEVICE::SPEC::RANDOM{}, (T)0, parameters.mdp.observation_noise.orientation, rng);
                    set(observation, 0, i, state.orientation[i] + noise);
                }
            }
            auto next_observation = view(device, observation, matrix::ViewSpec<1, OBS_SPEC::COLS - OBSERVATION::CURRENT_DIM>{}, 0, OBSERVATION::CURRENT_DIM);
            observe(device, env, parameters, state, typename OBSERVATION::NEXT_COMPONENT{}, next_observation, rng);
        }
        template<typename DEVICE, typename SPEC, typename OBSERVATION_SPEC, typename OBS_SPEC, typename RNG>
        RL_TOOLS_FUNCTION_PLACEMENT static void observe(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, typename rl::environments::Multirotor<SPEC>::Parameters& parameters, const typename rl::environments::Multirotor<SPEC>::State& state, rl::environments::l2f::observation::OrientationRotationMatrix<OBSERVATION_SPEC>, Matrix<OBS_SPEC>& observation, RNG& rng){
            using T = typename SPEC::T;
            using TI = typename DEVICE::index_t;
            using OBSERVATION = rl::environments::l2f::observation::OrientationRotationMatrix<OBSERVATION_SPEC>;
            static_assert(OBS_SPEC::COLS >= OBSERVATION::CURRENT_DIM);
            static_assert(OBS_SPEC::ROWS == 1);
            const typename SPEC::T* q = state.orientation;
            set(observation, 0, 0, (1 - 2*q[2]*q[2] - 2*q[3]*q[3]));
            set(observation, 0, 1, (    2*q[1]*q[2] - 2*q[0]*q[3]));
            set(observation, 0, 2, (    2*q[1]*q[3] + 2*q[0]*q[2]));
            set(observation, 0, 3, (    2*q[1]*q[2] + 2*q[0]*q[3]));
            set(observation, 0, 4, (1 - 2*q[1]*q[1] - 2*q[3]*q[3]));
            set(observation, 0, 5, (    2*q[2]*q[3] - 2*q[0]*q[1]));
            set(observation, 0, 6, (    2*q[1]*q[3] - 2*q[0]*q[2]));
            set(observation, 0, 7, (    2*q[2]*q[3] + 2*q[0]*q[1]));
            set(observation, 0, 8, (1 - 2*q[1]*q[1] - 2*q[2]*q[2]));
            if constexpr(!OBSERVATION_SPEC::PRIVILEGED || SPEC::STATIC_PARAMETERS::PRIVILEGED_OBSERVATION_NOISE){
                for(TI i = 0; i < OBSERVATION::CURRENT_DIM; i++){
                    T noise;
                    noise = random::normal_distribution::sample(typename DEVICE::SPEC::RANDOM(), (T)0, parameters.mdp.observation_noise.orientation, rng);
                    increment(observation, 0, i, noise);
                }
            }
            auto next_observation = view(device, observation, matrix::ViewSpec<1, OBS_SPEC::COLS - OBSERVATION::CURRENT_DIM>{}, 0, OBSERVATION::CURRENT_DIM);
            observe(device, env, parameters, state, typename OBSERVATION::NEXT_COMPONENT{}, next_observation, rng);
        }
        template<typename DEVICE, typename SPEC, typename OBSERVATION_SPEC, typename OBS_SPEC, typename RNG>
        RL_TOOLS_FUNCTION_PLACEMENT static void observe(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, typename rl::environments::Multirotor<SPEC>::Parameters& parameters, const typename rl::environments::Multirotor<SPEC>::State& state, rl::environments::l2f::observation::LinearVelocity<OBSERVATION_SPEC>, Matrix<OBS_SPEC>& observation, RNG& rng){
            using T = typename SPEC::T;
            using TI = typename DEVICE::index_t;
            using OBSERVATION = rl::environments::l2f::observation::LinearVelocity<OBSERVATION_SPEC>;
            static_assert(OBS_SPEC::COLS >= OBSERVATION::CURRENT_DIM);
            static_assert(OBS_SPEC::ROWS == 1);
            for(TI i = 0; i < OBSERVATION::CURRENT_DIM; i++){
                if constexpr(OBSERVATION_SPEC::PRIVILEGED && !SPEC::STATIC_PARAMETERS::PRIVILEGED_OBSERVATION_NOISE){
                    set(observation, 0, i, state.linear_velocity[i]);
                }
                else{
                    T noise = random::normal_distribution::sample(typename DEVICE::SPEC::RANDOM{}, (T)0, parameters.mdp.observation_noise.linear_velocity, rng);
                    set(observation, 0, i, state.linear_velocity[i] + noise);
                }
            }
            auto next_observation = view(device, observation, matrix::ViewSpec<1, OBS_SPEC::COLS - OBSERVATION::CURRENT_DIM>{}, 0, OBSERVATION::CURRENT_DIM);
            observe(device, env, parameters, state, typename OBSERVATION::NEXT_COMPONENT{}, next_observation, rng);
        }
        template<typename DEVICE, typename SPEC, typename OBSERVATION_SPEC, typename OBS_SPEC, typename RNG>
        RL_TOOLS_FUNCTION_PLACEMENT static void observe(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, typename rl::environments::Multirotor<SPEC>::Parameters& parameters, const typename rl::environments::Multirotor<SPEC>::State& state, rl::environments::l2f::observation::AngularVelocity<OBSERVATION_SPEC>, Matrix<OBS_SPEC>& observation, RNG& rng){
            using T = typename SPEC::T;
            using TI = typename DEVICE::index_t;
            using OBSERVATION = rl::environments::l2f::observation::AngularVelocity<OBSERVATION_SPEC>;
            static_assert(OBS_SPEC::COLS >= OBSERVATION::CURRENT_DIM);
            static_assert(OBS_SPEC::ROWS == 1);
            for(TI i = 0; i < OBSERVATION::CURRENT_DIM; i++){
                if constexpr(OBSERVATION_SPEC::PRIVILEGED && !SPEC::STATIC_PARAMETERS::PRIVILEGED_OBSERVATION_NOISE){
                    set(observation, 0, i, state.angular_velocity[i]);
                }
                else{
                    T noise = random::normal_distribution::sample(typename DEVICE::SPEC::RANDOM{}, (T)0, parameters.mdp.observation_noise.angular_velocity, rng);
                    set(observation, 0, i, state.angular_velocity[i] + noise);
                }
            }
            auto next_observation = view(device, observation, matrix::ViewSpec<1, OBS_SPEC::COLS - OBSERVATION::CURRENT_DIM>{}, 0, OBSERVATION::CURRENT_DIM);
            observe(device, env, parameters, state, typename OBSERVATION::NEXT_COMPONENT{}, next_observation, rng);
        }
        template<typename DEVICE, typename SPEC, typename OBSERVATION_SPEC, typename OBS_SPEC, typename RNG>
        RL_TOOLS_FUNCTION_PLACEMENT static void observe(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, typename rl::environments::Multirotor<SPEC>::Parameters& parameters, const typename rl::environments::Multirotor<SPEC>::State& state, rl::environments::l2f::observation::IMUAccelerometer<OBSERVATION_SPEC>, Matrix<OBS_SPEC>& observation, RNG& rng){
            using T = typename SPEC::T;
            using TI = typename DEVICE::index_t;
            using OBSERVATION = rl::environments::l2f::observation::IMUAccelerometer<OBSERVATION_SPEC>;
            static_assert(OBS_SPEC::COLS >= OBSERVATION::CURRENT_DIM);
            static_assert(OBS_SPEC::ROWS == 1);

//            observation = R_global_to_local * (acceleration - gravity)

            T conjugate_orientation[4];
            conjugate_orientation[0] = state.orientation[0];
            conjugate_orientation[1] = -state.orientation[1];
            conjugate_orientation[2] = -state.orientation[2];
            conjugate_orientation[3] = -state.orientation[3];

            T acceleration_observation_global[3];

            acceleration_observation_global[0] = state.linear_acceleration[0] - parameters.dynamics.gravity[0];
            acceleration_observation_global[1] = state.linear_acceleration[1] - parameters.dynamics.gravity[1];
            acceleration_observation_global[2] = state.linear_acceleration[2] - parameters.dynamics.gravity[2];

            T acceleration_observation[3];
            rotate_vector_by_quaternion<DEVICE, T>(conjugate_orientation, acceleration_observation_global, acceleration_observation);

            for(TI i = 0; i < OBSERVATION::CURRENT_DIM; i++){
                if constexpr(OBSERVATION_SPEC::PRIVILEGED && !SPEC::STATIC_PARAMETERS::PRIVILEGED_OBSERVATION_NOISE){
                    set(observation, 0, i, acceleration_observation[i]);
                }
                else{
                    T noise = random::normal_distribution::sample(typename DEVICE::SPEC::RANDOM{}, (T)0, parameters.mdp.observation_noise.imu_acceleration, rng);
                    set(observation, 0, i, acceleration_observation[i] + noise);
                }
            }
            auto next_observation = view(device, observation, matrix::ViewSpec<1, OBS_SPEC::COLS - OBSERVATION::CURRENT_DIM>{}, 0, OBSERVATION::CURRENT_DIM);
            observe(device, env, parameters, state, typename OBSERVATION::NEXT_COMPONENT{}, next_observation, rng);
        }
        template<typename DEVICE, typename SPEC, typename OBSERVATION_SPEC, typename OBS_SPEC, typename RNG>
        RL_TOOLS_FUNCTION_PLACEMENT static void observe(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, typename rl::environments::Multirotor<SPEC>::Parameters& parameters, const typename rl::environments::Multirotor<SPEC>::State& state, rl::environments::l2f::observation::Magnetometer<OBSERVATION_SPEC>, Matrix<OBS_SPEC>& observation, RNG& rng){
            using T = typename SPEC::T;
            using TI = typename DEVICE::index_t;
            using OBSERVATION = rl::environments::l2f::observation::Magnetometer<OBSERVATION_SPEC>;
            static_assert(OBS_SPEC::COLS >= OBSERVATION::CURRENT_DIM);
            static_assert(OBS_SPEC::ROWS == 1);

//            projecting the body x axis to the global xy plane


            T body_x_axis_local[3] = {1, 0, 0};
            T body_x_axis_world[3];
            rotate_vector_by_quaternion<DEVICE, T>(state.orientation, body_x_axis_local, body_x_axis_world);

            T pre_sqrt = body_x_axis_world[0]*body_x_axis_world[0] + body_x_axis_world[1]*body_x_axis_world[1];
            if(pre_sqrt > 0.01 * 0.01){
                T norm = math::sqrt(device.math, pre_sqrt);
                body_x_axis_world[0] /= norm;
                body_x_axis_world[1] /= norm;
            }
            else{
                body_x_axis_world[0] = 0;
                body_x_axis_world[1] = 0;
            }

            for(TI i = 0; i < OBSERVATION::CURRENT_DIM; i++){
                if constexpr(OBSERVATION_SPEC::PRIVILEGED && !SPEC::STATIC_PARAMETERS::PRIVILEGED_OBSERVATION_NOISE){
                    set(observation, 0, i, body_x_axis_world[i]);
                }
                else{
                    T noise = random::normal_distribution::sample(typename DEVICE::SPEC::RANDOM{}, (T)0, parameters.mdp.observation_noise.imu_acceleration, rng);
                    set(observation, 0, i, body_x_axis_world[i] + noise);
                }
            }
            auto next_observation = view(device, observation, matrix::ViewSpec<1, OBS_SPEC::COLS - OBSERVATION::CURRENT_DIM>{}, 0, OBSERVATION::CURRENT_DIM);
            observe(device, env, parameters, state, typename OBSERVATION::NEXT_COMPONENT{}, next_observation, rng);
        }
        template<typename DEVICE, typename SPEC, typename OBSERVATION_SPEC, typename OBS_SPEC, typename RNG>
        RL_TOOLS_FUNCTION_PLACEMENT static void observe(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, typename rl::environments::Multirotor<SPEC>::Parameters& parameters, const typename rl::environments::Multirotor<SPEC>::State& state, rl::environments::l2f::observation::RotorSpeeds<OBSERVATION_SPEC>, Matrix<OBS_SPEC>& observation, RNG& rng){
            using T = typename SPEC::T;
            using TI = typename DEVICE::index_t;
            using OBSERVATION = rl::environments::l2f::observation::RotorSpeeds<OBSERVATION_SPEC>;
            static_assert(OBS_SPEC::COLS >= OBSERVATION::CURRENT_DIM);
            static_assert(OBS_SPEC::ROWS == 1);
            for(TI action_i = 0; action_i < OBSERVATION::CURRENT_DIM; action_i++){
                T action_value = (state.rpm[action_i] - parameters.dynamics.action_limit.min)/(parameters.dynamics.action_limit.max - parameters.dynamics.action_limit.min) * 2 - 1;
                set(observation, 0, action_i, action_value);
            }
            auto next_observation = view(device, observation, matrix::ViewSpec<1, OBS_SPEC::COLS - OBSERVATION::CURRENT_DIM>{}, 0, OBSERVATION::CURRENT_DIM);
            observe(device, env, parameters, state, typename OBSERVATION::NEXT_COMPONENT{}, next_observation, rng);
        }
        template<typename DEVICE, typename SPEC, typename OBSERVATION_SPEC, typename OBS_SPEC, typename RNG>
        RL_TOOLS_FUNCTION_PLACEMENT static void observe(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, typename rl::environments::Multirotor<SPEC>::Parameters& parameters, const typename rl::environments::Multirotor<SPEC>::State& state, rl::environments::l2f::observation::ActionHistory<OBSERVATION_SPEC>, Matrix<OBS_SPEC>& observation, RNG& rng){
            using T = typename SPEC::T;
            using TI = typename DEVICE::index_t;
            using OBSERVATION = rl::environments::l2f::observation::ActionHistory<OBSERVATION_SPEC>;
            static_assert(OBS_SPEC::COLS >= OBSERVATION::CURRENT_DIM);
            static_assert(OBS_SPEC::ROWS == 1);
            static_assert(rl::environments::Multirotor<SPEC>::State::HISTORY_LENGTH == OBSERVATION::HISTORY_LENGTH);
            static_assert(rl::environments::Multirotor<SPEC>::State::ACTION_DIM == OBSERVATION::ACTION_DIM);
            static_assert(rl::environments::Multirotor<SPEC>::ACTION_DIM == OBSERVATION::ACTION_DIM);
            TI current_step = state.current_step;
            for(TI step_i = 0; step_i < OBSERVATION::HISTORY_LENGTH; step_i++){
                TI base = step_i*OBSERVATION::ACTION_DIM;
                for(TI action_i = 0; action_i < OBSERVATION::ACTION_DIM; action_i++){
                    set(observation, 0, base + action_i, state.action_history[current_step][action_i]);
                }
                current_step = (current_step + 1) % OBSERVATION::HISTORY_LENGTH;
            }
            auto next_observation = view(device, observation, matrix::ViewSpec<1, OBS_SPEC::COLS - OBSERVATION::CURRENT_DIM>{}, 0, OBSERVATION::CURRENT_DIM);
            observe(device, env, parameters, state, typename OBSERVATION::NEXT_COMPONENT{}, next_observation, rng);
        }
        template<typename DEVICE, typename SPEC, typename OBSERVATION_SPEC, typename OBS_SPEC, typename RNG>
        RL_TOOLS_FUNCTION_PLACEMENT static void observe(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, typename rl::environments::Multirotor<SPEC>::Parameters& parameters, const typename rl::environments::Multirotor<SPEC>::State& state, rl::environments::l2f::observation::RandomForce<OBSERVATION_SPEC>, Matrix<OBS_SPEC>& observation, RNG& rng){
            using T = typename SPEC::T;
            using TI = typename DEVICE::index_t;
            using OBSERVATION = rl::environments::l2f::observation::RandomForce<OBSERVATION_SPEC>;
            static_assert(OBS_SPEC::COLS >= OBSERVATION::CURRENT_DIM);
            static_assert(OBS_SPEC::ROWS == 1);
            for(TI i = 0; i < 3; i++){
                set(observation, 0, i, state.force[i]);
                set(observation, 0, 3 + i, state.torque[i]);
            }
            auto next_observation = view(device, observation, matrix::ViewSpec<1, OBS_SPEC::COLS - OBSERVATION::CURRENT_DIM>{}, 0, OBSERVATION::CURRENT_DIM);
            observe(device, env, parameters, state, typename OBSERVATION::NEXT_COMPONENT{}, next_observation, rng);
        }
    }
    template<typename DEVICE, typename SPEC, typename STATE, typename OBSERVATION, typename OBS_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT static void observe(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, typename rl::environments::Multirotor<SPEC>::Parameters& parameters, const STATE& state, const OBSERVATION& observation_type, Matrix<OBS_SPEC>& observation, RNG& rng){
        using ENVIRONMENT = rl::environments::Multirotor<SPEC>;
        static_assert(OBS_SPEC::COLS == OBSERVATION::DIM);
        static_assert(OBS_SPEC::ROWS == 1);
        rl::environments::l2f::observations::observe(device, env, parameters, state, observation_type, observation, rng);
    }
    template<typename DEVICE, typename SPEC, typename ACTION_SPEC, typename T_S, typename TI_S, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void post_integration(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, typename rl::environments::Multirotor<SPEC>::Parameters& parameters, const typename rl::environments::l2f::StateBase<T_S, TI_S>& state, const Matrix<ACTION_SPEC>& action, typename rl::environments::l2f::StateBase<T_S, TI_S>& next_state, RNG& rng) {
        using T = T_S;
        using TI = TI_S;
        T quaternion_norm = 0;
        for(TI state_i = 0; state_i < 4; state_i++){
            quaternion_norm += next_state.orientation[state_i] * next_state.orientation[state_i];
        }
        quaternion_norm = math::sqrt(device.math, quaternion_norm);
        for(TI state_i = 0; state_i < 4; state_i++){
            next_state.orientation[state_i] /= quaternion_norm;
        }

    }
    template<typename DEVICE, typename SPEC, typename ACTION_SPEC, typename T_S, typename TI_S, typename NEXT_COMPONENT, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void post_integration(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, typename rl::environments::Multirotor<SPEC>::Parameters& parameters, const typename rl::environments::l2f::StateLinearAcceleration<T_S, TI_S, NEXT_COMPONENT>& state, const Matrix<ACTION_SPEC>& action, typename rl::environments::l2f::StateLinearAcceleration<T_S, TI_S, NEXT_COMPONENT>& next_state, RNG& rng) {
        using T = T_S;
        using TI = TI_S;
        post_integration(device, env, parameters, static_cast<const NEXT_COMPONENT&>(state), action, static_cast<NEXT_COMPONENT&>(next_state), rng);
        for(TI state_i = 0; state_i < 3; state_i++){
            next_state.linear_acceleration[state_i] = (next_state.linear_velocity[state_i] - state.linear_velocity[state_i])/parameters.integration.dt;
        }
    }
//    template<typename DEVICE, typename SPEC, typename T, typename TI, typename NEXT_COMPONENT>
//    RL_TOOLS_FUNCTION_PLACEMENT void post_integration(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, rl::environments::l2f::StateRotors<T, TI, NEXT_COMPONENT>& state) {
    template<typename DEVICE, typename SPEC, typename ACTION_SPEC, typename T_S, typename TI_S, bool T_CLOSED_FORM, typename NEXT_COMPONENT, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void post_integration(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, typename rl::environments::Multirotor<SPEC>::Parameters& parameters, const typename rl::environments::l2f::StateRotors<T_S, TI_S, T_CLOSED_FORM, NEXT_COMPONENT>& state, const Matrix<ACTION_SPEC>& action, typename rl::environments::l2f::StateRotors<T_S, TI_S, T_CLOSED_FORM, NEXT_COMPONENT>& next_state, RNG& rng) {
        post_integration(device, env, parameters, static_cast<const NEXT_COMPONENT&>(state), action, static_cast<NEXT_COMPONENT&>(next_state), rng);
        using MULTIROTOR = rl::environments::Multirotor<SPEC>;
        using T = T_S;
        for(typename DEVICE::index_t rpm_i = 0; rpm_i < MULTIROTOR::ACTION_DIM; rpm_i++){
            if constexpr(T_CLOSED_FORM) {
                T setpoint_clamped = math::clamp(typename DEVICE::SPEC::MATH{}, get(action, 0, rpm_i), parameters.dynamics.action_limit.min, parameters.dynamics.action_limit.max);
                T alpha = math::exp(device.math, - parameters.integration.dt / parameters.dynamics.motor_time_constant);
                next_state.rpm[rpm_i] = alpha * state.rpm[rpm_i] + (1 - alpha) * setpoint_clamped;
            }
            else {
                next_state.rpm[rpm_i] = math::clamp(typename DEVICE::SPEC::MATH{}, next_state.rpm[rpm_i], parameters.dynamics.action_limit.min, parameters.dynamics.action_limit.max);
            }
        }
    }
    template<typename DEVICE, typename SPEC, typename ACTION_SPEC, typename T_S, typename TI_S, typename NEXT_COMPONENT, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void post_integration(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, typename rl::environments::Multirotor<SPEC>::Parameters& parameters, const typename rl::environments::l2f::StateRandomForce<T_S, TI_S, NEXT_COMPONENT>& state, const Matrix<ACTION_SPEC>& action, typename rl::environments::l2f::StateRandomForce<T_S, TI_S, NEXT_COMPONENT>& next_state, RNG& rng) {
        post_integration(device, env, parameters, static_cast<const NEXT_COMPONENT&>(state), action, static_cast<NEXT_COMPONENT&>(next_state), rng);
        next_state.force[0] = state.force[0];
        next_state.force[1] = state.force[1];
        next_state.force[2] = state.force[2];
        next_state.torque[0] = state.torque[0];
        next_state.torque[1] = state.torque[1];
        next_state.torque[2] = state.torque[2];
    }
    template<typename DEVICE, typename T_S, typename TI_S, typename NEXT_STATE_COMPONENT, TI_S HISTORY_LENGTH, bool T_CLOSED_FORM, typename SPEC, typename ACTION_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void post_integration(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, typename rl::environments::Multirotor<SPEC>::Parameters& parameters, const typename rl::environments::l2f::StateRotorsHistory<T_S, TI_S, HISTORY_LENGTH, T_CLOSED_FORM, NEXT_STATE_COMPONENT>& state, const Matrix<ACTION_SPEC>& action, typename rl::environments::l2f::StateRotorsHistory<T_S, TI_S, HISTORY_LENGTH, T_CLOSED_FORM, NEXT_STATE_COMPONENT>& next_state, RNG& rng) {
        using MULTIROTOR = rl::environments::Multirotor<SPEC>;
        using TI = typename DEVICE::index_t;
        static_assert(ACTION_SPEC::COLS == MULTIROTOR::ACTION_DIM);
        post_integration(device, env, parameters, static_cast<const rl::environments::l2f::StateRotors<T_S, TI_S, T_CLOSED_FORM, NEXT_STATE_COMPONENT>&>(state), action, static_cast<rl::environments::l2f::StateRotors<T_S, TI_S, T_CLOSED_FORM, NEXT_STATE_COMPONENT>&>(next_state), rng);
        if constexpr(HISTORY_LENGTH > 0){
            // for(TI step_i = 0; step_i < HISTORY_LENGTH-1; step_i++){
            //     for(TI action_i = 0; action_i < MULTIROTOR::ACTION_DIM; action_i++){
            //         next_state.action_history[step_i][action_i] = state.action_history[step_i+1][action_i];
            //     }
            // }
            // for(TI action_i = 0; action_i < MULTIROTOR::ACTION_DIM; action_i++){
            //     next_state.action_history[HISTORY_LENGTH-1][action_i] = get(action, 0, action_i);
            // }
            TI current_step = state.current_step;
            for(TI action_i = 0; action_i < MULTIROTOR::ACTION_DIM; action_i++){
                next_state.action_history[current_step][action_i] = get(action, 0, action_i);
            }
            next_state.current_step = (state.current_step + 1) % HISTORY_LENGTH;
        }
    }
//    template<typename DEVICE, typename SPEC, typename ACTION_SPEC, typename T_S, typename TI_S, typename STATE, typename RNG>
//    RL_TOOLS_FUNCTION_PLACEMENT void post_integration(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, const STATE& state, const Matrix<ACTION_SPEC>& action, STATE& next_state, RNG& rng) {
//        static_assert(!STATE::REQUIRES_INTEGRATION);
//        post_integration(device, env, static_cast<typename STATE::NEXT_COMPONENT&>(state), action, static_cast<typename STATE::NEXT_COMPONENT&>(next_state), rng);
//    }
    // todo: make state const again
    template<typename DEVICE, typename SPEC, typename ACTION_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT static typename SPEC::T step(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, typename rl::environments::Multirotor<SPEC>::Parameters& parameters, const typename rl::environments::Multirotor<SPEC>::State& state, const Matrix<ACTION_SPEC>& action, typename rl::environments::Multirotor<SPEC>::State& next_state, RNG& rng) {
        using STATE = typename rl::environments::Multirotor<SPEC>::State;
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
        utils::integrators::rk4  <DEVICE, typename SPEC::T, typename SPEC::PARAMETERS, STATE, ACTION_DIM, rl::environments::l2f::multirotor_dynamics_dispatch<DEVICE, typename SPEC::T, typename SPEC::PARAMETERS, STATE>>(device, parameters, state, action_scaled, parameters.integration.dt, next_state);
//        utils::integrators::euler<DEVICE, typename SPEC::T, typename SPEC::PARAMETERS, STATE, ACTION_DIM, rl::environments::l2f::multirotor_dynamics_dispatch<DEVICE, typename SPEC::T, typename SPEC::PARAMETERS, STATE>>(device, parameters, state, action_scaled, parameters.integration.dt, next_state);

        post_integration(device, env, parameters, state, action, next_state, rng);

//        utils::assert_exit(device, !math::is_nan(device.math, next_state.position_integral), "pi nan");
//        utils::assert_exit(device, !math::is_nan(device.math, next_state.orientation_integral), "oi nan");
//        utils::assert_exit(device, !is_nan(device, action), "action nan");
//        utils::assert_exit(device, !is_nan(device, next_state), "nan");
        return parameters.integration.dt;
    }

    template<typename DEVICE, typename SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT static bool terminated(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, const typename rl::environments::Multirotor<SPEC>::Parameters& parameters, const typename rl::environments::Multirotor<SPEC>::State& state, RNG& rng){
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
    template<typename DEVICE, typename SPEC, typename ACTION_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT static typename SPEC::T reward(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, typename rl::environments::Multirotor<SPEC>::Parameters& parameters, const typename rl::environments::Multirotor<SPEC>::State& state, const Matrix<ACTION_SPEC>& action, const typename rl::environments::Multirotor<SPEC>::State& next_state, RNG& rng) {
        return rl::environments::l2f::parameters::reward_functions::reward(device, env, parameters, parameters.mdp.reward, state, action, next_state, rng);
    }
    template<typename DEVICE, typename SPEC, typename ACTION_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void log_reward(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, typename rl::environments::Multirotor<SPEC>::Parameters& parameters, const typename rl::environments::Multirotor<SPEC>::State& state, const Matrix<ACTION_SPEC>& action, const typename rl::environments::Multirotor<SPEC>::State& next_state, RNG& rng, typename DEVICE::index_t cadence = 1) {
        rl::environments::l2f::parameters::reward_functions::log_reward(device, env, parameters, parameters.mdp.reward, state, action, next_state, rng, cadence);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#include "parameters/default.h"

#endif